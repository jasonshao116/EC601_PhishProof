import os
import argparse
import pandas as pd
from datasets import Dataset
import torch
from transformers import TrainingArguments

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

def build_argparser():
    ap = argparse.ArgumentParser(description="LoRA fine-tuning for phishing classification")
    ap.add_argument("--csv_path", type=str, default="spam_dataset.csv",
                    help="Path to CSV with columns: message_content, is_spam (0/1)")
    ap.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                    help="HF model id of base causal LM")
    ap.add_argument("--output_dir", type=str, default="phishsense_lora_adapter",
                    help="Where to save the LoRA adapter")
    ap.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    ap.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    ap.add_argument("--batch_size", type=int, default=2, help="Per-device train batch size")
    ap.add_argument("--eval_batch_size", type=int, default=2, help="Per-device eval batch size")
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8,
                    help="To simulate larger batches")
    ap.add_argument("--max_length", type=int, default=1024, help="Max sequence length")
    ap.add_argument("--use_4bit", action="store_true", help="Load base model in 4-bit (bitsandbytes)")
    ap.add_argument("--hf_token", type=str, default=None, help="HF token if the base model is gated")
    ap.add_argument("--save_steps", type=int, default=500, help="Save every N steps")
    ap.add_argument("--eval_ratio", type=float, default=0.1, help="Eval split ratio")
    return ap

PROMPT_PREFIX = (
    "Classify the following text as phishing or not. "
    "Respond with 'TRUE' or 'FALSE':\n\n"
)

def make_prompt(text: str, label_bool: int):
    label = "TRUE" if int(label_bool) == 1 else "FALSE"
    return f"{PROMPT_PREFIX}{text}\nAnswer: {label}"

def main():
    args = build_argparser().parse_args()

    # -------- Load & check data --------
    df = pd.read_csv(args.csv_path)
    if "message_content" not in df.columns or "is_spam" not in df.columns:
        raise ValueError("CSV must contain 'message_content' and 'is_spam' columns.")

    # Drop empty rows
    df = df.dropna(subset=["message_content", "is_spam"]).reset_index(drop=True)

    # Build supervised prompts
    df["prompt"] = df.apply(lambda r: make_prompt(r["message_content"], r["is_spam"]), axis=1)

    # Split train/eval
    ds = Dataset.from_pandas(df[["prompt"]])
    ds = ds.train_test_split(test_size=args.eval_ratio, seed=42)

    # -------- Tokenizer --------
    tok = AutoTokenizer.from_pretrained(args.base_model, token=args.hf_token, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # typical for LLaMA family

    def tokenize(batch):
        return tok(
            batch["prompt"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    ds_tok = ds.map(tokenize, batched=True, remove_columns=["prompt"])

    # -------- Base model (with optional 4-bit) --------
    load_kwargs = {}
    if args.use_4bit:
        # requires: pip install bitsandbytes accelerate
        load_kwargs.update(
            dict(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        token=args.hf_token,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        **load_kwargs,
    )

    if args.use_4bit:
        base_model = prepare_model_for_kbit_training(base_model)

    # -------- LoRA config --------
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )

    model = get_peft_model(base_model, lora_cfg)

    # -------- Training setup --------
    data_collator = DataCollatorForLanguageModeling(tok, mlm=False)

    common_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        logging_steps=25,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,  # prefer bf16 when available
        report_to="none",
        remove_unused_columns=False,  # important for causal LM
    )

    # Try modern strategy args; fall back if this build doesn't support them
    try:
        args_train = TrainingArguments(
            evaluation_strategy="epoch",
            save_strategy="steps",
            **common_kwargs,
            dataloader_pin_memory=False,   # removes warning
            gradient_checkpointing=True,   # saves memory on MPS
        )
    except TypeError:
        # Old API: drop strategy args; we'll just eval manually later
        args_train = TrainingArguments(**common_kwargs)

    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["test"],
        data_collator=data_collator,
        tokenizer=tok,
    )

    trainer.train()

    # Ensure we evaluate once at the end even if evaluation_strategy wasn't accepted
    try:
        eval_metrics = trainer.evaluate()
        print("Final eval metrics:", eval_metrics)
    except Exception as e:
        print("Evaluation skipped or failed:", e)

    # -------- Save adapter --------
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)

    print(f"\n✅ Finished. LoRA adapter saved to: {args.output_dir}\n")
    print("Use it like:\n"
          "  base = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')\n"
          "  model = PeftModel.from_pretrained(base, '<output_dir>')\n")

if __name__ == "__main__":
    main()
