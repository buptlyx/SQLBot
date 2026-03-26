from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Qwen LoRA model for SQL grading.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="models/Qwen2.5-Coder-1.5B-Instruct",
        help="Base instruct model used for LoRA fine-tuning.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/processed/spider_sql_grading_sft/train.jsonl",
        help="Training data in json/jsonl format with a messages field.",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default="data/processed/spider_sql_grading_sft/validation.jsonl",
        help="Validation data in json/jsonl format with a messages field.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/qwen2.5-coder-1.5b-sql-grader-lora",
        help="Directory where checkpoints and adapters are saved.",
    )
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--num_train_epochs", type=float, default=3.0, help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Train batch size.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Eval batch size.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Gradient accumulation steps.",
    )
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps.")
    parser.add_argument("--save_steps", type=int, default=200, help="Checkpoint save steps.")
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluation steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Compute dtype for model loading.",
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target modules, or all-linear.",
    )
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Enable 4-bit loading for a QLoRA-style setup if bitsandbytes is available.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce VRAM usage.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume training from a checkpoint directory.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints to keep.",
    )
    return parser.parse_args()


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_quantization_config(args: argparse.Namespace, torch_dtype: torch.dtype):
    if not args.use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch_dtype,
    )


def load_model(args: argparse.Namespace, torch_dtype: torch.dtype):
    quantization_config = build_quantization_config(args, torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map="auto" if args.use_4bit else None,
        low_cpu_mem_usage=True,
    )

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    target_modules: str | list[str]
    if args.target_modules.strip() == "all-linear":
        target_modules = "all-linear"
    else:
        target_modules = [module.strip() for module in args.target_modules.split(",") if module.strip()]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def tokenize_example(example: dict, tokenizer, max_length: int) -> dict:
    messages = example["messages"]
    prompt_messages = messages[:-1]
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_ids = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    if not full_ids:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    if full_ids[-1] != tokenizer.eos_token_id:
        full_ids = full_ids[: max_length - 1] + [tokenizer.eos_token_id]

    prompt_length = min(len(prompt_ids), len(full_ids))
    labels = full_ids.copy()
    labels[:prompt_length] = [-100] * prompt_length

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def has_supervision(example: dict) -> bool:
    return any(label != -100 for label in example["labels"])


def main() -> None:
    args = parse_args()
    torch_dtype = resolve_torch_dtype(args.torch_dtype)
    tokenizer = load_tokenizer(args.model_name_or_path)
    model = load_model(args, torch_dtype)

    data_files = {"train": args.train_file}
    if args.eval_file and Path(args.eval_file).exists():
        data_files["validation"] = args.eval_file
    dataset_dict = load_dataset("json", data_files=data_files)

    column_names = dataset_dict["train"].column_names
    tokenized = dataset_dict.map(
        lambda example: tokenize_example(example, tokenizer, args.max_length),
        remove_columns=column_names,
        desc="Tokenizing chat-format SQL grading data",
    )
    tokenized = tokenized.filter(has_supervision, desc="Filtering truncated or empty samples")

    training_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "lr_scheduler_type": "cosine",
        "bf16": torch_dtype == torch.bfloat16,
        "fp16": torch_dtype == torch.float16,
        "optim": "paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        "report_to": "none",
        "seed": args.seed,
        "remove_unused_columns": False,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": "validation" in tokenized,
    }
    eval_strategy_value = "steps" if "validation" in tokenized else "no"
    signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in signature.parameters:
        training_kwargs["evaluation_strategy"] = eval_strategy_value
    else:
        training_kwargs["eval_strategy"] = eval_strategy_value

    training_args = TrainingArguments(**training_kwargs)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"] if "validation" in tokenized else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
