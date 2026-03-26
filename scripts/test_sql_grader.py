from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from sql_rag import SqlKnowledgeRetriever, format_rag_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for the SQL grading LoRA model.")
    parser.add_argument(
        "--base_model",
        type=str,
        default="models/Qwen2.5-Coder-1.5B-Instruct",
        help="Base model path.",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="outputs/qwen2.5-coder-1.5b-sql-grader-lora",
        help="LoRA adapter path.",
    )
    parser.add_argument(
        "--dataset_file",
        type=Path,
        default=Path("data/processed/spider_sql_grading_sft/validation.jsonl"),
        help="Validation or test jsonl file with messages records.",
    )
    parser.add_argument(
        "--sample_indices",
        type=str,
        default="0,1,2",
        help="Comma-separated sample indices to test.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=384, help="Generation length.")
    parser.add_argument(
        "--use_rag",
        action="store_true",
        help="Enable local RAG retrieval before generation.",
    )
    parser.add_argument(
        "--knowledge_base",
        type=Path,
        default=Path("data/knowledge/sql_knowledge_base.jsonl"),
        help="Knowledge base used when --use_rag is enabled.",
    )
    parser.add_argument("--top_k", type=int, default=4, help="Number of retrieved knowledge items.")
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Inference dtype.",
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


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def parse_indices(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    dtype = resolve_torch_dtype(args.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    device = next(model.parameters()).device
    records = load_records(args.dataset_file)
    sample_indices = parse_indices(args.sample_indices)
    retriever = SqlKnowledgeRetriever.from_jsonl(args.knowledge_base) if args.use_rag else None

    for sample_index in sample_indices:
        record = records[sample_index]
        prompt_messages = record["messages"][:-1]
        gold_answer = record["messages"][-1]["content"]
        retrieved_docs = []
        if retriever is not None:
            prompt_messages[0]["content"] = (
                prompt_messages[0]["content"]
                + "\n\n如果提供了参考知识，请把它们当作辅助依据，但必须先识别学生 SQL 的主要错误，只做最小必要修改，不要改动已经正确的部分，也不要凭空引入新的错误类型。"
            )
            retrieved_docs = retriever.search(
                question=record.get("question", ""),
                schema_text=record.get("schema_text", ""),
                wrong_sql=record.get("wrong_sql", ""),
                top_k=args.top_k,
            )
            rag_context = format_rag_context(retrieved_docs)
            if rag_context:
                user_content = prompt_messages[-1]["content"]
                user_content = (
                    user_content
                    + "\n\n可参考的 SQL 知识：\n"
                    + rag_context
                    + "\n\n请优先参考这些知识来判断错误原因，但最终答案仍要结合题目、Schema 和学生 SQL。只修主要错误，不要改动已经正确的 SQL 片段。"
                )
                prompt_messages = [
                    prompt_messages[0],
                    {"role": "user", "content": user_content},
                ]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        print("=" * 100)
        print(f"sample_index: {sample_index}")
        print(f"id: {record.get('id', '<unknown>')}")
        print(f"error_type: {record.get('error_type', '<unknown>')}")
        print(f"question: {record.get('question', '<unknown>')}")
        print(f"wrong_sql: {record.get('wrong_sql', '<unknown>')}")
        print(f"correct_sql: {record.get('correct_sql', '<unknown>')}")
        if retrieved_docs:
            print("retrieved_knowledge:")
            for item in retrieved_docs:
                print(f"- {item['title']} (score={item['score']:.4f})")
        print("-" * 100)
        print("gold_answer:")
        print(gold_answer)
        print("-" * 100)
        print("model_output:")
        print(prediction)
        print()


if __name__ == "__main__":
    main()
