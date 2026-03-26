from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.sql_rag import SqlKnowledgeRetriever, format_rag_context


SYSTEM_PROMPT = """你是数据库课程中的 SQL 智能批阅助手。
你需要根据题目、相关表结构和学生提交的 SQL，输出标准答案以及清晰的纠错解析。
如果提供了参考知识，请把它们当作辅助依据，但必须先识别学生 SQL 的主要错误，只做最小必要修改，不要改动已经正确的部分，也不要凭空引入新的错误类型。
请始终使用下面的固定格式作答：

正确SQL：
```sql
...
```

纠错解析：
错误类型：...
错误定位：...
原因说明：...
修改建议：...
"""

USER_TEMPLATE = """请批阅下面这道 SQL 练习。

题目：
{question}

数据库 ID：
{db_id}

相关 Schema：
{schema_text}

学生 SQL：
```sql
{wrong_sql}
```"""

DEFAULT_BASE_MODEL = "models/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_ADAPTER = "outputs/qwen2.5-coder-1.5b-sql-grader-lora"
DEFAULT_DATASET = "data/processed/spider_sql_grading_sft/validation.jsonl"
DEFAULT_KNOWLEDGE_BASE = "data/knowledge/sql_knowledge_base.jsonl"


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


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_showcase_presets(dataset_path: Path) -> dict[str, dict]:
    records = load_jsonl(dataset_path)
    preset_map = {
        "演示 1: 聚合函数错误": 0,
        "演示 2: GROUP BY 缺失": 19,
        "演示 3: JOIN 条件遗漏": 38,
    }
    presets = {}
    for label, index in preset_map.items():
        record = records[index]
        presets[label] = {
            "db_id": record["db_id"],
            "question": record["question"],
            "schema_text": record["schema_text"],
            "wrong_sql": record["wrong_sql"],
            "correct_sql": record["correct_sql"],
            "gold_answer": record["messages"][-1]["content"],
            "error_type": record["error_type"],
            "record_id": record["id"],
        }
    return presets


@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(base_model_path: str, adapter_path: str, dtype_name: str):
    dtype = resolve_torch_dtype(dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner=False)
def load_retriever(knowledge_base_path: str):
    return SqlKnowledgeRetriever.from_jsonl(knowledge_base_path)


def generate_feedback(
    tokenizer,
    model,
    question: str,
    db_id: str,
    schema_text: str,
    wrong_sql: str,
    rag_context: str,
    max_new_tokens: int,
) -> str:
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                question=question.strip(),
                db_id=db_id.strip(),
                schema_text=schema_text.strip(),
                wrong_sql=wrong_sql.strip(),
            ),
        },
    ]
    if rag_context:
        prompt_messages[-1]["content"] = (
            prompt_messages[-1]["content"]
            + "\n\n可参考的 SQL 知识：\n"
            + rag_context
            + "\n\n请优先参考这些知识来判断错误原因，但最终答案仍要结合题目、Schema 和学生 SQL。只修主要错误，不要改动已经正确的 SQL 片段。"
        )

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    device = next(model.parameters()).device
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def render_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(247, 208, 96, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(71, 164, 233, 0.16), transparent 26%),
                linear-gradient(180deg, #f7f3ea 0%, #fbfaf7 100%);
        }
        .hero-card, .info-card, .result-card {
            border-radius: 24px;
            padding: 1.1rem 1.2rem;
            border: 1px solid rgba(26, 41, 61, 0.10);
            box-shadow: 0 12px 30px rgba(26, 41, 61, 0.08);
            background: rgba(255, 255, 255, 0.92);
            backdrop-filter: blur(10px);
        }
        .hero-title {
            font-size: 2.1rem;
            font-weight: 800;
            line-height: 1.1;
            color: #142235;
            margin-bottom: 0.4rem;
        }
        .hero-subtitle {
            color: #39506d;
            font-size: 1rem;
            line-height: 1.6;
        }
        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.75rem;
        }
        .pill {
            border-radius: 999px;
            padding: 0.35rem 0.75rem;
            background: #13263b;
            color: #f7f3ea;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .section-label {
            font-size: 0.85rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #8a6a18;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .result-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #102033;
        }
        .tiny-note {
            color: #55708f;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state(presets: dict[str, dict]) -> None:
    default_preset = next(iter(presets))
    preset = presets[default_preset]
    st.session_state.setdefault("selected_preset", default_preset)
    st.session_state.setdefault("db_id", preset["db_id"])
    st.session_state.setdefault("question", preset["question"])
    st.session_state.setdefault("schema_text", preset["schema_text"])
    st.session_state.setdefault("wrong_sql", preset["wrong_sql"])
    st.session_state.setdefault("prediction", "")
    st.session_state.setdefault("retrieved_docs", [])
    st.session_state.setdefault("show_reference", True)


def load_preset_into_state(preset: dict) -> None:
    st.session_state["db_id"] = preset["db_id"]
    st.session_state["question"] = preset["question"]
    st.session_state["schema_text"] = preset["schema_text"]
    st.session_state["wrong_sql"] = preset["wrong_sql"]
    st.session_state["prediction"] = ""
    st.session_state["retrieved_docs"] = []


def main() -> None:
    st.set_page_config(
        page_title="SQL 智能批阅 Demo",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_styles()

    dataset_path = Path(DEFAULT_DATASET)
    presets = build_showcase_presets(dataset_path)
    init_session_state(presets)

    with st.sidebar:
        st.markdown("### Demo 配置")
        base_model_path = st.text_input("Base Model", value=DEFAULT_BASE_MODEL)
        adapter_path = st.text_input("Adapter Path", value=DEFAULT_ADAPTER)
        knowledge_base_path = st.text_input("Knowledge Base", value=DEFAULT_KNOWLEDGE_BASE)
        dtype_name = st.selectbox("推理精度", options=["auto", "bfloat16", "float16", "float32"], index=0)
        max_new_tokens = st.slider("最大生成长度", min_value=128, max_value=768, value=384, step=32)
        use_rag = st.toggle("启用 RAG 检索", value=True)
        top_k = st.slider("检索条数", min_value=1, max_value=6, value=4, step=1, disabled=not use_rag)
        st.session_state["show_reference"] = st.toggle("显示参考答案", value=st.session_state["show_reference"])

        st.markdown("### 模型状态")
        device_text = "CUDA" if torch.cuda.is_available() else "CPU"
        st.write(f"当前设备: `{device_text}`")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            st.write(f"显卡: `{gpu_name}`")
        st.caption("首次点击“开始批阅”时会加载模型，后续会复用缓存。")
        st.caption("启用 RAG 后，模型会先检索本地 SQL 规则与纠错案例，再生成最终批阅结果。")

    st.markdown(
        """
        <div class="hero-card">
            <div class="section-label">SQL Grading Showcase</div>
            <div class="hero-title">SQL 智能批阅 Demo</div>
            <div class="hero-subtitle">
                输入英文题目、Schema 和学生 SQL，模型会返回正确 SQL 与中文纠错解析。
                这个界面已经连接到你训练好的 Qwen2.5-Coder-1.5B-Instruct + LoRA 适配器。
            </div>
            <div class="pill-row">
                <div class="pill">英文题目输入</div>
                <div class="pill">中文批阅输出</div>
                <div class="pill">结构化纠错解释</div>
                <div class="pill">RAG 知识增强</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.05, 0.95], gap="large")

    with left_col:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("#### 选择展示样例")
        preset_label = st.selectbox("预置案例", options=list(presets.keys()), key="selected_preset")
        preset = presets[preset_label]
        if st.button("载入这个样例", use_container_width=True):
            load_preset_into_state(preset)
            st.rerun()

        st.markdown("#### 输入区")
        st.text_input("数据库 ID", key="db_id")
        st.text_area("英文题目", key="question", height=120)
        st.text_area("相关 Schema", key="schema_text", height=220)
        st.text_area("学生错误 SQL", key="wrong_sql", height=180)

        generate = st.button("开始批阅", type="primary", use_container_width=True)
        st.markdown(
            '<div class="tiny-note">建议展示时先点“载入这个样例”，再点“开始批阅”，效果最稳。</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="result-title">模型输出</div>', unsafe_allow_html=True)
        st.caption("模型会按照训练时的固定格式输出正确 SQL 和纠错解析。")

        if generate:
            if not st.session_state["question"].strip() or not st.session_state["schema_text"].strip() or not st.session_state["wrong_sql"].strip():
                st.warning("请先补全题目、Schema 和学生 SQL。")
            else:
                with st.spinner("正在加载模型并生成批阅结果..."):
                    tokenizer, model = load_model_and_tokenizer(base_model_path, adapter_path, dtype_name)
                    retrieved_docs = []
                    rag_context = ""
                    if use_rag:
                        retriever = load_retriever(knowledge_base_path)
                        retrieved_docs = retriever.search(
                            question=st.session_state["question"],
                            schema_text=st.session_state["schema_text"],
                            wrong_sql=st.session_state["wrong_sql"],
                            top_k=top_k,
                        )
                        rag_context = format_rag_context(retrieved_docs)
                    prediction = generate_feedback(
                        tokenizer=tokenizer,
                        model=model,
                        question=st.session_state["question"],
                        db_id=st.session_state["db_id"],
                        schema_text=st.session_state["schema_text"],
                        wrong_sql=st.session_state["wrong_sql"],
                        rag_context=rag_context,
                        max_new_tokens=max_new_tokens,
                    )
                    st.session_state["prediction"] = prediction
                    st.session_state["retrieved_docs"] = retrieved_docs

        if st.session_state["prediction"]:
            st.markdown(st.session_state["prediction"])
        else:
            st.info("点击“开始批阅”后，这里会显示模型生成的结果。")

        if st.session_state["retrieved_docs"]:
            with st.expander("查看检索到的 SQL 知识", expanded=False):
                for item in st.session_state["retrieved_docs"]:
                    st.markdown(f"**{item['title']}**")
                    st.caption(f"类别: {item['category']} | score={item['score']:.4f}")
                    st.write(item["content"])
                    st.write(f"错误示例: {item['example_error']}")
                    st.write(f"纠正方式: {item['example_fix']}")
                    st.markdown("---")

        if st.session_state["show_reference"]:
            with st.expander("查看这个样例的参考答案", expanded=False):
                st.write(f"错误类型: `{preset['error_type']}`")
                st.write(f"记录 ID: `{preset['record_id']}`")
                st.markdown(preset["gold_answer"])
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")
    st.markdown(
        """
        <div class="info-card">
            <div class="section-label">Run Command</div>
            <div class="tiny-note">
                启动命令：<code>streamlit run demo_app.py</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
