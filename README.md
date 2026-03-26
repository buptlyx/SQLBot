# SQLBot

在使用本项目之前，请在hugging face官网下载**Qwen2.5-Coder-1.5B-Instruct**的模型参数文件。

这个仓库现在包含两条可直接运行的基础流水线：

1. `scripts/build_sft_dataset.py`
把 Spider 数据集转换成适合 SQL 智能批阅任务的 `messages` 格式 SFT 数据。

2. `scripts/train_lora.py`
基于 `Qwen2.5-Coder-1.5B-Instruct` 做 LoRA 微调，目标任务是：
`题目 + 相关 Schema + 学生错误 SQL -> 正确 SQL + 纠错解析`

## 目录约定

- 原始数据：`data/raw/spider`
- 处理后数据：`data/processed/spider_sql_grading_sft`
- 本地模型：`models/Qwen2.5-Coder-1.5B-Instruct`
- 训练输出：`outputs/qwen2.5-coder-1.5b-sql-grader-lora`

## 1. 生成 SFT 数据

默认会读取：

- 训练集：`train_spider.json` + `train_others.json`
- 验证集：`dev.json`

并自动合成几类常见学生错误：

- 字段选择错误
- 聚合函数错误
- 分组与聚合不匹配
- 连接条件遗漏
- 排序方向错误
- 筛选条件错误

运行命令：

```powershell
python scripts/build_sft_dataset.py
```

如果你只想保留部分错误类型，可以这样跑：

```powershell
python scripts/build_sft_dataset.py --error_types select_column,join_condition,group_by
```

处理完成后会得到：

- `data/processed/spider_sql_grading_sft/train.jsonl`
- `data/processed/spider_sql_grading_sft/validation.jsonl`
- `data/processed/spider_sql_grading_sft/stats.json`
- `data/processed/spider_sql_grading_sft/preview.json`

每条样本都包含：

- `wrong_sql`
- `correct_sql`
- `error_type`
- `messages`

其中 `messages` 可直接用于聊天式 SFT。

## 2. LoRA 微调

先安装依赖：

```powershell
pip install -r requirements.txt
```

标准 LoRA 训练：

```powershell
python scripts/train_lora.py ^
  --model_name_or_path models/Qwen2.5-Coder-1.5B-Instruct ^
  --train_file data/processed/spider_sql_grading_sft/train.jsonl ^
  --eval_file data/processed/spider_sql_grading_sft/validation.jsonl ^
  --output_dir outputs/qwen2.5-coder-1.5b-sql-grader-lora ^
  --max_length 2048 ^
  --per_device_train_batch_size 1 ^
  --gradient_accumulation_steps 16 ^
  --num_train_epochs 3 ^
  --learning_rate 2e-4 ^
  --gradient_checkpointing
```

如果你的环境支持 `bitsandbytes`，可以额外打开 `4bit`：

```powershell
python scripts/train_lora.py --use_4bit --gradient_checkpointing
```

## 3. 本地展示 Demo

仓库根目录提供了一个基于 Streamlit 的演示界面：

```powershell
streamlit run demo_app.py
```

这个 Demo 会加载你训练好的 LoRA 适配器，并支持：

- 手动输入英文题目、Schema 和学生 SQL
- 一键载入预置展示样例
- 可选启用本地 RAG 检索
- 输出“正确 SQL + 中文纠错解析”

默认会读取：

- Base Model: `models/Qwen2.5-Coder-1.5B-Instruct`
- Adapter: `outputs/qwen2.5-coder-1.5b-sql-grader-lora`
- Knowledge Base: `data/knowledge/sql_knowledge_base.jsonl`

## 4. 训练脚本说明

- 训练输入默认读取 `messages` 字段。
- 脚本会用 `tokenizer.apply_chat_template` 构造 Qwen 对话格式。
- 损失只计算在最后一轮 `assistant` 回复上，避免把用户提示部分也当成监督目标。
- 默认目标模块是：
  `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`

## 5. 建议的实验顺序

1. 先跑 `build_sft_dataset.py`，确认生成样本是否符合你的批阅风格。
2. 先做一版不带 RAG 的 LoRA 微调，建立基线。
3. 再把 SQL 规则库和纠错案例接入 RAG，对比解释质量和事实性。

## 6. 一个实用提醒

你当前系统里的 Python 版本是 `3.13.5`。如果后续遇到训练依赖安装问题，建议单独建一个更偏训练友好的虚拟环境再跑这套脚本。
