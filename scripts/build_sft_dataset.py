from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


SYSTEM_PROMPT = """你是数据库课程中的 SQL 智能批阅助手。
你需要根据题目、相关表结构和学生提交的 SQL，输出标准答案以及清晰的纠错解析。
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

ASSISTANT_TEMPLATE = """正确SQL：
```sql
{correct_sql}
```

纠错解析：
错误类型：{error_type}
错误定位：{error_location}
原因说明：{analysis}
修改建议：{fix_suggestion}"""

AGGREGATIONS = {
    0: "",
    1: "MAX",
    2: "MIN",
    3: "COUNT",
    4: "SUM",
    5: "AVG",
}

RESERVED_TOKENS = {
    "SELECT",
    "FROM",
    "WHERE",
    "GROUP",
    "BY",
    "HAVING",
    "ORDER",
    "LIMIT",
    "JOIN",
    "ON",
    "AS",
    "UNION",
    "INTERSECT",
    "EXCEPT",
    "AND",
    "OR",
}


@dataclass(frozen=True)
class ColumnInfo:
    column_id: int
    table_id: int
    table_name: str
    column_name: str
    column_type: str

    @property
    def qualified_name(self) -> str:
        return f"{self.table_name}.{self.column_name}"


@dataclass
class SchemaInfo:
    db_id: str
    table_names: list[str]
    columns_by_id: dict[int, ColumnInfo]
    table_to_columns: dict[int, list[ColumnInfo]]
    primary_keys: set[int]
    foreign_key_map: dict[int, list[int]]

    def candidate_columns_for_replacement(self, original: ColumnInfo) -> list[ColumnInfo]:
        same_table = [
            column
            for column in self.table_to_columns.get(original.table_id, [])
            if column.column_id != original.column_id
        ]
        same_type = [column for column in same_table if column.column_type == original.column_type]
        return same_type or same_table


@dataclass
class Mutation:
    error_key: str
    error_type: str
    error_location: str
    analysis: str
    fix_suggestion: str
    wrong_sql: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Spider-based SFT data for SQL grading.")
    parser.add_argument(
        "--spider_dir",
        type=Path,
        default=Path("data/raw/spider"),
        help="Directory that contains Spider json files and tables.json.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed/spider_sql_grading_sft"),
        help="Output directory for jsonl files.",
    )
    parser.add_argument(
        "--train_files",
        type=str,
        default="train_spider.json,train_others.json",
        help="Comma-separated Spider files used for training.",
    )
    parser.add_argument(
        "--validation_files",
        type=str,
        default="dev.json",
        help="Comma-separated Spider files used for validation.",
    )
    parser.add_argument(
        "--error_types",
        type=str,
        default="select_column,aggregate_function,group_by,join_condition,order_direction,filter_operator",
        help="Comma-separated mutation types to enable.",
    )
    parser.add_argument(
        "--max_mutations_per_example",
        type=int,
        default=2,
        help="Maximum number of synthetic mistakes generated from one Spider example.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_schema_map(tables_path: Path) -> dict[str, SchemaInfo]:
    schema_map: dict[str, SchemaInfo] = {}
    for raw_schema in load_json(tables_path):
        table_names = raw_schema["table_names_original"]
        primary_keys = set(raw_schema["primary_keys"])
        foreign_key_map: dict[int, list[int]] = defaultdict(list)
        for source_id, target_id in raw_schema["foreign_keys"]:
            foreign_key_map[source_id].append(target_id)

        columns_by_id: dict[int, ColumnInfo] = {}
        table_to_columns: dict[int, list[ColumnInfo]] = defaultdict(list)
        for column_id, (table_id, column_name) in enumerate(raw_schema["column_names_original"]):
            if table_id < 0:
                continue
            column = ColumnInfo(
                column_id=column_id,
                table_id=table_id,
                table_name=table_names[table_id],
                column_name=column_name,
                column_type=raw_schema["column_types"][column_id],
            )
            columns_by_id[column_id] = column
            table_to_columns[table_id].append(column)

        schema_map[raw_schema["db_id"]] = SchemaInfo(
            db_id=raw_schema["db_id"],
            table_names=table_names,
            columns_by_id=columns_by_id,
            table_to_columns=dict(table_to_columns),
            primary_keys=primary_keys,
            foreign_key_map=dict(foreign_key_map),
        )
    return schema_map


def normalize_sql_text(sql: str) -> str:
    sql = re.sub(r"\s+", " ", sql).strip()
    return sql.replace(" ;", ";")


def untokenize_sql(tokens: list[str]) -> str:
    if not tokens:
        return ""

    sql = ""
    in_single_quote = False
    for token in tokens:
        if not sql:
            sql = token
        else:
            no_space = False
            if token in {",", ")", ";", "("}:
                no_space = True
            elif sql.endswith("("):
                no_space = True
            elif token == "'":
                no_space = True
            sql += ("" if no_space else " ") + token

        if token == "'":
            in_single_quote = not in_single_quote
        elif token.startswith("'") and not token.endswith("'"):
            in_single_quote = True
        elif token.endswith("'") and in_single_quote:
            in_single_quote = False

    sql = sql.replace("( ", "(").replace(" )", ")")
    sql = sql.replace(" ,", ",").replace(" ;", ";")
    return sql.strip()


def is_simple_single_select(query_tokens: list[str], sql_ast: dict) -> bool:
    top_level_selects = sum(1 for token in query_tokens if token.upper() == "SELECT")
    return (
        top_level_selects == 1
        and not sql_ast["intersect"]
        and not sql_ast["union"]
        and not sql_ast["except"]
    )


def find_clause_spans(tokens: list[str]) -> dict[str, tuple[int, int]]:
    positions: dict[str, int] = {}
    depth = 0
    index = 0
    while index < len(tokens):
        token = tokens[index]
        upper = token.upper()

        if token == "(":
            depth += 1
            index += 1
            continue
        if token == ")":
            depth -= 1
            index += 1
            continue

        if depth == 0:
            if upper == "SELECT" and "select" not in positions:
                positions["select"] = index
            elif upper == "FROM" and "from" not in positions:
                positions["from"] = index
            elif upper == "WHERE" and "where" not in positions:
                positions["where"] = index
            elif upper == "GROUP" and index + 1 < len(tokens) and tokens[index + 1].upper() == "BY":
                positions.setdefault("group_by", index)
            elif upper == "HAVING" and "having" not in positions:
                positions["having"] = index
            elif upper == "ORDER" and index + 1 < len(tokens) and tokens[index + 1].upper() == "BY":
                positions.setdefault("order_by", index)
            elif upper == "LIMIT" and "limit" not in positions:
                positions["limit"] = index
            elif upper == "INTERSECT" and "intersect" not in positions:
                positions["intersect"] = index
            elif upper == "UNION" and "union" not in positions:
                positions["union"] = index
            elif upper == "EXCEPT" and "except" not in positions:
                positions["except"] = index
        index += 1

    ordered = sorted((start, name) for name, start in positions.items())
    spans: dict[str, tuple[int, int]] = {}
    for current_index, (start, name) in enumerate(ordered):
        end = ordered[current_index + 1][0] if current_index + 1 < len(ordered) else len(tokens)
        spans[name] = (start, end)
    return spans


def split_top_level_ranges(tokens: list[str], start_index: int) -> list[tuple[int, int, list[str]]]:
    ranges: list[tuple[int, int, list[str]]] = []
    depth = 0
    part_start = 0
    for index, token in enumerate(tokens):
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
        elif depth == 0 and token == ",":
            if tokens[part_start:index]:
                ranges.append((start_index + part_start, start_index + index, tokens[part_start:index]))
            part_start = index + 1
    if tokens[part_start:]:
        ranges.append((start_index + part_start, start_index + len(tokens), tokens[part_start:]))
    return ranges


def get_select_expression_ranges(tokens: list[str], spans: dict[str, tuple[int, int]]) -> list[tuple[int, int, list[str]]]:
    if "select" not in spans or "from" not in spans:
        return []

    start = spans["select"][0] + 1
    end = spans["from"][0]
    select_tokens = tokens[start:end]
    if select_tokens and select_tokens[0].upper() == "DISTINCT":
        start += 1
        select_tokens = select_tokens[1:]
    return split_top_level_ranges(select_tokens, start)


def extract_select_column_id(select_item: list) -> int | None:
    if not select_item or len(select_item) != 2:
        return None
    value_unit = select_item[1]
    if not value_unit or len(value_unit) < 2:
        return None
    if value_unit[0] != 0:
        return None
    column_unit = value_unit[1]
    if not column_unit or len(column_unit) < 2:
        return None
    return column_unit[1]


def table_ids_from_ast(sql_ast: dict) -> set[int]:
    table_ids: set[int] = set()
    for table_unit in sql_ast["from"]["table_units"]:
        if table_unit and table_unit[0] == "table_unit":
            table_ids.add(table_unit[1])
    return table_ids


def format_schema(schema: SchemaInfo, used_table_ids: set[int]) -> str:
    relevant_tables = sorted(used_table_ids) if used_table_ids else list(range(len(schema.table_names)))
    lines = []
    for table_id in relevant_tables:
        table_name = schema.table_names[table_id]
        column_texts = []
        for column in schema.table_to_columns.get(table_id, []):
            markers = []
            if column.column_id in schema.primary_keys:
                markers.append("PK")
            fk_targets = schema.foreign_key_map.get(column.column_id, [])
            if fk_targets:
                targets = []
                for target_id in fk_targets:
                    target = schema.columns_by_id.get(target_id)
                    if target:
                        targets.append(target.qualified_name)
                if targets:
                    markers.append("FK->" + "/".join(targets))
            marker_text = f" [{' | '.join(markers)}]" if markers else ""
            column_texts.append(f"{column.column_name} ({column.column_type}){marker_text}")
        lines.append(f"- {table_name}: " + ", ".join(column_texts))
    return "\n".join(lines)


def is_same_sql(candidate: str, reference: str) -> bool:
    return normalize_sql_text(candidate).upper() == normalize_sql_text(reference).upper()


def mutate_select_column(example: dict, schema: SchemaInfo, rng: random.Random) -> Mutation | None:
    query_tokens = list(example["query_toks"])
    sql_ast = example["sql"]
    if not is_simple_single_select(query_tokens, sql_ast):
        return None

    spans = find_clause_spans(query_tokens)
    expression_ranges = get_select_expression_ranges(query_tokens, spans)
    select_items = sql_ast["select"][1]
    if not expression_ranges or len(expression_ranges) != len(select_items):
        return None

    candidate_slots = []
    table_count = len(sql_ast["from"]["table_units"])
    for (start, end, expression_tokens), select_item in zip(expression_ranges, select_items):
        if len(expression_tokens) != 1:
            continue
        outer_agg = select_item[0]
        column_id = extract_select_column_id(select_item)
        if outer_agg != 0 or not column_id or column_id <= 0:
            continue
        original_token = expression_tokens[0]
        if "." not in original_token and table_count > 1:
            continue

        original_column = schema.columns_by_id.get(column_id)
        if not original_column:
            continue
        replacement_candidates = schema.candidate_columns_for_replacement(original_column)
        if not replacement_candidates:
            continue
        candidate_slots.append((start, end, original_token, original_column, replacement_candidates))

    if not candidate_slots:
        return None

    start, end, original_token, original_column, replacement_candidates = rng.choice(candidate_slots)
    replacement_column = rng.choice(replacement_candidates)
    if "." in original_token:
        qualifier, _ = original_token.split(".", 1)
        replacement_token = f"{qualifier}.{replacement_column.column_name}"
    else:
        replacement_token = replacement_column.column_name

    mutated_tokens = query_tokens[:start] + [replacement_token] + query_tokens[end:]
    wrong_sql = untokenize_sql(mutated_tokens)
    if is_same_sql(wrong_sql, example["query"]):
        return None

    return Mutation(
        error_key="select_column",
        error_type="字段选择错误",
        error_location="SELECT 子句",
        analysis=(
            f"学生把应返回的列 `{original_column.qualified_name}` 误写成了 "
            f"`{replacement_column.qualified_name}`，导致结果列的含义和题目要求不一致。"
        ),
        fix_suggestion=(
            f"将 SELECT 中的 `{replacement_token}` 改回与题意对应的 "
            f"`{original_token}`，其余连接、筛选和排序逻辑保持不变。"
        ),
        wrong_sql=wrong_sql,
    )


def mutate_aggregate_function(example: dict, schema: SchemaInfo, rng: random.Random) -> Mutation | None:
    del schema
    query_tokens = list(example["query_toks"])
    sql_ast = example["sql"]
    if not is_simple_single_select(query_tokens, sql_ast):
        return None

    spans = find_clause_spans(query_tokens)
    expression_ranges = get_select_expression_ranges(query_tokens, spans)
    select_items = sql_ast["select"][1]
    if not expression_ranges or len(expression_ranges) != len(select_items):
        return None

    candidate_slots = []
    for (start, _end, expression_tokens), select_item in zip(expression_ranges, select_items):
        outer_agg = select_item[0]
        if outer_agg not in AGGREGATIONS or outer_agg == 0:
            continue
        if not expression_tokens:
            continue
        token_upper = expression_tokens[0].upper()
        if token_upper not in {"MAX", "MIN", "COUNT", "SUM", "AVG"}:
            continue
        replacement = "MAX" if token_upper == "COUNT" else "COUNT"
        candidate_slots.append((start, token_upper, replacement))

    if not candidate_slots:
        return None

    start, original_agg, replacement_agg = rng.choice(candidate_slots)
    mutated_tokens = list(query_tokens)
    mutated_tokens[start] = replacement_agg
    wrong_sql = untokenize_sql(mutated_tokens)
    if is_same_sql(wrong_sql, example["query"]):
        return None

    return Mutation(
        error_key="aggregate_function",
        error_type="聚合函数错误",
        error_location="SELECT 子句中的聚合表达式",
        analysis=(
            f"学生把原本应该使用的聚合函数 `{original_agg}` 误写成了 `{replacement_agg}`。"
            "聚合函数一旦写错，统计口径就会变化，结果不再对应题目想问的内容。"
        ),
        fix_suggestion=(
            f"将聚合函数 `{replacement_agg}` 改回 `{original_agg}`，并保持被聚合的列不变。"
        ),
        wrong_sql=wrong_sql,
    )


def mutate_group_by(example: dict, schema: SchemaInfo, rng: random.Random) -> Mutation | None:
    del schema, rng
    query_tokens = list(example["query_toks"])
    sql_ast = example["sql"]
    if not is_simple_single_select(query_tokens, sql_ast):
        return None

    spans = find_clause_spans(query_tokens)
    if "group_by" not in spans or "having" in spans:
        return None

    start, end = spans["group_by"]
    mutated_tokens = query_tokens[:start] + query_tokens[end:]
    wrong_sql = untokenize_sql(mutated_tokens)
    if is_same_sql(wrong_sql, example["query"]):
        return None

    return Mutation(
        error_key="group_by",
        error_type="分组与聚合不匹配",
        error_location="GROUP BY 子句",
        analysis=(
            "题目需要按照某个维度分组后再统计，但学生 SQL 去掉了 GROUP BY，"
            "会导致非聚合列与聚合结果之间不再匹配，无法正确表达每组的统计含义。"
        ),
        fix_suggestion="补回与题意一致的 GROUP BY 子句，让分组维度和聚合结果重新对应起来。",
        wrong_sql=wrong_sql,
    )


def find_on_condition_ranges(tokens: list[str], spans: dict[str, tuple[int, int]]) -> list[tuple[int, int]]:
    if "from" not in spans:
        return []

    start, end = spans["from"]
    ranges = []
    depth = 0
    index = start
    while index < end:
        token = tokens[index]
        upper = token.upper()

        if token == "(":
            depth += 1
            index += 1
            continue
        if token == ")":
            depth -= 1
            index += 1
            continue

        if depth == 0 and upper == "ON":
            condition_start = index + 1
            condition_end = condition_start
            inner_depth = 0
            while condition_end < end:
                current = tokens[condition_end]
                current_upper = current.upper()
                if current == "(":
                    inner_depth += 1
                elif current == ")":
                    inner_depth -= 1
                if inner_depth == 0 and current_upper in {"JOIN", "WHERE", "GROUP", "HAVING", "ORDER", "LIMIT"}:
                    break
                condition_end += 1
            if condition_start < condition_end:
                ranges.append((condition_start, condition_end))
            index = condition_end
            continue
        index += 1
    return ranges


def mutate_join_condition(example: dict, schema: SchemaInfo, rng: random.Random) -> Mutation | None:
    del schema
    query_tokens = list(example["query_toks"])
    sql_ast = example["sql"]
    if not is_simple_single_select(query_tokens, sql_ast):
        return None
    if "JOIN" not in {token.upper() for token in query_tokens}:
        return None

    spans = find_clause_spans(query_tokens)
    ranges = find_on_condition_ranges(query_tokens, spans)
    if not ranges:
        return None

    start, end = rng.choice(ranges)
    mutated_tokens = query_tokens[:start] + ["1", "=", "1"] + query_tokens[end:]
    wrong_sql = untokenize_sql(mutated_tokens)
    if is_same_sql(wrong_sql, example["query"]):
        return None

    return Mutation(
        error_key="join_condition",
        error_type="连接条件遗漏",
        error_location="JOIN ... ON 连接条件",
        analysis=(
            "学生在表连接时没有保留正确的关联条件，实际效果接近把两张表直接做笛卡尔拼接，"
            "这样会放大结果集并破坏原本的表间对应关系。"
        ),
        fix_suggestion="恢复正确的 ON 条件，确保两张表通过主外键或题目要求的字段准确关联。",
        wrong_sql=wrong_sql,
    )


def mutate_order_direction(example: dict, schema: SchemaInfo, rng: random.Random) -> Mutation | None:
    del schema, rng
    query_tokens = list(example["query_toks"])
    sql_ast = example["sql"]
    if not is_simple_single_select(query_tokens, sql_ast):
        return None

    spans = find_clause_spans(query_tokens)
    if "order_by" not in spans:
        return None
    if not sql_ast["orderBy"] or len(sql_ast["orderBy"][1]) != 1:
        return None

    start, end = spans["order_by"]
    mutated_tokens = list(query_tokens)
    replaced = False
    for index in range(start, end):
        upper = mutated_tokens[index].upper()
        if upper == "ASC":
            mutated_tokens[index] = "DESC"
            replaced = True
            break
        if upper == "DESC":
            mutated_tokens[index] = "ASC"
            replaced = True
            break

    if not replaced:
        insert_at = end
        if insert_at > start and mutated_tokens[insert_at - 1] == ";":
            insert_at -= 1
        mutated_tokens = mutated_tokens[:insert_at] + ["DESC"] + mutated_tokens[insert_at:]

    wrong_sql = untokenize_sql(mutated_tokens)
    if is_same_sql(wrong_sql, example["query"]):
        return None

    original_direction = sql_ast["orderBy"][0].upper()
    wrong_direction = "DESC" if original_direction == "ASC" else "ASC"
    return Mutation(
        error_key="order_direction",
        error_type="排序方向错误",
        error_location="ORDER BY 子句",
        analysis=(
            f"学生把排序方向写成了 `{wrong_direction}`，而题目需要的是 `{original_direction}`。"
            "排序方向写反会直接改变最终返回记录的先后顺序。"
        ),
        fix_suggestion=f"将 ORDER BY 的排序方向改回 `{original_direction}`。",
        wrong_sql=wrong_sql,
    )


def mutate_filter_operator(example: dict, schema: SchemaInfo, rng: random.Random) -> Mutation | None:
    del schema, rng
    query_tokens = list(example["query_toks"])
    sql_ast = example["sql"]
    if not is_simple_single_select(query_tokens, sql_ast):
        return None

    spans = find_clause_spans(query_tokens)
    if "where" not in spans:
        return None

    opposite_map = {
        ">": "<",
        "<": ">",
        ">=": "<=",
        "<=": ">=",
        "=": "!=",
        "!=": "=",
        "<>": "=",
    }

    start, end = spans["where"]
    mutated_tokens = list(query_tokens)
    depth = 0
    changed = False
    original_operator = ""
    wrong_operator = ""
    for index in range(start + 1, end):
        token = mutated_tokens[index]
        if token == "(":
            depth += 1
            continue
        if token == ")":
            depth -= 1
            continue
        if depth == 0 and token in opposite_map:
            original_operator = token
            wrong_operator = opposite_map[token]
            mutated_tokens[index] = wrong_operator
            changed = True
            break

    if not changed:
        return None

    wrong_sql = untokenize_sql(mutated_tokens)
    if is_same_sql(wrong_sql, example["query"]):
        return None

    return Mutation(
        error_key="filter_operator",
        error_type="筛选条件错误",
        error_location="WHERE 子句",
        analysis=(
            f"学生把筛选运算符从 `{original_operator}` 写成了 `{wrong_operator}`，"
            "会把题目要求保留的数据过滤方向反过来，导致结果集范围错误。"
        ),
        fix_suggestion=f"将 WHERE 中的运算符 `{wrong_operator}` 改回 `{original_operator}`。",
        wrong_sql=wrong_sql,
    )


def build_messages(question: str, db_id: str, schema_text: str, wrong_sql: str, correct_sql: str, mutation: Mutation) -> list[dict]:
    user_prompt = USER_TEMPLATE.format(
        question=question,
        db_id=db_id,
        schema_text=schema_text,
        wrong_sql=wrong_sql,
    )
    assistant_response = ASSISTANT_TEMPLATE.format(
        correct_sql=correct_sql,
        error_type=mutation.error_type,
        error_location=mutation.error_location,
        analysis=mutation.analysis,
        fix_suggestion=mutation.fix_suggestion,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_response},
    ]


def read_examples(spider_dir: Path, filenames: list[str]) -> list[tuple[str, int, dict]]:
    examples = []
    for filename in filenames:
        file_path = spider_dir / filename
        for index, item in enumerate(load_json(file_path)):
            examples.append((filename, index, item))
    return examples


def build_records(
    examples: list[tuple[str, int, dict]],
    split_name: str,
    schema_map: dict[str, SchemaInfo],
    enabled_generators: list[Callable[[dict, SchemaInfo, random.Random], Mutation | None]],
    rng: random.Random,
    max_mutations_per_example: int,
) -> tuple[list[dict], Counter]:
    records = []
    error_counter: Counter = Counter()

    for source_file, source_index, example in examples:
        schema = schema_map[example["db_id"]]
        schema_text = format_schema(schema, table_ids_from_ast(example["sql"]))
        correct_sql = normalize_sql_text(example["query"])

        candidate_mutations = []
        for generator in enabled_generators:
            mutation = generator(example, schema, rng)
            if mutation and not is_same_sql(mutation.wrong_sql, correct_sql):
                candidate_mutations.append(mutation)

        deduplicated: dict[str, Mutation] = {}
        for mutation in candidate_mutations:
            deduplicated.setdefault(mutation.error_key, mutation)

        selected = list(deduplicated.values())
        rng.shuffle(selected)
        selected = selected[:max_mutations_per_example]

        for mutation in selected:
            record_id = f"{split_name}-{source_file.replace('.json', '')}-{source_index}-{mutation.error_key}"
            record = {
                "id": record_id,
                "split": split_name,
                "source_file": source_file,
                "source_index": source_index,
                "db_id": example["db_id"],
                "question": example["question"],
                "schema_text": schema_text,
                "wrong_sql": normalize_sql_text(mutation.wrong_sql),
                "correct_sql": correct_sql,
                "error_type": mutation.error_type,
                "messages": build_messages(
                    question=example["question"],
                    db_id=example["db_id"],
                    schema_text=schema_text,
                    wrong_sql=normalize_sql_text(mutation.wrong_sql),
                    correct_sql=correct_sql,
                    mutation=mutation,
                ),
            }
            records.append(record)
            error_counter[mutation.error_type] += 1

    return records, error_counter


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    schema_map = build_schema_map(args.spider_dir / "tables.json")
    train_files = [item.strip() for item in args.train_files.split(",") if item.strip()]
    validation_files = [item.strip() for item in args.validation_files.split(",") if item.strip()]

    generator_registry: dict[str, Callable[[dict, SchemaInfo, random.Random], Mutation | None]] = {
        "select_column": mutate_select_column,
        "aggregate_function": mutate_aggregate_function,
        "group_by": mutate_group_by,
        "join_condition": mutate_join_condition,
        "order_direction": mutate_order_direction,
        "filter_operator": mutate_filter_operator,
    }
    requested_error_types = [item.strip() for item in args.error_types.split(",") if item.strip()]
    enabled_generators = [generator_registry[name] for name in requested_error_types]

    train_examples = read_examples(args.spider_dir, train_files)
    validation_examples = read_examples(args.spider_dir, validation_files)

    train_records, train_counter = build_records(
        examples=train_examples,
        split_name="train",
        schema_map=schema_map,
        enabled_generators=enabled_generators,
        rng=rng,
        max_mutations_per_example=args.max_mutations_per_example,
    )
    validation_records, validation_counter = build_records(
        examples=validation_examples,
        split_name="validation",
        schema_map=schema_map,
        enabled_generators=enabled_generators,
        rng=rng,
        max_mutations_per_example=args.max_mutations_per_example,
    )

    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    write_jsonl(train_path, train_records)
    write_jsonl(validation_path, validation_records)

    stats = {
        "seed": args.seed,
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "train_error_distribution": dict(train_counter),
        "validation_error_distribution": dict(validation_counter),
        "enabled_error_types": requested_error_types,
        "train_files": train_files,
        "validation_files": validation_files,
        "max_mutations_per_example": args.max_mutations_per_example,
    }
    (args.output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    preview = train_records[:3] + validation_records[:2]
    (args.output_dir / "preview.json").write_text(
        json.dumps(preview, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Train records: {len(train_records)} -> {train_path}")
    print(f"Validation records: {len(validation_records)} -> {validation_path}")
    print("Train error distribution:", dict(train_counter))
    print("Validation error distribution:", dict(validation_counter))


if __name__ == "__main__":
    main()
