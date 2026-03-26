from __future__ import annotations

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer


class SqlKnowledgeRetriever:
    def __init__(self, records: list[dict]) -> None:
        self.records = records
        corpus = [self._record_to_text(record) for record in records]
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(corpus)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "SqlKnowledgeRetriever":
        records = []
        with Path(path).open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    records.append(json.loads(line))
        return cls(records)

    def search(self, question: str, schema_text: str, wrong_sql: str, top_k: int = 4) -> list[dict]:
        query = "\n".join(
            [
                question.strip(),
                schema_text.strip(),
                wrong_sql.strip(),
            ]
        )
        query_vector = self.vectorizer.transform([query])
        scores = (query_vector @ self.matrix.T).toarray().ravel()
        ranked_indices = scores.argsort()[::-1]
        best_score = float(scores[ranked_indices[0]]) if len(ranked_indices) else 0.0
        min_score = max(0.12, best_score * 0.90)

        results = []
        for index in ranked_indices:
            score = float(scores[index])
            if score < min_score:
                continue
            record = dict(self.records[index])
            record["score"] = score
            results.append(record)
            if len(results) >= top_k:
                break
        if not results and len(ranked_indices):
            top_index = int(ranked_indices[0])
            top_score = float(scores[top_index])
            if top_score > 0:
                record = dict(self.records[top_index])
                record["score"] = top_score
                results.append(record)
        return results

    @staticmethod
    def _record_to_text(record: dict) -> str:
        keywords = " ".join(record.get("keywords", []))
        parts = [
            record.get("title", ""),
            record.get("category", ""),
            keywords,
            record.get("content", ""),
            record.get("example_error", ""),
            record.get("example_fix", ""),
        ]
        return "\n".join(part for part in parts if part)


def format_rag_context(records: list[dict]) -> str:
    blocks = []
    for index, record in enumerate(records, start=1):
        keywords = ", ".join(record.get("keywords", []))
        block = "\n".join(
            [
                f"[知识 {index}] {record.get('title', 'Untitled')}",
                f"类别: {record.get('category', 'general')}",
                f"关键词: {keywords}" if keywords else "关键词: None",
                f"内容: {record.get('content', '')}",
                f"错误示例: {record.get('example_error', '')}",
                f"纠正方式: {record.get('example_fix', '')}",
            ]
        )
        blocks.append(block)
    return "\n\n".join(blocks)
