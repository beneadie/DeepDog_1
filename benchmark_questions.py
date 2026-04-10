"""Benchmark questions for deep research evaluation.

Loads all 100 questions from data/prompt_data/query.jsonl and provides
helper functions for filtering by id, topic, or language.
"""

import json
from pathlib import Path

_QUERY_FILE = Path(__file__).parent / "data" / "prompt_data" / "query.jsonl"

QUESTIONS: list[dict] = []
with open(_QUERY_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            QUESTIONS.append(json.loads(line))


def get_question(question_id: int) -> dict | None:
    """Return the question with the given id, or None if not found."""
    for q in QUESTIONS:
        if q["id"] == question_id:
            return q
    return None


def get_questions_by_topic(topic: str) -> list[dict]:
    """Return all questions matching the given topic (case-insensitive)."""
    topic_lower = topic.lower()
    return [q for q in QUESTIONS if q["topic"].lower() == topic_lower]


def get_questions_by_language(lang: str) -> list[dict]:
    """Return all questions matching the given language code."""
    return [q for q in QUESTIONS if q["language"] == lang]


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    print(f"Total questions: {len(QUESTIONS)}\n")
    for q in QUESTIONS:
        print(f"[{q['id']:>3}] ({q['language']}) [{q['topic']}]")
        print(f"      {q['prompt'][:120]}")
        print()
