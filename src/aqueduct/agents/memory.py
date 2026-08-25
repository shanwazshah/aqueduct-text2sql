"""Error memory — how the crew stops repeating itself.

When the Fixer corrects a query, the correction is written here. Later questions
retrieve relevant past corrections and put them in the Writer's prompt, so a
mistake made on Monday is not made again on Friday.

This is Reflexion's idea (Shinn et al., 2023) with one change that matters:
lessons are only stored when the fix is *verified* — the repaired query ran and
the original did not. The source notebook stores a lesson whenever iteration
count exceeded one, which records the crew's guesses alongside its knowledge. A
memory of unverified corrections is worse than no memory, because wrong lessons
get injected into future prompts with the same authority as right ones.

Retrieval is keyword overlap rather than embeddings. It is explainable, has no
model dependency, and at the scale this operates on (hundreds of lessons, not
millions) the ranking difference does not justify the extra moving part.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..config import DATA_DIR

MEMORY_PATH = DATA_DIR / "memory.json"

_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is",
    "are", "was", "were", "how", "what", "which", "who", "many", "much", "show",
    "list", "give", "me", "all", "each", "every", "do", "does", "did", "that",
    "this", "with", "by", "from", "have", "has", "их", "be", "it",
}


@dataclass
class Lesson:
    """One verified correction."""

    question: str
    broken_sql: str
    error: str
    fixed_sql: str
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def render(self) -> str:
        """Render as a rule, not as an example.

        The first version showed the broken query, the error, and the corrected
        query in full. Measured effect: the Writer copied the *shape* of the
        corrected SQL into unrelated questions. A lesson about joining `orders`
        instead of `order_items` was recalled for a revenue-by-department
        question, and the draft came back with the same `LEFT JOIN` chain plus an
        invented `WHERE o.status = 'shipped'` — turning a correct answer into a
        wrong one.

        Showing SQL invites imitation. Showing the rule does not.
        """
        return f"  - {self.error}"

    @property
    def keywords(self) -> set[str]:
        return _keywords(f"{self.question} {self.error}")


class ErrorMemory:
    """Persistent store of verified corrections."""

    def __init__(self, path: Path | None = None, max_lessons: int = 200):
        self.path = path or MEMORY_PATH
        self.max_lessons = max_lessons
        self.lessons: list[Lesson] = self._load()

    def _load(self) -> list[Lesson]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return [Lesson(**entry) for entry in raw]
        except (json.JSONDecodeError, TypeError, OSError):
            return []  # a corrupt memory file should not stop the crew working

    def save(self) -> None:
        try:
            self.path.write_text(
                json.dumps([asdict(l) for l in self.lessons], indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    def record(self, question: str, broken_sql: str, error: str, fixed_sql: str) -> Lesson | None:
        """Store a correction, if it is worth storing.

        Rejected: fixes that changed nothing, and duplicates of a lesson already
        held. Both would dilute retrieval without adding information.
        """
        if _one_line(broken_sql) == _one_line(fixed_sql):
            return None

        lesson = Lesson(
            question=question.strip(),
            broken_sql=broken_sql.strip(),
            error=error.strip(),
            fixed_sql=fixed_sql.strip(),
        )

        for existing in self.lessons:
            if (
                existing.error == lesson.error
                and _one_line(existing.broken_sql) == _one_line(lesson.broken_sql)
            ):
                return None

        self.lessons.append(lesson)
        if len(self.lessons) > self.max_lessons:
            self.lessons = self.lessons[-self.max_lessons :]
        self.save()
        return lesson

    def recall(self, question: str, k: int = 3, min_relevance: float = 0.25) -> list[Lesson]:
        """Return the k most relevant past corrections.

        Relevance is Jaccard overlap, not raw shared-word count. The first
        version admitted any lesson sharing a single keyword, which is how a
        lesson about "employees who never handled an order" was served to a
        question about revenue per department — they share "order" and
        "employee", and nothing else that matters.

        The threshold is the important part. A lesson that is merely
        topically adjacent is worse than no lesson: it occupies context and
        steers the model toward a pattern that does not apply here.
        """
        if not self.lessons:
            return []

        wanted = _keywords(question)
        if not wanted:
            return []

        scored = []
        for i, lesson in enumerate(self.lessons):
            overlap = wanted & lesson.keywords
            union = wanted | lesson.keywords
            relevance = len(overlap) / len(union) if union else 0.0
            if relevance >= min_relevance:
                scored.append((relevance, i, lesson))

        # Ties break towards the more recent lesson.
        scored.sort(key=lambda s: (-s[0], -s[1]))
        return [lesson for _, _, lesson in scored[:k]]

    def render_for_prompt(self, question: str, k: int = 3) -> str:
        """Relevant lessons, formatted for a prompt. Empty if none apply."""
        lessons = self.recall(question, k)
        if not lessons:
            return ""
        body = "\n".join(l.render() for l in lessons)
        return f"Errors seen on similar questions before - avoid them:\n{body}\n"

    def clear(self) -> int:
        count = len(self.lessons)
        self.lessons = []
        self.save()
        return count

    def __len__(self) -> int:
        return len(self.lessons)


def _keywords(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z_]{3,}", text.lower())
        if token not in _STOPWORDS
    }


def _one_line(sql: str) -> str:
    return " ".join(sql.split())
