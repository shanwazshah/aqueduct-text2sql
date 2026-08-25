"""The Critic agent — reviews SQL before anyone trusts it.

This agent exists to catch what execution cannot. A query with the wrong join,
the wrong aggregate, or a plausible-but-wrong column runs perfectly and returns a
confidently incorrect answer. The database has no opinion about whether the
answer is *right*; only about whether the SQL is *legal*.

That is the whole case for a critic — and the Phase 1 finding is the whole case
against relying on it. `qwen2.5-coder:3b` reviewed `SELECT dept FROM employees`,
having been told the column is `department`, and returned
`schema_ok: true, confidence: 0.9`.

So this agent is built with a hard rule: **checks that can be made mechanically
are made mechanically, and the model is only asked about what is left.** Column
existence is verified against the real schema in Python before the model is
consulted, because that is a set-membership test and a language model is the
wrong tool for it. What reaches the model is the genuinely semantic question:
does this query answer what was asked?
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from ..db.introspect import Schema
from ..llm.client import LLMClient, LLMError
from ..observability.trace import Trace
from .writer import _NullSpan

SYSTEM_PROMPT = """You review SQL queries for correctness against a question.

You are NOT checking syntax or whether columns exist - that has already been \
verified mechanically. Judge only whether the query answers the question asked.

Look for:
- Wrong aggregate: COUNT where SUM was meant, AVG over the wrong column.
- Wrong join direction or type: an INNER JOIN that silently drops rows the \
question wants included.
- Missing or wrong filter: a condition the question implies but the query omits.
- Wrong grouping: results grouped at a different level than the question asks.
- Answering a different question than the one asked.

Be specific. "The query counts orders but the question asks for total value" is \
useful. "The query may be incorrect" is not.

If the query correctly answers the question, say so with high confidence and an \
empty issues list. Do not invent problems."""

USER_PROMPT = """Question: {question}

Schema:
{schema}

SQL:
{sql}
{execution}
Does this query correctly answer the question?"""


class Critique(BaseModel):
    """The Critic's structured verdict."""

    answers_question: bool = Field(description="Does the query answer what was asked?")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence from 0 to 1.")
    issues: list[str] = Field(default_factory=list, description="Specific problems found.")
    suggestion: str = Field(default="", description="One-line fix, or empty if none needed.")


@dataclass
class Review:
    """A critique combined with the mechanical checks that preceded it."""

    schema_errors: list[str] = field(default_factory=list)
    critique: Critique | None = None
    voted_bad: int = 0
    votes_cast: int = 0

    @property
    def has_schema_error(self) -> bool:
        return bool(self.schema_errors)

    @property
    def is_clean(self) -> bool:
        if self.schema_errors:
            return False
        if self.critique is None:
            return True
        return self.critique.answers_question and not self.critique.issues

    def feedback(self) -> str:
        """Render as repair instructions for the Fixer."""
        parts = []
        if self.schema_errors:
            parts.append(
                "Schema errors (verified against the real database):\n"
                + "\n".join(f"  - {e}" for e in self.schema_errors)
            )
        if self.critique and self.critique.issues:
            parts.append(
                "Review found:\n"
                + "\n".join(f"  - {i}" for i in self.critique.issues)
            )
        if self.critique and self.critique.suggestion:
            parts.append(f"Suggested fix: {self.critique.suggestion}")
        return "\n\n".join(parts)


# ── mechanical checks ────────────────────────────────────────────────

# Identifiers that appear after FROM/JOIN, or in table.column form. Deliberately
# simple: it is a screen for obvious hallucination, not a SQL parser.
_TABLE_REF = re.compile(r"\b(?:FROM|JOIN)\s+[\"'`\[]?(\w+)", re.IGNORECASE)
_QUALIFIED = re.compile(r"\b(\w+)\.(\w+)\b")

_SQL_KEYWORDS = {
    "select", "from", "where", "group", "order", "by", "having", "join", "on",
    "as", "and", "or", "not", "in", "is", "null", "count", "sum", "avg", "min",
    "max", "distinct", "limit", "offset", "left", "right", "inner", "outer",
    "full", "cross", "union", "all", "case", "when", "then", "else", "end",
    "asc", "desc", "with", "over", "partition",
}


def check_against_schema(sql: str, schema: Schema) -> list[str]:
    """Verify tables and columns exist, without asking a model.

    This is a set-membership test. Handing it to a language model — as the source
    notebooks do via `schema_ok` in the critique JSON — converts a decidable
    question into a probabilistic one, and Phase 1 showed a 3B model getting it
    backwards with 0.9 confidence.

    Aliases are resolved so `e.department` is checked against `employees`.
    """
    errors: list[str] = []
    known_tables = {t.name.lower(): t for t in schema.tables}
    columns_by_table = {
        t.name.lower(): {c.name.lower() for c in t.columns} for t in schema.tables
    }
    all_columns = {c for cols in columns_by_table.values() for c in cols}

    referenced = {m.lower() for m in _TABLE_REF.findall(sql)}
    for table in referenced:
        if table not in known_tables:
            close = _closest(table, known_tables.keys())
            hint = f" Did you mean '{close}'?" if close else ""
            errors.append(f"table '{table}' does not exist.{hint}")

    aliases = _alias_map(sql, set(known_tables))

    for qualifier, column in _QUALIFIED.findall(sql):
        q, c = qualifier.lower(), column.lower()
        if c in _SQL_KEYWORDS:
            continue
        table = aliases.get(q, q)
        if table not in columns_by_table:
            continue  # unknown table already reported above
        if c not in columns_by_table[table]:
            close = _closest(c, columns_by_table[table])
            hint = f" Did you mean '{close}'?" if close else ""
            errors.append(f"column '{column}' does not exist on table '{table}'.{hint}")

    # Unqualified columns. When the query touches exactly one table, that
    # table's columns are the candidate set — which makes the "did you mean"
    # hint useful. Searching every column in the database instead produced
    # `dept -> budget` when the answer was plainly `department_id`.
    single_table = referenced & set(columns_by_table)
    candidates = (
        columns_by_table[next(iter(single_table))]
        if len(single_table) == 1
        else all_columns
    )

    for token in _unqualified_identifiers(sql):
        if token not in candidates and token not in known_tables:
            close = _closest(token, candidates)
            hint = f" Did you mean '{close}'?" if close else ""
            errors.append(f"column '{token}' does not exist.{hint}")

    return errors


def _alias_map(sql: str, tables: set[str]) -> dict[str, str]:
    """Map alias -> real table name, for `FROM employees AS e` and `FROM employees e`."""
    pattern = re.compile(
        r"\b(?:FROM|JOIN)\s+[\"'`\[]?(\w+)[\"'`\]]?\s+(?:AS\s+)?(\w+)", re.IGNORECASE
    )
    aliases = {t: t for t in tables}
    for table, alias in pattern.findall(sql):
        if alias.lower() not in _SQL_KEYWORDS and table.lower() in tables:
            aliases[alias.lower()] = table.lower()
    return aliases


def _unqualified_identifiers(sql: str) -> set[str]:
    """Bare identifiers in the SELECT list, for single-table queries.

    Only applied when the query has no alias-qualified references, since
    resolving a bare column across joined tables needs a real parser and the
    guesses would produce false positives.
    """
    if "." in sql:
        return set()
    match = re.search(r"\bSELECT\b(.*?)\bFROM\b", sql, re.IGNORECASE | re.DOTALL)
    if not match:
        return set()
    return {
        token.lower()
        for token in re.findall(r"\b[a-zA-Z_]\w*\b", match.group(1))
        if token.lower() not in _SQL_KEYWORDS
    }


def _closest(name: str, candidates) -> str | None:
    """Nearest known identifier, for a 'did you mean' hint.

    The hint matters more than it looks: `no such column: dept` tells the Fixer
    something is wrong, while `did you mean 'department'` tells it what to write.
    """
    import difflib

    matches = difflib.get_close_matches(name, list(candidates), n=1, cutoff=0.6)
    return matches[0] if matches else None


# ── the agent ────────────────────────────────────────────────────────


class Critic:
    """Reviews a query. Mechanical checks first, model second."""

    name = "critic"

    def __init__(self, client: LLMClient | None = None, schema: Schema | None = None):
        self.client = client or LLMClient(role="critic")
        self.schema = schema

    def review(
        self,
        question: str,
        sql: str,
        *,
        schema: Schema | None = None,
        execution_error: str | None = None,
        ask_model: bool = True,
        trace: Trace | None = None,
        label: str = "reviewing SQL",
    ) -> Review:
        """Review a query and return findings."""
        active_schema = schema or self.schema
        review = Review()

        span_ctx = trace.span(self.name, label) if trace else _NullSpan()
        with span_ctx as span:
            if active_schema is not None:
                review.schema_errors = check_against_schema(sql, active_schema)

            # If the schema is provably wrong there is nothing to deliberate
            # about, and asking the model would only spend 18 seconds to agree.
            if review.schema_errors:
                span.finish(verdict="schema-error", issues=review.schema_errors)
                return review

            if not ask_model:
                span.finish(verdict="mechanical-only")
                return review

            execution_note = ""
            if execution_error:
                execution_note = f"\nThe database rejected this query: {execution_error}\n"

            try:
                review.critique = self.client.structured(
                    SYSTEM_PROMPT,
                    USER_PROMPT.format(
                        question=question,
                        schema=active_schema.render() if active_schema else "(unavailable)",
                        sql=sql,
                        execution=execution_note,
                    ),
                    Critique,
                )
            except LLMError as e:
                # A critic that cannot answer must not block the pipeline; the
                # query still faces execution, which is the stronger check.
                span.fail(str(e))
                return review

            span.finish(
                verdict="ok" if review.is_clean else "issues",
                confidence=review.critique.confidence,
                issues=review.critique.issues,
            )

        return review
