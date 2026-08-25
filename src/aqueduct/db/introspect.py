"""Reading the database's structure.

What the Writer agent sees about the schema determines almost everything about
whether it writes correct SQL. Two details matter more than they look:

  * **Foreign keys are shown explicitly.** Without them a model guesses join
    conditions from column names, which is how `JOIN orders ON orders.id =
    employees.id` happens.
  * **Sample values are included.** A model that cannot see that `status` holds
    `'shipped'` will confidently filter for `'Shipped'` or `'complete'` and get
    zero rows back — valid SQL, wrong answer, and no error to learn from.

The second one is the difference between a query that runs and a query that is
right, and it is invisible from column names alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import inspect

from ..config import settings
from .engine import get_engine


@dataclass
class Column:
    name: str
    type: str
    nullable: bool
    primary_key: bool = False
    samples: list[str] = field(default_factory=list)

    def render(self) -> str:
        bits = [f"{self.name} {self.type}"]
        if self.primary_key:
            bits.append("PK")
        if not self.nullable:
            bits.append("NOT NULL")
        line = "    " + " ".join(bits)
        if self.samples:
            line += f"  -- e.g. {', '.join(self.samples)}"
        return line


@dataclass
class ForeignKey:
    column: str
    ref_table: str
    ref_column: str

    def render(self) -> str:
        return f"    {self.column} -> {self.ref_table}.{self.ref_column}"


@dataclass
class Table:
    name: str
    columns: list[Column]
    foreign_keys: list[ForeignKey] = field(default_factory=list)
    row_count: int | None = None

    def render(self) -> str:
        header = f"TABLE {self.name}"
        if self.row_count is not None:
            header += f"  ({self.row_count} rows)"
        parts = [header, *(c.render() for c in self.columns)]
        if self.foreign_keys:
            parts.append("  FOREIGN KEYS:")
            parts.extend(fk.render() for fk in self.foreign_keys)
        return "\n".join(parts)


@dataclass
class Schema:
    tables: list[Table]

    @property
    def table_names(self) -> list[str]:
        return [t.name for t in self.tables]

    def subset(self, names: list[str]) -> "Schema":
        """Narrow to specific tables.

        The Scout agent uses this: on a wide database, sending every table wastes
        context and gives the model more chances to pick the wrong one.
        """
        wanted = {n.lower() for n in names}
        return Schema([t for t in self.tables if t.name.lower() in wanted])

    def render(self) -> str:
        """The schema as the model sees it."""
        return "\n\n".join(t.render() for t in self.tables)

    def render_compact(self) -> str:
        """One line per table. For cheap calls like routing, where the Lead
        needs to know what exists but not the details."""
        return "\n".join(
            f"{t.name}({', '.join(c.name for c in t.columns)})" for t in self.tables
        )


def load_schema(
    db_url: str | None = None,
    *,
    sample_values: bool = True,
    max_samples: int = 3,
    include_row_counts: bool = True,
) -> Schema:
    """Introspect the database into a Schema."""
    engine = get_engine(db_url or settings.db_url)
    inspector = inspect(engine)

    tables: list[Table] = []
    for name in sorted(inspector.get_table_names()):
        pk_columns = set(inspector.get_pk_constraint(name).get("constrained_columns") or [])

        columns = [
            Column(
                name=c["name"],
                type=str(c["type"]),
                nullable=bool(c.get("nullable", True)),
                primary_key=c["name"] in pk_columns,
            )
            for c in inspector.get_columns(name)
        ]

        foreign_keys = [
            ForeignKey(
                column=fk["constrained_columns"][0],
                ref_table=fk["referred_table"],
                ref_column=fk["referred_columns"][0],
            )
            for fk in inspector.get_foreign_keys(name)
            if fk.get("constrained_columns") and fk.get("referred_columns")
        ]

        table = Table(name=name, columns=columns, foreign_keys=foreign_keys)

        if include_row_counts:
            table.row_count = _count_rows(engine, name)
        if sample_values:
            _attach_samples(engine, table, max_samples)

        tables.append(table)

    return Schema(tables)


def _count_rows(engine, table: str) -> int | None:
    from sqlalchemy import text

    try:
        with engine.connect() as conn:
            return conn.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar_one()
    except Exception:
        return None


_DATE_HINTS = ("_on", "_at", "date", "time")
_MAX_DISTINCT = 12


def _is_texty(column: Column) -> bool:
    upper = column.type.upper()
    return "CHAR" in upper or "TEXT" in upper or "STRING" in upper


def _attach_samples(engine, table: Table, max_samples: int) -> None:
    """Attach sample values to columns where they change how SQL gets written.

    Two cases earn their place in the prompt:

      * **Categorical columns** — `status`, `category`, `city`. The model has to
        match the stored spelling exactly, and `'Shipped'` instead of
        `'shipped'` returns zero rows with no error to learn from.
      * **Date-like columns** — one sample is enough to show that `hired_on`
        holds `'2018-03-12'` text rather than a native date, which decides
        whether the model reaches for `strftime` or `EXTRACT`.

    Everything else is skipped. A column whose values are nearly all distinct
    (names, emails, ids) is not categorical, and listing three of them teaches
    the model nothing while spending context and inviting it to filter on a
    value it happened to see.
    """
    from sqlalchemy import text

    for column in table.columns:
        if column.primary_key or not _is_texty(column):
            continue

        looks_like_date = any(h in column.name.lower() for h in _DATE_HINTS)

        try:
            with engine.connect() as conn:
                rows = conn.execute(
                    text(
                        f'SELECT DISTINCT "{column.name}" FROM "{table.name}" '
                        f'WHERE "{column.name}" IS NOT NULL LIMIT {_MAX_DISTINCT + 1}'
                    )
                ).fetchall()
        except Exception:
            continue

        values = [str(r[0]) for r in rows]
        if not values:
            continue

        if looks_like_date:
            column.samples = [repr(values[0])]  # format hint only
            continue

        distinct = len(values)
        total = table.row_count or 0
        if distinct > _MAX_DISTINCT:
            continue

        # Categorical if the values repeat, or the table is small enough that
        # they plainly enumerate the domain.
        if distinct < total or total <= 6:
            column.samples = [repr(v) for v in values[:max_samples]]


def schema_card(db_url: str | None = None, tables: list[str] | None = None) -> str:
    """Convenience: the rendered schema string agents put in their prompts."""
    schema = load_schema(db_url)
    if tables:
        schema = schema.subset(tables)
    return schema.render()
