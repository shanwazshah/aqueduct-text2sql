"""Demo database.

The notebooks used a single flat `employees` table, which is fine for showing a
tool call but useless for testing an agent crew — there is nothing to join, so
the Lead agent never has a reason to spin up a specialist.

This schema is deliberately just awkward enough to be interesting:
  * five tables that require real joins to answer anything useful
  * a self-referencing manager column (employees.manager_id)
  * a nullable FK (employees.department_id can be NULL for contractors)
  * two columns that are easy to confuse (`orders.amount` vs `order_items.price`)
  * a name collision across tables (`name` exists on three of them)

Those last two exist on purpose. They are the traps a small model falls into,
which is exactly what the critic and repair agents are built to catch.
"""

from __future__ import annotations

from sqlalchemy import text

from ..config import settings
from .engine import get_engine

SCHEMA = """
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS employees;
DROP TABLE IF EXISTS departments;

CREATE TABLE departments (
    id          INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    city        TEXT    NOT NULL,
    budget      REAL    NOT NULL
);

CREATE TABLE employees (
    id            INTEGER PRIMARY KEY,
    name          TEXT    NOT NULL,
    email         TEXT    UNIQUE,
    department_id INTEGER,          -- NULL for contractors
    manager_id    INTEGER,          -- self-reference
    salary        REAL    NOT NULL,
    hired_on      TEXT    NOT NULL,
    FOREIGN KEY (department_id) REFERENCES departments(id),
    FOREIGN KEY (manager_id)    REFERENCES employees(id)
);

CREATE TABLE products (
    id        INTEGER PRIMARY KEY,
    name      TEXT    NOT NULL,
    category  TEXT    NOT NULL,
    unit_cost REAL    NOT NULL
);

CREATE TABLE orders (
    id          INTEGER PRIMARY KEY,
    customer    TEXT    NOT NULL,
    employee_id INTEGER NOT NULL,
    ordered_on  TEXT    NOT NULL,
    status      TEXT    NOT NULL,   -- placed | shipped | cancelled
    amount      REAL    NOT NULL,   -- order total; NOT the per-item price
    FOREIGN KEY (employee_id) REFERENCES employees(id)
);

CREATE TABLE order_items (
    id         INTEGER PRIMARY KEY,
    order_id   INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity   INTEGER NOT NULL,
    price      REAL    NOT NULL,    -- per-unit price at time of sale
    FOREIGN KEY (order_id)   REFERENCES orders(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);
"""

DEPARTMENTS = [
    (1, "Engineering", "Berlin", 2_400_000),
    (2, "Marketing",   "Prague", 890_000),
    (3, "Sales",       "London", 1_350_000),
    (4, "Support",     "Berlin", 610_000),
]

EMPLOYEES = [
    # id, name, email, dept, manager, salary, hired_on
    (1,  "Alice Wong",    "alice@acme.io",   1, None, 142_000, "2018-03-12"),
    (2,  "Bob Iyer",      "bob@acme.io",     1, 1,     98_000, "2019-07-01"),
    (3,  "Carol Mensah",  "carol@acme.io",   1, 1,    105_000, "2020-01-20"),
    (4,  "David Novak",   "david@acme.io",   2, None,  88_000, "2017-11-05"),
    (5,  "Eve Lindqvist", "eve@acme.io",     2, 4,     72_000, "2021-06-14"),
    (6,  "Frank Osei",    "frank@acme.io",   3, None, 115_000, "2016-02-29"),
    (7,  "Grace Bianchi", "grace@acme.io",   3, 6,     94_000, "2019-09-30"),
    (8,  "Heidi Ahmed",   "heidi@acme.io",   3, 6,     81_000, "2022-04-18"),
    (9,  "Ivan Petrov",   "ivan@acme.io",    4, None,  67_000, "2020-08-03"),
    (10, "Judy Kaur",     "judy@acme.io",    4, 9,     59_000, "2023-01-09"),
    (11, "Ken Sato",      "ken@acme.io",     1, 1,    121_000, "2021-10-25"),
    (12, "Lena Fischer",  "lena@ext.io",  None, None,  90_000, "2023-05-02"),  # contractor
]

PRODUCTS = [
    (1, "Atlas Laptop",     "Hardware",     820.00),
    (2, "Nimbus Monitor",   "Hardware",     210.00),
    (3, "Orbit Keyboard",   "Accessories",   45.00),
    (4, "Vertex Dock",      "Accessories",   95.00),
    (5, "Pulse Analytics",  "Software",     150.00),
    (6, "Pulse Analytics+", "Software",     390.00),
]

ORDERS = [
    # id, customer, employee_id, ordered_on, status, amount
    (1,  "Northwind Ltd",  6,  "2024-01-15", "shipped",   4_310.00),
    (2,  "Contoso GmbH",   7,  "2024-01-28", "shipped",   1_265.00),
    (3,  "Fabrikam SA",    6,  "2024-02-04", "cancelled",   980.00),
    (4,  "Northwind Ltd",  8,  "2024-02-19", "shipped",   2_040.00),
    (5,  "Tailspin BV",    7,  "2024-03-02", "placed",      735.00),
    (6,  "Contoso GmbH",   6,  "2024-03-11", "shipped",   6_150.00),
    (7,  "Adventure Inc",  8,  "2024-03-22", "cancelled", 1_420.00),
    (8,  "Fabrikam SA",    7,  "2024-04-07", "shipped",     585.00),
    (9,  "Tailspin BV",    6,  "2024-04-25", "shipped",   3_275.00),
    (10, "Adventure Inc",  8,  "2024-05-09", "placed",    1_890.00),
]

ORDER_ITEMS = [
    # id, order_id, product_id, quantity, price
    (1,  1, 1, 5, 820.00), (2,  1, 3, 4, 45.00),
    (3,  2, 2, 6, 210.00), (4,  2, 3, 1, 45.00),
    (5,  3, 5, 6, 150.00), (6,  4, 1, 2, 820.00),
    (7,  4, 4, 4, 95.00),  (8,  5, 6, 1, 390.00),
    (9,  5, 5, 2, 150.00), (10, 6, 1, 7, 820.00),
    (11, 6, 2, 2, 210.00), (12, 7, 6, 3, 390.00),
    (13, 7, 4, 2, 95.00),  (14, 8, 3, 13, 45.00),
    (15, 9, 1, 3, 820.00), (16, 9, 2, 3, 210.00),
    (17, 10, 6, 4, 390.00), (18, 10, 4, 3, 95.00),
]

_INSERTS = [
    ("departments", "(:id, :name, :city, :budget)",
     ["id", "name", "city", "budget"], DEPARTMENTS),
    ("employees", "(:id, :name, :email, :department_id, :manager_id, :salary, :hired_on)",
     ["id", "name", "email", "department_id", "manager_id", "salary", "hired_on"], EMPLOYEES),
    ("products", "(:id, :name, :category, :unit_cost)",
     ["id", "name", "category", "unit_cost"], PRODUCTS),
    ("orders", "(:id, :customer, :employee_id, :ordered_on, :status, :amount)",
     ["id", "customer", "employee_id", "ordered_on", "status", "amount"], ORDERS),
    ("order_items", "(:id, :order_id, :product_id, :quantity, :price)",
     ["id", "order_id", "product_id", "quantity", "price"], ORDER_ITEMS),
]


def _split_statements(script: str) -> list[str]:
    """Split a DDL script into statements.

    Comments are stripped first, because a `;` inside a `--` comment would
    otherwise split a statement in half — which is exactly what happened the
    first time this ran.
    """
    without_comments = "\n".join(
        line.split("--", 1)[0].rstrip() for line in script.splitlines()
    )
    return [s.strip() for s in without_comments.split(";") if s.strip()]


def seed(db_url: str | None = None) -> dict[str, int]:
    """Create the demo schema and load it. Returns row counts per table.

    Safe to re-run; it drops and rebuilds. This bypasses the read-only guard on
    purpose — setup is not agent-driven, and the guard exists to constrain
    generated SQL, not our own fixtures.
    """
    engine = get_engine(db_url or settings.db_url)

    with engine.begin() as conn:
        for statement in _split_statements(SCHEMA):
            conn.execute(text(statement))

        for table, placeholders, columns, rows in _INSERTS:
            conn.execute(
                text(f"INSERT INTO {table} VALUES {placeholders}"),
                [dict(zip(columns, row)) for row in rows],
            )

    with engine.connect() as conn:
        return {
            table: conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
            for table, *_ in _INSERTS
        }


if __name__ == "__main__":
    counts = seed()
    print(f"Seeded {settings.db_url}")
    for table, n in counts.items():
        print(f"  {table:<14} {n:>4} rows")
