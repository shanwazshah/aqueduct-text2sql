"""Evaluation questions for the demo database.

Twelve questions with reference SQL, graded by comparing result sets rather than
query text — there are many correct ways to write the same query, and string
comparison would fail all but one of them.

The `trap` field records which deliberate schema hazard each question probes.
That is the useful part: when a strategy scores 8/12, the traps tell you *which*
kind of mistake it is still making, which is what decides what to build next.

This is the fast feedback loop. BIRD is the real benchmark, but it needs the
Kaggle tier; this runs on a laptop in a couple of minutes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    gold_sql: str
    difficulty: str  # easy | medium | hard
    trap: str | None = None


QUESTIONS: list[Question] = [
    Question(
        id="q01",
        question="How many employees are there in total?",
        gold_sql="SELECT COUNT(*) FROM employees",
        difficulty="easy",
    ),
    Question(
        id="q02",
        question="List every department name and the city it is in.",
        gold_sql="SELECT name, city FROM departments",
        difficulty="easy",
    ),
    Question(
        id="q03",
        question="What is the average salary in the Engineering department?",
        gold_sql=(
            "SELECT AVG(e.salary) FROM employees e "
            "JOIN departments d ON e.department_id = d.id "
            "WHERE d.name = 'Engineering'"
        ),
        difficulty="easy",
    ),
    Question(
        id="q04",
        question="How many employees work in each department? Include departments with nobody in them.",
        gold_sql=(
            "SELECT d.name, COUNT(e.id) FROM departments d "
            "LEFT JOIN employees e ON e.department_id = d.id GROUP BY d.name"
        ),
        difficulty="medium",
        trap="nullable-fk",
    ),
    Question(
        id="q05",
        question="Which employees do not belong to any department?",
        gold_sql="SELECT name FROM employees WHERE department_id IS NULL",
        difficulty="medium",
        trap="nullable-fk",
    ),
    Question(
        id="q06",
        question="What is the total value of all shipped orders?",
        gold_sql="SELECT SUM(amount) FROM orders WHERE status = 'shipped'",
        difficulty="medium",
        trap="amount-vs-price",
    ),
    Question(
        id="q07",
        question="Which product category generated the most revenue across all order items?",
        gold_sql=(
            "SELECT p.category, SUM(oi.quantity * oi.price) AS revenue "
            "FROM order_items oi JOIN products p ON oi.product_id = p.id "
            "GROUP BY p.category ORDER BY revenue DESC LIMIT 1"
        ),
        difficulty="hard",
        trap="amount-vs-price",
    ),
    Question(
        id="q08",
        question="List each employee together with the name of their manager.",
        gold_sql=(
            "SELECT e.name, m.name FROM employees e "
            "LEFT JOIN employees m ON e.manager_id = m.id"
        ),
        difficulty="hard",
        trap="self-join",
    ),
    Question(
        id="q09",
        question="Which employees earn more than the average salary across the company?",
        gold_sql=(
            "SELECT name FROM employees "
            "WHERE salary > (SELECT AVG(salary) FROM employees)"
        ),
        difficulty="medium",
    ),
    Question(
        id="q10",
        question="How many orders did each salesperson handle? Show their name.",
        gold_sql=(
            "SELECT e.name, COUNT(o.id) FROM employees e "
            "JOIN orders o ON o.employee_id = e.id GROUP BY e.name"
        ),
        difficulty="medium",
        trap="name-collision",
    ),
    Question(
        id="q11",
        question="Which department has the highest total salary spend?",
        gold_sql=(
            "SELECT d.name, SUM(e.salary) AS total FROM departments d "
            "JOIN employees e ON e.department_id = d.id "
            "GROUP BY d.name ORDER BY total DESC LIMIT 1"
        ),
        difficulty="medium",
        trap="name-collision",
    ),
    Question(
        id="q12",
        question="How many employees were hired in 2021 or later?",
        gold_sql="SELECT COUNT(*) FROM employees WHERE hired_on >= '2021-01-01'",
        difficulty="hard",
        trap="date-as-text",
    ),
]


# ── the hard set ─────────────────────────────────────────────────────
#
# The twelve questions above turned out to be too easy to be useful for
# comparing anything. Every repair mode scored an identical 91.7% and not a
# single repair was triggered — the Writer got them right first time, so the
# critics and the Fixer never ran. A test set that nothing fails cannot rank
# strategies.
#
# These are built to break a 3B model specifically: multi-hop joins, per-group
# extremes, correlated subqueries, integer-division ratios, and the
# amount-vs-price trap made unavoidable rather than optional.

HARD_QUESTIONS: list[Question] = [
    Question(
        id="h01",
        question="For each department, who is the highest paid employee? Show the department and the employee name.",
        gold_sql=(
            "SELECT d.name, e.name FROM employees e "
            "JOIN departments d ON e.department_id = d.id "
            "WHERE e.salary = (SELECT MAX(e2.salary) FROM employees e2 "
            "                  WHERE e2.department_id = e.department_id)"
        ),
        difficulty="hard",
        trap="per-group-extreme",
    ),
    Question(
        id="h02",
        question="Which employees have never handled an order?",
        gold_sql=(
            "SELECT e.name FROM employees e "
            "WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.employee_id = e.id)"
        ),
        difficulty="hard",
        trap="anti-join",
    ),
    Question(
        id="h03",
        question="What percentage of all orders were cancelled?",
        gold_sql=(
            "SELECT 100.0 * SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) "
            "/ COUNT(*) FROM orders"
        ),
        difficulty="hard",
        trap="integer-division",
    ),
    Question(
        id="h04",
        question="What is the total revenue from order items, broken down by the department of the salesperson who handled the order?",
        gold_sql=(
            "SELECT d.name, SUM(oi.quantity * oi.price) FROM order_items oi "
            "JOIN orders o ON oi.order_id = o.id "
            "JOIN employees e ON o.employee_id = e.id "
            "JOIN departments d ON e.department_id = d.id "
            "GROUP BY d.name"
        ),
        difficulty="hard",
        trap="multi-hop-join",
    ),
    Question(
        id="h05",
        question="Which manager has the most direct reports? Show their name and the count.",
        gold_sql=(
            "SELECT m.name, COUNT(e.id) AS reports FROM employees e "
            "JOIN employees m ON e.manager_id = m.id "
            "GROUP BY m.name ORDER BY reports DESC LIMIT 1"
        ),
        difficulty="hard",
        trap="self-join",
    ),
    Question(
        id="h06",
        question="For each product, how many distinct customers have bought it?",
        gold_sql=(
            "SELECT p.name, COUNT(DISTINCT o.customer) FROM products p "
            "JOIN order_items oi ON oi.product_id = p.id "
            "JOIN orders o ON oi.order_id = o.id "
            "GROUP BY p.name"
        ),
        difficulty="hard",
        trap="distinct-across-join",
    ),
    Question(
        id="h07",
        question="Which departments have an average salary above the company-wide average salary?",
        gold_sql=(
            "SELECT d.name FROM departments d "
            "JOIN employees e ON e.department_id = d.id "
            "GROUP BY d.name "
            "HAVING AVG(e.salary) > (SELECT AVG(salary) FROM employees)"
        ),
        difficulty="hard",
        trap="having-with-subquery",
    ),
    Question(
        id="h08",
        question="For each order, show the order id, the recorded order amount, and the sum of its line items.",
        gold_sql=(
            "SELECT o.id, o.amount, SUM(oi.quantity * oi.price) FROM orders o "
            "JOIN order_items oi ON oi.order_id = o.id GROUP BY o.id, o.amount"
        ),
        difficulty="hard",
        trap="amount-vs-price",
    ),
    Question(
        id="h09",
        question="How many orders were placed in the first quarter of 2024?",
        gold_sql=(
            "SELECT COUNT(*) FROM orders "
            "WHERE ordered_on >= '2024-01-01' AND ordered_on <= '2024-03-31'"
        ),
        difficulty="hard",
        trap="date-as-text",
    ),
    Question(
        id="h10",
        question="Which product category has the highest total revenue from shipped orders only?",
        gold_sql=(
            "SELECT p.category, SUM(oi.quantity * oi.price) AS revenue "
            "FROM order_items oi "
            "JOIN products p ON oi.product_id = p.id "
            "JOIN orders o ON oi.order_id = o.id "
            "WHERE o.status = 'shipped' "
            "GROUP BY p.category ORDER BY revenue DESC LIMIT 1"
        ),
        difficulty="hard",
        trap="filter-across-join",
    ),
]


ALL_QUESTIONS: list[Question] = QUESTIONS + HARD_QUESTIONS

BY_ID = {q.id: q for q in ALL_QUESTIONS}


def by_difficulty(level: str) -> list[Question]:
    return [q for q in ALL_QUESTIONS if q.difficulty == level]
