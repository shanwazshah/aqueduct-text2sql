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


BY_ID = {q.id: q for q in QUESTIONS}


def by_difficulty(level: str) -> list[Question]:
    return [q for q in QUESTIONS if q.difficulty == level]
