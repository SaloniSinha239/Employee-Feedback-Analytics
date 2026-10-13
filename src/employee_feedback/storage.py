"""SQLite storage layer for feedback and analysis results."""

import sqlite3

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    employee_id INTEGER PRIMARY KEY,
    department TEXT,
    tenure_years REAL,
    engagement_score REAL,
    workload_score REAL,
    salary_satisfaction REAL,
    comment TEXT
);
CREATE TABLE IF NOT EXISTS sentiment_results (
    employee_id INTEGER PRIMARY KEY,
    sentiment_score REAL,
    sentiment_label TEXT,
    FOREIGN KEY (employee_id) REFERENCES feedback(employee_id)
);
CREATE TABLE IF NOT EXISTS topic_results (
    employee_id INTEGER PRIMARY KEY,
    dominant_topic INTEGER,
    FOREIGN KEY (employee_id) REFERENCES feedback(employee_id)
);
CREATE TABLE IF NOT EXISTS segment_results (
    employee_id INTEGER PRIMARY KEY,
    segment TEXT,
    FOREIGN KEY (employee_id) REFERENCES feedback(employee_id)
);
"""


def save_all(db_path, df):
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
        feedback_cols = [
            "employee_id", "department", "tenure_years", "engagement_score",
            "workload_score", "salary_satisfaction", "comment",
        ]
        df[feedback_cols].to_sql("feedback", conn, if_exists="replace", index=False)
        df[["employee_id", "sentiment_score", "sentiment_label"]].to_sql(
            "sentiment_results", conn, if_exists="replace", index=False
        )
        df[["employee_id", "dominant_topic"]].to_sql(
            "topic_results", conn, if_exists="replace", index=False
        )
        df[["employee_id", "segment"]].to_sql(
            "segment_results", conn, if_exists="replace", index=False
        )


def query(db_path, sql):
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(sql, conn)