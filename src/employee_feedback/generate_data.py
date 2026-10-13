"""Synthetic employee survey data generator for the analytics pipeline."""

import numpy as np
import pandas as pd

TOPIC_TEMPLATES = {
    "compensation": {
        "positive": [
            "the pay raise this year was fair and competitive",
            "salary and benefits package meet my expectations",
            "compensation is transparent and rewards performance",
        ],
        "negative": [
            "the salary is too low for the workload",
            "benefits have been cut and bonuses delayed",
            "pay does not match the cost of living increase",
        ],
    },
    "management": {
        "positive": [
            "my manager gives clear direction and useful feedback",
            "leadership communicates goals well and trusts the team",
            "regular one on ones with my manager are helpful",
        ],
        "negative": [
            "management never listens to team concerns",
            "unclear expectations from leadership cause rework",
            "my manager micromanages every task",
        ],
    },
    "work_life_balance": {
        "positive": [
            "flexible hours let me manage family commitments",
            "the remote work policy supports a healthy balance",
            "workload is reasonable and overtime is rare",
        ],
        "negative": [
            "constant overtime is burning the team out",
            "no support for flexible schedules here",
            "work follows me home every weekend",
        ],
    },
    "career_growth": {
        "positive": [
            "training budget and promotion paths are clear",
            "I see real opportunities to grow my career here",
            "mentorship program helped me develop new skills",
        ],
        "negative": [
            "there is no path for promotion in this role",
            "requests for training are always rejected",
            "career growth discussions never happen",
        ],
    },
    "culture": {
        "positive": [
            "colleagues collaborate and celebrate wins together",
            "the team culture is inclusive and supportive",
            "great atmosphere and friendly coworkers",
        ],
        "negative": [
            "blame culture makes people afraid to speak up",
            "teams work in silos and rarely share knowledge",
            "office politics hurt morale",
        ],
    },
}

TOPICS = list(TOPIC_TEMPLATES)
NEUTRAL_FILLER = [
    "the survey asks about my experience this quarter",
    "please share results with the team",
    "I completed the annual engagement survey today",
]


def generate(n=500, seed=42):
    rng = np.random.default_rng(seed)
    departments = rng.choice(
        ["Engineering", "Sales", "HR", "Operations", "Finance"], size=n
    )
    tenure = rng.gamma(2.0, 2.5, size=n).clip(0.2, 20).round(1)
    engagement = rng.normal(6.5, 1.8, size=n).clip(1, 10).round(1)
    workload = rng.normal(6.0, 2.0, size=n).clip(1, 10).round(1)
    salary_satisfaction = rng.normal(5.8, 2.0, size=n).clip(1, 10).round(1)

    rows = []
    for i in range(n):
        topic = rng.choice(TOPICS)
        # Low engagement/workload pressure skews comments negative
        p_neg = 0.15 + 0.05 * (10 - engagement[i]) + 0.03 * (workload[i] - 5)
        p_neg = float(np.clip(p_neg, 0.1, 0.85))
        p_neutral = 0.1
        u = rng.random()
        if u < p_neg:
            text = rng.choice(TOPIC_TEMPLATES[topic]["negative"])
        elif u < p_neg + p_neutral:
            text = rng.choice(NEUTRAL_FILLER)
        else:
            text = rng.choice(TOPIC_TEMPLATES[topic]["positive"])
        rows.append(
            {
                "employee_id": i + 1,
                "department": departments[i],
                "tenure_years": tenure[i],
                "engagement_score": engagement[i],
                "workload_score": workload[i],
                "salary_satisfaction": salary_satisfaction[i],
                "comment": text,
                "true_topic": topic,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/survey.csv", index=False)
    print(f"Wrote data/survey.csv with {len(df)} rows")