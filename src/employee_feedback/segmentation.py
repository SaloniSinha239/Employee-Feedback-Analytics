"""K-Means employee segmentation."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "engagement_score",
    "workload_score",
    "salary_satisfaction",
    "tenure_years",
    "sentiment_score",
]

SEGMENT_NAMES = {
    0: "Segment 0",
    1: "Segment 1",
    2: "Segment 2",
    3: "Segment 3",
    4: "Segment 4",
}


def fit(df, k_range=range(2, 7), seed=42):
    X = df[FEATURES].values
    X_scaled = StandardScaler().fit_transform(X)

    best_k, best_score = None, -1.0
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X_scaled)
        score = silhouette_score(X_scaled, km.labels_)
        if score > best_score:
            best_k, best_score, best_model = k, score, km

    out = df.copy()
    out["segment"] = best_model.labels_
    out["segment"] = out["segment"].map(SEGMENT_NAMES)

    profile = (
        out.groupby("segment")
        .agg(
            headcount=("segment", "size"),
            engagement=("engagement_score", "mean"),
            workload=("workload_score", "mean"),
            salary_sat=("salary_satisfaction", "mean"),
            tenure=("tenure_years", "mean"),
            sentiment=("sentiment_score", "mean"),
        )
        .round(2)
        .reset_index()
    )
    return out, profile, {"best_k": best_k, "silhouette": round(best_score, 3)}