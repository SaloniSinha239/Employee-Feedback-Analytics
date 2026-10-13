"""Sentiment analysis of survey comments using VADER."""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def analyze(df, text_col="comment"):
    analyzer = SentimentIntensityAnalyzer()
    scores = df[text_col].apply(lambda t: analyzer.polarity_scores(str(t))["compound"])

    out = df.copy()
    out["sentiment_score"] = scores.round(3)
    out["sentiment_label"] = pd.cut(
        scores,
        bins=[-1.0, -0.05, 0.05, 1.0],
        labels=["negative", "neutral", "positive"],
    ).astype(str)
    return out


def summary(df):
    by_dept = (
        df.groupby("department")
        .agg(
            n=("sentiment_score", "size"),
            mean_sentiment=("sentiment_score", "mean"),
            pct_positive=("sentiment_label", lambda s: (s == "positive").mean()),
            pct_negative=("sentiment_label", lambda s: (s == "negative").mean()),
        )
        .round(3)
        .reset_index()
    )
    return by_dept