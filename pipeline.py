"""End-to-end employee feedback analytics pipeline."""

import argparse
import sys

sys.path.insert(0, "src")

from employee_feedback import generate_data, reporting, segmentation, sentiment, storage, topics


def run(n=500, seed=42, skip_generate=False):
    if skip_generate:
        import pandas as pd

        df = pd.read_csv("data/survey.csv")
    else:
        df = generate_data.generate(n=n, seed=seed)
        df.to_csv("data/survey.csv", index=False)

    print(f"[1/5] Data: {len(df)} responses, {df['department'].nunique()} departments")

    df = sentiment.analyze(df)
    print(f"[2/5] Sentiment: {df['sentiment_label'].value_counts().to_dict()}")

    df, topic_summary = topics.fit(df, n_topics=5)
    print(f"[3/5] LDA topics fitted: {len(topic_summary)} topics")

    df, segment_profile, kmeans_meta = segmentation.fit(df)
    print(f"[4/5] K-Means: k={kmeans_meta['best_k']} (silhouette={kmeans_meta['silhouette']})")

    storage.save_all("outputs/feedback.db", df)
    sample_sql = """
        SELECT f.department, AVG(s.sentiment_score) AS avg_sentiment
        FROM feedback f JOIN sentiment_results s USING(employee_id)
        GROUP BY f.department ORDER BY avg_sentiment
    """
    print("[5/5] SQL check (lowest-sentiment departments):")
    print(storage.query("outputs/feedback.db", sample_sql).to_string(index=False))

    xlsx = reporting.export_excel(df, sentiment.summary(df), topic_summary, segment_profile)
    pptx = reporting.export_pptx(sentiment.summary(df), topic_summary, segment_profile, kmeans_meta)
    print(f"\nDeliverables:\n  {xlsx}\n  {pptx}\n  outputs/feedback.db")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-generate", action="store_true")
    args = parser.parse_args()
    run(n=args.n, seed=args.seed, skip_generate=args.skip_generate)