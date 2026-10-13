import os
import sys
import tempfile

sys.path.insert(0, "src")

import pandas as pd
import streamlit as st

from employee_feedback import generate_data, reporting, segmentation, sentiment, topics

REQUIRED_COLS = [
    "department", "tenure_years", "engagement_score",
    "workload_score", "salary_satisfaction", "comment",
]

st.set_page_config(page_title="Employee Feedback Analytics", page_icon="bar chart", layout="wide")


@st.cache_data
def load_data(source, file_bytes):
    if source == "Sample data (synthetic)":
        return generate_data.generate()
    df = pd.read_csv(file_bytes)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if "employee_id" not in df.columns:
        df.insert(0, "employee_id", range(1, len(df) + 1))
    return df


@st.cache_data
def run_sentiment(df):
    return sentiment.analyze(df)


@st.cache_data
def run_topics(df, n_topics):
    return topics.fit(df, n_topics=n_topics)


@st.cache_data
def run_segments(df):
    return segmentation.fit(df)


@st.cache_data
def build_exports(df, sentiment_summary, topic_summary, segment_profile, kmeans_meta):
    with tempfile.TemporaryDirectory() as td:
        xlsx = reporting.export_excel(df, sentiment_summary, topic_summary, segment_profile, out_dir=td)
        pptx = reporting.export_pptx(sentiment_summary, topic_summary, segment_profile, kmeans_meta, out_dir=td)
        with open(xlsx, "rb") as f:
            xlsx_bytes = f.read()
        with open(pptx, "rb") as f:
            pptx_bytes = f.read()
    return xlsx_bytes, pptx_bytes


with st.sidebar:
    st.title("Employee Feedback Analytics")
    st.caption("Sentiment + LDA topics + K-Means segmentation")
    source = st.radio("Data source", ["Sample data (synthetic)", "Upload CSV"])
    uploaded = None
    if source == "Upload CSV":
        uploaded = st.file_uploader("Survey CSV", type="csv")
        st.caption("Required columns: " + ", ".join(REQUIRED_COLS))
    n_topics = st.slider("LDA topics", 2, 8, 5)
    max_k = st.slider("Max K-Means clusters", 3, 8, 6)
    st.divider()
    st.markdown("[Source code](https://github.com/SaloniSinha239/Employee-Feedback-Analytics) · Built with Streamlit + scikit-learn")

try:
    if uploaded is None and source == "Upload CSV":
        st.info("Upload a CSV to begin, or switch to the sample data.")
        st.stop()
    df = load_data(source, uploaded if uploaded is not None else None)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

with st.spinner("Running pipeline..."):
    df = run_sentiment(df)
    df, topic_summary = run_topics(df, n_topics)
    df, segment_profile, kmeans_meta = run_segments(df)

sentiment_summary = sentiment.summary(df)

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Topics", "Segments", "Exports"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Responses", len(df))
    c2.metric("Mean sentiment", f"{df['sentiment_score'].mean():.2f}")
    c3.metric("Positive", f"{(df['sentiment_label'] == 'positive').mean():.0%}")
    c4.metric("Negative", f"{(df['sentiment_label'] == 'negative').mean():.0%}")
    st.bar_chart(
        sentiment_summary.set_index("department")[["pct_positive", "pct_negative"]],
        color=["#2ecc71", "#e74c3c"],
    )
    st.dataframe(sentiment_summary, use_container_width=True, hide_index=True)

with tab2:
    st.caption("Top words per topic, comment volume, and average sentiment per theme")
    st.bar_chart(topic_summary.set_index("topic_id")["n_comments"])
    st.dataframe(topic_summary[["topic_id", "top_words", "n_comments", "mean_sentiment"]],
                 use_container_width=True, hide_index=True)
    st.dataframe(
        df[df["sentiment_label"] == "negative"][["department", "topic_label", "comment"]].head(10),
        use_container_width=True, hide_index=True,
    )

with tab3:
    c1, c2 = st.columns([1, 3])
    c1.metric("Best k", kmeans_meta["best_k"])
    c1.metric("Silhouette", kmeans_meta["silhouette"])
    c2.caption("K-Means over engagement, workload, salary satisfaction, tenure and sentiment")
    st.dataframe(segment_profile, use_container_width=True, hide_index=True)
    st.bar_chart(segment_profile.set_index("segment")[["engagement", "salary_sat", "sentiment"]])

with tab4:
    xlsx_bytes, pptx_bytes = build_exports(df, sentiment_summary, topic_summary, segment_profile, kmeans_meta)
    d1, d2 = st.columns(2)
    d1.download_button("Download Excel workbook", xlsx_bytes,
                       file_name="feedback_insights.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    d2.download_button("Download PowerPoint deck", pptx_bytes,
                       file_name="feedback_insights.pptx",
                       mime="application/vnd.openxmlformats-officedocument.presentationml.presentation")
