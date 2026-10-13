"""LDA topic modelling over survey comments."""

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

STOPWORDS = [
    "the", "is", "and", "my", "to", "of", "in", "for", "this", "that",
    "with", "are", "be", "was", "it", "on", "i", "a", "an", "have",
    "has", "there", "at", "from", "here", "every", "always",
]


def fit(df, text_col="comment", n_topics=5, top_words=8, seed=42):
    vectorizer = CountVectorizer(
        max_df=0.9,
        min_df=5 if len(df) >= 50 else 1,
        stop_words=STOPWORDS,
        ngram_range=(1, 2),
    )
    dtm = vectorizer.fit_transform(df[text_col])

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=20,
        learning_method="online",
        random_state=seed,
    )
    doc_topic = lda.fit_transform(dtm)

    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for k, topic in enumerate(lda.components_):
        words = [feature_names[i] for i in topic.argsort()[:-top_words - 1:-1]]
        topics.append({"topic_id": k, "top_words": ", ".join(words)})

    topic_labels = {t["topic_id"]: f"T{t['topic_id']}: {t['top_words'][:40]}" for t in topics}

    out = df.copy()
    out["dominant_topic"] = doc_topic.argmax(axis=1)
    out["topic_label"] = out["dominant_topic"].map(topic_labels)

    topic_summary = (
        out.groupby("dominant_topic")
        .agg(
            n_comments=("dominant_topic", "size"),
            mean_sentiment=("sentiment_score", "mean"),
        )
        .reset_index()
        .merge(pd.DataFrame(topics), left_on="dominant_topic", right_on="topic_id")
        .drop(columns="topic_id")
    )
    return out, topic_summary