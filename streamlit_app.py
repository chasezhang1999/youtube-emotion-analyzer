from pathlib import Path
import os

import pandas as pd
import streamlit as st

from youtube_emotion.core import build_marketing_recommendation, parse_video_id, summarize_predictions
from youtube_emotion.model_runner import (
    DEFAULT_EMOTION_MODEL,
    DEFAULT_SENTIMENT_MODEL,
    load_text_classification_pipeline,
    predict_comment_emotions,
)
from youtube_emotion.youtube_client import fetch_top_comments


PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_COMMENTS_PATH = PROJECT_ROOT / "data" / "sample_comments.csv"


st.set_page_config(
    page_title="YouTube Audience Emotion Analyzer",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_pipeline(model_name: str):
    return load_text_classification_pipeline(model_name)


@st.cache_data(show_spinner=False)
def load_sample_comments() -> list[str]:
    sample_df = pd.read_csv(SAMPLE_COMMENTS_PATH)
    return sample_df["comment"].dropna().astype(str).tolist()


def get_secret_value(name: str) -> str:
    env_value = os.getenv(name, "")
    if env_value:
        return env_value

    try:
        return str(st.secrets.get(name, ""))
    except Exception:
        return ""


def display_summary(rows: list[dict]) -> None:
    summary = summarize_predictions(rows)
    recommendation = build_marketing_recommendation(summary)
    result_df = pd.DataFrame(rows)

    metric_cols = st.columns(3)
    metric_cols[0].metric("Comments analyzed", summary["total_comments"])
    metric_cols[1].metric("Main emotion", summary["main_emotion"].title())
    metric_cols[2].metric(
        "Negative emotion ratio",
        f"{summary['negative_emotion_ratio']:.1f}%",
    )

    st.subheader("Marketing Recommendation")
    if summary["negative_emotion_ratio"] >= 40:
        st.warning(recommendation)
    else:
        st.info(recommendation)

    chart_cols = st.columns(2)
    emotion_chart_df = pd.DataFrame(
        {
            "emotion": list(summary["emotion_percentages"].keys()),
            "percentage": list(summary["emotion_percentages"].values()),
        }
    ).set_index("emotion")
    chart_cols[0].subheader("Seven-Emotion Distribution")
    chart_cols[0].bar_chart(emotion_chart_df)

    sentiment_chart_df = (
        result_df["sentiment"]
        .value_counts()
        .rename_axis("sentiment")
        .reset_index(name="count")
        .set_index("sentiment")
    )
    chart_cols[1].subheader("Sentiment Distribution")
    chart_cols[1].bar_chart(sentiment_chart_df)

    st.subheader("Comment-Level Results")
    st.dataframe(
        result_df[
            [
                "comment",
                "emotion",
                "emotion_score",
                "sentiment",
                "sentiment_score",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Download results as CSV",
        result_df.to_csv(index=False).encode("utf-8"),
        file_name="youtube_comment_emotion_results.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("YouTube Audience Emotion Analyzer")
    st.caption("A deep learning application for digital marketing campaign evaluation.")

    with st.sidebar:
        st.header("Settings")
        api_key = get_secret_value("YOUTUBE_API_KEY")
        if not api_key:
            api_key = st.text_input("YouTube API key", type="password")

        emotion_model = st.text_input("Emotion model", value=DEFAULT_EMOTION_MODEL)
        sentiment_model = st.text_input("Sentiment model", value=DEFAULT_SENTIMENT_MODEL)
        comment_order = st.selectbox("Comment order", ["relevance", "time"], index=0)
        use_sample_comments = st.checkbox(
            "Use sample comments",
            value=False,
            help="Use this for local testing when no YouTube API key is available.",
        )

    video_url = st.text_input(
        "YouTube video URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )
    analyze_clicked = st.button("Analyze comments", type="primary")

    if not analyze_clicked:
        st.write("Enter a YouTube video URL, then analyze the first 50 comments.")
        return

    try:
        if use_sample_comments:
            comments = load_sample_comments()[:50]
            st.info("Using local sample comments for demonstration.")
        else:
            video_id = parse_video_id(video_url)
            with st.spinner("Fetching YouTube comments..."):
                comments = fetch_top_comments(
                    video_id=video_id,
                    api_key=api_key,
                    max_results=50,
                    order=comment_order,
                )

        if not comments:
            st.warning("No comments were found for this video.")
            return

        with st.spinner("Loading Hugging Face models..."):
            emotion_pipeline = get_pipeline(emotion_model)
            sentiment_pipeline = get_pipeline(sentiment_model)

        with st.spinner("Analyzing audience emotions..."):
            rows = predict_comment_emotions(
                comments=comments,
                emotion_pipeline=emotion_pipeline,
                sentiment_pipeline=sentiment_pipeline,
            )

        display_summary(rows)

    except Exception as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()
