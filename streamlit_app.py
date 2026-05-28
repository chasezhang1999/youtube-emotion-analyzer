from pathlib import Path
import os

import pandas as pd
import streamlit as st

from youtube_emotion.core import (
    build_campaign_decision,
    build_marketing_recommendation,
    clean_comment_text,
    normalize_emotion_label,
    normalize_sentiment_label,
    parse_video_id,
    summarize_predictions,
)
from youtube_emotion import model_runner
from youtube_emotion.youtube_client import fetch_top_comments


PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_COMMENTS_PATH = PROJECT_ROOT / "data" / "sample_comments.csv"

DEFAULT_SENTIMENT_MODEL = getattr(
    model_runner,
    "DEFAULT_SENTIMENT_MODEL",
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
)
SENTIMENT_MODEL_OPTIONS = {
    "CardiffNLP Twitter RoBERTa (recommended)": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "lxyuan DistilBERT Multilingual": "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    "FiniteAutomata BERTweet": "finiteautomata/bertweet-base-sentiment-analysis",
}
EMOTION_MODEL_OPTIONS = getattr(
    model_runner,
    "EMOTION_MODEL_OPTIONS",
    {
        "YouTube-domain adapted DistilBERT (recommended)": "chase1zhang/youtube-emotion-distilbert-domain-adapted",
        "GoEmotions DistilBERT (previous fine-tuned)": "chase1zhang/youtube-emotion-distilbert",
        "Public GoEmotions RoBERTa (SamLowe)": "SamLowe/roberta-base-go_emotions",
        "Public DistilRoBERTa 7-emotion (j-hartmann)": "j-hartmann/emotion-english-distilroberta-base",
        "Public RoBERTa-large 7-emotion (j-hartmann)": "j-hartmann/emotion-english-roberta-large",
    },
)
DEFAULT_COMPARISON_MODEL_LABELS = [
    label
    for label in getattr(
        model_runner,
        "DEFAULT_COMPARISON_MODEL_LABELS",
        [
            "YouTube-domain adapted DistilBERT (recommended)",
            "YouTube-domain adapted RoBERTa (SamLowe)",
            "YouTube-domain adapted DistilRoBERTa (j-hartmann)",
        ],
    )
    if label in EMOTION_MODEL_OPTIONS
]
PIPELINE_KWARGS = {"truncation": True, "return_token_type_ids": False}


st.set_page_config(
    page_title="YouTube Audience Emotion Analyzer",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_pipeline(model_name: str):
    return model_runner.load_text_classification_pipeline(model_name)


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


def extract_top_prediction(output) -> tuple[str, float]:
    extractor = getattr(model_runner, "extract_top_prediction", None)
    if extractor is not None:
        return extractor(output)

    if isinstance(output, dict):
        return str(output.get("label", "")), float(output.get("score", 0.0))

    if isinstance(output, list) and output:
        ranked = [item for item in output if isinstance(item, dict)]
        if ranked:
            best = max(ranked, key=lambda item: float(item.get("score", 0.0)))
            return str(best.get("label", "")), float(best.get("score", 0.0))

    return "neutral", 0.0


def predict_comment_emotion_only(comments: list[str], emotion_pipeline) -> list[dict]:
    predictor = getattr(model_runner, "predict_comment_emotion_only", None)
    if predictor is not None:
        return predictor(comments=comments, emotion_pipeline=emotion_pipeline)

    cleaned_comments = [clean_comment_text(comment) for comment in comments]
    cleaned_comments = [comment for comment in cleaned_comments if comment]
    if not cleaned_comments:
        return []

    outputs = emotion_pipeline(cleaned_comments, **PIPELINE_KWARGS)
    rows = []
    for comment, output in zip(cleaned_comments, outputs):
        label, score = extract_top_prediction(output)
        rows.append(
            {
                "comment": comment,
                "emotion": normalize_emotion_label(label),
                "emotion_score": round(score, 4),
            }
        )
    return rows


def predict_comment_sentiments(comments: list[str], sentiment_pipeline) -> list[dict]:
    predictor = getattr(model_runner, "predict_comment_sentiments", None)
    if predictor is not None:
        return predictor(comments=comments, sentiment_pipeline=sentiment_pipeline)

    cleaned_comments = [clean_comment_text(comment) for comment in comments]
    cleaned_comments = [comment for comment in cleaned_comments if comment]
    if not cleaned_comments:
        return []

    outputs = sentiment_pipeline(cleaned_comments, **PIPELINE_KWARGS)
    rows = []
    for comment, output in zip(cleaned_comments, outputs):
        label, score = extract_top_prediction(output)
        rows.append(
            {
                "comment": comment,
                "sentiment": normalize_sentiment_label(label),
                "sentiment_score": round(score, 4),
            }
        )
    return rows


def predict_comment_emotions(comments: list[str], emotion_pipeline, sentiment_pipeline) -> list[dict]:
    predictor = getattr(model_runner, "predict_comment_emotions", None)
    if predictor is not None:
        return predictor(
            comments=comments,
            emotion_pipeline=emotion_pipeline,
            sentiment_pipeline=sentiment_pipeline,
        )

    return merge_emotion_and_sentiment_rows(
        emotion_rows=predict_comment_emotion_only(comments, emotion_pipeline),
        sentiment_rows=predict_comment_sentiments(comments, sentiment_pipeline),
    )


def display_summary(rows: list[dict]) -> None:
    summary = summarize_predictions(rows)
    recommendation = build_marketing_recommendation(summary)
    decision = build_campaign_decision(rows)
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

    st.subheader("Decision Pipeline")
    decision_cols = st.columns(3)
    decision_cols[0].metric("Decision", decision["decision"].replace("_", " ").title())
    decision_cols[1].metric("Risk level", decision["risk_level"].title())
    decision_cols[2].metric(
        "Positive sentiment",
        f"{decision['positive_sentiment_ratio']:.1f}%",
    )
    if decision["risk_level"] == "high":
        st.warning(decision["rationale"])
    elif decision["risk_level"] == "low":
        st.success(decision["rationale"])
    else:
        st.info(decision["rationale"])
    st.markdown(
        "\n".join(f"- {action}" for action in decision["next_actions"])
    )

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


def merge_emotion_and_sentiment_rows(
    emotion_rows: list[dict],
    sentiment_rows: list[dict],
) -> list[dict]:
    rows: list[dict] = []
    for emotion_row, sentiment_row in zip(emotion_rows, sentiment_rows):
        rows.append(
            {
                **emotion_row,
                "sentiment": sentiment_row["sentiment"],
                "sentiment_score": sentiment_row["sentiment_score"],
            }
        )
    return rows


def display_model_comparison(comparison_results: dict[str, dict]) -> None:
    summary_rows = []
    distribution_rows = []
    combined_rows = []

    for model_label, result in comparison_results.items():
        rows = result["rows"]
        summary = summarize_predictions(rows)
        decision = build_campaign_decision(rows)
        result["summary"] = summary
        result["decision"] = decision

        summary_rows.append(
            {
                "Model": model_label,
                "Hugging Face Repo": result["model_name"],
                "Comments": summary["total_comments"],
                "Main Emotion": summary["main_emotion"].title(),
                "Negative Emotion Ratio": f"{summary['negative_emotion_ratio']:.1f}%",
                "Decision": decision["decision"].replace("_", " ").title(),
                "Risk": decision["risk_level"].title(),
            }
        )

        for emotion, percentage in summary["emotion_percentages"].items():
            distribution_rows.append(
                {
                    "emotion": emotion,
                    "model": model_label,
                    "percentage": percentage,
                }
            )

        for row in rows:
            combined_rows.append(
                {
                    "model": model_label,
                    "model_name": result["model_name"],
                    **row,
                }
            )

    st.subheader("Model Comparison Summary")
    st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

    st.subheader("Emotion Distribution by Model")
    distribution_df = pd.DataFrame(distribution_rows)
    distribution_pivot = (
        distribution_df.pivot(index="emotion", columns="model", values="percentage")
        .fillna(0.0)
        .sort_index()
    )
    st.bar_chart(distribution_pivot)

    st.subheader("Per-Model Comment Results")
    tabs = st.tabs(list(comparison_results.keys()))
    for tab, (model_label, result) in zip(tabs, comparison_results.items()):
        with tab:
            summary = result["summary"]
            decision = result["decision"]
            rows = result["rows"]
            model_df = pd.DataFrame(rows)

            metric_cols = st.columns(4)
            metric_cols[0].metric("Comments analyzed", summary["total_comments"])
            metric_cols[1].metric("Main emotion", summary["main_emotion"].title())
            metric_cols[2].metric(
                "Negative emotion ratio",
                f"{summary['negative_emotion_ratio']:.1f}%",
            )
            metric_cols[3].metric("Decision risk", decision["risk_level"].title())
            st.caption(f"Hugging Face model: `{result['model_name']}`")
            st.info(
                f"Decision pipeline: {decision['decision'].replace('_', ' ').title()} — "
                f"{decision['rationale']}"
            )

            emotion_chart_df = pd.DataFrame(
                {
                    "emotion": list(summary["emotion_percentages"].keys()),
                    "percentage": list(summary["emotion_percentages"].values()),
                }
            ).set_index("emotion")
            st.bar_chart(emotion_chart_df)
            st.dataframe(
                model_df[
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

    combined_df = pd.DataFrame(combined_rows)
    st.download_button(
        "Download model comparison CSV",
        combined_df.to_csv(index=False).encode("utf-8"),
        file_name="youtube_comment_emotion_model_comparison.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.header("🎯 Final Campaign Decision Model Selection")
    st.write(
        "Choose one of the compared models to generate the final campaign decision, "
        "marketing recommendations, and decision pipeline outcomes:"
    )

    decision_model_label = st.selectbox(
        "Select decision model",
        options=list(comparison_results.keys()),
        index=0,
        help="Select the model whose predictions will be used to make the final campaign decision.",
        key="final_decision_model_selector",
    )

    st.write(f"Generating final campaign decision using predictions from **{decision_model_label}**:")
    display_summary(comparison_results[decision_model_label]["rows"])


def main() -> None:
    st.title("YouTube Audience Emotion Analyzer")
    st.caption("A deep learning application for digital marketing campaign evaluation.")

    with st.sidebar:
        st.header("Settings")
        api_key = get_secret_value("YOUTUBE_API_KEY")
        if not api_key:
            api_key = st.text_input("YouTube API key", type="password")

        analysis_mode = st.radio(
            "Analysis mode",
            options=["Single model", "Compare emotion models"],
            index=0,
            help="All modes run emotion, sentiment, and decision pipelines. Compare mode runs the same comments through multiple emotion models.",
        )

        if analysis_mode == "Single model":
            emotion_model_label = st.selectbox(
                "Emotion model",
                options=list(EMOTION_MODEL_OPTIONS.keys()),
                index=0,
                help="The recommended model was further fine-tuned on YouTube-domain comments.",
            )
            selected_emotion_model_labels = [emotion_model_label]
        else:
            selected_emotion_model_labels = st.multiselect(
                "Emotion models to compare",
                options=list(EMOTION_MODEL_OPTIONS.keys()),
                default=DEFAULT_COMPARISON_MODEL_LABELS,
                help="The default comparison uses three fine-tuned emotion models.",
            )

        selected_emotion_models = {
            label: EMOTION_MODEL_OPTIONS[label]
            for label in selected_emotion_model_labels
        }
        for label, model_name in selected_emotion_models.items():
            st.caption(f"{label}: `{model_name}`")

        sentiment_model_label = st.selectbox(
            "Sentiment model",
            options=list(SENTIMENT_MODEL_OPTIONS.keys()),
            index=0,
            help="Sentiment analysis baseline model for Pipeline 2.",
        )
        sentiment_model = SENTIMENT_MODEL_OPTIONS[sentiment_model_label]
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
        st.write("Enter a YouTube video URL, then analyze the first 100 comments.")
        return

    try:
        if use_sample_comments:
            comments = load_sample_comments()[:100]
            st.info("Using local sample comments for demonstration.")
        else:
            video_id = parse_video_id(video_url)
            with st.spinner("Fetching YouTube comments..."):
                comments = fetch_top_comments(
                    video_id=video_id,
                    api_key=api_key,
                    max_results=100,
                    order=comment_order,
                )

        if not comments:
            st.warning("No comments were found for this video.")
            return

        if not selected_emotion_models:
            st.warning("Please select at least one emotion model.")
            return

        if analysis_mode == "Compare emotion models" and len(selected_emotion_models) < 3:
            st.warning("For the project comparison requirement, select at least three emotion models.")

        with st.spinner("Loading Hugging Face models..."):
            emotion_pipelines = {
                label: get_pipeline(model_name)
                for label, model_name in selected_emotion_models.items()
            }
            sentiment_pipeline = get_pipeline(sentiment_model)

        if analysis_mode == "Compare emotion models":
            with st.spinner("Analyzing comments with selected emotion models..."):
                sentiment_rows = predict_comment_sentiments(
                    comments=comments,
                    sentiment_pipeline=sentiment_pipeline,
                )
                comparison_results = {}
                for label, emotion_pipeline in emotion_pipelines.items():
                    emotion_rows = predict_comment_emotion_only(
                        comments=comments,
                        emotion_pipeline=emotion_pipeline,
                    )
                    comparison_results[label] = {
                        "model_name": selected_emotion_models[label],
                        "rows": merge_emotion_and_sentiment_rows(
                            emotion_rows=emotion_rows,
                            sentiment_rows=sentiment_rows,
                        ),
                    }

            display_model_comparison(comparison_results)
        else:
            selected_label = selected_emotion_model_labels[0]
            with st.spinner("Analyzing audience emotions..."):
                rows = predict_comment_emotions(
                    comments=comments,
                    emotion_pipeline=emotion_pipelines[selected_label],
                    sentiment_pipeline=sentiment_pipeline,
                )

            display_summary(rows)

    except Exception as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()
