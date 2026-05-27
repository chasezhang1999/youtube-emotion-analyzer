from typing import Any, Callable

from youtube_emotion.core import clean_comment_text, normalize_emotion_label, normalize_sentiment_label


DEFAULT_EMOTION_MODEL = "chase1zhang/youtube-emotion-distilbert-domain-adapted"
DEFAULT_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

EMOTION_MODEL_OPTIONS = {
    "YouTube-domain adapted DistilBERT (recommended)": DEFAULT_EMOTION_MODEL,
    "GoEmotions DistilBERT (previous fine-tuned)": "chase1zhang/youtube-emotion-distilbert",
    "Public GoEmotions RoBERTa (SamLowe)": "SamLowe/roberta-base-go_emotions",
    "Public DistilRoBERTa 7-emotion (j-hartmann)": "j-hartmann/emotion-english-distilroberta-base",
    "Public RoBERTa-large 7-emotion (j-hartmann)": "j-hartmann/emotion-english-roberta-large",
}

DEFAULT_COMPARISON_MODEL_LABELS = [
    "YouTube-domain adapted DistilBERT (recommended)",
    "Public GoEmotions RoBERTa (SamLowe)",
    "Public DistilRoBERTa 7-emotion (j-hartmann)",
]

PIPELINE_KWARGS = {"truncation": True, "return_token_type_ids": False}


def load_text_classification_pipeline(model_name: str):
    from transformers import pipeline

    return pipeline("text-classification", model=model_name)


def extract_top_prediction(output: Any) -> tuple[str, float]:
    if isinstance(output, dict):
        return str(output.get("label", "")), float(output.get("score", 0.0))

    if isinstance(output, list) and output:
        ranked = [item for item in output if isinstance(item, dict)]
        if ranked:
            best = max(ranked, key=lambda item: float(item.get("score", 0.0)))
            return str(best.get("label", "")), float(best.get("score", 0.0))

    return "neutral", 0.0


def predict_comment_emotions(
    comments: list[str],
    emotion_pipeline: Callable[..., list[Any]],
    sentiment_pipeline: Callable[..., list[Any]],
) -> list[dict[str, Any]]:
    emotion_rows = predict_comment_emotion_only(comments, emotion_pipeline)
    sentiment_rows = predict_comment_sentiments(comments, sentiment_pipeline)

    rows: list[dict[str, Any]] = []
    for emotion_row, sentiment_row in zip(emotion_rows, sentiment_rows):
        rows.append(
            {
                **emotion_row,
                "sentiment": sentiment_row["sentiment"],
                "sentiment_score": sentiment_row["sentiment_score"],
            }
        )

    return rows


def predict_comment_emotion_only(
    comments: list[str],
    emotion_pipeline: Callable[..., list[Any]],
) -> list[dict[str, Any]]:
    cleaned_comments = [clean_comment_text(comment) for comment in comments]
    cleaned_comments = [comment for comment in cleaned_comments if comment]

    if not cleaned_comments:
        return []

    emotion_outputs = emotion_pipeline(cleaned_comments, **PIPELINE_KWARGS)

    rows: list[dict[str, Any]] = []
    for comment, emotion_output in zip(cleaned_comments, emotion_outputs):
        emotion_label, emotion_score = extract_top_prediction(emotion_output)

        rows.append(
            {
                "comment": comment,
                "emotion": normalize_emotion_label(emotion_label),
                "emotion_score": round(emotion_score, 4),
            }
        )

    return rows


def predict_comment_sentiments(
    comments: list[str],
    sentiment_pipeline: Callable[..., list[Any]],
) -> list[dict[str, Any]]:
    cleaned_comments = [clean_comment_text(comment) for comment in comments]
    cleaned_comments = [comment for comment in cleaned_comments if comment]

    if not cleaned_comments:
        return []

    sentiment_outputs = sentiment_pipeline(cleaned_comments, **PIPELINE_KWARGS)

    rows: list[dict[str, Any]] = []
    for comment, sentiment_output in zip(cleaned_comments, sentiment_outputs):
        sentiment_label, sentiment_score = extract_top_prediction(sentiment_output)

        rows.append(
            {
                "comment": comment,
                "sentiment": normalize_sentiment_label(sentiment_label),
                "sentiment_score": round(sentiment_score, 4),
            }
        )

    return rows
