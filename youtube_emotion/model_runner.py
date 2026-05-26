from typing import Any, Callable

from youtube_emotion.core import clean_comment_text, normalize_emotion_label, normalize_sentiment_label


DEFAULT_EMOTION_MODEL = "chase1zhang/youtube-emotion-distilbert-domain-adapted"
DEFAULT_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

EMOTION_MODEL_OPTIONS = {
    "YouTube-domain adapted DistilBERT (recommended)": DEFAULT_EMOTION_MODEL,
    "GoEmotions DistilBERT (previous fine-tuned)": "chase1zhang/youtube-emotion-distilbert",
    "DistilRoBERTa baseline": "j-hartmann/emotion-english-distilroberta-base",
}


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
    cleaned_comments = [clean_comment_text(comment) for comment in comments]
    cleaned_comments = [comment for comment in cleaned_comments if comment]

    if not cleaned_comments:
        return []

    pipeline_kwargs = {"truncation": True, "return_token_type_ids": False}
    emotion_outputs = emotion_pipeline(cleaned_comments, **pipeline_kwargs)
    sentiment_outputs = sentiment_pipeline(cleaned_comments, **pipeline_kwargs)

    rows: list[dict[str, Any]] = []
    for comment, emotion_output, sentiment_output in zip(
        cleaned_comments, emotion_outputs, sentiment_outputs
    ):
        emotion_label, emotion_score = extract_top_prediction(emotion_output)
        sentiment_label, sentiment_score = extract_top_prediction(sentiment_output)

        rows.append(
            {
                "comment": comment,
                "emotion": normalize_emotion_label(emotion_label),
                "emotion_score": round(emotion_score, 4),
                "sentiment": normalize_sentiment_label(sentiment_label),
                "sentiment_score": round(sentiment_score, 4),
            }
        )

    return rows
