import re
from collections import Counter
from html import unescape
from typing import Any
from urllib.parse import parse_qs, urlparse


EMOTION_LABELS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
]

NEGATIVE_EMOTIONS = {"anger", "disgust", "fear", "sadness"}

EMOTION_LABEL_MAP = {
    "admiration": "joy",
    "amusement": "joy",
    "anger": "anger",
    "annoyance": "anger",
    "approval": "joy",
    "caring": "joy",
    "confusion": "neutral",
    "curiosity": "neutral",
    "desire": "joy",
    "disappointment": "sadness",
    "disapproval": "disgust",
    "disgust": "disgust",
    "embarrassment": "sadness",
    "excitement": "joy",
    "fear": "fear",
    "gratitude": "joy",
    "grief": "sadness",
    "joy": "joy",
    "love": "joy",
    "nervousness": "fear",
    "optimism": "joy",
    "pride": "joy",
    "realization": "surprise",
    "relief": "joy",
    "remorse": "sadness",
    "sadness": "sadness",
    "surprise": "surprise",
    "neutral": "neutral",
}

SENTIMENT_LABEL_MAP = {
    "label_0": "negative",
    "label_1": "neutral",
    "label_2": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
}

VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


def parse_video_id(value: str) -> str:
    """Extract a YouTube video ID from common URL formats or a raw ID."""
    candidate = (value or "").strip()
    if VIDEO_ID_PATTERN.match(candidate):
        return candidate

    parsed = urlparse(candidate)
    host = parsed.netloc.lower()

    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/").split("/")[0]
        if VIDEO_ID_PATTERN.match(video_id):
            return video_id

    if host.endswith("youtube.com") or host.endswith("youtube-nocookie.com"):
        query_video_id = parse_qs(parsed.query).get("v", [""])[0]
        if VIDEO_ID_PATTERN.match(query_video_id):
            return query_video_id

        path_parts = [part for part in parsed.path.split("/") if part]
        if len(path_parts) >= 2 and path_parts[0] in {"shorts", "embed", "live"}:
            video_id = path_parts[1]
            if VIDEO_ID_PATTERN.match(video_id):
                return video_id

    raise ValueError("Please enter a valid YouTube video URL or 11-character video ID.")


def clean_comment_text(text: str) -> str:
    """Normalize comment text from the YouTube API for display and inference."""
    cleaned = unescape(text or "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def normalize_emotion_label(label: str) -> str:
    normalized = (label or "").strip().lower()
    return EMOTION_LABEL_MAP.get(normalized, "neutral")


def normalize_sentiment_label(label: str) -> str:
    normalized = (label or "").strip().lower()
    return SENTIMENT_LABEL_MAP.get(normalized, normalized)


def summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    emotions = [normalize_emotion_label(row.get("emotion", "")) for row in rows]
    total_comments = len(emotions)
    counts = Counter(emotions)
    emotion_counts = {label: counts.get(label, 0) for label in EMOTION_LABELS}

    if total_comments == 0:
        return {
            "total_comments": 0,
            "main_emotion": "none",
            "emotion_counts": emotion_counts,
            "emotion_percentages": {label: 0.0 for label in EMOTION_LABELS},
            "negative_emotion_count": 0,
            "negative_emotion_ratio": 0.0,
        }

    emotion_percentages = {
        label: round((count / total_comments) * 100, 2)
        for label, count in emotion_counts.items()
    }
    main_emotion = max(EMOTION_LABELS, key=lambda label: emotion_counts[label])
    negative_count = sum(emotion_counts[label] for label in NEGATIVE_EMOTIONS)

    return {
        "total_comments": total_comments,
        "main_emotion": main_emotion,
        "emotion_counts": emotion_counts,
        "emotion_percentages": emotion_percentages,
        "negative_emotion_count": negative_count,
        "negative_emotion_ratio": round((negative_count / total_comments) * 100, 2),
    }


def build_marketing_recommendation(summary: dict[str, Any]) -> str:
    main_emotion = summary.get("main_emotion", "none")
    negative_ratio = float(summary.get("negative_emotion_ratio", 0.0))

    if main_emotion == "none":
        return "No usable comments were found. The agency should test another video or broaden the comment collection."

    if negative_ratio >= 40:
        return (
            "Negative emotion risk is high. The agency should review comments with anger, "
            "sadness, fear, or disgust to identify campaign issues and adjust the message."
        )

    if main_emotion in {"joy", "surprise"}:
        return (
            f"The main audience emotion is {main_emotion}. The campaign is generating a strong "
            "engagement signal, and similar creative direction can be considered for future videos."
        )

    if main_emotion == "neutral":
        return (
            "The main audience emotion is neutral. The agency may improve the hook, call to action, "
            "or storytelling to create a stronger emotional response."
        )

    return (
        f"The main audience emotion is {main_emotion}. The agency should inspect representative "
        "comments and adjust content tone, targeting, or messaging before scaling the campaign."
    )


def build_campaign_decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_predictions(rows)
    total_comments = summary["total_comments"]
    sentiments = [
        normalize_sentiment_label(str(row.get("sentiment", "")))
        for row in rows
        if row.get("sentiment")
    ]
    sentiment_counts = Counter(sentiments)

    if total_comments == 0:
        return {
            "decision": "collect_more_comments",
            "risk_level": "unknown",
            "main_emotion": "none",
            "negative_emotion_ratio": 0.0,
            "positive_sentiment_ratio": 0.0,
            "negative_sentiment_ratio": 0.0,
            "rationale": "No usable comments were available for a campaign decision.",
            "next_actions": ["Test another video or broaden the comment collection window."],
        }

    sentiment_total = max(len(sentiments), 1)
    positive_sentiment_ratio = round(
        (sentiment_counts.get("positive", 0) / sentiment_total) * 100,
        2,
    )
    negative_sentiment_ratio = round(
        (sentiment_counts.get("negative", 0) / sentiment_total) * 100,
        2,
    )
    main_emotion = summary["main_emotion"]
    negative_emotion_ratio = summary["negative_emotion_ratio"]

    if negative_emotion_ratio >= 40 or negative_sentiment_ratio >= 40:
        decision = "review_before_scaling"
        risk_level = "high"
        rationale = "Negative emotion or sentiment is high enough to require human review before scaling."
        next_actions = [
            "Human review the highest-risk anger, disgust, fear, and sadness comments.",
            "Identify repeated complaint themes and adjust message framing before more paid amplification.",
            "Re-run analysis after the revised creative or response plan is ready.",
        ]
    elif main_emotion in {"joy", "surprise"} and positive_sentiment_ratio >= 50:
        decision = "scale_positive_creative"
        risk_level = "low"
        rationale = "Positive audience response is strong and fine-grained emotions are favorable."
        next_actions = [
            "Reuse the strongest hook, topic, or creator angle in the next campaign asset.",
            "Collect representative positive comments for campaign reporting.",
            "Monitor comments after scaling to catch sentiment drift early.",
        ]
    elif main_emotion == "neutral":
        decision = "improve_engagement_hook"
        risk_level = "medium"
        rationale = "The audience is not strongly negative, but the emotional response is muted."
        next_actions = [
            "Improve the opening hook, title, call to action, or storytelling frame.",
            "A/B test a stronger emotional angle before expanding distribution.",
            "Track whether neutral comments shift toward joy or surprise in the next run.",
        ]
    else:
        decision = "monitor_and_review_samples"
        risk_level = "medium"
        rationale = "The signal is mixed and should be treated as decision support rather than an automatic call."
        next_actions = [
            "Review a sample of comments from the dominant emotion class.",
            "Compare emotion and sentiment outputs for disagreement.",
            "Use the CSV export to inspect uncertain or low-confidence predictions.",
        ]

    return {
        "decision": decision,
        "risk_level": risk_level,
        "main_emotion": main_emotion,
        "negative_emotion_ratio": negative_emotion_ratio,
        "positive_sentiment_ratio": positive_sentiment_ratio,
        "negative_sentiment_ratio": negative_sentiment_ratio,
        "rationale": rationale,
        "next_actions": next_actions,
    }
