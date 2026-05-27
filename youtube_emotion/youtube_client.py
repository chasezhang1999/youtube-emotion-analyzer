from collections.abc import Callable
from typing import Any

import requests

from youtube_emotion.core import clean_comment_text


YOUTUBE_COMMENT_THREADS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"


def fetch_top_comments(
    video_id: str,
    api_key: str,
    max_results: int = 100,
    order: str = "relevance",
    request_get: Callable[..., Any] = requests.get,
) -> list[str]:
    if not api_key:
        raise ValueError("YouTube API key is required.")

    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": max(1, min(int(max_results), 100)),
        "order": order,
        "textFormat": "plainText",
        "key": api_key,
    }

    response = request_get(YOUTUBE_COMMENT_THREADS_URL, params=params, timeout=20)
    if response.status_code != 200:
        raise RuntimeError(f"YouTube API request failed: {response.status_code} {response.text}")

    payload = response.json()
    comments: list[str] = []
    for item in payload.get("items", []):
        comment_snippet = (
            item.get("snippet", {})
            .get("topLevelComment", {})
            .get("snippet", {})
        )
        text = comment_snippet.get("textOriginal") or comment_snippet.get("textDisplay") or ""
        cleaned = clean_comment_text(text)
        if cleaned:
            comments.append(cleaned)

    return comments
