from __future__ import annotations

import os
import tomllib
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from youtube_emotion.core import clean_comment_text

COMMENTS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"
SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
REQUEST_TIMEOUT_SECONDS = 10
NEW_VIDEO_TARGET_COUNT = 10

EXISTING_CANDIDATE_PATHS = [
    PROJECT_ROOT / "experiments" / "youtube_training_video_candidates.csv",
    PROJECT_ROOT / "experiments" / "youtube_training_video_candidates_extra.csv",
]

EXPANDED_CANDIDATES_PATH = PROJECT_ROOT / "experiments" / "youtube_training_video_candidates_expanded.csv"
NEW_CANDIDATES_PATH = PROJECT_ROOT / "experiments" / "youtube_training_video_candidates_new10.csv"
EXPANDED_COMMENTS_PATH = PROJECT_ROOT / "data" / "youtube_domain_training_comments_expanded_unlabeled.csv"

NEW_VIDEO_SEARCH_SPECS = [
    {
        "theme": "beauty_brand_campaign",
        "video_short": "beauty_brand_campaign",
        "query": "Dove Real Beauty Sketches official campaign",
        "selection_reason": "Beauty brand campaign; likely joy, sadness, neutral, and self-image discussion.",
    },
    {
        "theme": "classic_brand_ad",
        "video_short": "classic_brand_ad",
        "query": "Old Spice The Man Your Man Could Smell Like official commercial",
        "selection_reason": "Classic humorous ad; likely joy, surprise, and brand nostalgia.",
    },
    {
        "theme": "automotive_brand_ad",
        "video_short": "automotive_brand_ad",
        "query": "Volvo Trucks The Epic Split official",
        "selection_reason": "Automotive brand ad; likely surprise, joy, and admiration.",
    },
    {
        "theme": "social_campaign",
        "video_short": "social_campaign",
        "query": "Always Like A Girl official commercial",
        "selection_reason": "Social-impact brand campaign; likely joy, sadness, anger, and neutral discussion.",
    },
    {
        "theme": "sports_brand_purpose",
        "video_short": "sports_brand_purpose",
        "query": "Nike Dream Crazy official ad Colin Kaepernick",
        "selection_reason": "Brand purpose campaign; likely joy, anger, surprise, and polarized brand discussion.",
    },
    {
        "theme": "tech_brand_story",
        "video_short": "tech_brand_story",
        "query": "Google Parisian Love official ad",
        "selection_reason": "Technology brand storytelling; likely joy and emotional response.",
    },
    {
        "theme": "controversial_brand_purpose",
        "video_short": "controversial_brand_purpose",
        "query": "Gillette We Believe official ad",
        "selection_reason": "Controversial brand purpose campaign; likely anger, disgust, joy, and neutral debate.",
    },
    {
        "theme": "fast_food_brand_launch",
        "video_short": "fast_food_brand_launch",
        "query": "McDonald's Grimace Shake official commercial",
        "selection_reason": "Fast-food brand launch; likely joy, surprise, humor, and neutral discussion.",
    },
    {
        "theme": "gaming_product_launch",
        "video_short": "gaming_product_launch",
        "query": "PlayStation 5 reveal trailer official",
        "selection_reason": "Gaming product launch; likely joy, surprise, anger, and brand comparison.",
    },
    {
        "theme": "streaming_trailer_reaction",
        "video_short": "streaming_trailer_reaction",
        "query": "Netflix Wednesday official trailer",
        "selection_reason": "Entertainment marketing trailer; likely joy, surprise, neutral, and fandom reactions.",
    },
    {
        "theme": "app_brand_campaign",
        "video_short": "app_brand_campaign",
        "query": "Duolingo official commercial campaign",
        "selection_reason": "App brand campaign; likely joy, surprise, and humor.",
    },
]


def get_api_key() -> str:
    env_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if env_key:
        return env_key

    secrets_path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        with secrets_path.open("rb") as handle:
            secrets = tomllib.load(handle)
        return str(secrets.get("YOUTUBE_API_KEY", "")).strip()

    raise RuntimeError("YOUTUBE_API_KEY was not found in environment or .streamlit/secrets.toml")


def youtube_get(url: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise RuntimeError(f"YouTube API request failed: {exc}") from exc
    if response.status_code != 200:
        raise RuntimeError(f"YouTube API request failed: {response.status_code} {response.text}")
    return response.json()


def load_existing_candidates() -> pd.DataFrame:
    frames = []
    for path in EXISTING_CANDIDATE_PATHS:
        frame = pd.read_csv(path).fillna("")
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["video_id"]).reset_index(drop=True)
    return df


def fetch_video_stats(video_ids: list[str], api_key: str) -> dict[str, dict[str, Any]]:
    if not video_ids:
        return {}

    stats: dict[str, dict[str, Any]] = {}
    for start in range(0, len(video_ids), 50):
        chunk = video_ids[start : start + 50]
        payload = youtube_get(
            VIDEOS_URL,
            {
                "part": "snippet,statistics",
                "id": ",".join(chunk),
                "key": api_key,
            },
        )
        for item in payload.get("items", []):
            snippet = item.get("snippet", {})
            statistics = item.get("statistics", {})
            stats[item["id"]] = {
                "video_id": item["id"],
                "url": f"https://www.youtube.com/watch?v={item['id']}",
                "title": snippet.get("title", ""),
                "channel": snippet.get("channelTitle", ""),
                "published_at": snippet.get("publishedAt", "")[:10],
                "view_count": int(statistics.get("viewCount", 0)),
                "comment_count": int(statistics.get("commentCount", 0)),
            }
    return stats


def fetch_comments(video_id: str, api_key: str, target_count: int = 100) -> list[str]:
    comments: list[str] = []
    seen: set[str] = set()

    for order in ["relevance", "time"]:
        payload = youtube_get(
            COMMENTS_URL,
            {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": 100,
                "order": order,
                "textFormat": "plainText",
                "key": api_key,
            },
        )
        for item in payload.get("items", []):
            snippet = (
                item.get("snippet", {})
                .get("topLevelComment", {})
                .get("snippet", {})
            )
            text = snippet.get("textOriginal") or snippet.get("textDisplay") or ""
            cleaned = clean_comment_text(text)
            key = cleaned.lower()
            if cleaned and key not in seen:
                comments.append(cleaned)
                seen.add(key)
            if len(comments) >= target_count:
                return comments[:target_count]

    return comments[:target_count]


def search_video_candidates(query: str, api_key: str, max_results: int = 10) -> list[str]:
    payload = youtube_get(
        SEARCH_URL,
        {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": max_results,
            "relevanceLanguage": "en",
            "safeSearch": "none",
            "key": api_key,
        },
    )
    video_ids = []
    for item in payload.get("items", []):
        video_id = item.get("id", {}).get("videoId")
        if video_id:
            video_ids.append(video_id)
    return video_ids


def choose_new_videos(existing_ids: set[str], api_key: str) -> pd.DataFrame:
    selected_rows = []
    selected_ids = set(existing_ids)

    for spec in NEW_VIDEO_SEARCH_SPECS:
        if len(selected_rows) >= NEW_VIDEO_TARGET_COUNT:
            break

        print(f"Searching: {spec['query']}", flush=True)
        candidate_ids = search_video_candidates(spec["query"], api_key)
        stats = fetch_video_stats(candidate_ids, api_key)

        selected = None
        selected_comments: list[str] = []
        for video_id in candidate_ids:
            if video_id in selected_ids:
                continue
            video_stats = stats.get(video_id)
            if not video_stats or video_stats["comment_count"] < 100:
                continue
            try:
                comments = fetch_comments(video_id, api_key, target_count=100)
            except RuntimeError:
                continue
            if len(comments) >= 80:
                selected = video_stats
                selected_comments = comments
                break

        if selected is None:
            print(f"Skipped search spec without enough comments: {spec['query']}", flush=True)
            continue

        selected_ids.add(selected["video_id"])
        selected_rows.append(
            {
                **spec,
                **selected,
                "fetched_comments": len(selected_comments),
                "sample_comment": selected_comments[0] if selected_comments else "",
            }
        )
        print(f"Selected {spec['video_short']}: {selected['video_id']} ({len(selected_comments)} comments)", flush=True)

    return pd.DataFrame(selected_rows)


def build_comment_rows(candidates: pd.DataFrame, api_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    video_rows = []
    comment_rows = []

    for _, row in candidates.iterrows():
        video_id = str(row["video_id"])
        try:
            comments = fetch_comments(video_id, api_key, target_count=100)
        except RuntimeError as exc:
            print(f"Skipped comments for {video_id}: {exc}", flush=True)
            comments = []
        video_url = row.get("url") or f"https://www.youtube.com/watch?v={video_id}"

        video_row = row.to_dict()
        video_row["url"] = video_url
        video_row["fetched_comments"] = len(comments)
        video_rows.append(video_row)

        for index, comment in enumerate(comments, start=1):
            comment_rows.append(
                {
                    "theme": row.get("theme", ""),
                    "video_short": row.get("video_short", ""),
                    "video_id": video_id,
                    "video": video_url,
                    "comment_index": index,
                    "comment": comment,
                }
            )

        print(f"Fetched {len(comments):3d} comments: {row.get('video_short', video_id)}", flush=True)

    return pd.DataFrame(video_rows), pd.DataFrame(comment_rows)


def main() -> None:
    api_key = get_api_key()
    existing_candidates = load_existing_candidates()
    new_candidates = choose_new_videos(set(existing_candidates["video_id"].astype(str)), api_key)
    combined_candidates = pd.concat([existing_candidates, new_candidates], ignore_index=True)
    combined_candidates = combined_candidates.drop_duplicates(subset=["video_id"]).reset_index(drop=True)

    expanded_candidates, comment_rows = build_comment_rows(combined_candidates, api_key)

    NEW_CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPANDED_COMMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_candidates.to_csv(NEW_CANDIDATES_PATH, index=False)
    expanded_candidates.to_csv(EXPANDED_CANDIDATES_PATH, index=False)
    comment_rows.to_csv(EXPANDED_COMMENTS_PATH, index=False)

    print(f"\nSaved new video candidates: {NEW_CANDIDATES_PATH}")
    print(f"Saved expanded candidates: {EXPANDED_CANDIDATES_PATH}")
    print(f"Saved expanded comments: {EXPANDED_COMMENTS_PATH}")
    print(f"Videos: {expanded_candidates.shape[0]}")
    print(f"Comments: {comment_rows.shape[0]}")


if __name__ == "__main__":
    main()
