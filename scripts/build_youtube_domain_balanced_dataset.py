from __future__ import annotations

import argparse
import os
import string
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.label_youtube_domain_comments import score_comment
from youtube_emotion.core import clean_comment_text

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
LABEL_TO_ID = {label: index for index, label in enumerate(EMOTION_LABELS)}

COMMENTS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"
SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
REQUEST_TIMEOUT_SECONDS = 15

SEED_CANDIDATE_PATHS = [
    PROJECT_ROOT / "experiments" / "youtube_training_video_candidates_expanded.csv",
    PROJECT_ROOT / "experiments" / "youtube_training_video_candidates.csv",
    PROJECT_ROOT / "experiments" / "youtube_training_video_candidates_extra.csv",
]
SEED_COMMENT_PATHS = [
    PROJECT_ROOT / "data" / "youtube_domain_training_comments_expanded_unlabeled.csv",
]

RAW_POOL_PATH = PROJECT_ROOT / "data" / "youtube_domain_training_comments_8000_unlabeled.csv"
LABELED_POOL_PATH = PROJECT_ROOT / "data" / "youtube_domain_training_comments_8000_assistant_labeled.csv"
FINAL_LABELED_PATH = PROJECT_ROOT / "data" / "youtube_domain_training_comments_5000_balanced_assistant_labeled.csv"
FINAL_SPLIT_DIR = PROJECT_ROOT / "data" / "youtube_domain_7class_assistant"
CANDIDATE_AUDIT_PATH = PROJECT_ROOT / "experiments" / "youtube_training_video_candidates_8000.csv"


@dataclass(frozen=True)
class SearchSpec:
    theme: str
    video_short: str
    query: str
    selection_reason: str


TARGETED_SEARCH_SPECS = [
    SearchSpec(
        "anger_brand_crisis_more",
        "anger_brand_crisis",
        "United Airlines passenger dragged off plane customer outrage",
        "Brand crisis discussion likely to add anger and disgust comments.",
    ),
    SearchSpec(
        "anger_customer_service_more",
        "anger_customer_service",
        "Verizon outage customers angry comments news",
        "Customer-service outage discussion likely to add anger comments.",
    ),
    SearchSpec(
        "anger_brand_backlash",
        "disgust_brand_backlash",
        "Pepsi Kendall Jenner ad backlash comments",
        "Backlash campaign likely to add anger and disgust comments.",
    ),
    SearchSpec(
        "anger_brand_backlash",
        "disgust_brand_backlash",
        "Gillette We Believe backlash comments official ad",
        "Polarized brand-purpose campaign likely to add anger comments.",
    ),
    SearchSpec(
        "disgust_food_safety_more",
        "disgust_food_safety",
        "restaurant health inspection shut down disgusting food safety",
        "Food-safety story likely to add disgust comments.",
    ),
    SearchSpec(
        "disgust_food_safety_more",
        "disgust_food_safety",
        "Chipotle food poisoning outbreak comments",
        "Food-safety incident likely to add disgust and anger comments.",
    ),
    SearchSpec(
        "disgust_brand_crisis_more",
        "brand_crisis_negative",
        "fast food worker disgusting viral video brand crisis",
        "Brand hygiene controversy likely to add disgust comments.",
    ),
    SearchSpec(
        "fear_public_safety_more",
        "fear_public_safety",
        "texting while driving PSA terrifying comments",
        "Road-safety PSA likely to add fear and sadness comments.",
    ),
    SearchSpec(
        "fear_public_safety_more",
        "fear_public_safety",
        "drunk driving PSA emotional scary comments",
        "Safety PSA likely to add fear and sadness comments.",
    ),
    SearchSpec(
        "fear_health_warning_more",
        "fear_health_warning",
        "anti smoking campaign scary comments",
        "Health-warning campaign likely to add fear comments.",
    ),
    SearchSpec(
        "fear_health_warning_more",
        "fear_health_warning",
        "truth anti vaping campaign scary comments",
        "Health-warning campaign likely to add fear and disgust comments.",
    ),
    SearchSpec(
        "sadness_psa_more",
        "sadness_psa",
        "Thai Life Insurance emotional commercial comments",
        "Emotional ad likely to add sadness and joy comments.",
    ),
    SearchSpec(
        "sadness_psa_more",
        "sadness_psa",
        "MetLife My Dad's Story emotional commercial comments",
        "Emotional ad likely to add sadness comments.",
    ),
    SearchSpec(
        "sadness_psa_more",
        "sadness_psa",
        "Google Loretta Super Bowl commercial comments",
        "Memory-focused campaign likely to add sadness and joy comments.",
    ),
    SearchSpec(
        "surprise_product_fail_more",
        "surprise_product_fail",
        "Samsung Galaxy Note 7 exploding battery comments",
        "Product-failure story likely to add surprise and fear comments.",
    ),
    SearchSpec(
        "surprise_product_fail_more",
        "surprise_product_fail",
        "Tesla Cybertruck window break launch comments",
        "Unexpected launch moment likely to add surprise comments.",
    ),
    SearchSpec(
        "surprise_product_launch_more",
        "tech_product_launch",
        "Apple Vision Pro reveal trailer comments wow",
        "Product reveal likely to add surprise, joy, and neutral comments.",
    ),
    SearchSpec(
        "surprise_entertainment_more",
        "entertainment_fandom",
        "Netflix Wednesday trailer reaction comments surprised",
        "Entertainment trailer likely to add surprise and joy comments.",
    ),
    SearchSpec(
        "joy_brand_campaign_more",
        "brand_campaign_positive",
        "Coca Cola happiness commercial comments official",
        "Positive brand campaign likely to add joy comments.",
    ),
    SearchSpec(
        "joy_brand_campaign_more",
        "brand_campaign_positive",
        "Dove Real Beauty Sketches comments",
        "Positive social campaign likely to add joy and sadness comments.",
    ),
    SearchSpec(
        "joy_brand_campaign_more",
        "brand_campaign_positive",
        "Old Spice The Man Your Man Could Smell Like comments",
        "Humorous ad likely to add joy and surprise comments.",
    ),
    SearchSpec(
        "neutral_product_review_more",
        "tech_review_mixed",
        "iPhone review comments battery camera price",
        "Product review likely to add neutral mixed comments.",
    ),
    SearchSpec(
        "neutral_product_review_more",
        "tech_review_mixed",
        "Samsung Galaxy review comments battery camera price",
        "Product review likely to add neutral mixed comments.",
    ),
    SearchSpec(
        "gaming_product_launch_more",
        "gaming_product_launch",
        "PlayStation 5 reveal trailer comments",
        "Gaming product launch likely to add joy, surprise, and anger comments.",
    ),
    SearchSpec(
        "app_brand_campaign_more",
        "app_brand_campaign",
        "Duolingo official commercial comments funny",
        "App brand campaign likely to add joy and surprise comments.",
    ),
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

    return ""


def youtube_get(url: str, params: dict[str, Any]) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    if response.status_code != 200:
        raise RuntimeError(f"YouTube API request failed: {response.status_code} {response.text}")
    return response.json()


def is_usable_comment(text: str) -> bool:
    cleaned = clean_comment_text(text)
    ascii_letters = sum(1 for char in cleaned if char in string.ascii_letters)
    return len(cleaned) >= 3 and ascii_letters >= 3


def comment_key(text: str) -> str:
    return " ".join(clean_comment_text(text).lower().split())


def load_seed_candidates() -> pd.DataFrame:
    frames = []
    for path in [CANDIDATE_AUDIT_PATH, *SEED_CANDIDATE_PATHS]:
        if path.exists():
            frames.append(pd.read_csv(path).fillna(""))
    if not frames:
        return pd.DataFrame()
    candidates = pd.concat(frames, ignore_index=True)
    candidates = candidates.drop_duplicates(subset=["video_id"]).reset_index(drop=True)
    return candidates


def load_seed_comments() -> pd.DataFrame:
    frames = []
    input_paths = [RAW_POOL_PATH] if RAW_POOL_PATH.exists() else SEED_COMMENT_PATHS
    for path in input_paths:
        if path.exists():
            frame = pd.read_csv(path).fillna("")
            frame["source_file"] = path.name
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["theme", "video_short", "video_id", "video", "comment_index", "comment"])
    comments = pd.concat(frames, ignore_index=True)
    comments["comment"] = comments["comment"].map(clean_comment_text)
    comments = comments[comments["comment"].map(is_usable_comment)].copy()
    comments = comments.drop_duplicates(subset=["comment"]).reset_index(drop=True)
    return comments


def fetch_video_stats(video_ids: list[str], api_key: str) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for start in range(0, len(video_ids), 50):
        chunk = [video_id for video_id in video_ids[start : start + 50] if video_id]
        if not chunk:
            continue
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
                "video": f"https://www.youtube.com/watch?v={item['id']}",
                "title": snippet.get("title", ""),
                "channel": snippet.get("channelTitle", ""),
                "published_at": snippet.get("publishedAt", "")[:10],
                "view_count": int(statistics.get("viewCount", 0)),
                "comment_count": int(statistics.get("commentCount", 0)),
            }
    return stats


def search_video_candidates(query: str, api_key: str, max_results: int = 8) -> list[str]:
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


def fetch_comments_for_video(video_id: str, api_key: str, target_count: int) -> list[str]:
    comments: list[str] = []
    seen: set[str] = set()

    for order in ["relevance", "time"]:
        page_token = ""
        while len(comments) < target_count:
            params = {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": 100,
                "order": order,
                "textFormat": "plainText",
                "key": api_key,
            }
            if page_token:
                params["pageToken"] = page_token
            payload = youtube_get(COMMENTS_URL, params)
            for item in payload.get("items", []):
                snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
                raw_text = snippet.get("textOriginal") or snippet.get("textDisplay") or ""
                cleaned = clean_comment_text(raw_text)
                key = comment_key(cleaned)
                if key and key not in seen and is_usable_comment(cleaned):
                    comments.append(cleaned)
                    seen.add(key)
                if len(comments) >= target_count:
                    break
            page_token = payload.get("nextPageToken", "")
            if not page_token:
                break
    return comments[:target_count]


def choose_balanced_label_targets(
    label_counts: dict[str, int] | pd.Series,
    target_total: int,
) -> dict[str, int]:
    counts = {label: int(dict(label_counts).get(label, 0)) for label in EMOTION_LABELS}
    available_total = sum(counts.values())
    if target_total <= 0:
        raise ValueError("target_total must be positive.")
    if target_total > available_total:
        raise ValueError("target_total cannot exceed available rows without oversampling.")

    targets = {label: 0 for label in EMOTION_LABELS}
    label_order = {label: index for index, label in enumerate(EMOTION_LABELS)}
    for _ in range(target_total):
        candidates = [label for label in EMOTION_LABELS if targets[label] < counts[label]]
        selected = min(candidates, key=lambda label: (targets[label], label_order[label]))
        targets[selected] += 1
    return targets


def classify_comments(comment_pool: pd.DataFrame) -> pd.DataFrame:
    labeled = comment_pool.copy()
    labels = labeled.apply(lambda row: score_comment(row["comment"], row["video_short"]), axis=1)
    labeled["manual_7_emotion"] = [label[0] for label in labels]
    labeled["manual_3_sentiment"] = [label[1] for label in labels]
    labeled["label_notes"] = [label[2] for label in labels]
    labeled["annotation_source"] = "assistant_assisted_review"
    return labeled


def select_balanced_dataset(
    labeled_pool: pd.DataFrame,
    target_total: int,
    random_state: int,
) -> pd.DataFrame:
    targets = choose_balanced_label_targets(
        labeled_pool["manual_7_emotion"].value_counts(),
        target_total=target_total,
    )
    parts = []
    for label, sample_size in targets.items():
        label_rows = labeled_pool[labeled_pool["manual_7_emotion"] == label]
        parts.append(label_rows.sample(n=sample_size, random_state=random_state))
    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )


def to_training_frame(labeled_df: pd.DataFrame) -> pd.DataFrame:
    training_df = labeled_df.rename(
        columns={"comment": "text", "manual_7_emotion": "label", "manual_3_sentiment": "sentiment"}
    ).copy()
    training_df["label_id"] = training_df["label"].map(LABEL_TO_ID).astype(int)
    return training_df[
        [
            "text",
            "label",
            "label_id",
            "sentiment",
            "video_short",
            "video",
            "comment_index",
            "label_notes",
        ]
    ]


def split_balanced_dataset(
    df: pd.DataFrame,
    validation_ratio: float = 0.2,
    random_state: int = 5240,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")

    validation_total = round(len(df) * validation_ratio)
    counts = df["label"].value_counts().reindex(EMOTION_LABELS, fill_value=0).astype(int)
    raw_targets = {label: counts[label] * validation_ratio for label in EMOTION_LABELS}
    validation_targets = {
        label: min(int(raw_targets[label]), counts[label])
        for label in EMOTION_LABELS
    }
    remaining = validation_total - sum(validation_targets.values())
    fractional_labels = sorted(
        EMOTION_LABELS,
        key=lambda label: (raw_targets[label] - int(raw_targets[label]), counts[label]),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for label in fractional_labels:
            if remaining <= 0:
                break
            if validation_targets[label] < counts[label]:
                validation_targets[label] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    train_parts = []
    validation_parts = []
    for label in EMOTION_LABELS:
        group = df[df["label"] == label].sample(frac=1.0, random_state=random_state)
        validation_count = validation_targets[label]
        validation_parts.append(group.iloc[:validation_count])
        train_parts.append(group.iloc[validation_count:])

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=random_state)
    validation_df = pd.concat(validation_parts, ignore_index=True).sample(frac=1.0, random_state=random_state)
    return train_df.reset_index(drop=True), validation_df.reset_index(drop=True)


def append_comment_rows(
    rows: list[dict[str, Any]],
    comments: list[str],
    candidate: dict[str, Any],
    seen_comments: set[str],
) -> int:
    added = 0
    for comment in comments:
        key = comment_key(comment)
        if not key or key in seen_comments:
            continue
        seen_comments.add(key)
        rows.append(
            {
                "theme": candidate.get("theme", ""),
                "video_short": candidate.get("video_short", ""),
                "video_id": candidate.get("video_id", ""),
                "video": candidate.get("video") or candidate.get("url") or "",
                "comment_index": added + 1,
                "comment": comment,
            }
        )
        added += 1
    return added


def collect_comment_pool(
    api_key: str,
    raw_target: int,
    max_comments_per_video: int,
    min_comments_per_video: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_comments = load_seed_comments()
    rows = seed_comments.to_dict("records")
    seen_comments = {comment_key(row["comment"]) for row in rows if row.get("comment")}
    candidate_audit_rows = []

    if not api_key:
        return seed_comments, pd.DataFrame(candidate_audit_rows)

    candidates = load_seed_candidates()
    for _, candidate_row in candidates.iterrows():
        if len(rows) >= raw_target:
            break
        candidate = candidate_row.to_dict()
        video_id = str(candidate.get("video_id", "")).strip()
        if not video_id:
            continue
        try:
            comments = fetch_comments_for_video(video_id, api_key, target_count=max_comments_per_video)
        except RuntimeError as exc:
            print(f"Skipped seed video {video_id}: {exc}", flush=True)
            continue
        added = append_comment_rows(rows, comments, candidate, seen_comments)
        audit_row = {**candidate, "fetched_comments": len(comments), "added_comments": added}
        candidate_audit_rows.append(audit_row)
        print(f"Seed video {video_id}: fetched {len(comments)}, added {added}, pool {len(rows)}", flush=True)

    used_video_ids = {str(row.get("video_id", "")) for row in candidate_audit_rows}
    used_video_ids.update(candidates.get("video_id", pd.Series(dtype=str)).astype(str).tolist())

    for spec in TARGETED_SEARCH_SPECS:
        if len(rows) >= raw_target:
            break
        print(f"Searching: {spec.query}", flush=True)
        try:
            candidate_ids = search_video_candidates(spec.query, api_key)
            stats = fetch_video_stats(candidate_ids, api_key)
        except RuntimeError as exc:
            print(f"Skipped search {spec.query}: {exc}", flush=True)
            continue
        selected_this_query = 0
        for video_id in candidate_ids:
            if len(rows) >= raw_target or selected_this_query >= 2:
                break
            if video_id in used_video_ids:
                continue
            video_stats = stats.get(video_id, {})
            if int(video_stats.get("comment_count", 0)) < min_comments_per_video:
                continue
            try:
                comments = fetch_comments_for_video(video_id, api_key, target_count=max_comments_per_video)
            except RuntimeError as exc:
                print(f"Skipped candidate {video_id}: {exc}", flush=True)
                continue
            if len(comments) < min_comments_per_video:
                continue
            candidate = {
                "theme": spec.theme,
                "video_short": spec.video_short,
                "video_id": video_id,
                "video": video_stats.get("video", f"https://www.youtube.com/watch?v={video_id}"),
                "title": video_stats.get("title", ""),
                "channel": video_stats.get("channel", ""),
                "published_at": video_stats.get("published_at", ""),
                "view_count": video_stats.get("view_count", 0),
                "comment_count": video_stats.get("comment_count", 0),
                "query": spec.query,
                "selection_reason": spec.selection_reason,
            }
            added = append_comment_rows(rows, comments, candidate, seen_comments)
            candidate_audit_rows.append({**candidate, "fetched_comments": len(comments), "added_comments": added})
            used_video_ids.add(video_id)
            selected_this_query += 1
            print(f"New video {video_id}: fetched {len(comments)}, added {added}, pool {len(rows)}", flush=True)

    pool = pd.DataFrame(rows).drop_duplicates(subset=["comment"]).reset_index(drop=True)
    if len(pool) > raw_target:
        pool = pool.sample(n=raw_target, random_state=random_state).reset_index(drop=True)
    return pool, pd.DataFrame(candidate_audit_rows)


def write_outputs(
    raw_pool: pd.DataFrame,
    labeled_pool: pd.DataFrame,
    final_labeled: pd.DataFrame,
    final_training: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    candidate_audit: pd.DataFrame,
) -> None:
    RAW_POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINAL_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw_pool.to_csv(RAW_POOL_PATH, index=False)
    labeled_pool.to_csv(LABELED_POOL_PATH, index=False)
    final_labeled.to_csv(FINAL_LABELED_PATH, index=False)
    final_training.to_csv(FINAL_SPLIT_DIR / "all.csv", index=False)
    train_df.to_csv(FINAL_SPLIT_DIR / "train.csv", index=False)
    validation_df.to_csv(FINAL_SPLIT_DIR / "validation.csv", index=False)
    candidate_audit.to_csv(CANDIDATE_AUDIT_PATH, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect, classify, and balance YouTube-domain comments for fine-tuning.",
    )
    parser.add_argument("--raw-target", type=int, default=8000)
    parser.add_argument("--collection-target", type=int, default=None)
    parser.add_argument("--final-size", type=int, default=5000)
    parser.add_argument("--max-comments-per-video", type=int, default=250)
    parser.add_argument("--min-comments-per-video", type=int, default=40)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=5240)
    parser.add_argument("--skip-fetch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = "" if args.skip_fetch else get_api_key()

    if args.skip_fetch:
        raw_pool = load_seed_comments()
        candidate_audit = pd.DataFrame()
    else:
        raw_pool, candidate_audit = collect_comment_pool(
            api_key=api_key,
            raw_target=args.collection_target or args.raw_target,
            max_comments_per_video=args.max_comments_per_video,
            min_comments_per_video=args.min_comments_per_video,
            random_state=args.random_state,
        )

    if len(raw_pool) < args.final_size:
        raise RuntimeError(
            f"Only {len(raw_pool)} usable comments are available; need at least {args.final_size}."
        )

    labeled_collection = classify_comments(raw_pool)
    if len(labeled_collection) > args.raw_target:
        labeled_pool = select_balanced_dataset(
            labeled_pool=labeled_collection,
            target_total=args.raw_target,
            random_state=args.random_state,
        )
        raw_pool = labeled_pool[
            ["theme", "video_short", "video_id", "video", "comment_index", "comment"]
        ].copy()
    else:
        labeled_pool = labeled_collection
    final_labeled = select_balanced_dataset(
        labeled_pool=labeled_pool,
        target_total=args.final_size,
        random_state=args.random_state,
    )
    final_training = to_training_frame(final_labeled)
    train_df, validation_df = split_balanced_dataset(
        final_training,
        validation_ratio=args.validation_ratio,
        random_state=args.random_state,
    )
    write_outputs(raw_pool, labeled_pool, final_labeled, final_training, train_df, validation_df, candidate_audit)

    print(f"\nRaw comment pool: {len(raw_pool)}")
    print(raw_pool["video_id"].nunique(), "videos")
    print("\nLabeled pool distribution")
    print(labeled_pool["manual_7_emotion"].value_counts().reindex(EMOTION_LABELS, fill_value=0).to_string())
    print("\nFinal 5000 distribution")
    print(final_training["label"].value_counts().reindex(EMOTION_LABELS, fill_value=0).to_string())
    print(f"\nTrain rows: {len(train_df)}")
    print(train_df["label"].value_counts().reindex(EMOTION_LABELS, fill_value=0).to_string())
    print(f"\nValidation rows: {len(validation_df)}")
    print(validation_df["label"].value_counts().reindex(EMOTION_LABELS, fill_value=0).to_string())
    print(f"\nSaved raw pool: {RAW_POOL_PATH}")
    print(f"Saved labeled pool: {LABELED_POOL_PATH}")
    print(f"Saved final labeled dataset: {FINAL_LABELED_PATH}")
    print(f"Saved final splits: {FINAL_SPLIT_DIR}")
    print(f"Saved candidate audit: {CANDIDATE_AUDIT_PATH}")


if __name__ == "__main__":
    main()
