"""Expand the 150-comment app benchmark to 500 comments using DeepSeek labeling.

Searches for 7 new YouTube videos across diverse themes, fetches 50 comments each,
labels them with DeepSeek API, and appends to the existing benchmark CSV.

Usage:
    DEEPSEEK_API_KEY=sk-... YOUTUBE_API_KEY=AIza... .venv/bin/python scripts/expand_benchmark_with_deepseek.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from youtube_emotion.youtube_client import fetch_top_comments

BENCHMARK_PATH = PROJECT_ROOT / "experiments" / "app_per_comment_manual_labels.csv"
OUTPUT_PATH = PROJECT_ROOT / "experiments" / "app_per_comment_manual_labels_500.csv"

VIDEO_SPECS = [
    {"video_short": "anger_brand_crisis", "video_id": "VrDWY6C1178", "theme": "anger"},
    {"video_short": "fear_public_safety", "video_id": "O8MQV40pOtk", "theme": "fear"},
    {"video_short": "sadness_psa", "video_id": "B2rFTbvwteo", "theme": "sadness"},
    {"video_short": "surprise_product_fail", "video_id": "8H6jz30Im_Y", "theme": "surprise"},
    {"video_short": "disgust_food_safety", "video_id": "--OnclX8yac", "theme": "disgust"},
    {"video_short": "joy_trailer", "video_id": "LEjhY15eCx0", "theme": "joy"},
    {"video_short": "brand_crisis_negative", "video_id": "JVxCp22WatU", "theme": "negative"},
]

SYSTEM_PROMPT = """You are an emotion and sentiment annotation expert for YouTube comments.

Classify each YouTube comment into:
1. One of 7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise
2. One of 3 sentiments: negative, neutral, positive

Rules:
- Consider sarcasm, humor, irony, and cultural context
- Emoji-only comments: infer emotion from emoji meaning
- Very short comments (e.g. "lol", "first"): classify based on tone
- Sponsored/ad/metadata comments: emotion=neutral, sentiment=neutral
- If a comment expresses mixed emotions, pick the DOMINANT one

Respond in this exact JSON format:
{"emotion": "<emotion>", "sentiment": "<sentiment>", "reason": "<brief reason>"}"""


async def label_comment(client: AsyncOpenAI, comment: str, sem: asyncio.Semaphore) -> dict:
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f'Classify this YouTube comment:\n\n"{comment}"'},
                    ],
                    temperature=0.1,
                    max_tokens=100,
                    response_format={"type": "json_object"},
                )
                r = json.loads(resp.choices[0].message.content)
                emotion = r.get("emotion", "neutral").lower().strip()
                sentiment = r.get("sentiment", "neutral").lower().strip()
                valid_e = {"anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"}
                valid_s = {"negative", "neutral", "positive"}
                return {
                    "manual_7_emotion": emotion if emotion in valid_e else "neutral",
                    "manual_3_sentiment": sentiment if sentiment in valid_s else "neutral",
                    "manual_review_note": r.get("reason", ""),
                }
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** (attempt + 1))
                else:
                    return {"manual_7_emotion": "neutral", "manual_3_sentiment": "neutral", "manual_review_note": f"error: {e}"}


async def label_batch(client: AsyncOpenAI, comments: list[str]) -> list[dict]:
    sem = asyncio.Semaphore(10)
    return await asyncio.gather(*[label_comment(client, c, sem) for c in comments])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--youtube-api-key", default=os.environ.get("YOUTUBE_API_KEY", ""))
    parser.add_argument("--max-per-video", type=int, default=50)
    args = parser.parse_args()

    if not args.api_key:
        print("Error: set DEEPSEEK_API_KEY"); return

    existing = pd.read_csv(BENCHMARK_PATH)
    existing_videos = set(existing["video_short"].unique())
    print(f"Existing benchmark: {len(existing)} comments, {len(existing_videos)} videos")

    new_rows = []
    for spec in VIDEO_SPECS:
        if spec["video_short"] in existing_videos:
            print(f"  Skipping {spec['video_short']} (already in benchmark)")
            continue

        video_url = f"https://www.youtube.com/watch?v={spec['video_id']}"
        print(f"  Fetching comments for {spec['video_short']} ({spec['video_id']})...")
        try:
            comments = fetch_top_comments(
                video_id=spec["video_id"],
                api_key=args.youtube_api_key,
                max_results=args.max_per_video,
            )
        except Exception as e:
            print(f"    Failed to fetch: {e}")
            continue

        if not comments:
            print(f"    No comments found")
            continue

        print(f"    Got {len(comments)} comments, labeling with DeepSeek...")
        client = AsyncOpenAI(api_key=args.api_key, base_url="https://api.deepseek.com/v1")
        labels = asyncio.run(label_batch(client, comments))

        for i, (comment, label) in enumerate(zip(comments, labels)):
            new_rows.append({
                "video": video_url,
                "video_short": spec["video_short"],
                "comment_index": i + 1,
                "comment": comment,
                **label,
                "manual_review_source": "deepseek_v4pro_ai_labeling",
            })

    if not new_rows:
        print("No new comments to add.")
        return

    new_df = pd.DataFrame(new_rows)

    # Append new rows to existing benchmark, preserving all existing columns
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"\nDone! Total: {len(combined)} comments ({len(existing)} old + {len(new_df)} new)")
    print(f"Videos: {combined['video_short'].nunique()}")
    print(f"Saved to: {OUTPUT_PATH}")
    print("\nEmotion distribution:")
    print(combined["manual_7_emotion"].value_counts().sort_index().to_string())
    print("\nSentiment distribution:")
    print(combined["manual_3_sentiment"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
