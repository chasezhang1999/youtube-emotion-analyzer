"""Label YouTube comments using DeepSeek API (OpenAI-compatible).

Replaces the keyword-based label_youtube_domain_comments.py with actual AI labeling.
Output format is identical to youtube_domain_training_comments_expanded_assistant_labeled.csv.

Usage:
    DEEPSEEK_API_KEY=sk-... .venv/bin/python scripts/label_youtube_with_deepseek.py
    .venv/bin/python scripts/label_youtube_with_deepseek.py --limit 10   # test on first 10 rows
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "youtube_domain_training_comments_8000_unlabeled.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "youtube_domain_training_comments_8000_deepseek_labeled.csv"

EMOTION_LABELS = {"anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"}
SENTIMENT_LABELS = {"negative", "neutral", "positive"}

SYSTEM_PROMPT = """You are an emotion and sentiment annotation expert for YouTube comments.

Classify each YouTube comment into:
1. One of 7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise
2. One of 3 sentiments: negative, neutral, positive
3. A brief reason (max 20 words)

Rules:
- Consider sarcasm, humor, irony, and cultural context
- Emoji-only comments: infer emotion from emoji meaning
- Very short comments (e.g. "lol", "first"): classify based on tone
- Sponsored/ad/metadata comments: emotion=neutral, sentiment=neutral
- If a comment expresses mixed emotions, pick the DOMINANT one
- "neutral" emotion means no strong emotional signal detected

Respond in this exact JSON format:
{"emotion": "<emotion>", "sentiment": "<sentiment>", "reason": "<brief reason>"}"""

USER_PROMPT_TEMPLATE = 'Classify this YouTube comment:\n\n"{comment}"'


def build_client(api_key: str) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")


async def label_comment(
    client: AsyncOpenAI,
    comment: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> dict:
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(comment=comment)},
                    ],
                    temperature=0.1,
                    max_tokens=100,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content.strip()
                result = json.loads(content)

                emotion = result.get("emotion", "neutral").strip().lower()
                sentiment = result.get("sentiment", "neutral").strip().lower()
                reason = result.get("reason", "")

                if emotion not in EMOTION_LABELS:
                    emotion = "neutral"
                if sentiment not in SENTIMENT_LABELS:
                    sentiment = "neutral"

                return {"emotion": emotion, "sentiment": sentiment, "reason": reason}

            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    await asyncio.sleep(wait)
                else:
                    return {"emotion": "neutral", "sentiment": "neutral", "reason": f"API error: {e}"}


async def label_batch(
    client: AsyncOpenAI,
    comments: list[str],
    concurrency: int = 10,
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [label_comment(client, comment, semaphore) for comment in comments]
    return await asyncio.gather(*tasks)


def load_progress(output_path: Path) -> set[tuple[str, int]]:
    """Load already-labeled (video_short, comment_index) pairs for resume."""
    if not output_path.exists():
        return set()
    df = pd.read_csv(output_path)
    return set(zip(df["video_short"], df["comment_index"]))


def save_progress(output_path: Path, rows: list[dict]) -> None:
    """Append labeled rows to output CSV."""
    df_new = pd.DataFrame(rows)
    if output_path.exists():
        df_new.to_csv(output_path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(output_path, index=False)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Label YouTube comments with DeepSeek API")
    parser.add_argument("--limit", type=int, default=0, help="Limit to first N rows (0 = all)")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent API requests")
    parser.add_argument("--batch-size", type=int, default=50, help="Save progress every N rows")
    parser.add_argument("--api-key", type=str, default="", help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("Error: Set DEEPSEEK_API_KEY env var or pass --api-key")
        return

    df = pd.read_csv(INPUT_PATH)
    if args.limit > 0:
        df = df.head(args.limit)

    # Resume support: skip already-labeled rows
    done = load_progress(OUTPUT_PATH)
    remaining = df[~df.apply(lambda r: (r["video_short"], r["comment_index"]) in done, axis=1)]
    print(f"Total rows: {len(df)}, already labeled: {len(done)}, remaining: {len(remaining)}")

    if remaining.empty:
        print("All rows already labeled.")
        return

    client = build_client(api_key)

    # Process in batches for incremental saving
    batch_size = args.batch_size
    total_labeled = len(done)

    for start in range(0, len(remaining), batch_size):
        batch = remaining.iloc[start : start + batch_size]
        comments = batch["comment"].fillna("").astype(str).tolist()

        print(f"Labeling rows {start + 1}-{start + len(batch)} of {len(remaining)}...")
        results = await label_batch(client, comments, concurrency=args.concurrency)

        rows = []
        for (_, row), result in zip(batch.iterrows(), results):
            rows.append({
                "theme": row["theme"],
                "video_short": row["video_short"],
                "video_id": row["video_id"],
                "video": row["video"],
                "comment_index": row["comment_index"],
                "comment": row["comment"],
                "source_file": INPUT_PATH.name,
                "manual_7_emotion": result["emotion"],
                "manual_3_sentiment": result["sentiment"],
                "label_notes": result["reason"],
                "annotation_source": "deepseek_v4pro_ai_labeling",
            })

        save_progress(OUTPUT_PATH, rows)
        total_labeled += len(rows)
        print(f"  Saved {len(rows)} rows (total labeled: {total_labeled})")

    print(f"\nDone! Output: {OUTPUT_PATH}")

    # Print label distribution
    final_df = pd.read_csv(OUTPUT_PATH)
    print("\n7-emotion distribution:")
    print(final_df["manual_7_emotion"].value_counts().sort_index().to_string())
    print("\n3-sentiment distribution:")
    print(final_df["manual_3_sentiment"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    asyncio.run(main())
