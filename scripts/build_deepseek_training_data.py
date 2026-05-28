"""Build training dataset from DeepSeek-labeled YouTube comments.

Reads the DeepSeek-labeled CSV and produces:
  - data/youtube_domain_7class_deepseek/all.csv
  - data/youtube_domain_7class_deepseek/train.csv
  - data/youtube_domain_7class_deepseek/validation.csv

These files are compatible with scripts/train_youtube_domain_all_models.py.

Usage:
    .venv/bin/python scripts/build_deepseek_training_data.py
    .venv/bin/python scripts/build_deepseek_training_data.py --target-size 8000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "youtube_domain_training_comments_8000_deepseek_labeled.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "youtube_domain_7class_deepseek"

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
LABEL_TO_ID = {label: i for i, label in enumerate(EMOTION_LABELS)}


def load_and_convert(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df = df.rename(columns={
        "comment": "text",
        "manual_7_emotion": "label",
        "manual_3_sentiment": "sentiment",
    })
    df = df[df["label"].isin(EMOTION_LABELS)].copy()
    df["label_id"] = df["label"].map(LABEL_TO_ID)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    return df


def choose_balanced_targets(df: pd.DataFrame, target_size: int) -> pd.DataFrame:
    counts = df["label"].value_counts()
    per_class = target_size // len(EMOTION_LABELS)
    parts = []
    for label in EMOTION_LABELS:
        subset = df[df["label"] == label]
        n = min(per_class, len(subset))
        parts.append(subset.sample(n=n, random_state=5240))
    balanced = pd.concat(parts).sample(frac=1.0, random_state=5240).reset_index(drop=True)
    return balanced


def split_dataset(df: pd.DataFrame, val_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts, val_parts = [], []
    for label in EMOTION_LABELS:
        subset = df[df["label"] == label].sample(frac=1.0, random_state=5240)
        n_val = max(1, round(len(subset) * val_ratio))
        val_parts.append(subset.iloc[:n_val])
        train_parts.append(subset.iloc[n_val:])
    train = pd.concat(train_parts).sample(frac=1.0, random_state=5240).reset_index(drop=True)
    val = pd.concat(val_parts).sample(frac=1.0, random_state=5240).reset_index(drop=True)
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-size", type=int, default=0, help="Balanced target size (0 = use all)")
    parser.add_argument("--input", type=str, default=str(INPUT_PATH))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_convert(input_path)
    print(f"Loaded {len(df)} labeled comments from {input_path.name}")

    if args.target_size > 0:
        df = choose_balanced_targets(df, args.target_size)
        print(f"Balanced to {len(df)} comments")

    train_df, val_df = split_dataset(df)
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}")

    cols = ["text", "label", "label_id", "sentiment", "video_short", "video", "comment_index", "label_notes"]
    cols = [c for c in cols if c in df.columns]

    df[cols].to_csv(output_dir / "all.csv", index=False)
    train_df[cols].to_csv(output_dir / "train.csv", index=False)
    val_df[cols].to_csv(output_dir / "validation.csv", index=False)

    print(f"\nSaved to {output_dir}/")
    print("\nLabel distribution (train):")
    print(train_df["label"].value_counts().reindex(EMOTION_LABELS).to_string())
    print("\nLabel distribution (validation):")
    print(val_df["label"].value_counts().reindex(EMOTION_LABELS).to_string())


if __name__ == "__main__":
    main()
