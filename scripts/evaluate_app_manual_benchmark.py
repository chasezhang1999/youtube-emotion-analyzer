from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from youtube_emotion.core import normalize_emotion_label, normalize_sentiment_label
from youtube_emotion.model_runner import extract_top_prediction


MANUAL_LABELS_PATH = PROJECT_ROOT / "experiments" / "app_per_comment_manual_labels.csv"
DETAIL_PATH = PROJECT_ROOT / "experiments" / "app_per_comment_model_predictions.csv"
SUMMARY_PATH = PROJECT_ROOT / "experiments" / "app_performance_model_comparison.csv"
RUNTIME_PATH = PROJECT_ROOT / "experiments" / "app_runtime_summary.csv"

PIPELINE_KWARGS = {"truncation": True, "return_token_type_ids": False}

EMOTION_MODELS = [
    {
        "stage": "pre-tuning public baseline",
        "column_prefix": "pretrained",
        "model_name": "j-hartmann/emotion-english-distilroberta-base",
        "task": "7-emotion",
    },
    {
        "stage": "GoEmotions fine-tuned DistilBERT",
        "column_prefix": "finetuned",
        "model_name": "chase1zhang/youtube-emotion-distilbert",
        "task": "7-emotion",
    },
    {
        "stage": "YouTube-domain adapted DistilBERT",
        "column_prefix": "domain_adapted",
        "model_name": "chase1zhang/youtube-emotion-distilbert-domain-adapted",
        "task": "7-emotion",
    },
    {
        "stage": "public GoEmotions RoBERTa",
        "column_prefix": "samlowe_roberta",
        "model_name": "SamLowe/roberta-base-go_emotions",
        "task": "7-emotion",
    },
    {
        "stage": "public RoBERTa-large seven-emotion",
        "column_prefix": "jhartmann_roberta_large",
        "model_name": "j-hartmann/emotion-english-roberta-large",
        "task": "7-emotion",
    },
]

SENTIMENT_MODEL = {
    "stage": "supporting sentiment pipeline",
    "column_prefix": "sentiment_pipeline",
    "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "task": "3-sentiment",
}


def distribution_string(values: list[str]) -> str:
    counts = Counter(values)
    return "; ".join(f"{label}:{counts[label]}" for label in sorted(counts))


def main_label(values: list[str]) -> str:
    if not values:
        return ""
    counts = Counter(values)
    return counts.most_common(1)[0][0]


def load_pipeline(model_name: str, task_name: str):
    from transformers import pipeline

    return pipeline(task_name, model=model_name, device=-1)


def run_model(
    df: pd.DataFrame,
    model_spec: dict[str, str],
    task_name: str,
    label_normalizer,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    texts = df["comment"].astype(str).tolist()

    load_start = time.time()
    pipe = load_pipeline(model_spec["model_name"], task_name)
    load_seconds = time.time() - load_start

    predict_start = time.time()
    outputs = pipe(texts, batch_size=16, **PIPELINE_KWARGS)
    predict_seconds = time.time() - predict_start

    rows = []
    for row, output in zip(df.itertuples(index=False), outputs):
        raw_label, score = extract_top_prediction(output)
        normalized_label = label_normalizer(raw_label)
        manual_label = (
            row.manual_7_emotion
            if model_spec["task"] == "7-emotion"
            else row.manual_3_sentiment
        )
        rows.append(
            {
                "video": row.video,
                "video_short": row.video_short,
                "comment_index": int(row.comment_index),
                "comment": row.comment,
                "evaluation_task": model_spec["task"],
                "model_stage": model_spec["stage"],
                "model_name": model_spec["model_name"],
                "manual_label": manual_label,
                "predicted_label": normalized_label,
                "raw_model_label": raw_label,
                "prediction_score": round(float(score), 4),
                "match": "Yes" if normalized_label == manual_label else "No",
            }
        )

    runtime = {
        "model_name": model_spec["model_name"],
        "dataset": "150 reviewed Streamlit app comments",
        "device": "CPU",
        "num_comments": len(df),
        "runtime_with_model_loading_seconds": round(load_seconds + predict_seconds, 4),
        "runtime_without_model_loading_seconds": round(predict_seconds, 4),
        "notes": model_spec["stage"],
    }
    return rows, runtime


def summarize_model(detail_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for (model_stage, model_name, task, video), group in detail_df.groupby(
        ["model_stage", "model_name", "evaluation_task", "video"],
        sort=False,
    ):
        matches = int((group["match"] == "Yes").sum())
        total = int(len(group))
        summary_rows.append(
            {
                "video": video,
                "evaluation_task": task,
                "model_stage": model_stage,
                "model_name": model_name,
                "num_comments": total,
                "matched_comments": matches,
                "accuracy": round(matches / total, 4) if total else 0,
                "manual_main_label": main_label(group["manual_label"].tolist()),
                "model_main_label": main_label(group["predicted_label"].tolist()),
                "manual_distribution": distribution_string(group["manual_label"].tolist()),
                "model_distribution": distribution_string(group["predicted_label"].tolist()),
            }
        )

    for (model_stage, model_name, task), group in detail_df.groupby(
        ["model_stage", "model_name", "evaluation_task"],
        sort=False,
    ):
        matches = int((group["match"] == "Yes").sum())
        total = int(len(group))
        summary_rows.append(
            {
                "video": "OVERALL",
                "evaluation_task": task,
                "model_stage": model_stage,
                "model_name": model_name,
                "num_comments": total,
                "matched_comments": matches,
                "accuracy": round(matches / total, 4) if total else 0,
                "manual_main_label": main_label(group["manual_label"].tolist()),
                "model_main_label": main_label(group["predicted_label"].tolist()),
                "manual_distribution": distribution_string(group["manual_label"].tolist()),
                "model_distribution": distribution_string(group["predicted_label"].tolist()),
            }
        )
    return pd.DataFrame(summary_rows)


def update_wide_manual_file(df: pd.DataFrame, detail_df: pd.DataFrame) -> pd.DataFrame:
    updated = df.copy()
    for spec in EMOTION_MODELS:
        model_rows = detail_df[
            (detail_df["model_name"] == spec["model_name"])
            & (detail_df["evaluation_task"] == "7-emotion")
        ].sort_values(["video_short", "comment_index"])
        prefix = spec["column_prefix"]

        lookup = model_rows.set_index(["video_short", "comment_index"])
        labels = []
        scores = []
        matches = []
        for row in updated.itertuples(index=False):
            pred_row = lookup.loc[(row.video_short, int(row.comment_index))]
            labels.append(pred_row["predicted_label"])
            scores.append(pred_row["prediction_score"])
            matches.append(pred_row["match"])

        updated[f"{prefix}_7_emotion"] = labels
        updated[f"{prefix}_7_score"] = scores
        updated[f"{prefix}_7_match"] = matches

    sentiment_rows = detail_df[
        (detail_df["model_name"] == SENTIMENT_MODEL["model_name"])
        & (detail_df["evaluation_task"] == "3-sentiment")
    ].sort_values(["video_short", "comment_index"])
    lookup = sentiment_rows.set_index(["video_short", "comment_index"])
    sentiment_labels = []
    sentiment_scores = []
    sentiment_matches = []
    for row in updated.itertuples(index=False):
        pred_row = lookup.loc[(row.video_short, int(row.comment_index))]
        sentiment_labels.append(pred_row["predicted_label"])
        sentiment_scores.append(pred_row["prediction_score"])
        sentiment_matches.append(pred_row["match"])

    updated["sentiment_pipeline_3"] = sentiment_labels
    updated["sentiment_pipeline_3_score"] = sentiment_scores
    updated["sentiment_3_match"] = sentiment_matches
    return updated


def main() -> None:
    df = pd.read_csv(MANUAL_LABELS_PATH)
    detail_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    for spec in EMOTION_MODELS:
        print(f"Running {spec['stage']}: {spec['model_name']}", flush=True)
        rows, runtime = run_model(
            df=df,
            model_spec=spec,
            task_name="text-classification",
            label_normalizer=normalize_emotion_label,
        )
        detail_rows.extend(rows)
        runtime_rows.append(runtime)

    print(f"Running sentiment model: {SENTIMENT_MODEL['model_name']}", flush=True)
    rows, runtime = run_model(
        df=df,
        model_spec=SENTIMENT_MODEL,
        task_name="sentiment-analysis",
        label_normalizer=normalize_sentiment_label,
    )
    detail_rows.extend(rows)
    runtime_rows.append(runtime)

    detail_df = pd.DataFrame(detail_rows)
    summary_df = summarize_model(detail_df)
    runtime_df = pd.DataFrame(runtime_rows)
    updated_manual_df = update_wide_manual_file(df, detail_df)

    detail_df.to_csv(DETAIL_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    runtime_df.to_csv(RUNTIME_PATH, index=False)
    updated_manual_df.to_csv(MANUAL_LABELS_PATH, index=False)

    print(f"Saved detail predictions: {DETAIL_PATH}")
    print(f"Saved model comparison summary: {SUMMARY_PATH}")
    print(f"Saved runtime summary: {RUNTIME_PATH}")
    print(f"Updated wide benchmark file: {MANUAL_LABELS_PATH}")
    print("\nOverall summary")
    print(
        summary_df[summary_df["video"] == "OVERALL"][
            ["evaluation_task", "model_stage", "matched_comments", "num_comments", "accuracy"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
