from __future__ import annotations

import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from youtube_emotion.core import normalize_emotion_label, normalize_sentiment_label
from youtube_emotion.model_runner import extract_top_prediction


PIPELINE_KWARGS = {
    "truncation": True,
    "max_length": 512,
    "return_token_type_ids": False,
}

GO_EMOTIONS_TEST_PATH = PROJECT_ROOT / "data" / "go_emotions_7class" / "test.csv"
YOUTUBE_VALIDATION_PATH = (
    PROJECT_ROOT / "data" / "youtube_domain_7class_assistant" / "validation.csv"
)
MANUAL_LABELS_PATH = PROJECT_ROOT / "experiments" / "app_per_comment_manual_labels.csv"

MODEL_SELECTION_PATH = PROJECT_ROOT / "experiments" / "experimental_results_template.csv"
YOUTUBE_VALIDATION_RESULTS_PATH = (
    PROJECT_ROOT / "experiments" / "youtube_domain_validation_performance.csv"
)
APP_DETAIL_PATH = PROJECT_ROOT / "experiments" / "app_per_comment_model_predictions.csv"
APP_SUMMARY_PATH = PROJECT_ROOT / "experiments" / "app_performance_model_comparison.csv"
APP_RUNTIME_PATH = PROJECT_ROOT / "experiments" / "app_runtime_summary.csv"
APP_5MODEL_PATH = PROJECT_ROOT / "experiments" / "app_model_comparison_5models.csv"
REVISION_SUMMARY_PATH = PROJECT_ROOT / "experiments" / "model_revision_summary.csv"

EMOTION_TO_SENTIMENT = {
    "anger": "negative",
    "disgust": "negative",
    "fear": "negative",
    "sadness": "negative",
    "joy": "positive",
    "surprise": "positive",
    "neutral": "neutral",
}


@dataclass(frozen=True)
class ModelSpec:
    stage: str
    display_name: str
    column_prefix: str
    model_name: str
    task: str
    pipeline_task: str


EMOTION_MODELS = [
    ModelSpec(
        stage="pre-tuning public baseline",
        display_name="Pre-tuning baseline",
        column_prefix="pretrained",
        model_name="j-hartmann/emotion-english-distilroberta-base",
        task="7-emotion",
        pipeline_task="text-classification",
    ),
    ModelSpec(
        stage="GoEmotions fine-tuned DistilBERT",
        display_name="GoEmotions fine-tuned",
        column_prefix="finetuned",
        model_name="chase1zhang/youtube-emotion-distilbert",
        task="7-emotion",
        pipeline_task="text-classification",
    ),
    ModelSpec(
        stage="YouTube-domain adapted DistilBERT",
        display_name="YouTube-domain adapted (your model)",
        column_prefix="domain_adapted",
        model_name="chase1zhang/youtube-emotion-distilbert-domain-adapted",
        task="7-emotion",
        pipeline_task="text-classification",
    ),
    ModelSpec(
        stage="public GoEmotions RoBERTa",
        display_name="Public GoEmotions RoBERTa (SamLowe)",
        column_prefix="samlowe_roberta",
        model_name="SamLowe/roberta-base-go_emotions",
        task="7-emotion",
        pipeline_task="text-classification",
    ),
    ModelSpec(
        stage="public RoBERTa-large seven-emotion",
        display_name="Public RoBERTa-large 7-emotion",
        column_prefix="jhartmann_roberta_large",
        model_name="j-hartmann/emotion-english-roberta-large",
        task="7-emotion",
        pipeline_task="text-classification",
    ),
]

SENTIMENT_MODEL = ModelSpec(
    stage="supporting sentiment pipeline",
    display_name="3-sentiment pipeline",
    column_prefix="sentiment_pipeline",
    model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
    task="3-sentiment",
    pipeline_task="sentiment-analysis",
)

GO_EMOTIONS_MODEL_SELECTION = [
    EMOTION_MODELS[0],
    EMOTION_MODELS[1],
    EMOTION_MODELS[2],
    SENTIMENT_MODEL,
]


def distribution_string(values: list[str]) -> str:
    counts = Counter(values)
    return "; ".join(f"{label}:{counts[label]}" for label in sorted(counts))


def main_label(values: list[str]) -> str:
    if not values:
        return ""
    return Counter(values).most_common(1)[0][0]


def resolve_model_source(model_name: str) -> tuple[str, dict[str, Any]]:
    """Pin project-owned Hugging Face repos to the current remote SHA."""
    metadata: dict[str, Any] = {"model_name": model_name, "resolved_source": model_name}

    try:
        from huggingface_hub import HfApi, snapshot_download

        info = HfApi().model_info(model_name)
        metadata["remote_sha"] = info.sha
        metadata["last_modified"] = str(info.last_modified)

        if model_name.startswith("chase1zhang/"):
            local_path = snapshot_download(
                repo_id=model_name,
                revision=info.sha,
                allow_patterns=[
                    ".gitattributes",
                    "README.md",
                    "config.json",
                    "model.safetensors",
                    "pytorch_model.bin",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.txt",
                    "merges.txt",
                ],
            )
            metadata["resolved_source"] = local_path
            return local_path, metadata
    except Exception as exc:  # pragma: no cover - defensive for offline reruns
        metadata["resolution_warning"] = str(exc)

    return model_name, metadata


def load_pipeline(model_name: str, pipeline_task: str):
    from transformers import pipeline

    source, metadata = resolve_model_source(model_name)
    load_start = time.time()
    pipe = pipeline(pipeline_task, model=source, tokenizer=source, device=-1)
    load_seconds = time.time() - load_start
    return pipe, load_seconds, metadata


def run_predictions(
    pipe,
    texts: list[str],
    label_normalizer: Callable[[str], str],
) -> tuple[list[tuple[str, str, float]], float]:
    predict_start = time.time()
    outputs = pipe(texts, batch_size=16, **PIPELINE_KWARGS)
    predict_seconds = time.time() - predict_start

    predictions = []
    for output in outputs:
        raw_label, score = extract_top_prediction(output)
        normalized_label = label_normalizer(raw_label)
        predictions.append((normalized_label, raw_label, round(float(score), 4)))

    return predictions, predict_seconds


def summarize_accuracy(
    model_name: str,
    task: str,
    dataset: str,
    device: str,
    labels: list[str],
    predictions: list[str],
    load_seconds: float,
    predict_seconds: float,
    notes: str,
) -> dict[str, Any]:
    matches = sum(predicted == label for predicted, label in zip(predictions, labels))
    total = len(labels)
    return {
        "model_name": model_name,
        "task": task,
        "dataset": dataset,
        "device": device,
        "num_samples": total,
        "matched_samples": matches,
        "accuracy": round(matches / total, 4) if total else 0.0,
        "runtime_with_model_loading_seconds": round(load_seconds + predict_seconds, 4),
        "runtime_without_model_loading_seconds": round(predict_seconds, 4),
        "notes": notes,
    }


def summarize_app_predictions(detail_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    group_cols = ["model_stage", "model_name", "evaluation_task"]
    for (model_stage, model_name, task, video), group in detail_df.groupby(
        group_cols + ["video"], sort=False
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

    for (model_stage, model_name, task), group in detail_df.groupby(group_cols, sort=False):
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


def update_wide_manual_file(
    manual_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    specs: list[ModelSpec],
) -> pd.DataFrame:
    updated = manual_df.copy()

    for spec in specs:
        model_rows = detail_df[
            (detail_df["model_name"] == spec.model_name)
            & (detail_df["evaluation_task"] == "7-emotion")
        ].sort_values(["video_short", "comment_index"])
        lookup = model_rows.set_index(["video_short", "comment_index"])
        labels, scores, matches = [], [], []

        for row in updated.itertuples(index=False):
            pred_row = lookup.loc[(row.video_short, int(row.comment_index))]
            labels.append(pred_row["predicted_label"])
            scores.append(pred_row["prediction_score"])
            matches.append(pred_row["match"])

        updated[f"{spec.column_prefix}_7_emotion"] = labels
        updated[f"{spec.column_prefix}_7_score"] = scores
        updated[f"{spec.column_prefix}_7_match"] = matches

    sentiment_rows = detail_df[
        (detail_df["model_name"] == SENTIMENT_MODEL.model_name)
        & (detail_df["evaluation_task"] == "3-sentiment")
    ].sort_values(["video_short", "comment_index"])
    lookup = sentiment_rows.set_index(["video_short", "comment_index"])
    labels, scores, matches = [], [], []
    for row in updated.itertuples(index=False):
        pred_row = lookup.loc[(row.video_short, int(row.comment_index))]
        labels.append(pred_row["predicted_label"])
        scores.append(pred_row["prediction_score"])
        matches.append(pred_row["match"])

    updated["sentiment_pipeline_3"] = labels
    updated["sentiment_pipeline_3_score"] = scores
    updated["sentiment_3_match"] = matches
    return updated


def make_app_5model_summary(
    app_summary: pd.DataFrame,
    runtime_df: pd.DataFrame,
    specs: list[ModelSpec],
) -> pd.DataFrame:
    runtime_lookup = runtime_df.set_index("model_name")
    display_lookup = {spec.model_name: spec.display_name for spec in specs + [SENTIMENT_MODEL]}

    rows = []
    for row in app_summary.itertuples(index=False):
        runtime_row = runtime_lookup.loc[row.model_name]
        inference = float(runtime_row["runtime_without_model_loading_seconds"])
        loading = float(runtime_row["runtime_with_model_loading_seconds"]) - inference
        rows.append(
            {
                "video": row.video,
                "model": display_lookup.get(row.model_name, row.model_stage),
                "repo": row.model_name,
                "task": row.evaluation_task,
                "matched": int(row.matched_comments),
                "total": int(row.num_comments),
                "accuracy": float(row.accuracy),
                "load_s": round(loading, 4),
                "infer_s": round(inference, 4),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    go_df = pd.read_csv(GO_EMOTIONS_TEST_PATH)
    youtube_df = pd.read_csv(YOUTUBE_VALIDATION_PATH)
    manual_df = pd.read_csv(MANUAL_LABELS_PATH)

    go_df["sentiment_label"] = go_df["label_name"].map(EMOTION_TO_SENTIMENT)

    revision_rows: list[dict[str, Any]] = []
    model_selection_rows: list[dict[str, Any]] = []
    youtube_validation_rows: list[dict[str, Any]] = []
    app_detail_rows: list[dict[str, Any]] = []
    app_runtime_rows: list[dict[str, Any]] = []

    all_specs = EMOTION_MODELS + [SENTIMENT_MODEL]

    for spec in all_specs:
        print(f"Loading {spec.stage}: {spec.model_name}", flush=True)
        normalizer = (
            normalize_emotion_label if spec.task == "7-emotion" else normalize_sentiment_label
        )
        pipe, load_seconds, metadata = load_pipeline(spec.model_name, spec.pipeline_task)
        revision_rows.append(
            {
                **metadata,
                "model_stage": spec.stage,
                "task": spec.task,
                "load_seconds": round(load_seconds, 4),
            }
        )

        if spec in GO_EMOTIONS_MODEL_SELECTION:
            true_col = "label_name" if spec.task == "7-emotion" else "sentiment_label"
            predictions, predict_seconds = run_predictions(
                pipe=pipe,
                texts=go_df["text"].astype(str).tolist(),
                label_normalizer=normalizer,
            )
            predicted_labels = [item[0] for item in predictions]
            model_selection_rows.append(
                {
                    "experiment_id": len(model_selection_rows) + 1,
                    **summarize_accuracy(
                        model_name=spec.model_name,
                        task=(
                            "seven-emotion classification"
                            if spec.task == "7-emotion"
                            else "sentiment classification"
                        ),
                        dataset="GoEmotions 7-class test split",
                        device="CPU",
                        labels=go_df[true_col].astype(str).tolist(),
                        predictions=predicted_labels,
                        load_seconds=load_seconds,
                        predict_seconds=predict_seconds,
                        notes=f"{spec.stage}; refreshed after 5,000/1,000 split expansion",
                    ),
                }
            )

        true_col = "label" if spec.task == "7-emotion" else "sentiment"
        predictions, predict_seconds = run_predictions(
            pipe=pipe,
            texts=youtube_df["text"].astype(str).tolist(),
            label_normalizer=normalizer,
        )
        predicted_labels = [item[0] for item in predictions]
        youtube_validation_rows.append(
            summarize_accuracy(
                model_name=spec.model_name,
                task=(
                    "seven-emotion classification"
                    if spec.task == "7-emotion"
                    else "sentiment classification"
                ),
                dataset="YouTube-domain validation comments",
                device="CPU",
                labels=youtube_df[true_col].astype(str).tolist(),
                predictions=predicted_labels,
                load_seconds=load_seconds,
                predict_seconds=predict_seconds,
                notes=f"{spec.stage}; refreshed on expanded 592-comment validation split",
            )
        )

        app_true_col = "manual_7_emotion" if spec.task == "7-emotion" else "manual_3_sentiment"
        predictions, predict_seconds = run_predictions(
            pipe=pipe,
            texts=manual_df["comment"].astype(str).tolist(),
            label_normalizer=normalizer,
        )
        app_runtime_rows.append(
            {
                "model_name": spec.model_name,
                "dataset": "150 reviewed Streamlit app comments",
                "device": "CPU",
                "num_comments": len(manual_df),
                "runtime_with_model_loading_seconds": round(load_seconds + predict_seconds, 4),
                "runtime_without_model_loading_seconds": round(predict_seconds, 4),
                "notes": spec.stage,
            }
        )

        true_labels = manual_df[app_true_col].astype(str).tolist()
        for row, true_label, prediction in zip(
            manual_df.itertuples(index=False),
            true_labels,
            predictions,
        ):
            predicted_label, raw_label, score = prediction
            app_detail_rows.append(
                {
                    "video": row.video,
                    "video_short": row.video_short,
                    "comment_index": int(row.comment_index),
                    "comment": row.comment,
                    "evaluation_task": spec.task,
                    "model_stage": spec.stage,
                    "model_name": spec.model_name,
                    "manual_label": true_label,
                    "predicted_label": predicted_label,
                    "raw_model_label": raw_label,
                    "prediction_score": score,
                    "match": "Yes" if predicted_label == true_label else "No",
                }
            )

    model_selection_df = pd.DataFrame(model_selection_rows)
    youtube_validation_df = pd.DataFrame(youtube_validation_rows)
    app_detail_df = pd.DataFrame(app_detail_rows)
    app_summary_df = summarize_app_predictions(app_detail_df)
    app_runtime_df = pd.DataFrame(app_runtime_rows)
    app_5model_df = make_app_5model_summary(
        app_summary=app_summary_df,
        runtime_df=app_runtime_df,
        specs=EMOTION_MODELS,
    )
    updated_manual_df = update_wide_manual_file(manual_df, app_detail_df, EMOTION_MODELS)
    revision_df = pd.DataFrame(revision_rows)

    model_selection_df.to_csv(MODEL_SELECTION_PATH, index=False)
    youtube_validation_df.to_csv(YOUTUBE_VALIDATION_RESULTS_PATH, index=False)
    app_detail_df.to_csv(APP_DETAIL_PATH, index=False)
    app_summary_df.to_csv(APP_SUMMARY_PATH, index=False)
    app_runtime_df.to_csv(APP_RUNTIME_PATH, index=False)
    app_5model_df.to_csv(APP_5MODEL_PATH, index=False)
    updated_manual_df.to_csv(MANUAL_LABELS_PATH, index=False)
    revision_df.to_csv(REVISION_SUMMARY_PATH, index=False)

    print(f"Saved model selection results: {MODEL_SELECTION_PATH}")
    print(f"Saved YouTube validation results: {YOUTUBE_VALIDATION_RESULTS_PATH}")
    print(f"Saved app detail predictions: {APP_DETAIL_PATH}")
    print(f"Saved app performance summary: {APP_SUMMARY_PATH}")
    print(f"Saved app runtime summary: {APP_RUNTIME_PATH}")
    print(f"Saved app 5-model summary: {APP_5MODEL_PATH}")
    print(f"Updated wide manual label file: {MANUAL_LABELS_PATH}")
    print(f"Saved model revision summary: {REVISION_SUMMARY_PATH}")

    print("\nModel selection summary")
    print(
        model_selection_df[
            ["model_name", "task", "num_samples", "matched_samples", "accuracy"]
        ].to_string(index=False)
    )

    print("\nApp overall summary")
    print(
        app_summary_df[app_summary_df["video"] == "OVERALL"][
            ["evaluation_task", "model_stage", "matched_comments", "num_comments", "accuracy"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
