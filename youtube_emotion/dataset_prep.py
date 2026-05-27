from __future__ import annotations

from pathlib import Path

import pandas as pd


TARGET_EMOTION_LABELS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
]

LABEL_TO_ID = {label: index for index, label in enumerate(TARGET_EMOTION_LABELS)}

GO_EMOTIONS_PARQUET_URLS = {
    "train": "https://huggingface.co/datasets/SetFit/go_emotions/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
    "validation": "https://huggingface.co/datasets/SetFit/go_emotions/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet",
    "test": "https://huggingface.co/datasets/SetFit/go_emotions/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
}


def filter_single_target_emotions(
    df: pd.DataFrame,
    all_label_columns: list[str],
) -> pd.DataFrame:
    single_label_df = df[df[all_label_columns].sum(axis=1) == 1].copy()
    target_df = single_label_df[single_label_df[TARGET_EMOTION_LABELS].sum(axis=1) == 1].copy()

    target_df["label_name"] = target_df[TARGET_EMOTION_LABELS].idxmax(axis=1)
    target_df["label"] = target_df["label_name"].map(LABEL_TO_ID).astype(int)

    return target_df[["text", "label_name", "label"]].reset_index(drop=True)


def balance_by_label(
    df: pd.DataFrame,
    max_per_label: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    counts = df["label_name"].value_counts()
    if counts.empty:
        return df.copy()

    sample_size = min(counts.min(), max_per_label) if max_per_label else counts.min()
    balanced_parts = []
    for label_name in sorted(counts.index):
        part = df[df["label_name"] == label_name]
        balanced_parts.append(part.sample(n=sample_size, random_state=random_state))

    return (
        pd.concat(balanced_parts, ignore_index=True)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )


def compute_label_sample_sizes(
    label_counts: dict[str, int] | pd.Series,
    target_total: int,
) -> dict[str, int]:
    counts = {
        str(label): int(count)
        for label, count in dict(label_counts).items()
        if int(count) > 0
    }
    if target_total <= 0:
        raise ValueError("target_total must be positive.")
    if target_total > sum(counts.values()):
        raise ValueError("target_total cannot exceed available rows without replacement.")

    allocations = {label: 0 for label in sorted(counts)}
    for _ in range(target_total):
        candidates = [
            label
            for label in allocations
            if allocations[label] < counts[label]
        ]
        label = min(candidates, key=lambda item: (allocations[item], item))
        allocations[label] += 1

    return allocations


def stratified_sample_by_label(
    df: pd.DataFrame,
    target_total: int,
    random_state: int = 42,
    label_column: str = "label_name",
) -> pd.DataFrame:
    sample_sizes = compute_label_sample_sizes(
        df[label_column].value_counts(),
        target_total=target_total,
    )
    sampled_parts = []
    for label_name, sample_size in sample_sizes.items():
        part = df[df[label_column] == label_name]
        sampled_parts.append(part.sample(n=sample_size, random_state=random_state))

    return (
        pd.concat(sampled_parts, ignore_index=True)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )


def prepare_go_emotions_split(
    parquet_url: str,
    balance: bool,
    target_total: int | None = None,
    max_per_label: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    raw_df = pd.read_parquet(parquet_url)
    all_label_columns = [column for column in raw_df.columns if column != "text"]
    filtered = filter_single_target_emotions(raw_df, all_label_columns)
    if target_total is not None:
        return stratified_sample_by_label(
            filtered,
            target_total=target_total,
            random_state=random_state,
        )
    if balance:
        return balance_by_label(filtered, max_per_label=max_per_label, random_state=random_state)
    return filtered


def prepare_all_splits(
    output_dir: Path,
    balance: bool = True,
    max_train_per_label: int | None = None,
    target_train_size: int | None = None,
    target_test_size: int | None = None,
    random_state: int = 42,
) -> dict[str, dict[str, int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, int]] = {}

    for split, url in GO_EMOTIONS_PARQUET_URLS.items():
        max_per_label = max_train_per_label if split == "train" else None
        target_total = None
        if split == "train":
            target_total = target_train_size
        elif split == "test":
            target_total = target_test_size
        prepared = prepare_go_emotions_split(
            parquet_url=url,
            balance=balance,
            target_total=target_total,
            max_per_label=max_per_label,
            random_state=random_state,
        )
        prepared.to_csv(output_dir / f"{split}.csv", index=False)
        summary[split] = prepared["label_name"].value_counts().sort_index().to_dict()

    return summary
