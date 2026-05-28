from __future__ import annotations

import argparse
import inspect
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class TrainingSpec:
    key: str
    display_name: str
    base_model: str
    output_slug: str
    notes: str


TRAINING_SPECS = [
    TrainingSpec(
        key="distilbert",
        display_name="GoEmotions fine-tuned DistilBERT",
        base_model="chase1zhang/youtube-emotion-distilbert",
        output_slug="youtube-emotion-distilbert-domain-adapted",
        notes="Project-owned DistilBERT baseline rerun on the shared YouTube-domain split.",
    ),
    TrainingSpec(
        key="samlowe_roberta",
        display_name="Public GoEmotions RoBERTa (SamLowe)",
        base_model="SamLowe/roberta-base-go_emotions",
        output_slug="youtube-emotion-samlowe-roberta-domain-adapted",
        notes="Strong public GoEmotions RoBERTa adapted to the project seven-emotion taxonomy.",
    ),
    TrainingSpec(
        key="jhartmann_distilroberta",
        display_name="Public DistilRoBERTa 7-emotion (j-hartmann)",
        base_model="j-hartmann/emotion-english-distilroberta-base",
        output_slug="youtube-emotion-jhartmann-distilroberta-domain-adapted",
        notes="Public seven-emotion baseline adapted to YouTube-domain comments.",
    ),
    TrainingSpec(
        key="jhartmann_roberta_large",
        display_name="Public RoBERTa-large 7-emotion (j-hartmann)",
        base_model="j-hartmann/emotion-english-roberta-large",
        output_slug="youtube-emotion-roberta-large-domain-adapted",
        notes="Larger public seven-emotion baseline; train on Colab GPU due to size.",
    ),
]


def build_repo_id(namespace: str, spec: TrainingSpec) -> str:
    return f"{namespace.strip('/')}/{spec.output_slug}"


def resolve_selected_specs(selection: str) -> list[TrainingSpec]:
    if selection == "all":
        return TRAINING_SPECS

    lookup = {spec.key: spec for spec in TRAINING_SPECS}
    if selection not in lookup:
        valid = ", ".join(["all", *lookup])
        raise ValueError(f"Unknown model selection {selection!r}. Valid choices: {valid}")
    return [lookup[selection]]


def parse_args() -> argparse.Namespace:
    choices = ["all", *[spec.key for spec in TRAINING_SPECS]]
    parser = argparse.ArgumentParser(
        description="Fine-tune all seven-emotion benchmark models on YouTube-domain comments.",
    )
    parser.add_argument("--model", choices=choices, default="all")
    parser.add_argument(
        "--train-csv",
        default=str(PROJECT_ROOT / "data" / "youtube_domain_7class_assistant" / "train.csv"),
    )
    parser.add_argument(
        "--validation-csv",
        default=str(
            PROJECT_ROOT / "data" / "youtube_domain_7class_assistant" / "validation.csv"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "fine_tuned_model_files" / "youtube_domain_all_models"),
    )
    parser.add_argument("--hub-namespace", default="chase1zhang")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=5240)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_labeled_frame(csv_path: str):
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "text" not in df.columns:
        raise ValueError(f"{csv_path} must contain a text column.")

    if "label_id" in df.columns:
        if "label" in df.columns:
            sample_val = df["label"].iloc[0]
            if isinstance(sample_val, str) or not isinstance(sample_val, (int, float)):
                df = df.rename(columns={"label": "label_name"})
        df = df.rename(columns={"label_id": "label"})

    if "label" not in df.columns:
        if "label_name" not in df.columns:
            raise ValueError(f"{csv_path} must contain label or label_name.")
        df["label"] = df["label_name"].map(LABEL_TO_ID)

    if "label_name" not in df.columns:
        df["label_name"] = df["label"].map(ID_TO_LABEL)

    df = df[["text", "label", "label_name"]].copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    invalid = sorted(set(df["label"]) - set(ID_TO_LABEL))
    if invalid:
        raise ValueError(f"{csv_path} contains labels outside the seven-emotion taxonomy: {invalid}")
    return df


def compute_class_weights(labels: list[int]):
    import torch

    counts = {label_id: labels.count(label_id) for label_id in range(len(LABELS))}
    total = max(len(labels), 1)
    weights = [
        total / (len(LABELS) * counts[label_id]) if counts[label_id] else 0.0
        for label_id in range(len(LABELS))
    ]
    return torch.tensor(weights, dtype=torch.float)


def make_training_arguments(output_dir: Path, args: argparse.Namespace):
    from transformers import TrainingArguments

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "logging_steps": 25,
        "save_strategy": "epoch",
        "report_to": "none",
        "fp16": args.fp16,
        "seed": args.seed,
    }
    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in signature:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return TrainingArguments(**kwargs)


def make_trainer(
    model,
    training_args,
    train_dataset,
    eval_dataset,
    tokenizer,
    class_weights,
    early_stopping_patience: int,
):
    import torch
    from transformers import EarlyStoppingCallback, Trainer

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            weights = class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "tokenizer": tokenizer,
        "compute_metrics": compute_metrics,
    }
    if early_stopping_patience > 0:
        trainer_kwargs["callbacks"] = [
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
        ]
    signature = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in signature:
        trainer_kwargs["processing_class"] = tokenizer
        trainer_kwargs.pop("tokenizer")
    return WeightedTrainer(**trainer_kwargs)


def compute_metrics(eval_pred):
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    weighted = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )
    macro = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    per_class = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(len(LABELS))),
        average=None,
        zero_division=0,
    )
    metrics = {
        "accuracy": accuracy,
        "weighted_precision": weighted[0],
        "weighted_recall": weighted[1],
        "weighted_f1": weighted[2],
        "macro_precision": macro[0],
        "macro_recall": macro[1],
        "macro_f1": macro[2],
    }
    for index, label_name in enumerate(LABELS):
        metrics[f"{label_name}_precision"] = per_class[0][index]
        metrics[f"{label_name}_recall"] = per_class[1][index]
        metrics[f"{label_name}_f1"] = per_class[2][index]
    return metrics


def prepare_datasets(train_csv: str, validation_csv: str, tokenizer, max_length: int):
    from datasets import Dataset, DatasetDict

    train_df = load_labeled_frame(train_csv)
    validation_df = load_labeled_frame(validation_csv)
    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, preserve_index=False),
            "validation": Dataset.from_pandas(validation_df, preserve_index=False),
        }
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized, train_df, validation_df


def train_one_model(spec: TrainingSpec, args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    output_dir = Path(args.output_root) / spec.output_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(spec.base_model, use_fast=True)
    tokenized, train_df, validation_df = prepare_datasets(
        args.train_csv,
        args.validation_csv,
        tokenizer,
        args.max_length,
    )
    class_weights = compute_class_weights(train_df["label"].astype(int).tolist())
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.base_model,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )
    training_args = make_training_arguments(output_dir, args)
    trainer = make_trainer(
        model=model,
        training_args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        class_weights=class_weights,
        early_stopping_patience=args.early_stopping_patience,
    )

    trainer.train()
    metrics = trainer.evaluate(tokenized["validation"])

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    repo_id = build_repo_id(args.hub_namespace, spec)
    if args.push_to_hub:
        trainer.model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)

    validation_pipe = pipeline(
        "text-classification",
        model=str(output_dir),
        tokenizer=str(output_dir),
        device=-1,
    )
    sample_outputs = validation_pipe(
        validation_df["text"].head(10).astype(str).tolist(),
        truncation=True,
        max_length=args.max_length,
    )

    result = {
        "spec": asdict(spec),
        "repo_id": repo_id,
        "output_dir": str(output_dir),
        "train_samples": int(len(train_df)),
        "validation_samples": int(len(validation_df)),
        "class_weights": [round(float(value), 6) for value in class_weights.tolist()],
        "metrics": {
            key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))
        },
        "sample_outputs": sample_outputs,
        "elapsed_seconds": round(time.time() - start, 4),
        "pushed_to_hub": bool(args.push_to_hub),
    }
    metrics_path = output_dir / "youtube_domain_training_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def dry_run_summary(specs: list[TrainingSpec], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "selected_models": [asdict(spec) for spec in specs],
        "train_csv": args.train_csv,
        "validation_csv": args.validation_csv,
        "output_root": args.output_root,
        "hub_namespace": args.hub_namespace,
        "push_to_hub": args.push_to_hub,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "early_stopping_patience": args.early_stopping_patience,
        "fp16": args.fp16,
    }


def main() -> None:
    args = parse_args()
    specs = resolve_selected_specs(args.model)
    set_seed(args.seed)

    if args.dry_run:
        print(json.dumps(dry_run_summary(specs, args), indent=2))
        return

    all_results = []
    for spec in specs:
        print(f"Training {spec.display_name}: {spec.base_model}", flush=True)
        all_results.append(train_one_model(spec, args))

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "youtube_domain_all_model_training_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Saved training summary: {summary_path}")


if __name__ == "__main__":
    main()
