from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from youtube_emotion.dataset_prep import prepare_all_splits  # noqa: E402


def main() -> None:
    output_dir = PROJECT_ROOT / "data" / "go_emotions_7class"
    summary = prepare_all_splits(
        output_dir=output_dir,
        balance=True,
        max_train_per_label=None,
        random_state=42,
    )

    print(f"Prepared CSV files in: {output_dir}")
    for split, counts in summary.items():
        total = sum(counts.values())
        print(f"{split}: total={total}, counts={counts}")


if __name__ == "__main__":
    main()
