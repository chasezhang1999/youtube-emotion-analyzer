import unittest

import pandas as pd

from scripts.build_youtube_domain_balanced_dataset import (
    EMOTION_LABELS,
    choose_balanced_label_targets,
    split_balanced_dataset,
)


class YouTubeDomainBalancingTest(unittest.TestCase):
    def test_choose_balanced_label_targets_caps_majority_classes(self):
        counts = {
            "anger": 900,
            "disgust": 600,
            "fear": 500,
            "joy": 2000,
            "neutral": 2500,
            "sadness": 700,
            "surprise": 800,
        }

        targets = choose_balanced_label_targets(counts, target_total=5000)

        self.assertEqual(sum(targets.values()), 5000)
        self.assertEqual(targets["anger"], 800)
        self.assertEqual(targets["disgust"], 600)
        self.assertEqual(targets["fear"], 500)
        self.assertEqual(targets["joy"], 800)
        self.assertEqual(targets["neutral"], 800)
        self.assertEqual(targets["sadness"], 700)
        self.assertEqual(targets["surprise"], 800)

    def test_choose_balanced_label_targets_uses_available_rows_without_oversampling(self):
        counts = {label: 3 for label in EMOTION_LABELS}
        counts["fear"] = 1

        targets = choose_balanced_label_targets(counts, target_total=10)

        self.assertEqual(sum(targets.values()), 10)
        self.assertEqual(targets["fear"], 1)
        self.assertTrue(all(targets[label] <= counts[label] for label in EMOTION_LABELS))

    def test_split_balanced_dataset_keeps_each_label_in_train_and_validation(self):
        rows = []
        for label in EMOTION_LABELS:
            rows.extend({"text": f"{label}-{index}", "label": label} for index in range(10))
        df = pd.DataFrame(rows)

        train_df, validation_df = split_balanced_dataset(df, validation_ratio=0.2, random_state=7)

        self.assertEqual(len(train_df), 56)
        self.assertEqual(len(validation_df), 14)
        self.assertEqual(train_df["label"].value_counts().sort_index().to_dict(), {label: 8 for label in EMOTION_LABELS})
        self.assertEqual(validation_df["label"].value_counts().sort_index().to_dict(), {label: 2 for label in EMOTION_LABELS})

    def test_split_balanced_dataset_preserves_rare_labels_in_train(self):
        counts = {
            "anger": 20,
            "disgust": 10,
            "fear": 5,
            "joy": 40,
            "neutral": 40,
            "sadness": 8,
            "surprise": 7,
        }
        rows = []
        for label, count in counts.items():
            rows.extend({"text": f"{label}-{index}", "label": label} for index in range(count))
        df = pd.DataFrame(rows)

        train_df, validation_df = split_balanced_dataset(df, validation_ratio=0.2, random_state=7)

        self.assertEqual(len(validation_df), 26)
        self.assertEqual(validation_df["label"].value_counts().sort_index().to_dict(), {
            "anger": 4,
            "disgust": 2,
            "fear": 1,
            "joy": 8,
            "neutral": 8,
            "sadness": 2,
            "surprise": 1,
        })
        self.assertEqual(train_df["label"].value_counts()["fear"], 4)


if __name__ == "__main__":
    unittest.main()
