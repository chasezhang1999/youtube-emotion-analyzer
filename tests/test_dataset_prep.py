import unittest

import pandas as pd

from youtube_emotion.dataset_prep import (
    TARGET_EMOTION_LABELS,
    balance_by_label,
    filter_single_target_emotions,
)


class DatasetPrepTest(unittest.TestCase):
    def test_filter_single_target_emotions_keeps_only_clean_target_labels(self):
        all_label_columns = TARGET_EMOTION_LABELS + ["admiration", "annoyance"]
        df = pd.DataFrame(
            [
                {"text": "angry text", "anger": 1, "disgust": 0, "fear": 0, "joy": 0, "neutral": 0, "sadness": 0, "surprise": 0, "admiration": 0, "annoyance": 0},
                {"text": "multi target", "anger": 1, "disgust": 1, "fear": 0, "joy": 0, "neutral": 0, "sadness": 0, "surprise": 0, "admiration": 0, "annoyance": 0},
                {"text": "outside label", "anger": 0, "disgust": 0, "fear": 0, "joy": 0, "neutral": 0, "sadness": 0, "surprise": 0, "admiration": 1, "annoyance": 0},
                {"text": "mixed outside", "anger": 0, "disgust": 0, "fear": 0, "joy": 1, "neutral": 0, "sadness": 0, "surprise": 0, "admiration": 1, "annoyance": 0},
                {"text": "neutral text", "anger": 0, "disgust": 0, "fear": 0, "joy": 0, "neutral": 1, "sadness": 0, "surprise": 0, "admiration": 0, "annoyance": 0},
            ]
        )

        filtered = filter_single_target_emotions(df, all_label_columns)

        self.assertEqual(filtered["text"].tolist(), ["angry text", "neutral text"])
        self.assertEqual(filtered["label_name"].tolist(), ["anger", "neutral"])
        self.assertEqual(filtered["label"].tolist(), [0, 4])

    def test_balance_by_label_downsamples_each_label_to_limit(self):
        df = pd.DataFrame(
            {
                "text": ["a1", "a2", "a3", "j1", "j2"],
                "label_name": ["anger", "anger", "anger", "joy", "joy"],
                "label": [0, 0, 0, 3, 3],
            }
        )

        balanced = balance_by_label(df, max_per_label=2, random_state=7)

        self.assertEqual(len(balanced), 4)
        self.assertEqual(balanced["label_name"].value_counts().to_dict(), {"anger": 2, "joy": 2})


if __name__ == "__main__":
    unittest.main()
