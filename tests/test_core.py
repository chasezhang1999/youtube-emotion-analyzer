import unittest

from youtube_emotion.core import (
    build_marketing_recommendation,
    normalize_sentiment_label,
    parse_video_id,
    summarize_predictions,
)


class CoreLogicTest(unittest.TestCase):
    def test_parse_video_id_accepts_common_youtube_url_formats(self):
        cases = {
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ": "dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ?si=abc123": "dQw4w9WgXcQ",
            "https://www.youtube.com/shorts/dQw4w9WgXcQ": "dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ": "dQw4w9WgXcQ",
            "dQw4w9WgXcQ": "dQw4w9WgXcQ",
        }

        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(parse_video_id(value), expected)

    def test_parse_video_id_rejects_invalid_values(self):
        invalid_values = ["", "https://example.com/video", "not a youtube link"]

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_video_id(value)

    def test_summarize_predictions_returns_main_emotion_and_negative_ratio(self):
        rows = [
            {"comment": "I love this campaign", "emotion": "joy"},
            {"comment": "Amazing launch", "emotion": "joy"},
            {"comment": "This feels disappointing", "emotion": "sadness"},
            {"comment": "This made me angry", "emotion": "anger"},
            {"comment": "The video is okay", "emotion": "neutral"},
        ]

        summary = summarize_predictions(rows)

        self.assertEqual(summary["total_comments"], 5)
        self.assertEqual(summary["main_emotion"], "joy")
        self.assertEqual(summary["emotion_counts"]["joy"], 2)
        self.assertEqual(summary["emotion_percentages"]["joy"], 40.0)
        self.assertEqual(summary["negative_emotion_count"], 2)
        self.assertEqual(summary["negative_emotion_ratio"], 40.0)

    def test_build_marketing_recommendation_flags_high_negative_emotion(self):
        summary = {
            "main_emotion": "anger",
            "negative_emotion_ratio": 55.0,
            "emotion_percentages": {"anger": 55.0},
        }

        recommendation = build_marketing_recommendation(summary)

        self.assertIn("negative", recommendation.lower())
        self.assertIn("review", recommendation.lower())

    def test_normalize_sentiment_label_accepts_cardiff_and_plain_labels(self):
        self.assertEqual(normalize_sentiment_label("LABEL_0"), "negative")
        self.assertEqual(normalize_sentiment_label("LABEL_1"), "neutral")
        self.assertEqual(normalize_sentiment_label("LABEL_2"), "positive")
        self.assertEqual(normalize_sentiment_label("Positive"), "positive")


if __name__ == "__main__":
    unittest.main()
