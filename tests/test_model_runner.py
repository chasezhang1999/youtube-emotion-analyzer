import unittest

from youtube_emotion.model_runner import extract_top_prediction, predict_comment_emotions


class FakePipeline:
    def __init__(self, outputs):
        self.outputs = outputs

    def __call__(self, comments, truncation=True):
        return self.outputs[: len(comments)]


class ModelRunnerTest(unittest.TestCase):
    def test_extract_top_prediction_accepts_dict_or_ranked_list(self):
        self.assertEqual(
            extract_top_prediction({"label": "joy", "score": 0.91}),
            ("joy", 0.91),
        )
        self.assertEqual(
            extract_top_prediction(
                [
                    {"label": "sadness", "score": 0.23},
                    {"label": "joy", "score": 0.84},
                ]
            ),
            ("joy", 0.84),
        )

    def test_predict_comment_emotions_merges_emotion_and_sentiment_outputs(self):
        comments = ["Love this launch", "This is disappointing"]
        emotion_pipe = FakePipeline(
            [
                {"label": "joy", "score": 0.88},
                {"label": "sadness", "score": 0.76},
            ]
        )
        sentiment_pipe = FakePipeline(
            [
                {"label": "LABEL_2", "score": 0.92},
                {"label": "LABEL_0", "score": 0.81},
            ]
        )

        rows = predict_comment_emotions(comments, emotion_pipe, sentiment_pipe)

        self.assertEqual(rows[0]["comment"], "Love this launch")
        self.assertEqual(rows[0]["emotion"], "joy")
        self.assertEqual(rows[0]["sentiment"], "positive")
        self.assertEqual(rows[1]["emotion"], "sadness")
        self.assertEqual(rows[1]["sentiment"], "negative")


if __name__ == "__main__":
    unittest.main()
