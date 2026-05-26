import unittest

from youtube_emotion.model_runner import DEFAULT_EMOTION_MODEL, EMOTION_MODEL_OPTIONS


class DefaultModelsTest(unittest.TestCase):
    def test_default_emotion_model_uses_domain_adapted_hugging_face_repo(self):
        self.assertEqual(
            DEFAULT_EMOTION_MODEL,
            "chase1zhang/youtube-emotion-distilbert-domain-adapted",
        )

    def test_emotion_model_options_keep_previous_models_available(self):
        self.assertIn(DEFAULT_EMOTION_MODEL, EMOTION_MODEL_OPTIONS.values())
        self.assertIn("chase1zhang/youtube-emotion-distilbert", EMOTION_MODEL_OPTIONS.values())
        self.assertIn("j-hartmann/emotion-english-distilroberta-base", EMOTION_MODEL_OPTIONS.values())


if __name__ == "__main__":
    unittest.main()
