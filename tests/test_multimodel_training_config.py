import unittest

from scripts.train_youtube_domain_all_models import (
    LABEL_TO_ID,
    TRAINING_SPECS,
    build_repo_id,
    resolve_selected_specs,
)


class MultiModelTrainingConfigTest(unittest.TestCase):
    def test_label_mapping_uses_project_seven_emotion_order(self):
        self.assertEqual(
            LABEL_TO_ID,
            {
                "anger": 0,
                "disgust": 1,
                "fear": 2,
                "joy": 3,
                "neutral": 4,
                "sadness": 5,
                "surprise": 6,
            },
        )

    def test_training_specs_include_all_emotion_baselines(self):
        keys = [spec.key for spec in TRAINING_SPECS]

        self.assertEqual(
            keys,
            [
                "distilbert",
                "samlowe_roberta",
                "jhartmann_distilroberta",
                "jhartmann_roberta_large",
            ],
        )
        self.assertIn("SamLowe/roberta-base-go_emotions", [spec.base_model for spec in TRAINING_SPECS])
        self.assertIn(
            "j-hartmann/emotion-english-distilroberta-base",
            [spec.base_model for spec in TRAINING_SPECS],
        )

    def test_resolve_selected_specs_supports_all_and_single_model(self):
        all_specs = resolve_selected_specs("all")
        samlowe_specs = resolve_selected_specs("samlowe_roberta")

        self.assertEqual(all_specs, TRAINING_SPECS)
        self.assertEqual(len(samlowe_specs), 1)
        self.assertEqual(samlowe_specs[0].base_model, "SamLowe/roberta-base-go_emotions")

    def test_build_repo_id_uses_namespace_and_model_slug(self):
        spec = resolve_selected_specs("jhartmann_distilroberta")[0]

        self.assertEqual(
            build_repo_id("chase1zhang", spec),
            "chase1zhang/youtube-emotion-jhartmann-distilroberta-domain-adapted",
        )


if __name__ == "__main__":
    unittest.main()
