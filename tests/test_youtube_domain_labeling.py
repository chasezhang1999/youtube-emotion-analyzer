import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.label_youtube_domain_comments import build_labeled_dataset


class YouTubeDomainLabelingTest(unittest.TestCase):
    def test_build_labeled_dataset_combines_multiple_input_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            first_path = Path(tmp_dir) / "first.csv"
            second_path = Path(tmp_dir) / "second.csv"

            base_columns = [
                "video_short",
                "video",
                "video_id",
                "comment_index",
                "comment",
                "manual_7_emotion",
                "manual_3_sentiment",
                "label_notes",
            ]
            pd.DataFrame(
                [
                    {
                        "video_short": "brand_crisis_negative",
                        "video": "https://www.youtube.com/watch?v=aaaaaaaaaaa",
                        "video_id": "aaaaaaaaaaa",
                        "comment_index": 1,
                        "comment": "That is disgusting and the health department should inspect it.",
                    },
                    {
                        "video_short": "tech_product_launch",
                        "video": "https://www.youtube.com/watch?v=bbbbbbbbbbb",
                        "video_id": "bbbbbbbbbbb",
                        "comment_index": 1,
                        "comment": "This launch looks amazing!",
                    },
                ],
                columns=base_columns,
            ).to_csv(first_path, index=False)
            pd.DataFrame(
                [
                    {
                        "video_short": "fear_public_safety",
                        "video": "https://www.youtube.com/watch?v=ccccccccccc",
                        "video_id": "ccccccccccc",
                        "comment_index": 1,
                        "comment": "This road safety ad is scary.",
                    },
                ],
                columns=base_columns,
            ).to_csv(second_path, index=False)

            labeled = build_labeled_dataset([first_path, second_path])

        self.assertEqual(len(labeled), 3)
        self.assertEqual(set(labeled["source_file"]), {"first.csv", "second.csv"})
        self.assertEqual(labeled["manual_7_emotion"].isna().sum(), 0)
        self.assertEqual(labeled["manual_3_sentiment"].isna().sum(), 0)
        self.assertEqual(labeled.iloc[0]["manual_7_emotion"], "disgust")
        self.assertEqual(labeled.iloc[1]["manual_7_emotion"], "joy")
        self.assertEqual(labeled.iloc[2]["manual_7_emotion"], "fear")


if __name__ == "__main__":
    unittest.main()
