from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "experiments" / "app_per_comment_manual_labels.csv"
OUTPUT_PATH = PROJECT_ROOT / "experiments" / "app_per_comment_manual_labels_reviewed.csv"


# Expert review pass for the independent 150-comment app benchmark.
# Key: (video_short, comment_index)
# Value: (reviewed_7_emotion, reviewed_3_sentiment, reason)
LABEL_REVISIONS = {
    ("rare_earths", 6): ("anger", "negative", "rhetorical criticism about policy double standard"),
    ("rare_earths", 7): ("anger", "negative", "sarcastic criticism of weapon use"),
    ("rare_earths", 8): ("joy", "positive", "dominant emotion is pride and positive national progress"),
    ("rare_earths", 19): ("anger", "negative", "sarcastic criticism of economic vulnerability"),
    ("rare_earths", 25): ("neutral", "neutral", "mostly factual strategic analysis, not clear praise"),
    ("rare_earths", 26): ("joy", "positive", "dominant cue is amusement/laughter"),
    ("rare_earths", 27): ("joy", "positive", "laughing at wordplay"),
    ("rare_earths", 33): ("anger", "negative", "critical blame about technology transfer"),
    ("rare_earths", 34): ("fear", "negative", "business shutdown risk cue"),
    ("rare_earths", 35): ("anger", "negative", "sarcastic criticism of media editing"),
    ("rare_earths", 36): ("surprise", "neutral", "incredulous question with surprise cue"),
    ("rare_earths", 38): ("anger", "negative", "sarcastic criticism of war spending"),
    ("rare_earths", 41): ("joy", "positive", "hopeful cooperative framing"),
    ("rare_earths", 47): ("neutral", "neutral", "balanced factual policy comment with concern but no dominant emotion"),
    ("rare_earths", 48): ("anger", "negative", "complaint about excessive ads"),
    ("avatar_clip", 1): ("joy", "positive", "positive praise for the type of content"),
    ("avatar_clip", 25): ("anger", "negative", "criticism of world leaders lying"),
    ("avatar_clip", 26): ("joy", "positive", "anticipatory excitement"),
    ("shooting_news", 9): ("joy", "positive", "joking comparison to Agent 007"),
    ("shooting_news", 10): ("joy", "positive", "humorous laughing comment"),
    ("shooting_news", 28): ("joy", "positive", "emoji-only laughter"),
    ("shooting_news", 38): ("joy", "positive", "humorous explanation"),
    ("shooting_news", 49): ("sadness", "negative", "resigned negative comment about the country"),
}


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    if "previous_manual_7_emotion" not in df.columns:
        df["previous_manual_7_emotion"] = df["manual_7_emotion"]
    if "previous_manual_3_sentiment" not in df.columns:
        df["previous_manual_3_sentiment"] = df["manual_3_sentiment"]

    reviewed_emotions = []
    reviewed_sentiments = []
    review_notes = []
    changed_flags = []

    for row in df.itertuples(index=False):
        key = (row.video_short, int(row.comment_index))
        original_emotion = row.previous_manual_7_emotion
        original_sentiment = row.previous_manual_3_sentiment

        if key in LABEL_REVISIONS:
            reviewed_emotion, reviewed_sentiment, note = LABEL_REVISIONS[key]
            changed = reviewed_emotion != original_emotion or reviewed_sentiment != original_sentiment
        else:
            reviewed_emotion = original_emotion
            reviewed_sentiment = original_sentiment
            note = "confirmed after assistant manual review"
            changed = False

        reviewed_emotions.append(reviewed_emotion)
        reviewed_sentiments.append(reviewed_sentiment)
        review_notes.append(note)
        changed_flags.append("Yes" if changed else "No")

    df["manual_7_emotion"] = reviewed_emotions
    df["manual_3_sentiment"] = reviewed_sentiments
    df["manual_label_changed"] = changed_flags
    df["manual_review_note"] = review_notes
    df["manual_review_source"] = "assistant_manual_review_2026-05-27"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    df.to_csv(INPUT_PATH, index=False)

    print(f"Saved reviewed labels: {OUTPUT_PATH}")
    print(f"Updated benchmark labels: {INPUT_PATH}")
    print("\nChanged labels:", (df["manual_label_changed"] == "Yes").sum())
    print("\nSeven-emotion distribution")
    print(df["manual_7_emotion"].value_counts().sort_index().to_string())
    print("\nThree-sentiment distribution")
    print(df["manual_3_sentiment"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
