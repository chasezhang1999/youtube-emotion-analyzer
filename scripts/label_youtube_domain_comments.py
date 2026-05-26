from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
NEGATIVE_EMOTIONS = {"anger", "disgust", "fear", "sadness"}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATHS = [
    PROJECT_ROOT / "data" / "youtube_domain_training_comments_unlabeled.csv",
    PROJECT_ROOT / "data" / "youtube_domain_training_comments_extra_unlabeled.csv",
]
OUTPUT_PATH = PROJECT_ROOT / "data" / "youtube_domain_training_comments_assistant_labeled.csv"
SPLIT_DIR = PROJECT_ROOT / "data" / "youtube_domain_7class_assistant"


def has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def score_comment(text: str, video_short: str) -> tuple[str, str, str]:
    raw = normalize_text(text)
    low = raw.lower()

    if not raw:
        return "neutral", "neutral", "empty comment"

    # Channel pinned comments, correction notes, and long sponsor/link blocks are metadata-like.
    if (
        ("http://" in low or "https://" in low)
        and has_any(low, ("sponsored", "correction", "subscribe", "use the code", "raffle"))
    ) or len(raw) > 1200:
        return "neutral", "neutral", "metadata, sponsor, or link-heavy comment"

    positive_terms = (
        "love",
        "loved",
        "best",
        "great",
        "good",
        "well done",
        "amazing",
        "awesome",
        "beautiful",
        "brilliant",
        "brilliance",
        "perfect",
        "inspirational",
        "inspired",
        "uplift",
        "thank",
        "thanks",
        "fun",
        "cool",
        "cute",
        "adorable",
        "genius",
        "incredible",
        "precise",
        "better",
        "straight to the point",
        "hilarious",
        "funny",
        "laugh",
        "gem",
        "iconic",
        "epic",
        "fire",
        "🔥",
        "😂",
        "🤣",
        "😊",
        "❤",
        "❤️",
        "chills",
        "goosebumps",
        "nostalgia",
        "classic",
        "masterpiece",
        "goes so hard",
        "goes hard",
        "highest compliment",
        "real for that",
    )
    negative_terms = (
        "worst",
        "terrible",
        "awful",
        "horrible",
        "lame",
        "tone deaf",
        "trash",
        "garbage",
        "sucks",
        "scam",
        "ripoff",
        "refund",
        "outage",
        "lying",
        "stupid",
        "idiot",
        "dumb",
        "negligent",
        "complain",
        "liability",
        "never buying",
        "never touching",
        "can't stand",
        "shit",
        "$hit",
    )
    disgust_terms = (
        "disgusting",
        "gross",
        "nasty",
        "🤮",
        "puke",
        "dirty",
        "filthy",
        "health department",
        "food heat",
        "blackface",
        "sweater",
        "where everyone can see",
        "no one is looking",
        "under food",
        "lamp",
        "unsanitary",
        "sewage",
        "roaches",
        "food poisoning",
    )
    fear_terms = (
        "terrified",
        "terrifies",
        "scared",
        "scary",
        "afraid",
        "fear",
        "dangerous",
        "unsafe",
        "anxiety",
        "worry",
        "worried",
        "panic",
        "nightmare",
        "threat",
    )
    sadness_terms = (
        "sad",
        "depressed",
        "cry",
        "cried",
        "tears",
        "tear",
        "gone",
        "last time",
        "old days",
        "7 years old",
    )
    surprise_terms = (
        "wow",
        "mind blowing",
        "mind-blowing",
        "shocked",
        "shock",
        "surprised",
        "unexpected",
        "insane",
        "crazy",
        "wild",
        "unbelievable",
        "can't believe",
        "didn't expect",
        "what!",
    )

    pos = has_any(low, positive_terms)
    neg = has_any(low, negative_terms) or re.search(r"\b(hate|hated|so bad|pos)\b", low) is not None
    disgust = has_any(low, disgust_terms)
    fear = has_any(low, fear_terms)
    sadness = (
        has_any(low, sadness_terms)
        or re.search(r"\b(rip|died|dead)\b", low) is not None
        or re.search(r"\b(i|we|really|still|so|how)\s+miss\b|\bmiss you\b", low) is not None
    )
    surprise = has_any(low, surprise_terms)

    if video_short == "movie_trailer_joy" and has_any(low, ("anxiety", "disgust", "envy", "embarrassment")):
        if pos or not neg:
            return "joy", "positive", "movie character reference with positive trailer context"

    if video_short == "automotive_controversy" and has_any(
        low,
        (
            "whirlpool fridge",
            "garbage can",
            "trash cans",
            "design looks",
            "so bad",
            "financial decisions",
        ),
    ):
        return "anger", "negative", "controversial product criticism"

    if video_short == "anger_brand_crisis":
        if has_any(low, ("disgusting", "vile", "inhuman")):
            return "disgust", "negative", "brand-crisis disgust cue"
        if has_any(
            low,
            (
                "beat",
                "beating",
                "assault",
                "drag",
                "dragged",
                "sue",
                "sues",
                "sued",
                "lawsuit",
                "apologize",
                "overbooked",
                "held accountable",
                "never flying",
                "outraged",
                "united airlines",
                "passenger",
                "patient",
            ),
        ):
            return "anger", "negative", "airline brand-crisis criticism"

    if video_short == "anger_customer_service":
        if has_any(
            low,
            (
                "lying",
                "outage",
                "millions",
                "bill credit",
                "bs",
                "fuck",
                "refund",
                "switching carriers",
                "what the heck",
                "without service",
                "demand",
                "emergency",
                "no one else can call",
            ),
        ):
            return "anger", "negative", "customer-service outage criticism"

    if video_short in {"fear_public_safety", "sadness_psa"}:
        if has_any(
            low,
            (
                "heartbreaking",
                "regret",
                "mother died",
                "died",
                "death",
                "killed",
                "cry",
                "cried",
                "tears",
                "please... i've got my boy",
                "got my boy in the back",
            ),
        ):
            return "sadness", "negative", "road-safety sadness cue"
        if has_any(
            low,
            (
                "accident",
                "crash",
                "injury",
                "injuries",
                "spine",
                "terrifying",
                "disturbing",
                "dangerous",
                "unsafe",
                "chills up your spine",
                "not able to stop",
                "prevent car",
                "drive safely",
            ),
        ):
            return "fear", "negative", "road-safety fear cue"

    if video_short == "fear_health_warning":
        if has_any(low, ("rip", "rest in peace", "cancer", "lost her", "miss her", "long battle")):
            return "sadness", "negative", "health-warning sadness cue"
        if has_any(
            low,
            (
                "smoke",
                "smoking",
                "wake smokers",
                "quit smoking",
                "quitting smoking",
                "continue to smoke",
                "regret",
                "anti-smoking",
                "scary",
            ),
        ):
            return "fear", "negative", "health-warning fear cue"

    if video_short == "disgust_food_safety":
        if has_any(
            low,
            (
                "violation",
                "violations",
                "kitchen",
                "buffet",
                "food",
                "health inspector",
                "shut",
                "sewage",
                "unsanitary",
                "roaches",
                "not chicken",
                "eaten out",
                "food safety",
            ),
        ):
            return "disgust", "negative", "food-safety disgust cue"

    if video_short == "disgust_brand_backlash":
        if has_any(
            low,
            (
                "tone deaf",
                "backlash",
                "offended",
                "offensive",
                "lame",
                "suffer",
                "negative repercussions",
                "abuse of power",
                "sensitive matter",
                "preposterous",
                "should know better",
            ),
        ):
            return "anger", "negative", "brand-backlash criticism"

    if video_short == "surprise_product_fail":
        if has_any(low, ("terrified", "scary", "creepy", "horror")):
            return "fear", "negative", "product-failure fear cue"
        if has_any(
            low,
            (
                "explode",
                "explosion",
                "overheat",
                "overheating",
                "battery",
                "banned from planes",
                "boom",
            ),
        ):
            return "surprise", "negative", "product-failure surprise cue"

    if video_short == "brand_crisis_negative":
        if disgust or has_any(low, ("health", "food", "under food", "lamp", "no one is looking")):
            return "disgust", "negative", "brand-crisis hygiene or safety concern"
        if neg:
            return "anger", "negative", "brand-crisis criticism"

    if disgust:
        return "disgust", "negative", "disgust cue"

    if fear and not (video_short == "movie_trailer_joy" and pos):
        return "fear", "negative", "fear or safety cue"

    if sadness and not pos:
        return "sadness", "negative", "sadness cue"

    if neg and not pos:
        return "anger", "negative", "negative criticism cue"

    if pos and sadness:
        return "joy", "positive", "positive nostalgia or emotional appreciation"

    if pos:
        return "joy", "positive", "positive appreciation or humor cue"

    if surprise:
        sentiment = "positive" if video_short in {
            "tech_product_launch",
            "movie_trailer_joy",
            "entertainment_fandom",
            "brand_campaign_positive",
        } else "neutral"
        return "surprise", sentiment, "surprise cue"

    if neg:
        return "anger", "negative", "negative criticism cue"

    if sadness:
        return "sadness", "negative", "sadness cue"

    # Question-only or factual brand/product comparison comments usually do not express a clear emotion.
    return "neutral", "neutral", "no dominant affective cue"


def build_labeled_dataset(input_paths: list[Path]) -> pd.DataFrame:
    frames = []
    for input_path in input_paths:
        frame = pd.read_csv(input_path).fillna("")
        frame["source_file"] = input_path.name
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    labels = df.apply(lambda row: score_comment(row["comment"], row["video_short"]), axis=1)
    df["manual_7_emotion"] = [label[0] for label in labels]
    df["manual_3_sentiment"] = [label[1] for label in labels]
    df["label_notes"] = [label[2] for label in labels]
    df["annotation_source"] = "assistant_assisted_review"
    return df


def make_splits(df: pd.DataFrame, split_dir: Path = SPLIT_DIR) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    split_df = df.rename(
        columns={"comment": "text", "manual_7_emotion": "label", "manual_3_sentiment": "sentiment"}
    ).copy()
    split_df["label_id"] = split_df["label"].map({label: i for i, label in enumerate(EMOTION_LABELS)})
    split_df = split_df[
        [
            "text",
            "label",
            "label_id",
            "sentiment",
            "video_short",
            "video",
            "comment_index",
            "label_notes",
        ]
    ]

    train_parts = []
    validation_parts = []
    for _, group in split_df.groupby("label", sort=False):
        group = group.sample(frac=1.0, random_state=5240)
        validation_count = max(1, round(len(group) * 0.2))
        validation_parts.append(group.iloc[:validation_count])
        train_parts.append(group.iloc[validation_count:])

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=5240).reset_index(drop=True)
    validation_df = pd.concat(validation_parts).sample(frac=1.0, random_state=5240).reset_index(drop=True)

    split_df.to_csv(split_dir / "all.csv", index=False)
    train_df.to_csv(split_dir / "train.csv", index=False)
    validation_df.to_csv(split_dir / "validation.csv", index=False)


def main() -> None:
    existing_inputs = [input_path for input_path in INPUT_PATHS if input_path.exists()]
    df = build_labeled_dataset(existing_inputs)
    df.to_csv(OUTPUT_PATH, index=False)
    make_splits(df)

    print(f"Saved labeled comments: {OUTPUT_PATH}")
    print(f"Saved splits: {SPLIT_DIR}")
    print("\n7-emotion distribution")
    print(df["manual_7_emotion"].value_counts().reindex(EMOTION_LABELS, fill_value=0).to_string())
    print("\n3-sentiment distribution")
    print(df["manual_3_sentiment"].value_counts().reindex(["negative", "neutral", "positive"], fill_value=0).to_string())


if __name__ == "__main__":
    main()
