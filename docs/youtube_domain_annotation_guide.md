# YouTube Domain Comment Annotation Guide

This file documents the assistant-assisted annotation used to create the YouTube-domain fine-tuning data.

## Data source

Seventy-one public YouTube videos were selected to cover common digital marketing scenarios:
product launch, product review, controversial product discussion, inspirational brand advertising,
holiday advertising, fast food advertising, beauty/social campaign, movie trailer, entertainment fandom,
brand crisis news, customer-service failures, public-safety PSAs, health-warning campaigns,
product-failure incidents, food-safety issues, brand backlash, brand purpose advertising,
gaming/product reveals, streaming trailers, and app-brand campaigns.

The updated collection pass used YouTube Data API pagination to build an 8,000-comment raw pool.
The final adaptation dataset selects 5,000 labeled comments from that pool, balancing labels as far as
possible without duplicating comments from minority classes.

## Label set

The seven emotion labels match the project app:

- anger
- disgust
- fear
- joy
- neutral
- sadness
- surprise

The three sentiment labels are:

- negative
- neutral
- positive

## Annotation rule

Each comment is assigned the dominant emotion expressed by the viewer. If a comment is factual,
metadata-like, or does not clearly express emotion, it is labeled neutral.

Short YouTube comments often use sarcasm, emojis, and fandom language. The annotation pass therefore treats
obvious praise, humor, nostalgia, complaint, hygiene concerns, safety concerns, and sadness cues as stronger
signals than generic model outputs. Character names from *Inside Out 2*, such as Anxiety and Disgust, are not
treated as fear or disgust unless the comment itself expresses that emotion.

## Intended use

These labels are intended for YouTube-domain adaptation training. They should not replace the independent
150-comment application performance test set, because that set is used to evaluate whether the final app improves.

Recommended report wording:

> To reduce domain shift, we collected an 8,000-comment assistant-assisted YouTube-domain pool from
> 71 marketing-related videos, then selected a 5,000-comment balanced-as-possible adaptation dataset. The data was used only for further domain adaptation, while the original
> 150 manually reviewed comments remained an independent application-level test set.

## Current label distribution

The 8,000-comment labeled raw pool has the following seven-emotion distribution:

- neutral: 2,331
- joy: 2,332
- anger: 1,050
- sadness: 802
- fear: 608
- disgust: 505
- surprise: 372

The final 5,000-comment adaptation dataset has the following seven-emotion distribution:

- anger: 905
- disgust: 505
- fear: 608
- joy: 904
- neutral: 904
- sadness: 802
- surprise: 372

The final 5,000-comment three-sentiment distribution is:

- negative: 2,997
- neutral: 1,080
- positive: 923

The raw and selected datasets are stored in:

- `data/youtube_domain_training_comments_8000_unlabeled.csv`
- `data/youtube_domain_training_comments_8000_deepseek_labeled.csv`
- `data/youtube_domain_training_comments_5000_balanced_assistant_labeled.csv`
- `data/youtube_domain_7class_deepseek/`
