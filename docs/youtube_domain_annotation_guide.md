# YouTube Domain Comment Annotation Guide

This file documents the assistant-assisted annotation used to create the YouTube-domain fine-tuning data.

## Data source

Thirty public YouTube videos were selected to cover common digital marketing scenarios:
product launch, product review, controversial product discussion, inspirational brand advertising,
holiday advertising, fast food advertising, beauty/social campaign, movie trailer, entertainment fandom,
brand crisis news, customer-service failures, public-safety PSAs, health-warning campaigns,
product-failure incidents, food-safety issues, brand backlash, brand purpose advertising,
gaming/product reveals, streaming trailers, and app-brand campaigns.

For each video, up to 100 top-level comments were collected with the YouTube Data API using relevance order.
Most videos returned 100 usable comments; one older video returned fewer public top-level comments.
The resulting dataset contains 2,962 comments.

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

> To reduce domain shift, we added an assistant-assisted annotation set of 2,962 YouTube comments collected from
> 30 marketing-related videos. The data was used only for further domain adaptation, while the original
> 150 manually reviewed comments remained an independent application-level test set.

## Current label distribution

The final 2,962-comment dataset has the following seven-emotion distribution:

- neutral: 1,532
- joy: 943
- anger: 136
- disgust: 102
- surprise: 88
- sadness: 82
- fear: 79

The three-sentiment distribution is:

- neutral: 1,572
- positive: 947
- negative: 443
