# Presentation Outline

Target length: 8 to 9 minutes.

## Slide 1: Title

Title: YouTube Audience Emotion Analyzer for Digital Marketing Campaigns

Include:

- Team members and student IDs
- Course: ISOM5240 Deep Learning Business Applications with Python
- Streamlit app URL: https://youtube-emotion-analyzer.streamlit.app/

Speaker notes:

This project builds a deployed Streamlit app for a digital marketing agency. The app takes a YouTube video URL, analyzes the first 100 comments, and summarizes audience emotions and sentiment.

## Slide 2: Business Problem

Main message:

Digital marketing teams need a faster way to understand YouTube audience reaction.

Include:

- YouTube comments contain useful customer feedback.
- Manual review is slow and inconsistent.
- Brands need quick signals: positive excitement, neutral engagement, or negative risk.

Speaker notes:

For a marketing agency, comments are not just text. They are audience reaction data. The app helps the agency quickly identify whether a campaign is creating joy, surprise, anger, or other emotions.

## Slide 3: App Objective and Workflow

Main message:

The app turns a video URL into an emotion dashboard and recommendation.

Workflow:

```text
YouTube URL -> YouTube API -> first 100 comments -> emotion model + sentiment model -> charts, table, recommendation
```

Outputs:

- Main emotion
- Seven-emotion distribution
- Sentiment distribution
- Negative emotion ratio
- Comment-level results
- Marketing recommendation

Speaker notes:

The workflow is designed for business users. They only need to paste a YouTube link, and the app returns both a detailed emotion view and a simple positive / neutral / negative signal.

## Slide 4: Dataset

Main message:

The project uses GoEmotions plus YouTube-domain adaptation data.

Include:

- GoEmotions seven-class balanced dataset:
  - train: 3,010
  - validation: 406
  - test: 455
- Labels: anger, disgust, fear, joy, neutral, sadness, surprise
- YouTube-domain adaptation set:
  - 2,962 comments
  - 30 videos
  - up to 100 comments per video
  - assistant-assisted labels
- Independent app test:
  - 150 manually reviewed comments
  - 3 videos x 50 comments

Speaker notes:

GoEmotions gives a reliable starting point, but it is not YouTube-specific. To reduce domain shift, we added a YouTube-domain adaptation set. The final app test is separate from training data.

## Slide 5: Models

Main message:

The app uses two Hugging Face text classification pipelines.

Emotion pipeline:

- Base model: `distilbert-base-uncased`
- Original fine-tuned model: `chase1zhang/youtube-emotion-distilbert`
- Domain-adapted model: `chase1zhang/youtube-emotion-distilbert-domain-adapted`
- Public comparison model: `SamLowe/roberta-base-go_emotions`
- Public comparison model: `j-hartmann/emotion-english-distilroberta-base`
- Optional larger comparison model: `j-hartmann/emotion-english-roberta-large`

Sentiment pipeline:

- `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Labels: positive, neutral, negative

Speaker notes:

The seven-emotion model gives more detailed diagnosis. The sentiment model is easier to interpret and provides a stable business-level signal.

## Slide 6: Streamlit Cloud Demo

Main message:

The deployed app provides an end-to-end business workflow.

Show screenshots:

1. URL input
2. Summary metrics
3. Emotion distribution chart
4. Comment-level table
5. Three-model comparison mode
6. Marketing recommendation

Speaker notes:

During the demo, paste a YouTube URL, run the analysis, and explain what the main emotion and negative emotion ratio mean for campaign decisions.
Also switch to compare mode to show how the same comments are labeled by three fine-tuned emotion models.

## Slide 7: Experimental Results

Main message:

Fine-tuning improves benchmark accuracy, while real YouTube comments remain challenging.

Key results:

| Test | Model / Pipeline | Result |
|---|---|---:|
| GoEmotions test | Pre-tuning baseline | 0.7033 |
| GoEmotions test | Fine-tuned DistilBERT | 0.7604 |
| YouTube-domain validation | Domain-adapted model | 0.6400 |
| Streamlit app 7-emotion | Pre-tuning baseline | 73/150 |
| Streamlit app 7-emotion | GoEmotions fine-tuned | 73/150 |
| Streamlit app 7-emotion | Domain-adapted | 69/150 |
| Streamlit app 3-sentiment | Supporting pipeline | 108/150 |

Speaker notes:

The GoEmotions fine-tuned model performs best on the original test set. The domain-adapted model helps on some marketing-style videos but performs worse on anger-heavy news comments. The broad sentiment pipeline is the most stable on the app benchmark.

## Slide 8: Business Insights and Limitations

Main message:

The app is useful as a decision-support tool, not an automatic final decision maker.

Insights:

- Joy / surprise: campaign excitement
- Neutral: low emotional engagement
- High anger / disgust / fear / sadness: reputational risk
- Model disagreement: human review needed

Limitations:

- YouTube comments are noisy and sarcastic.
- Seven-emotion classification is harder than three-class sentiment.
- More manually verified YouTube labels are needed.

Speaker notes:

The practical value is early signal detection. The app can reduce manual reading workload and guide marketing teams toward videos or comments that deserve deeper review.

## Slide 9: Conclusion

Main message:

The project demonstrates a complete deep learning business application.

Include:

- Two Hugging Face pipelines
- Fine-tuned DistilBERT emotion classifier
- YouTube Data API integration
- Streamlit Cloud deployment
- Experimental comparison and manual app evaluation

Speaker notes:

Overall, this project shows how transformer pipelines can be used in a real marketing workflow. The app collects comments, classifies emotions, summarizes audience reaction, and supports data-driven campaign decisions.
