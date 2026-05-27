# YouTube Audience Emotion Analyzer for Digital Marketing Campaigns

## 1. Project Title and Student Names

**Project title:** YouTube Audience Emotion Analyzer for Digital Marketing Campaigns

**Student names:**
- [Student name 1, student ID]
- [Student name 2, student ID]

## 2. Company Name and Website URL

**Company:** InsightWave Digital Marketing Agency (course project scenario)

**Website / application URL:** https://youtube-emotion-analyzer.streamlit.app/

InsightWave Digital Marketing Agency helps brands evaluate video campaigns and audience engagement on social media. YouTube comments contain useful customer reactions, but manually reading comments is slow, subjective, and difficult to scale. This project builds a deep learning application that summarizes the main emotions in YouTube comments and turns them into practical marketing recommendations.

## 3. Project Objective

This project helps a digital marketing agency classify the first 100 comments under a YouTube video into seven emotions, summarize campaign reaction, flag negative feedback risk, and generate actionable recommendations for content and messaging improvement.

## 4. Strategy

The strategy is to build a Streamlit Cloud business application that accepts a YouTube video URL, retrieves the first 100 top-level comments through the YouTube Data API, and analyzes the comments with Hugging Face transformer pipelines.

The application uses two text classification pipelines:

1. A seven-emotion pipeline for detailed audience emotion analysis.
2. A three-class sentiment pipeline for a simpler positive / neutral / negative business signal.

The seven target emotions are anger, disgust, fear, joy, neutral, sadness, and surprise. The app calculates the dominant emotion, the full emotion distribution, the sentiment distribution, and a negative emotion ratio defined as anger + disgust + fear + sadness. The ratio is used as a campaign risk indicator. If negative emotions are high, the marketing team should review message framing, audience targeting, and potential public-reaction risks.

The dashboard provides:

- Main audience emotion
- Seven-emotion distribution chart
- Three-class sentiment distribution chart
- Three-model emotion comparison mode
- Negative emotion ratio metric
- Comment-level prediction table with model confidence scores
- Marketing recommendation text
- CSV download of all predictions

## 5. Model URL

**Primary deployed emotion model:** https://huggingface.co/chase1zhang/youtube-emotion-distilbert-domain-adapted

**Original fine-tuned emotion model:** https://huggingface.co/chase1zhang/youtube-emotion-distilbert

**Public GoEmotions fine-tuned comparison model:** https://huggingface.co/SamLowe/roberta-base-go_emotions

**Pre-tuning baseline emotion model:** https://huggingface.co/j-hartmann/emotion-english-distilroberta-base

**Optional larger seven-emotion model:** https://huggingface.co/j-hartmann/emotion-english-roberta-large

**Supporting sentiment model:** https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest

The final Streamlit app defaults to the domain-adapted model because it was further trained with YouTube-domain comments. The app also includes a comparison mode that runs the same comments through three emotion models: the YouTube-domain adapted DistilBERT, the public SamLowe GoEmotions RoBERTa model, and the public j-hartmann DistilRoBERTa seven-emotion model. The original GoEmotions DistilBERT and the larger j-hartmann RoBERTa-large model remain available in the sidebar selector.

## 6. App URL

**Deployed Streamlit Cloud app:** https://youtube-emotion-analyzer.streamlit.app/

## 7. GitHub URL

**GitHub repository:** https://github.com/chasezhang1999/youtube-emotion-analyzer

The repository is connected to Streamlit Cloud for automatic deployment. The YouTube Data API key is stored in Streamlit secrets and is not uploaded to GitHub.

## 8. Dataset

### 8.1 GoEmotions Fine-Tuning Dataset

The first training stage uses the GoEmotions dataset from Hugging Face:

- Dataset URL: https://huggingface.co/datasets/SetFit/go_emotions
- Input feature: `text`
- Target feature: seven-class emotion label
- Labels: anger, disgust, fear, joy, neutral, sadness, surprise

The raw GoEmotions dataset is multi-label. For this project, it was filtered to clean single-label examples and reduced to the seven target emotions. Each split was balanced by downsampling to the minority class.

| Split | Samples | Class balance |
|---|---:|---|
| Train | 3,010 | 430 per class |
| Validation | 406 | 58 per class |
| Test | 455 | 65 per class |

Preprocessing steps:

1. Keep examples where exactly one emotion label is active.
2. Keep only the seven project labels.
3. Convert label names to label IDs from 0 to 6.
4. Balance each split by class to avoid a majority-class model.

Prepared files:

- `data/go_emotions_7class/train.csv`
- `data/go_emotions_7class/validation.csv`
- `data/go_emotions_7class/test.csv`

### 8.2 YouTube-Domain Adaptation Dataset

Because Reddit-style GoEmotions text is different from YouTube comments, an additional YouTube-domain adaptation dataset was created. The updated dataset contains 2,962 comments collected from 30 YouTube videos related to technology, brands, entertainment, product issues, public announcements, brand purpose advertising, product launches, and audience reaction topics. The script attempts to collect up to 100 top-level comments per video; one older video returned fewer public top-level comments.

The comments were labeled with assistant-assisted annotation using the same seven emotion labels. These labels were used only for additional domain adaptation. The independent app evaluation still uses a separate manually reviewed set of 150 YouTube comments, so the reported app test is not measured on the training comments.

YouTube-domain label distribution:

| Label | Count |
|---|---:|
| neutral | 1,532 |
| joy | 943 |
| anger | 136 |
| disgust | 102 |
| surprise | 88 |
| sadness | 82 |
| fear | 79 |

Prepared files:

- `data/youtube_domain_7class_assistant/all.csv`
- `data/youtube_domain_7class_assistant/train.csv`
- `data/youtube_domain_7class_assistant/validation.csv`
- `docs/youtube_domain_annotation_guide.md`

### 8.3 Manual App Testing Dataset

The deployed app was evaluated on three YouTube videos with 50 manually reviewed comments per video, for 150 comments in total. This benchmark remains separate from the YouTube-domain adaptation training data. The production app now retrieves up to 100 comments per video, but the app-level accuracy table below is based on the manually reviewed 150-comment benchmark. The manual benchmark is stored in:

- `experiments/app_per_comment_manual_labels.csv`

## 9. Model

### 9.1 Pipeline 1: Seven-Emotion Classification

The main pipeline classifies each YouTube comment into one of seven emotion labels.

```python
pipeline(
    "text-classification",
    model="chase1zhang/youtube-emotion-distilbert-domain-adapted",
)
```

Model development:

- Base model: `distilbert-base-uncased`
- Stage 1: fine-tuned on the balanced seven-class GoEmotions dataset
- Stage 2: further fine-tuned on YouTube-domain comments; the updated repository dataset contains 2,962 comments for the final Colab rerun
- Training setup: learning rate 2e-5, batch size 16, 3 epochs for the first stage
- Model selection: validation accuracy and app-level manual testing
- Streamlit comparison: the app can compare the deployed model with `SamLowe/roberta-base-go_emotions`, `j-hartmann/emotion-english-distilroberta-base`, `chase1zhang/youtube-emotion-distilbert`, and `j-hartmann/emotion-english-roberta-large`

The SamLowe model predicts the 28-label GoEmotions label set. To compare it with this project's seven-emotion output, related labels are mapped into the seven target emotions. For example, admiration, amusement, love, gratitude, optimism, and excitement are mapped to joy; annoyance is mapped to anger; disappointment, grief, and remorse are mapped to sadness.

### 9.2 Pipeline 2: Sentiment Classification

The second pipeline classifies each comment into positive, neutral, or negative sentiment.

```python
pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
)
```

This supporting model gives stakeholders a simpler signal. In app testing, the three-class sentiment task was more stable than seven-emotion classification because YouTube comments are short, noisy, sarcastic, and sometimes multilingual.

### 9.3 Application Pipeline

```text
YouTube video URL
    -> parse video ID
    -> YouTube Data API commentThreads endpoint
    -> first 100 top-level comments
    -> clean comment text
    -> seven-emotion pipeline
    -> three-class sentiment pipeline
    -> emotion summary, sentiment summary, risk ratio
    -> Streamlit dashboard and CSV export
```

### 9.4 Code Structure

```text
youtube_emotion_project/
|-- streamlit_app.py
|-- youtube_emotion/
|   |-- core.py
|   |-- model_runner.py
|   |-- youtube_client.py
|   |-- dataset_prep.py
|   `-- __init__.py
|-- data/
|   |-- go_emotions_7class/
|   |-- youtube_domain_7class_assistant/
|   `-- sample_comments.csv
|-- notebooks/
|   |-- Fine_tune_Model.ipynb
|   |-- fine_tune_go_emotions_distilbert.ipynb
|   `-- testing_experiments.ipynb
|-- experiments/
|   |-- Experimental_results.xlsx
|   |-- app_performance_model_comparison.csv
|   |-- app_per_comment_manual_labels.csv
|   |-- app_runtime_summary.csv
|   `-- youtube_domain_validation_performance.csv
|-- tests/
|-- docs/
`-- requirements.txt
```

## 10. Deployment

The app is deployed on Streamlit Cloud and connected to the GitHub repository.

Deployment steps:

1. Push the Streamlit app files to GitHub.
2. Connect the GitHub repository to Streamlit Cloud.
3. Add `YOUTUBE_API_KEY` to Streamlit Cloud secrets.
4. Load Hugging Face models on first use and cache them with `st.cache_resource`.
5. Let users access the public app URL.

App usage:

1. Paste a YouTube video URL.
2. Choose single-model analysis or multi-model comparison from the sidebar.
3. Click the analyze button.
4. Review emotion and sentiment charts.
5. Read the comment-level prediction table.
6. Use the recommendation box for campaign interpretation.
7. Download the prediction CSV if needed.

Application screenshots:

![Input page with YouTube URL](screenshots/01_input_with_url.png)

![Summary metrics](screenshots/02_summary_metrics.png)

![Seven-emotion distribution](screenshots/03_emotion_distribution.png)

![Comment-level result table](screenshots/04_comment_level_results.png)

![Marketing recommendation](screenshots/05_marketing_recommendation.png)

![Three-model comparison mode](screenshots/06_model_comparison_mode.png)

## 11. Experiments

The experiments evaluate model accuracy, runtime, domain adaptation, and deployed app performance.

### 11.1 Model Selection on GoEmotions Test Set

This benchmark follows the course pipeline selection idea: compare accuracy and runtime with model loading and without model loading.

| Model | Device | Test samples | Accuracy | Runtime with loading | Runtime w/o loading | Notes |
|---|---:|---:|---:|---:|---:|---|
| `j-hartmann/emotion-english-distilroberta-base` | CPU | 455 | 0.7033 | 38.6122s | 27.8509s | Pre-tuning baseline |
| `j-hartmann/emotion-english-distilroberta-base` | GPU | 455 | 0.7033 | 4.8555s | 3.9927s | Pre-tuning baseline |
| `chase1zhang/youtube-emotion-distilbert` | CPU | 455 | 0.7604 | 30.7984s | 25.9997s | GoEmotions fine-tuned |
| `chase1zhang/youtube-emotion-distilbert` | GPU | 455 | 0.7604 | 3.2396s | 2.2477s | GoEmotions fine-tuned |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | CPU | 455 | 0.7363 | 57.5298s | 50.5400s | Supporting sentiment model |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | GPU | 455 | 0.7363 | 6.1080s | 4.2489s | Supporting sentiment model |
| `chase1zhang/youtube-emotion-distilbert-domain-adapted` | CPU | 455 | 0.7011 | 3.2007s | 2.3860s | Local Mac CPU, cached model |
| `chase1zhang/youtube-emotion-distilbert-domain-adapted` | MPS | 455 | 0.7011 | 4.9693s | 3.6406s | Local Mac MPS, not Colab T4 |

The original GoEmotions fine-tuned model performs best on the GoEmotions test set with 0.7604 accuracy. The domain-adapted model is lower on this test set because it was further optimized toward YouTube-style comments rather than the original GoEmotions distribution.

In addition to the benchmark table, the Streamlit app now supports an interactive three-model comparison mode. This lets business users run one YouTube video through three fine-tuned emotion models and compare the main emotion, negative emotion ratio, emotion distribution, and comment-level predictions side by side.

### 11.2 YouTube-Domain Validation

The domain-adapted model was also tested on the YouTube-domain validation split. The table below records the earlier benchmark from the first YouTube-domain adaptation run. After rerunning Colab on the expanded 2,962-comment dataset, this row should be refreshed with the new validation split result.

| Model | Dataset | Device | Samples | Accuracy | Runtime with loading | Runtime w/o loading |
|---|---|---:|---:|---:|---:|---:|
| `chase1zhang/youtube-emotion-distilbert-domain-adapted` | YouTube-domain validation comments | CPU | 200 | 0.6400 | 3.4064s | 2.6420s |

This result suggests that the YouTube-domain data helps the model learn platform-specific language patterns, but the dataset is still small and imbalanced. The model tends to over-predict neutral and joy for some difficult comments.

### 11.3 Deployed App Performance

The deployed app was tested on three YouTube videos. The manual benchmark uses 50 reviewed comments per video. Performance is calculated as:

```text
Accuracy = number of comments matching the manual label / 50
```

| Video | Task | Model stage | Matched / 50 | Accuracy | Manual main label | Model main label |
|---|---|---|---:|---:|---|---|
| `d2dgJGkw5p0` | 7-emotion | Pre-tuning baseline | 31 | 0.6200 | neutral | neutral |
| `d2dgJGkw5p0` | 7-emotion | GoEmotions fine-tuned | 32 | 0.6400 | neutral | neutral |
| `d2dgJGkw5p0` | 7-emotion | Domain-adapted fine-tuned | 34 | 0.6800 | neutral | neutral |
| `d2dgJGkw5p0` | 3-sentiment | Supporting pipeline | 31 | 0.6200 | neutral | neutral |
| `M8To7iorkxQ` | 7-emotion | Pre-tuning baseline | 28 | 0.5600 | joy | neutral |
| `M8To7iorkxQ` | 7-emotion | GoEmotions fine-tuned | 24 | 0.4800 | joy | neutral |
| `M8To7iorkxQ` | 7-emotion | Domain-adapted fine-tuned | 25 | 0.5000 | joy | neutral |
| `M8To7iorkxQ` | 3-sentiment | Supporting pipeline | 44 | 0.8800 | positive | positive |
| `-_-eIVAX1yQ` | 7-emotion | Pre-tuning baseline | 14 | 0.2800 | anger | neutral |
| `-_-eIVAX1yQ` | 7-emotion | GoEmotions fine-tuned | 17 | 0.3400 | anger | neutral |
| `-_-eIVAX1yQ` | 7-emotion | Domain-adapted fine-tuned | 10 | 0.2000 | anger | neutral |
| `-_-eIVAX1yQ` | 3-sentiment | Supporting pipeline | 33 | 0.6600 | negative | negative |

Overall app-level performance:

| Task | Model / Pipeline | Matched / Total | Accuracy |
|---|---|---:|---:|
| 7-emotion | Pre-tuning baseline | 73 / 150 | 0.4867 |
| 7-emotion | GoEmotions fine-tuned model | 73 / 150 | 0.4867 |
| 7-emotion | Domain-adapted fine-tuned model | 69 / 150 | 0.4600 |
| 3-sentiment | Supporting sentiment pipeline | 108 / 150 | 0.7200 |

Runtime for the deployed app workload:

| Model | Dataset | Device | Comments | Runtime with loading | Runtime w/o loading |
|---|---|---:|---:|---:|---:|
| `chase1zhang/youtube-emotion-distilbert-domain-adapted` | 150 manually reviewed app comments | CPU | 150 | 2.3695s | 1.5621s |

### 11.4 Key Findings

- The GoEmotions fine-tuned DistilBERT improves over the pre-tuning emotion baseline on the GoEmotions test set, increasing accuracy from 0.7033 to 0.7604.
- GPU inference is much faster than CPU inference in the Colab benchmark. This matters for training and batch experiments, while Streamlit Cloud CPU runtime is still acceptable because each app run analyzes up to 100 comments.
- Domain adaptation improved the app result on the rare-earths video from 32/50 to 34/50 and slightly improved the Avatar video from 24/50 to 25/50.
- Domain adaptation hurt the shooting-news video, dropping from 17/50 to 10/50, because the YouTube-domain training set is dominated by neutral and joy comments and has fewer anger / fear / sadness examples.
- The Streamlit comparison mode makes this trade-off visible by showing how the same comments are labeled by three fine-tuned emotion models.
- The three-class sentiment pipeline is the most stable app-level signal, achieving 108/150 accuracy. This shows that broad sentiment is easier than fine-grained seven-emotion classification on noisy YouTube comments.

## 12. Business Interpretation

For a digital marketing agency, the most useful output is not only the exact per-comment label but also the campaign-level pattern:

- If joy and surprise dominate, the campaign is likely generating positive excitement. The agency can continue similar storytelling, tone, and creative direction.
- If neutral dominates, the video may be informative but not emotionally engaging. The agency can improve the hook, call to action, or emotional framing.
- If anger, disgust, fear, or sadness rise above 40%, the agency should review comment themes manually and consider messaging changes before scaling the campaign.
- If the seven-emotion model and three-class sentiment model disagree, the team should treat the output as a signal for manual review instead of an automatic decision.

The final app is therefore a decision-support tool. It reduces the manual workload of comment reading and helps marketing teams quickly identify whether a video is generating enthusiasm, indifference, or reputational risk.

## 13. Limitations and Future Improvement

The main limitation is domain shift. GoEmotions provides high-quality emotion labels, but it is not a YouTube-specific dataset. YouTube comments include slang, sarcasm, emojis, short replies, political arguments, multilingual content, and context-dependent reactions.

The YouTube-domain adaptation dataset reduces this gap, but it is still small and class-imbalanced. The domain-adapted model became stronger for neutral and joyful marketing-style videos but weaker for anger-heavy news content. Future work should:

1. Collect more manually verified YouTube comments.
2. Balance the domain dataset across all seven emotion classes.
3. Add more product-launch, brand-crisis, entertainment, and public-issue videos.
4. Evaluate macro-F1 in addition to accuracy.
5. Use confidence thresholds to flag uncertain comments for human review.

## 14. Conclusion

This project demonstrates how transformer-based text classification can support digital marketing decisions. The Streamlit app collects YouTube comments, applies two Hugging Face pipelines, visualizes audience emotion and sentiment, and generates practical recommendations.

The experimental results show that fine-tuning improves performance on the original GoEmotions benchmark, while real YouTube app performance is harder because of noisy platform language. The supporting sentiment model performs best at the broad positive / neutral / negative level, and the seven-emotion model adds more detailed diagnostic insight. Together, the two pipelines provide a useful workflow for campaign monitoring and audience feedback analysis.

## 15. Submission Checklist

- [ ] Fill in student names and IDs.
- [x] Insert five Streamlit screenshots before exporting the final PDF.
- [x] GitHub repository: https://github.com/chasezhang1999/youtube-emotion-analyzer
- [x] Streamlit app: https://youtube-emotion-analyzer.streamlit.app/
- [x] Original fine-tuned model: https://huggingface.co/chase1zhang/youtube-emotion-distilbert
- [x] Domain-adapted model: https://huggingface.co/chase1zhang/youtube-emotion-distilbert-domain-adapted
- [x] Public comparison model: https://huggingface.co/SamLowe/roberta-base-go_emotions
- [x] Public comparison model: https://huggingface.co/j-hartmann/emotion-english-distilroberta-base
- [x] Experimental results Excel: `experiments/Experimental_results.xlsx`
- [x] App and dataset files prepared in the repository
- [x] Draft PDF generated: `docs/report/Project_report.pdf`
- [ ] Re-export final PDF after filling student names
- [ ] Prepare PPT slides
- [ ] Record MP4 presentation video
