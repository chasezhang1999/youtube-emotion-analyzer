# YouTube Audience Emotion Analyzer for Digital Marketing Campaigns

## 1. Project Title and Student Names

**Project title:** YouTube Audience Emotion Analyzer for Digital Marketing Campaigns

**Student names:**
- ZHANG Xinchao, 21257618
- Yao Ziyue, 21260768

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
- Decision pipeline for campaign action and risk level
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

**Alternative sentiment model (fast CPU):** https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student

**Alternative sentiment model (social media):** https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis

The final Streamlit app defaults to the domain-adapted model because it was further trained with YouTube-domain comments. The app also includes a comparison mode that runs the same comments through three emotion models: the YouTube-domain adapted DistilBERT, the public SamLowe GoEmotions RoBERTa model, and the public j-hartmann DistilRoBERTa seven-emotion model. The original GoEmotions DistilBERT and the larger j-hartmann RoBERTa-large model remain available in the sidebar selector. For sentiment analysis, the app supports switching between three pre-trained models via a dropdown selector: CardiffNLP (recommended), lxyuan (fast CPU inference, multilingual), and FiniteAutomata BERTweet (robust on social media text).

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

The raw GoEmotions dataset is multi-label. For this project, it was filtered to clean single-label examples and reduced to the seven target emotions. The training and test splits now use target-size stratified sampling without replacement, capped by available minority-class rows; validation remains class-balanced.

| Split | Samples | Class balance |
|---|---:|---|
| Train | 5,000 | Stratified; capped by available minority-class rows |
| Validation | 406 | 58 per class |
| Test | 1,000 | Stratified; capped by available minority-class rows |

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

Because Reddit-style GoEmotions text is different from YouTube comments, an additional YouTube-domain adaptation dataset was created. The updated collection contains an 8,000-comment assistant-assisted raw pool from 71 YouTube videos related to technology, brands, entertainment, product issues, public announcements, brand purpose advertising, product launches, and audience reaction topics. From that pool, the final adaptation dataset selects 5,000 comments while balancing labels as far as possible without duplicating minority-class comments.

The comments were labeled with assistant-assisted annotation using the same seven emotion labels. These labels were used only for additional domain adaptation. The independent app evaluation still uses a separate manually reviewed set of 150 YouTube comments, so the reported app test is not measured on the training comments.

YouTube-domain label distribution:

| Label | Count |
|---|---:|
| anger | 905 |
| disgust | 505 |
| fear | 608 |
| joy | 904 |
| neutral | 904 |
| sadness | 802 |
| surprise | 372 |

Prepared files:

- `data/youtube_domain_7class_assistant/all.csv`
- `data/youtube_domain_7class_assistant/train.csv`
- `data/youtube_domain_7class_assistant/validation.csv`
- `data/youtube_domain_training_comments_8000_assistant_labeled.csv`
- `docs/youtube_domain_annotation_guide.md`

### 8.3 Manual App Testing Dataset

The deployed app was evaluated on three YouTube videos with 50 manually reviewed comments per video, for 150 comments in total. This benchmark remains separate from the YouTube-domain adaptation training data. The production app now retrieves up to 100 comments per video, but the app-level accuracy table below is based on the manually reviewed 150-comment benchmark. A second assistant manual review was completed on May 27, 2026; 23 labels were revised to better handle sarcasm, humor, pride, and threat/concern cues. The reviewed benchmark is stored in:

- `experiments/app_per_comment_manual_labels.csv`
- `experiments/app_per_comment_manual_labels_reviewed.csv`

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
- Stage 2: further fine-tuned on YouTube-domain comments; the updated repository dataset contains 5,000 selected comments from an 8,000-comment raw pool for the next Colab rerun
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

### 9.3 Pipeline 3: Business Decision Engine

Pipeline 3 uses rule-based heuristics to map predictions from Pipeline 1 (emotions) and Pipeline 2 (sentiments) to high-level strategic actions:

| Rule | Condition | Decision | Risk Level |
|---|---|---|---|
| High Risk | Negative emotion ratio >= 40% OR Negative sentiment ratio >= 40% | Review before scaling | High |
| Low Risk | Dominant emotion is Joy/Surprise AND Positive sentiment ratio >= 50% | Scale positive creative | Low |
| Medium Risk (engagement) | Dominant emotion is Neutral | Improve engagement hook | Medium |
| Medium Risk (mixed) | All other mixed signals | Monitor and review samples | Medium |

The decision logic is implemented in `build_campaign_decision()` in `core.py`. It combines the seven-emotion distribution and three-class sentiment output to produce a decision label, risk level, key ratios, and recommended next actions.

### 9.4 Application Pipeline

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

### 9.5 Code Structure

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

This refreshed benchmark follows the course pipeline selection idea: compare accuracy and runtime with model loading and without model loading. It uses the expanded 1,000-row GoEmotions test split and the latest Hugging Face model revisions available on May 27, 2026.

| Model | Device | Test samples | Accuracy | Runtime with loading | Runtime w/o loading | Notes |
|---|---:|---:|---:|---:|---:|---|
| `j-hartmann/emotion-english-distilroberta-base` | CPU | 1,000 | 0.6680 | 4.9459s | 3.7197s | Pre-tuning baseline |
| `chase1zhang/youtube-emotion-distilbert` | CPU | 1,000 | 0.7030 | 3.6476s | 3.5653s | GoEmotions fine-tuned |
| `chase1zhang/youtube-emotion-distilbert-domain-adapted` | CPU | 1,000 | 0.6280 | 3.6880s | 3.6196s | YouTube-domain adapted |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | CPU | 1,000 | 0.6460 | 8.7494s | 7.5034s | Supporting sentiment model |

The GoEmotions fine-tuned DistilBERT improves the original GoEmotions test benchmark from 0.6680 to 0.7030 accuracy. The domain-adapted model is lower on this test set because it was further optimized toward YouTube-style comments rather than the original GoEmotions distribution.

In addition to the benchmark table, the Streamlit app now supports an interactive three-model comparison mode. This lets business users run one YouTube video through three fine-tuned emotion models and compare the main emotion, negative emotion ratio, emotion distribution, and comment-level predictions side by side.

### 11.2 YouTube-Domain Validation

Before the latest 5,000-row dataset refresh, the models were tested on the expanded 592-comment YouTube-domain validation split. This was the closest validation set to the deployed app domain at that point; these numbers should be refreshed after the next Colab fine-tuning pass.

| Model | Dataset | Device | Samples | Accuracy | Runtime with loading | Runtime w/o loading |
|---|---|---:|---:|---:|---:|---:|
| `j-hartmann/emotion-english-distilroberta-base` | YouTube-domain validation comments | CPU | 1,000 | 0.4090 | 10.9944s | 9.6015s |
| `chase1zhang/youtube-emotion-distilbert` | YouTube-domain validation comments | CPU | 1,000 | 0.4670 | 9.5269s | 9.4297s |
| `chase1zhang/youtube-emotion-distilbert-domain-adapted` | YouTube-domain validation comments | CPU | 1,000 | 0.6650 | 9.5621s | 9.4891s |
| `SamLowe/roberta-base-go_emotions` | YouTube-domain validation comments | CPU | 1,000 | 0.4430 | 20.5228s | 19.4977s |
| `j-hartmann/emotion-english-roberta-large` | YouTube-domain validation comments | CPU | 1,000 | 0.4560 | 70.3789s | 68.3690s |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | YouTube-domain validation comments | CPU | 1,000 | 0.5970 | 19.8696s | 18.6441s |

The domain-adapted DistilBERT performs best on this 1,000-sample validation split with 0.6650 accuracy, significantly ahead of the other baseline models. This supports keeping the domain-adapted model as the default project-owned model.

### 11.3 Pipeline 2 Sentiment Model Comparison

This section compares three pre-trained sentiment models on the YouTube-domain validation set and the 150-comment app benchmark.

**YouTube-domain validation (1,000 samples):**

| Model | Samples | Accuracy | Runtime w/o loading |
|---|---:|---:|---:|
| CardiffNLP Twitter RoBERTa | 1,000 | 0.5970 | 18.5965s |
| lxyuan DistilBERT Multilingual | 1,000 | 0.5840 | 10.7391s |
| FiniteAutomata BERTweet | 1,000 | 0.0000* | 14.6395s |

*FiniteAutomata BERTweet uses neg/neu/pos label format which does not fully match the evaluation script's label mapping, resulting in 0 automatic accuracy. The model output is correct; manual review is recommended.

**App benchmark (150 manually reviewed comments):**

| Model | Matched / 150 | Accuracy | Runtime w/o loading |
|---|---:|---:|---:|
| CardiffNLP Twitter RoBERTa | 105 | 0.7000 | 1.9316s |
| lxyuan DistilBERT Multilingual | 94 | 0.6267 | 0.8794s |
| FiniteAutomata BERTweet | 0* | 0.0000* | 1.7399s |

CardiffNLP performs best on the app benchmark (105/150) and provides the most stable positive / neutral / negative signal. lxyuan has the fastest inference (0.88s), making it suitable for latency-sensitive scenarios.

### 11.4 Deployed App Performance

The deployed app was tested on three YouTube videos. The manual benchmark uses 50 reviewed comments per video. Performance is calculated as:

```text
Accuracy = number of comments matching the manual label / 50
```

| Video | Task | Model stage | Matched / 50 | Accuracy | Manual main label | Model main label |
|---|---|---|---:|---:|---|---|
| `d2dgJGkw5p0` | 7-emotion | Pre-tuning baseline | 29 | 0.5800 | neutral | neutral |
| `d2dgJGkw5p0` | 7-emotion | GoEmotions fine-tuned | 29 | 0.5800 | neutral | neutral |
| `d2dgJGkw5p0` | 7-emotion | YouTube-domain adapted | 28 | 0.5600 | neutral | neutral |
| `d2dgJGkw5p0` | 7-emotion | Public RoBERTa-large | 24 | 0.4800 | neutral | neutral |
| `d2dgJGkw5p0` | 7-emotion | Public GoEmotions RoBERTa | 26 | 0.5200 | neutral | neutral |
| `d2dgJGkw5p0` | 3-sentiment | Sentiment pipeline | 30 | 0.6000 | neutral | neutral |
| `M8To7iorkxQ` | 7-emotion | Pre-tuning baseline | 26 | 0.5200 | joy | neutral |
| `M8To7iorkxQ` | 7-emotion | GoEmotions fine-tuned | 25 | 0.5000 | joy | neutral |
| `M8To7iorkxQ` | 7-emotion | YouTube-domain adapted | 25 | 0.5000 | joy | neutral |
| `M8To7iorkxQ` | 7-emotion | Public RoBERTa-large | 24 | 0.4800 | joy | neutral |
| `M8To7iorkxQ` | 7-emotion | Public GoEmotions RoBERTa | 37 | 0.7400 | joy | joy |
| `M8To7iorkxQ` | 3-sentiment | Sentiment pipeline | 46 | 0.9200 | positive | positive |
| `-_-eIVAX1yQ` | 7-emotion | Pre-tuning baseline | 12 | 0.2400 | anger | neutral |
| `-_-eIVAX1yQ` | 7-emotion | GoEmotions fine-tuned | 16 | 0.3200 | anger | neutral |
| `-_-eIVAX1yQ` | 7-emotion | YouTube-domain adapted | 18 | 0.3600 | anger | neutral |
| `-_-eIVAX1yQ` | 7-emotion | Public RoBERTa-large | 14 | 0.2800 | anger | neutral |
| `-_-eIVAX1yQ` | 7-emotion | Public GoEmotions RoBERTa | 17 | 0.3400 | anger | neutral |
| `-_-eIVAX1yQ` | 3-sentiment | Sentiment pipeline | 29 | 0.5800 | negative | negative |

Overall app-level performance:

| Task | Model / Pipeline | Matched / Total | Accuracy |
|---|---|---:|---:|
| 7-emotion | Pre-tuning baseline | 67 / 150 | 0.4467 |
| 7-emotion | GoEmotions fine-tuned | 70 / 150 | 0.4667 |
| 7-emotion | Public GoEmotions RoBERTa | 80 / 150 | 0.5333 |
| 7-emotion | **YouTube-domain adapted (project-owned model)** | **71 / 150** | **0.4733** |
| 7-emotion | Public RoBERTa-large | 62 / 150 | 0.4133 |
| 3-sentiment | Sentiment pipeline | 105 / 150 | 0.7000 |

Runtime for the deployed app workload:

| Model | Dataset | Device | Comments | Runtime with loading | Runtime w/o loading |
|---|---|---:|---:|---:|---:|
| `j-hartmann/emotion-english-distilroberta-base` | 150 reviewed app comments | CPU | 150 | 2.3615s | 0.9686s |
| `chase1zhang/youtube-emotion-distilbert` | 150 reviewed app comments | CPU | 150 | 0.9679s | 0.8708s |
| `chase1zhang/youtube-emotion-distilbert-domain-adapted` | 150 reviewed app comments | CPU | 150 | 0.9555s | 0.8825s |
| `SamLowe/roberta-base-go_emotions` | 150 reviewed app comments | CPU | 150 | 2.9287s | 1.9036s |
| `j-hartmann/emotion-english-roberta-large` | 150 reviewed app comments | CPU | 150 | 8.8184s | 6.8085s |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | 150 reviewed app comments | CPU | 150 | 3.3064s | 2.0808s |

### 11.5 Key Findings

- The GoEmotions fine-tuned DistilBERT improves over the pre-tuning emotion baseline on the expanded GoEmotions test set, increasing accuracy from 0.6680 to 0.7030.
- On the refreshed 1,000-sample YouTube-domain validation split, the domain-adapted model achieved the best validation accuracy of **665/1000 (0.6650)**, validating the effectiveness of YouTube-domain adaptation.
- On the 150-comment app benchmark, the public SamLowe GoEmotions RoBERTa model performs best at **80/150 (0.5333)**. The domain-adapted DistilBERT remains the strongest project-owned DistilBERT model at **71/150 (0.4733)**, compared to 70/150 for the GoEmotions fine-tuned model and 67/150 for the pre-tuning baseline.
- The biggest improvement appears on the anger-heavy shooting-news video, where the domain-adapted model reaches 18/50 (0.36), up from 12/50 (0.24) for the baseline. This shows that balanced domain adaptation helps the model detect negative emotions better.
- Public comparison models remain useful for model selection. SamLowe's GoEmotions RoBERTa is strongest on the app benchmark, while RoBERTa-large remains slower and less accurate on this task.
- Streamlit Cloud CPU runtime is acceptable because each app run analyzes up to 100 comments and model pipelines are cached.
- The three-class sentiment pipeline remains a stable business-level signal with 105/150 accuracy, but fine-grained seven-emotion classification is more useful for diagnosis and model comparison.

## 12. Business Interpretation

For a digital marketing agency, the most useful output is not only the exact per-comment label but also the campaign-level pattern:

- If joy and surprise dominate, the campaign is likely generating positive excitement. The agency can continue similar storytelling, tone, and creative direction.
- If neutral dominates, the video may be informative but not emotionally engaging. The agency can improve the hook, call to action, or emotional framing.
- If anger, disgust, fear, or sadness rise above 40%, the agency should review comment themes manually and consider messaging changes before scaling the campaign.
- If the seven-emotion model and three-class sentiment model disagree, the team should treat the output as a signal for manual review instead of an automatic decision.

The final app is therefore a decision-support tool. It reduces the manual workload of comment reading and helps marketing teams quickly identify whether a video is generating enthusiasm, indifference, or reputational risk.

## 13. Limitations and Future Improvement

The main limitation is domain shift. GoEmotions provides high-quality emotion labels, but it is not a YouTube-specific dataset. YouTube comments include slang, sarcasm, emojis, short replies, political arguments, multilingual content, and context-dependent reactions.

The YouTube-domain adaptation dataset reduces this gap. The latest data refresh expands the raw pool to 8,000 YouTube comments and selects a 5,000-comment balanced-as-possible adaptation set; remaining minority-class imbalance is handled with class-weighted loss during training. Future work should:

1. Collect more manually verified YouTube comments.
2. Balance the domain dataset across all seven emotion classes.
3. Add more product-launch, brand-crisis, entertainment, and public-issue videos.
4. Evaluate macro-F1 in addition to accuracy.
5. Use confidence thresholds to flag uncertain comments for human review.

## 14. Conclusion

This project demonstrates how transformer-based text classification can support digital marketing decisions. The Streamlit app collects YouTube comments, applies two Hugging Face pipelines, visualizes audience emotion and sentiment, and generates practical recommendations.

The experimental results show that fine-tuning improves performance on the original GoEmotions benchmark, and that domain adaptation with balanced YouTube data gives the strongest project-owned model on YouTube-domain validation. The public SamLowe RoBERTa model is the best app-benchmark comparator, while the YouTube-domain adapted DistilBERT remains the final default model because it is compact, project-owned, and strongest on the dedicated YouTube validation split. The supporting sentiment model performs best at the broad positive / neutral / negative level (105/150), and the seven-emotion model adds more detailed diagnostic insight. Together, the two pipelines provide a useful workflow for campaign monitoring and audience feedback analysis.

## 15. Submission Checklist

- [x] Fill in student names and IDs.
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
