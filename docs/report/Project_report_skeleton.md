# YouTube Audience Emotion Analyzer for Digital Marketing Campaigns

## 1. Project Title and Student Names

**Project title:** YouTube Audience Emotion Analyzer for Digital Marketing Campaigns

**Student names:**
- [Student name 1, student ID]
- [Student name 2, student ID]

## 2. Company Name and Website URL

**Company:** InsightWave Digital Marketing Agency

**Website URL:** https://insightwave-marketing.example.com

InsightWave Digital Marketing Agency helps brands evaluate social media campaigns and improve content strategies. YouTube is one of the major platforms used by clients for video marketing, product promotion, and audience engagement. However, manually reading comments is slow, subjective, and difficult to scale. This project provides an automated solution using deep learning to classify audience emotions from YouTube comments.

## 3. Project Objective

This project helps a digital marketing agency use deep learning to classify the first 50 YouTube comments into seven audience emotions, summarize campaign reaction, flag negative feedback risk, and generate actionable marketing recommendations.

## 4. Strategy

The strategy is to build a Streamlit Cloud business application that accepts a YouTube video URL, retrieves the first 50 comments through the YouTube Data API, and applies Hugging Face transformer pipelines to analyze audience emotions.

The app focuses on seven emotion categories: anger, disgust, fear, joy, neutral, sadness, and surprise. A fine-tuned DistilBERT model (fine-tuned on `SetFit/go_emotions`) serves as the primary emotion classifier, while a pre-trained sentiment analysis pipeline (`cardiffnlp/twitter-roberta-base-sentiment-latest`) provides supporting business signals by classifying each comment as positive, neutral, or negative.

The dominant emotion and emotion distribution are summarized to help marketing teams understand audience reaction. A negative emotion ratio (anger + disgust + fear + sadness) is calculated to flag potential campaign risks.

The application provides a dashboard with the following outputs:
- Main audience emotion
- Seven-emotion distribution (bar chart)
- Sentiment distribution (bar chart)
- Negative emotion ratio summary metric
- Comment-level prediction table with scores
- Marketing recommendation for campaign improvement
- CSV export of all predictions

## 5. Model URL

**Fine-tuned model on Hugging Face:** https://huggingface.co/chase1zhang/youtube-emotion-distilbert

Base model: `distilbert-base-uncased` (66M parameters)

Comparison model: `j-hartmann/emotion-english-distilroberta-base`
- URL: https://huggingface.co/j-hartmann/emotion-english-distilroberta-base

Sentiment pipeline model: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- URL: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest

## 6. App URL

**Deployed Streamlit Cloud app:** https://youtube-emotion-analyzer.streamlit.app/

## 7. GitHub URL

**GitHub repository:** https://github.com/chasezhang1999/youtube-emotion-analyzer

The repository is connected to Streamlit Cloud for automatic deployment. The app uses Streamlit secrets to manage the YouTube Data API key securely.

## 8. Dataset

The fine-tuning dataset is based on the GoEmotions corpus, filtered to seven Ekman emotion labels and balanced across classes.

**Dataset source:** `SetFit/go_emotions` on Hugging Face
- URL: https://huggingface.co/datasets/SetFit/go_emotions

**Dataset description:**

**(a) Number of labels:** 7 (anger, disgust, fear, joy, neutral, sadness, surprise)

**(b) Relevant features:**
- `text`: the input comment or short text (string)
- `label_name`: the target emotion class (string, one of 7 labels)
- `label`: the numeric label ID (integer, 0–6)

**(c) Number of samples:**
- Training: 3,010 (430 per class, balanced)
- Validation: 406 (58 per class, balanced)
- Testing: 455 (65 per class, balanced)

**(d) Data preprocessing steps:**
1. Filter to single-label examples where exactly one emotion label is active.
2. Keep only the seven Ekman emotion labels (anger, disgust, fear, joy, neutral, sadness, surprise).
3. Balance each split by downsampling to the minority class to prevent label imbalance.

**(e) Sources:**
- Hugging Face dataset: https://huggingface.co/datasets/SetFit/go_emotions
- Preprocessing code: `scripts/prepare_go_emotions_7class.py`
- Prepared CSV files: `data/go_emotions_7class/train.csv`, `validation.csv`, `test.csv`

## 9. Model

This project uses two Hugging Face transformer pipelines for text classification.

### 9.1 Pipeline 1: Seven-Emotion Classification (Primary)

The first pipeline classifies each YouTube comment into one of seven emotion categories. This is the main business pipeline because it provides detailed audience reactions beyond simple positive or negative sentiment.

```python
pipeline("text-classification", model="chase1zhang/youtube-emotion-distilbert")
```

**Target labels:** anger, disgust, fear, joy, neutral, sadness, surprise

**Architecture:**
- Base model: `distilbert-base-uncased` (66M parameters, 6 transformer layers)
- Fine-tuned on GoEmotions seven-class balanced dataset
- Training: 3 epochs, batch size 16, learning rate 2e-5
- Optimization: AdamW with weight decay 0.01
- Best model selected by validation accuracy

**Comparison model:** `j-hartmann/emotion-english-distilroberta-base` (used as baseline in experiments)

### 9.2 Pipeline 2: Sentiment Classification (Supporting)

The second pipeline classifies each comment into positive, neutral, or negative sentiment. It provides a simplified business signal that is easier for non-technical stakeholders to interpret.

```python
pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
```

**Output labels:** positive, neutral, negative

### 9.3 Model Pipeline Structure

```text
YouTube video URL
    ↓ parse_video_id()
Video ID
    ↓ YouTube Data API (commentThreads endpoint)
First 50 comments (raw text)
    ↓ clean_comment_text()
Cleaned text
    ↓ (parallel)
    ├── Pipeline 1: text-classification → 7 emotions + scores
    └── Pipeline 2: sentiment-analysis → sentiment + scores
    ↓
Prediction rows (list of dicts)
    ↓ summarize_predictions() + build_marketing_recommendation()
Dashboard: charts, metrics, table, recommendation
```

### 9.4 Application Code Structure

```text
youtube_emotion_project/
├── streamlit_app.py              # Streamlit UI and entry point
├── youtube_emotion/
│   ├── __init__.py               # Package init
│   ├── core.py                   # URL parsing, label normalization, summary, recommendation
│   ├── model_runner.py           # Pipeline loading and prediction formatting
│   ├── youtube_client.py         # YouTube Data API comment fetching
│   └── dataset_prep.py           # GoEmotions dataset preparation utilities
├── data/
│   ├── go_emotions_7class/       # Training, validation, test CSVs
│   └── sample_comments.csv       # Sample comments for demo without API key
├── notebooks/
│   ├── fine_tune_go_emotions_distilbert.ipynb    # Fine-tuning notebook
│   └── testing_experiments.ipynb                  # Experiments notebook
├── tests/                        # Unit tests for all modules
├── experiments/                  # Experiment CSV template
├── docs/report/                  # Project report
└── requirements.txt              # Streamlit Cloud dependencies
```

## 10. Deployment

The application is deployed on Streamlit Cloud and connected to a GitHub repository. Users can access the app through a public URL.

**Deployment steps:**
1. Code is pushed to a public GitHub repository.
2. Streamlit Cloud is connected to the repository.
3. The `YOUTUBE_API_KEY` is configured as a Streamlit secret.
4. Hugging Face models are downloaded on first use and cached by `@st.cache_resource`.
5. The app is accessible via a public Streamlit Cloud URL.

**Application usage steps:**
1. Paste a YouTube video URL into the input box.
2. Optionally, adjust the comment order (relevance or time) in the sidebar.
3. Click the "Analyze comments" button.
4. The app retrieves the first 50 comments from the YouTube video.
5. Both emotion and sentiment pipelines classify each comment.
6. The dashboard displays the main emotion, distributions, comment-level results, and marketing recommendation.
7. Results can be downloaded as a CSV file.

**Key configuration files:**
- `requirements.txt`: lists all Python dependencies (streamlit, transformers, torch, pandas, requests).
- `.streamlit/config.toml`: Streamlit theme and server settings.
- `.streamlit/secrets.toml`: local API key (excluded from GitHub via `.gitignore`).

**Screenshots to include:**

- **Screenshot 1:** Streamlit input page with YouTube URL input box
- **Screenshot 2:** Main emotion and summary metrics (comments analyzed, main emotion, negative ratio)
- **Screenshot 3:** Emotion distribution bar chart
- **Screenshot 4:** Comment-level prediction table with scores
- **Screenshot 5:** Marketing recommendation output

## 11. Experiments

The experiments have two objectives:

1. Select the best model based on accuracy and runtime (CPU vs GPU).
2. Evaluate the overall performance of the deployed Streamlit Cloud application.

### 11.1 Model Selection — Accuracy and Runtime Comparison

Models compared:

| Model | Device | Task | Evaluation metric |
|---|---|---|---|
| `j-hartmann/emotion-english-distilroberta-base` | CPU / GPU | Seven-emotion | Accuracy |
| Fine-tuned DistilBERT (`distilbert-base-uncased`) | CPU / GPU | Seven-emotion | Accuracy |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | CPU / GPU | Sentiment | Accuracy |

Runtime is measured in two ways (following Class 2 `Pipeline_ChoosingModel.pdf`):
- **Runtime with model loading:** time to load model + run inference on all test samples.
- **Runtime without model loading:** inference time only (model already loaded).

Both CPU (`device=-1`) and GPU (`device=0`) measurements are recorded. Streamlit Cloud runs on CPU, so CPU runtime is the most relevant for deployment performance.

**Results table:**

| Model | Device | Test Samples | Accuracy | Runtime (with loading) | Runtime (w/o loading) |
|---|---:|---:|---:|---:|---:|
| `j-hartmann/emotion-english-distilroberta-base` | CPU | 455 | 0.7033 | 38.6122s | 27.8509s |
| `j-hartmann/emotion-english-distilroberta-base` | GPU | 455 | 0.7033 | 4.8555s | 3.9927s |
| `chase1zhang/youtube-emotion-distilbert` | CPU | 455 | 0.7604 | 30.7984s | 25.9997s |
| `chase1zhang/youtube-emotion-distilbert` | GPU | 455 | 0.7604 | 3.2396s | 2.2477s |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | CPU | 455 | 0.7363 | 57.5298s | 50.5400s |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | GPU | 455 | 0.7363 | 6.1080s | 4.2489s |

### 11.2 Application Performance on Streamlit Cloud

The deployed app is tested using three YouTube videos. For each video, the first 50 comments are manually labeled comment by comment. The app performance score is calculated as:

```text
Accuracy = number of comments matching the manual label / 50 comments
```

The seven-emotion task compares the pre-tuning baseline model with the fine-tuned model. The three-class sentiment task compares the supporting sentiment pipeline with manual sentiment labels.

**App testing results:**

| Video | Task | Model Stage | Matched / 50 | Accuracy | Manual Main Label | Model Main Label |
|---|---|---|---:|---:|---|---|
| https://www.youtube.com/watch?v=d2dgJGkw5p0 | 7-emotion | Pre-tuning baseline | 31 | 0.6200 | neutral | neutral |
| https://www.youtube.com/watch?v=d2dgJGkw5p0 | 7-emotion | Fine-tuned | 32 | 0.6400 | neutral | neutral |
| https://www.youtube.com/watch?v=d2dgJGkw5p0 | 3-sentiment | Supporting pipeline | 31 | 0.6200 | neutral | neutral |
| https://www.youtube.com/watch?v=M8To7iorkxQ | 7-emotion | Pre-tuning baseline | 28 | 0.5600 | joy | neutral |
| https://www.youtube.com/watch?v=M8To7iorkxQ | 7-emotion | Fine-tuned | 24 | 0.4800 | joy | neutral |
| https://www.youtube.com/watch?v=M8To7iorkxQ | 3-sentiment | Supporting pipeline | 44 | 0.8800 | positive | positive |
| https://www.youtube.com/watch?v=-_-eIVAX1yQ | 7-emotion | Pre-tuning baseline | 14 | 0.2800 | anger | neutral |
| https://www.youtube.com/watch?v=-_-eIVAX1yQ | 7-emotion | Fine-tuned | 17 | 0.3400 | anger | neutral |
| https://www.youtube.com/watch?v=-_-eIVAX1yQ | 3-sentiment | Supporting pipeline | 33 | 0.6600 | negative | negative |

**Overall app performance:**

| Task | Model / Pipeline | Matched / Total | Accuracy |
|---|---|---:|---:|
| 7-emotion | Pre-tuning baseline | 73 / 150 | 0.4867 |
| 7-emotion | Fine-tuned model | 73 / 150 | 0.4867 |
| 3-sentiment | Supporting sentiment pipeline | 108 / 150 | 0.7200 |

Manual review rationale: the rare-earths video is mainly informational and neutral; the Avatar video is mostly joyful/positive because many comments include praise, hearts, and "amazing" language; the shooting news video contains anger/negative reactions because many comments include hostility, blame, insults, or violent sarcasm.

### 11.3 Key Findings

- The fine-tuned model improves accuracy compared with the pre-trained emotion baseline on the seven-emotion task, increasing accuracy from 0.7033 to 0.7604.
- GPU inference is much faster than CPU inference for all tested models, while CPU runtime remains acceptable for the Streamlit Cloud app because each app run analyzes only the first 50 comments.
- On the Streamlit Cloud app test, the fine-tuned seven-emotion model and the pre-tuning baseline both achieved 73/150 comment-level matches overall. The fine-tuned model improved slightly on the rare-earths and shooting-news videos, but performed worse on the Avatar video because many short positive comments were predicted as neutral.
- The three-class sentiment pipeline performed more consistently, achieving 108/150 comment-level matches overall. This suggests that broad sentiment is easier and more stable than fine-grained seven-emotion classification on short, multilingual, and sarcastic YouTube comments.

## 12. Conclusion

This project addresses the business problem of manually analyzing YouTube audience feedback for digital marketing campaigns. The deployed application uses deep learning to classify YouTube comments into seven emotions and summarizes the main audience reaction automatically.

The application helps InsightWave Digital Marketing Agency evaluate campaign effectiveness by identifying whether viewers mainly express joy, surprise, neutrality, sadness, anger, fear, or disgust. Key business insights include:

- **Positive signal:** If joy and surprise dominate, the agency can recommend continuing the current content strategy.
- **Risk alert:** If the negative emotion ratio (anger + disgust + fear + sadness) exceeds 40%, the agency should review comments and adjust messaging, content style, or campaign targeting.
- **Engagement gap:** If neutral dominates, the agency may improve the hook, call to action, or storytelling.

The final Streamlit Cloud app provides a practical, repeatable workflow:

1. Input a YouTube video URL.
2. Automatically collect the first 50 comments.
3. Run seven-emotion and sentiment analysis through two Hugging Face pipelines.
4. Display results in interactive charts and sortable tables.
5. Generate actionable business recommendations.
6. Export results as CSV for further analysis.

Overall, the project demonstrates how Hugging Face transformer models can support data-driven marketing decisions in a real business application. The modular code structure (separate core logic, model runner, and API client modules) makes the application maintainable and testable.

## 13. Submission Checklist

- [ ] Project report PDF (this document, <10 pages)
- [ ] Fine-tuning Colab notebook (`notebooks/fine_tune_go_emotions_distilbert.ipynb`)
- [ ] Testing and experiments Colab notebook (`notebooks/testing_experiments.ipynb`)
- [x] Streamlit app files (`streamlit_app.py`, `youtube_emotion/`, `requirements.txt`)
- [x] Dataset files (`data/go_emotions_7class/`)
- [x] Fine-tuned model files (`fine_tuned_model_files/youtube-emotion-distilbert/`)
- [ ] Experimental results Excel file (`Experimental_results.xlsx`)
- [x] GitHub repository URL: https://github.com/chasezhang1999/youtube-emotion-analyzer
- [x] Hugging Face model URL: https://huggingface.co/chase1zhang/youtube-emotion-distilbert
- [x] Streamlit Cloud app URL: https://youtube-emotion-analyzer.streamlit.app/
- [ ] PPT presentation
- [ ] MP4 presentation video
