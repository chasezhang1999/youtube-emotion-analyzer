# Fine-Tuning Plan for YouTube Audience Emotion Analyzer

## Current Models in the App

The current Streamlit app uses two Hugging Face text classification models.

1. Seven-emotion model:
   - default: `chase1zhang/youtube-emotion-distilbert-domain-adapted`
   - comparison options:
     - `chase1zhang/youtube-emotion-distilbert`
     - `SamLowe/roberta-base-go_emotions`
     - `j-hartmann/emotion-english-distilroberta-base`
     - `j-hartmann/emotion-english-roberta-large`
   - Used for: anger, disgust, fear, joy, neutral, sadness, surprise
   - Role: main audience emotion analysis pipeline

2. Sentiment model:
   - `cardiffnlp/twitter-roberta-base-sentiment-latest`
   - Used for: negative, neutral, positive
   - Role: supporting business signal for campaign interpretation

## Why Fine-Tune

The app already works with pre-trained models, but the course project requires a fine-tuned model. Fine-tuning also improves project quality because:

- It creates a model customized for our seven-emotion objective.
- It provides a model URL for the report.
- It gives us experimental results to compare against the pre-trained baseline.
- It makes the project less like a direct model demo and more like a business application.

## Dataset Search Process

1. Search Hugging Face Datasets for emotion classification datasets.
2. Prefer datasets with English short text because YouTube comments are usually short and informal.
3. Check whether the dataset has the target labels:
   - anger
   - disgust
   - fear
   - joy
   - neutral
   - sadness
   - surprise
4. Use the Hugging Face Dataset Viewer API to inspect:
   - whether the dataset is valid
   - train, validation, and test splits
   - row count
   - column names
   - label distribution
   - first rows
5. Reject datasets with suspicious quality.

## Dataset Decision

Rejected candidate:

- `Frankhihi/goemotion-ekman-emotions`
- Reason: It has seven labels, but only 10 unique text values in 10,000 rows. This is not suitable for serious fine-tuning.

Selected dataset:

- `SetFit/go_emotions`
- URL: https://huggingface.co/datasets/SetFit/go_emotions
- Reason: It is based on GoEmotions, has 54,263 rows, and contains short English text with emotion annotations.

## Prepared Dataset

The raw `SetFit/go_emotions` dataset has 28 one-hot emotion columns. Our project needs only seven emotions, so the preparation script keeps only clean single-label rows where exactly one label is active and that label is one of our seven target emotions.

The script then balances all seven emotion classes by downsampling to the smallest class size.

Prepared files:

- `data/go_emotions_7class/train.csv`
- `data/go_emotions_7class/validation.csv`
- `data/go_emotions_7class/test.csv`

Prepared split sizes:

| Split | Total rows | Rows per class |
|---|---:|---:|
| Train | 5,000 | Stratified; capped by available minority-class rows |
| Validation | 406 | 58 |
| Test | 1,000 | Stratified; capped by available minority-class rows |

## YouTube-Domain Adaptation Dataset

The original GoEmotions fine-tuned model was further adapted with YouTube-domain comments because YouTube comments are different from the original GoEmotions text style.

Prepared files:

- `data/youtube_domain_7class_deepseek/all.csv`
- `data/youtube_domain_7class_deepseek/train.csv`
- `data/youtube_domain_7class_deepseek/validation.csv`

Dataset summary:

- 8,000 DeepSeek AI-labeled YouTube comments in the raw pool
- 3,991 balanced comments in the final adaptation dataset
- 71 videos
- balanced as far as possible without duplicating minority-class comments
- seven target emotions
- assistant-assisted labels
- used only for domain adaptation
- independent app evaluation uses a separate manually reviewed 150-comment set

Current split sizes:

| Split | Rows |
|---|---:|
| Train | 3,193 |
| Validation | 798 |
| All | 3,991 |

## Fine-Tuning Steps

1. Open Colab and enable GPU.
2. Upload or copy the project notebook:
   - `notebooks/fine_tune_all_models.ipynb`
3. Install training dependencies.
4. Load the prepared CSV files.
5. Load `distilbert-base-uncased` tokenizer and model.
6. Tokenize the text column.
7. Train with `Trainer`.
8. Evaluate on validation and test sets.
9. Save the model and tokenizer.
10. Save and test the original GoEmotions fine-tuned model.
11. Continue training on the YouTube-domain adaptation dataset.
12. Save and test the domain-adapted model.
13. Push both model versions to Hugging Face.
14. Verify the uploaded models with `pipeline("text-classification")`.
15. Update the Streamlit app default model to the domain-adapted model.
16. Record accuracy and runtime in `experiments/Experimental_results.xlsx`.

## Recommended Hyperparameters

Use conservative settings so the notebook runs reliably in Colab:

- Base model: `distilbert-base-uncased`
- Epochs: 3
- Learning rate: `2e-5`
- Batch size: 16
- Max sequence length: 128
- Evaluation strategy: each epoch
- Save best model: yes

## Optimization Ideas

1. Use a balanced training set to avoid over-predicting neutral.
2. Compare the fine-tuned DistilBERT model against the pre-trained emotion model.
3. Compare against public emotion classifiers such as `SamLowe/roberta-base-go_emotions` and `j-hartmann/emotion-english-distilroberta-base`.
4. Measure runtime with model loading and without model loading.
5. Keep the final app model small enough for Streamlit Cloud.
6. Cache model pipelines in Streamlit using `st.cache_resource`.
7. If accuracy is weak, try:
   - more epochs
   - lower learning rate
   - `j-hartmann/emotion-english-distilroberta-base` as the base model
   - more training examples per class if using a non-balanced dataset
   - more manually verified YouTube comments for anger, fear, sadness, and disgust

## Commands Already Executed Locally

Prepare the seven-class dataset:

```bash
.venv/bin/python scripts/prepare_go_emotions_7class.py
```

Run tests:

```bash
.venv/bin/python -m unittest discover tests
```

## Important Upload Check

The Hugging Face model repository must contain model weights. Check the repo files after upload. A valid uploaded model should include files such as:

- `config.json`
- `model.safetensors` or `pytorch_model.bin`
- `tokenizer.json`
- `tokenizer_config.json`

If the repo only contains tokenizer files, rerun the upload section in the notebook using:

```python
repo_name = "chase1zhang/youtube-emotion-distilbert"
trainer.model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)
```

For the domain-adapted model, use:

```python
domain_repo_name = "chase1zhang/youtube-emotion-distilbert-domain-adapted"
domain_trainer.model.push_to_hub(domain_repo_name)
tokenizer.push_to_hub(domain_repo_name)
```
