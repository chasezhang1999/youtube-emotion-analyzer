# YouTube Audience Emotion Analyzer

ISOM5240 Group Project — Deep learning business application for a digital marketing agency.

## Business Scenario

InsightWave Digital Marketing Agency helps brands understand whether YouTube campaign videos generate the intended audience reaction. This app classifies the first 50 YouTube comments into seven emotions and provides actionable marketing recommendations.

## Project Status

- ✅ `youtube_emotion/` package fully built and tested (15 tests pass)
- ✅ Fine-tuning Colab notebook ready
- ✅ Testing/experiments Colab notebook ready
- ✅ Streamlit app built
- ✅ Dataset prepared (GoEmotions 7-class + YouTube-domain adaptation set)
- ✅ Project report drafted
- ✅ Original and domain-adapted Hugging Face models uploaded and verified
- ✅ GitHub repository pushed
- ✅ Streamlit Cloud app deployed and tested
- ✅ Experimental results workbook prepared
- ✅ Report screenshots captured
- ✅ Draft report PDF generated
- ✅ Three-model Streamlit comparison mode added

⏳ **Remaining: fill student names, re-export PDF, save final Colab notebook outputs, create PPT, record MP4, package Canvas submission**

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python -m unittest discover tests -v

# Run the Streamlit app locally
streamlit run streamlit_app.py
```

For local testing without a YouTube API key, check "Use sample comments" in the sidebar.

## Execution Steps (in order)

### Step 1: Run Fine-tuning on Google Colab

1. Open `notebooks/fine_tune_go_emotions_distilbert.ipynb` in Google Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Runtime → **Restart & Run All** (verify it runs end-to-end)
4. Keep all cell outputs visible (TA will verify this)

### Step 2: Upload Model to Hugging Face

1. Create a Hugging Face account at https://huggingface.co
2. Get a write token: Settings → Access Tokens
3. In Colab, add the token as a secret named `HF_TOKEN`
4. Run Section 12 to push the model and tokenizer to Hugging Face
5. Run Section 13 to verify the uploaded model can be loaded by `pipeline()`
6. Record the model URL: `https://huggingface.co/chase1zhang/youtube-emotion-distilbert`
7. Optional domain-adaptation upload URL: `https://huggingface.co/chase1zhang/youtube-emotion-distilbert-domain-adapted`

Status: completed and verified with `pipeline("text-classification")`.

### Step 3: Update App with Fine-tuned Model

Edit `youtube_emotion/model_runner.py`:

```python
DEFAULT_EMOTION_MODEL = "chase1zhang/youtube-emotion-distilbert-domain-adapted"
```

The original fine-tuned model is still available in the Streamlit sidebar model selector for comparison.

### Step 4: Create GitHub Repository and Deploy

```bash
# Initialize git and push
git init
git add .
git commit -m "Initial commit: YouTube emotion analyzer"
git branch -M main
git remote add origin https://github.com/chasezhang1999/youtube-emotion-analyzer.git
git push -u origin main
```

Then:
1. Go to https://streamlit.io/cloud
2. Connect your GitHub repo
3. Add secret: `YOUTUBE_API_KEY` = your YouTube Data API key
4. Deploy
5. Record the app URL: `https://youtube-emotion-analyzer.streamlit.app/`

Status: completed. GitHub repository: `https://github.com/chasezhang1999/youtube-emotion-analyzer`

### Step 5: Run Experiments

1. Open `notebooks/testing_experiments.ipynb` in Google Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Confirm the app default emotion model is set to `chase1zhang/youtube-emotion-distilbert-domain-adapted`
4. Runtime → **Restart & Run All**
5. Record all accuracy and runtime numbers
6. Download the generated CSV files
7. Convert the final results table to `Experimental_results.xlsx`

### Step 6: Fill Report Numbers

The report draft is now filled with model URLs, experiment numbers, and app testing results:

```text
docs/report/Project_report_skeleton.md
```

Only the student names/IDs need to be filled before re-exporting the final PDF.

### Step 7: Streamlit Model Comparison

The app now supports two analysis modes:

- Single model: run one selected emotion model.
- Compare emotion models: run the same comments through three fine-tuned emotion models.

Default comparison models:

- `chase1zhang/youtube-emotion-distilbert-domain-adapted`
- `chase1zhang/youtube-emotion-distilbert`
- `SamLowe/roberta-base-go_emotions`

### Step 8: App Screenshots

The required screenshots are saved in:

```text
docs/report/screenshots/
```

Replace them only if you want screenshots from Streamlit Cloud instead of local Streamlit.

### Step 9: Create PPT and Record MP4

- Presentation ≤ 10 minutes
- Faces must be visible in the video
- Demo the app workflow

### Step 10: Package and Submit

Organize files per Canvas submission structure:

```
GroupXX_documentation/
├── Project_report.pdf
└── Experimental_results.xlsx

GroupXX_program/
├── Python_notebooks/
│   ├── Fine_tune_Model.ipynb
│   └── Testing_Experiments.ipynb
├── GitHub_App_Files/
│   ├── streamlit_app.py
│   ├── requirements.txt
│   ├── youtube_emotion/
│   └── ...

GroupXX_Dataset_files/
├── data/go_emotions_7class/
├── data/youtube_domain_7class_assistant/
└── Fine-tuned_Model_files/
    ├── youtube-emotion-distilbert/
    └── youtube-emotion-distilbert-domain-adapted/

GroupXX_presentation/
├── Presentation_slide.pptx
└── grpXX.mp4
```

## Project Structure

```text
youtube_emotion_project/
├── streamlit_app.py                     # Main Streamlit application
├── youtube_emotion/                     # Core Python package
│   ├── __init__.py
│   ├── core.py                          # URL parsing, labels, summary, recommendations
│   ├── model_runner.py                  # Hugging Face pipeline loading & prediction
│   ├── youtube_client.py                # YouTube Data API comment fetching
│   └── dataset_prep.py                  # GoEmotions dataset preparation
├── data/
│   ├── go_emotions_7class/              # Balanced 7-class dataset
│   │   ├── train.csv                    (3,010 samples)
│   │   ├── validation.csv               (406 samples)
│   │   └── test.csv                     (455 samples)
│   ├── youtube_domain_7class_assistant/ # YouTube-domain adaptation dataset
│   └── sample_comments.csv              # Demo comments (no API key needed)
├── notebooks/
│   ├── fine_tune_go_emotions_distilbert.ipynb   # Colab fine-tuning
│   └── testing_experiments.ipynb                 # Colab experiments
├── tests/
│   ├── test_core.py
│   ├── test_model_runner.py
│   ├── test_youtube_client.py
│   └── test_dataset_prep.py
├── experiments/
│   ├── Experimental_results.xlsx
│   ├── app_performance_model_comparison.csv
│   ├── app_per_comment_manual_labels.csv
│   ├── app_runtime_summary.csv
│   └── youtube_domain_validation_performance.csv
├── docs/
│   ├── report/Project_report_skeleton.md
│   ├── final_submission_checklist.md
│   ├── fine_tuning_steps.md
│   ├── presentation_outline.md
│   └── youtube_domain_annotation_guide.md
├── scripts/
│   └── prepare_go_emotions_7class.py
├── requirements.txt                     # Streamlit Cloud dependencies
├── requirements_train.txt               # Colab fine-tuning dependencies
├── .streamlit/
│   ├── config.toml
│   └── secrets.example.toml
└── README.md
```

## Models and Pipelines

| Pipeline | Task | Model | Purpose |
|---|---|---|---|
| Pipeline 1 | `text-classification` | Domain-adapted fine-tuned `distilbert-base-uncased` | Seven-emotion classification |
| Pipeline 2 | `sentiment-analysis` | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Supporting sentiment signal |

Additional Streamlit comparison model:

- `SamLowe/roberta-base-go_emotions`, a public GoEmotions fine-tuned RoBERTa model

## Dataset

Source: `SetFit/go_emotions` on Hugging Face  
Filtered to 7 Ekman emotions, single-label, balanced  
Labels: anger, disgust, fear, joy, neutral, sadness, surprise

Additional YouTube-domain adaptation data:

- 1,000 assistant-assisted labeled YouTube comments
- 20 videos
- Stored in `data/youtube_domain_7class_assistant/`
- Independent app evaluation uses a separate manually reviewed 150-comment benchmark

## YouTube API Key

Set via environment variable or Streamlit secrets:

```toml
# .streamlit/secrets.toml
YOUTUBE_API_KEY = "your_key_here"
```

## Requirements

- Python ≥ 3.10
- streamlit ≥ 1.36
- transformers ≥ 4.41
- torch ≥ 2.2
- pandas ≥ 2.2
- requests ≥ 2.31
