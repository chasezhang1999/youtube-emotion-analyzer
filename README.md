# YouTube Audience Emotion Analyzer

ISOM5240 Group Project — Deep learning business application for a digital marketing agency.

## Business Scenario

InsightWave Digital Marketing Agency helps brands understand whether YouTube campaign videos generate the intended audience reaction. This app classifies the first 50 YouTube comments into seven emotions and provides actionable marketing recommendations.

## Project Status

✅ `youtube_emotion/` package fully built and tested (11 tests pass)  
✅ Fine-tuning Colab notebook ready  
✅ Testing/experiments Colab notebook ready  
✅ Streamlit app built  
✅ Dataset prepared (GoEmotions 7-class, balanced)  
✅ Project report drafted  

⏳ **Remaining: rerun Hugging Face upload verification, then deploy, run experiments, fill report numbers**

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

### Step 3: Update App with Fine-tuned Model

Edit `youtube_emotion/model_runner.py`:

```python
DEFAULT_EMOTION_MODEL = "chase1zhang/youtube-emotion-distilbert"
```

Only make this change after Section 13 of the fine-tuning notebook successfully loads the uploaded model from Hugging Face.

### Step 4: Create GitHub Repository and Deploy

```bash
# Initialize git and push
git init
git add .
git commit -m "Initial commit: YouTube emotion analyzer"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/youtube-emotion-analyzer.git
git push -u origin main
```

Then:
1. Go to https://streamlit.io/cloud
2. Connect your GitHub repo
3. Add secret: `YOUTUBE_API_KEY` = your YouTube Data API key
4. Deploy
5. Record the app URL: `https://your-app-name.streamlit.app`

### Step 5: Run Experiments

1. Open `notebooks/testing_experiments.ipynb` in Google Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Confirm the fine-tuned model is set to `chase1zhang/youtube-emotion-distilbert`
4. Runtime → **Restart & Run All**
5. Record all accuracy and runtime numbers
6. Download the generated CSV files

### Step 6: Fill Report Numbers

Fill in all `[Replace]` placeholders in `docs/report/Project_report_skeleton.md`:
- Model accuracy and runtime numbers from experiments
- App performance test results
- Hugging Face model URL, GitHub URL, Streamlit app URL
- Student names and IDs

### Step 7: Take App Screenshots (5 required)

1. Input page with YouTube URL
2. Main emotion and summary metrics
3. Emotion distribution bar chart
4. Comment-level prediction table
5. Marketing recommendation

### Step 8: Create PPT and Record MP4

- Presentation ≤ 10 minutes
- Faces must be visible in the video
- Demo the app workflow

### Step 9: Package and Submit

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
└── Fine-tuned_Model_files/

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
│   └── experimental_results_template.csv
├── docs/
│   ├── report/Project_report_skeleton.md
│   └── fine_tuning_steps.md
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
| Pipeline 1 | `text-classification` | Fine-tuned `distilbert-base-uncased` | Seven-emotion classification |
| Pipeline 2 | `sentiment-analysis` | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Supporting sentiment signal |

## Dataset

Source: `SetFit/go_emotions` on Hugging Face  
Filtered to 7 Ekman emotions, single-label, balanced  
Labels: anger, disgust, fear, joy, neutral, sadness, surprise

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
