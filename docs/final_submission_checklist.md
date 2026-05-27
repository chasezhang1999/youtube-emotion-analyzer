# ISOM5240 Final Submission Checklist

This checklist is based on `ISOM5240_project_requirements.pdf` and the current YouTube emotion analyzer project status.

## Completed

- Streamlit app built: `streamlit_app.py`
- GitHub repository uploaded: https://github.com/chasezhang1999/youtube-emotion-analyzer
- Streamlit Cloud app deployed: https://youtube-emotion-analyzer.streamlit.app/
- Original Hugging Face fine-tuned model uploaded: https://huggingface.co/chase1zhang/youtube-emotion-distilbert
- Domain-adapted Hugging Face model uploaded: https://huggingface.co/chase1zhang/youtube-emotion-distilbert-domain-adapted
- Domain-adapted model files downloaded locally: `fine_tuned_model_files/youtube-emotion-distilbert-domain-adapted/`
- App default model updated to the domain-adapted model
- Model selector added to the Streamlit sidebar for comparison
- Three-model Streamlit comparison mode added:
  - `chase1zhang/youtube-emotion-distilbert-domain-adapted`
  - `SamLowe/roberta-base-go_emotions`
  - `j-hartmann/emotion-english-distilroberta-base`
- Additional optional comparison models remain available:
  - `chase1zhang/youtube-emotion-distilbert`
  - `j-hartmann/emotion-english-roberta-large`
- App comment retrieval expanded to 100 comments per video
- GoEmotions seven-class dataset prepared: `data/go_emotions_7class/`
- YouTube-domain adaptation dataset prepared: `data/youtube_domain_7class_assistant/` with 2,962 comments from 30 videos
- Manual app testing dataset prepared: `experiments/app_per_comment_manual_labels.csv`
- Experimental results workbook prepared: `experiments/Experimental_results.xlsx`
- Report screenshots captured: `docs/report/screenshots/`
- Draft report PDF generated: `docs/report/Project_report.pdf`
- Unit tests pass locally

## Still Required Before Canvas Submission

### 1. Project Report PDF

Current draft:

```text
docs/report/Project_report_skeleton.md
```

Submit as:

```text
GroupXX_documentation/Project_report.pdf
```

Need to finish:

- Fill in student names and IDs.
- Review the inserted screenshots and replace them if you prefer screenshots from Streamlit Cloud instead of local Streamlit.
- After filling names, re-export the report as PDF.
- Keep the final PDF under 10 pages.

### 2. Python Notebooks

Folder to submit:

```text
GroupXX_program/Python_notebooks/
```

Files:

```text
Fine_tune_Model.ipynb
Testing_Experiments.ipynb
```

Use these project files:

```text
notebooks/Fine_tune_Model.ipynb
notebooks/testing_experiments.ipynb
```

Need to finish:

- Open both notebooks in Google Colab.
- Runtime: T4 GPU.
- Run `Runtime -> Restart & Run All`.
- Keep all outputs visible.
- Make sure notebook numbers match `experiments/Experimental_results.xlsx` and the report.
- Rerun the domain-adaptation section after pushing the expanded 2,962-comment dataset, then upload the refreshed model to `chase1zhang/youtube-emotion-distilbert-domain-adapted`.

### 3. GitHub App Files

Folder to submit:

```text
GroupXX_program/GitHub_App_Files/
```

Include:

```text
streamlit_app.py
requirements.txt
.streamlit/config.toml
.streamlit/secrets.example.toml
youtube_emotion/
data/sample_comments.csv
README.md
```

Do not include:

```text
.streamlit/secrets.toml
.venv/
__pycache__/
.playwright-cli/
```

### 4. Dataset and Model Files

Folder to submit:

```text
GroupXX_Dataset_files/
```

Include:

```text
data/go_emotions_7class/train.csv
data/go_emotions_7class/validation.csv
data/go_emotions_7class/test.csv
data/youtube_domain_7class_assistant/all.csv
data/youtube_domain_7class_assistant/train.csv
data/youtube_domain_7class_assistant/validation.csv
data/sample_comments.csv
fine_tuned_model_files/youtube-emotion-distilbert/
fine_tuned_model_files/youtube-emotion-distilbert-domain-adapted/
```

Recommended note in the report:

```text
The YouTube-domain labels are assistant-assisted annotations used for domain adaptation. The independent app performance test uses a separate manually reviewed 150-comment benchmark.
```

### 5. Experimental Results Excel

File prepared:

```text
experiments/Experimental_results.xlsx
```

Submit as:

```text
GroupXX_documentation/Experimental_results.xlsx
```

Workbook sheets:

- `Model_Selection`
- `Streamlit_App_Performance`
- `App_Runtime`
- `Manual_Comment_Labels`
- `Per_Comment_Predictions`
- `Label_Review_Summary`
- `YouTube_Domain_Validation`

Key numbers:

- GoEmotions fine-tuned DistilBERT: 0.7604 on the historical 455-sample GoEmotions test benchmark. Refresh this after retraining/evaluating on the expanded 1,000-sample test split.
- Domain-adapted model: 0.6400 on the earlier 200-sample YouTube-domain validation split; refresh this after rerunning on the expanded 592-sample validation split.
- Streamlit app seven-emotion manual benchmark:
  - pre-tuning public baseline: 67/150 = 0.4467
  - GoEmotions fine-tuned DistilBERT: 70/150 = 0.4667
  - YouTube-domain adapted DistilBERT: 65/150 = 0.4333
  - public SamLowe GoEmotions RoBERTa: 80/150 = 0.5333
  - public RoBERTa-large seven-emotion: 62/150 = 0.4133
- Streamlit app three-class sentiment benchmark: 105/150 = 0.7000
- Manual label review: 23 of 150 labels revised after assistant review.

### 6. Presentation PPT

File to submit:

```text
GroupXX_presentation/Presentation_slide.pptx
```

Use this outline:

```text
docs/presentation_outline.md
```

Suggested slide order:

1. Title and project objective
2. Business problem and target company scenario
3. Dataset and preprocessing
4. Model pipeline and app workflow
5. Fine-tuning and domain adaptation
6. Streamlit Cloud app demo
7. Experimental results
8. Business insights, limitations, and conclusion

### 7. MP4 Presentation Video

File to submit:

```text
GroupXX_presentation/grpXX.mp4
```

Requirements:

- MP4 format
- Group members' faces visible
- Include app demo
- Keep within 10 minutes
- File name must use two-digit group number, e.g. `grp01.mp4`

## Final Zip Structure

Final zip name:

```text
GroupXX_STUDENTID.zip
```

Recommended structure:

```text
GroupXX_documentation/
|-- Project_report.pdf
`-- Experimental_results.xlsx

GroupXX_program/
|-- Python_notebooks/
|   |-- Fine_tune_Model.ipynb
|   `-- Testing_Experiments.ipynb
`-- GitHub_App_Files/
    |-- streamlit_app.py
    |-- requirements.txt
    |-- README.md
    |-- .streamlit/
    |-- data/
    `-- youtube_emotion/

GroupXX_Dataset_files/
|-- data/
`-- fine_tuned_model_files/

GroupXX_presentation/
|-- Presentation_slide.pptx
`-- grpXX.mp4
```

Also paste the Streamlit Cloud app URL:

```text
https://youtube-emotion-analyzer.streamlit.app/
```

## Highest Priority Next Steps

1. Fill student names and IDs in the report.
2. Re-export the report to PDF after filling names.
3. Open both notebooks in Colab and save final outputs.
4. Build the PPT from `docs/presentation_outline.md`.
5. Record the MP4 presentation.
6. Package the final zip using the required folder names.
