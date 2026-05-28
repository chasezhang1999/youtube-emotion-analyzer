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
- YouTube-domain raw pool prepared: `data/youtube_domain_training_comments_8000_deepseek_labeled.csv` with 8,000 comments from 71 videos
- YouTube-domain adaptation dataset prepared: `data/youtube_domain_7class_deepseek/` with 3,991 balanced comments
- Manual app testing dataset prepared: `experiments/app_per_comment_manual_labels.csv`
- Experimental results workbook prepared: `experiments/Experimental_results.xlsx`
- Performance result workbook prepared: `experiments/Performance_result.xlsx`
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
fine_tune_all_models.ipynb
testing_experiments.ipynb
```

Use these project files:

```text
notebooks/fine_tune_all_models.ipynb
notebooks/testing_experiments.ipynb
```

Need to finish:

- Open the required notebooks in Google Colab.
- Runtime: T4 GPU.
- Run `Runtime -> Restart & Run All`.
- Keep all outputs visible.
- Make sure notebook numbers match `experiments/Experimental_results.xlsx` and the report.
- Refreshed model has been uploaded and evaluated; latest model revisions are recorded in `experiments/model_revision_summary.csv`.
- Use `notebooks/fine_tune_all_models.ipynb` for the next fair-comparison experiment: YouTube-domain fine-tuning for every seven-emotion baseline model.

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
data/youtube_domain_7class_deepseek/all.csv
data/youtube_domain_7class_deepseek/train.csv
data/youtube_domain_7class_deepseek/validation.csv
data/youtube_domain_training_comments_8000_deepseek_labeled.csv
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
- `YouTube_Domain_Validation`
- `Streamlit_App_Performance`
- `App_5Model_Comparison`
- `App_Runtime`
- `Model_Revisions`
- `Manual_Comment_Labels`

Key numbers:

- GoEmotions fine-tuned DistilBERT: 0.7030 on the expanded 1,000-sample GoEmotions test benchmark.
- Domain-adapted model: 0.6650 on the refreshed 1,000-sample YouTube-domain validation split.
- Streamlit app seven-emotion manual benchmark:
  - pre-tuning public baseline: 67/150 = 0.4467
  - GoEmotions fine-tuned DistilBERT: 70/150 = 0.4667
  - YouTube-domain adapted DistilBERT: 71/150 = 0.4733
  - public SamLowe GoEmotions RoBERTa: 80/150 = 0.5333
  - public RoBERTa-large seven-emotion: 62/150 = 0.4133
- Streamlit app three-class sentiment benchmark: 105/150 = 0.7000
- Manual label review: 23 of 150 labels revised after assistant review.

Dataset refresh note:

- The YouTube-domain dataset has now been rebuilt to 5,000 selected comments from an 8,000-comment pool.
- The validation metrics above have been refreshed on the 1,000-sample YouTube-domain validation split (resulting in 0.6650 accuracy for our domain-adapted model).
- Optional: Rerun fine-tuning on Colab if you want to perform multi-model domain adaptation.

### 6. Performance Result Excel

File prepared:

```text
experiments/Performance_result.xlsx
```

Original course template location:

```text
../Performance_result.xlsx
```

Submit as:

```text
GroupXX_documentation/Performance_result.xlsx
```

Workbook contents:

- Pipeline 1: seven-emotion classification, marked as the pipeline requiring fine-tuning.
- Step 1: comparable model selection on the 1,000-row GoEmotions test split.
- Step 2a: YouTube-domain validation performance on the refreshed 1,000-sample validation split (our model achieved 0.6650 accuracy).
- Step 2b: current implementation model selection and caveat about SamLowe's app benchmark strength.
- Step 3: deployed app benchmark on 150 manually reviewed YouTube comments.
- Pipeline 2: CardiffNLP three-class sentiment pipeline as supporting signal, no fine-tuning required.

Before final packaging:

- Copy or export the final project workbook to `../Performance_result.xlsx` if the course template file in the Assignment2 root must be the submitted file.
- Confirm the numbers match `experiments/Experimental_results.xlsx`, the report, and the PPT.

### 7. Presentation PPT

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

### 8. MP4 Presentation Video

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
|-- Experimental_results.xlsx
`-- Performance_result.xlsx

GroupXX_program/
|-- Python_notebooks/
|   |-- fine_tune_all_models.ipynb
|   `-- testing_experiments.ipynb
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
3. Open the required notebooks in Colab and save final outputs.
4. Verify `experiments/Performance_result.xlsx` against the final report numbers.
5. Build the PPT from `docs/presentation_outline.md`.
6. Record the MP4 presentation.
7. Package the final zip using the required folder names.
