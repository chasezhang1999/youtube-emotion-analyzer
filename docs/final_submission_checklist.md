# ISOM5240 Final Submission Checklist

This checklist is based on `ISOM5240_project_requirements.pdf` and the current YouTube emotion analyzer project status.

## Completed

- Streamlit app built: `streamlit_app.py`
- GitHub repository uploaded: https://github.com/chasezhang1999/youtube-emotion-analyzer
- Streamlit Cloud app deployed: https://youtube-emotion-analyzer.streamlit.app/
- Hugging Face fine-tuned model uploaded: https://huggingface.co/chase1zhang/youtube-emotion-distilbert
- Fine-tuned model files downloaded locally: `fine_tuned_model_files/youtube-emotion-distilbert/`
- Dataset prepared: `data/go_emotions_7class/`
- App tested with sample comments and a real YouTube video URL
- Unit tests pass locally

## Still Required

### 1. Project Report PDF

File to prepare:

```text
GroupXX_documentation/Project_report.pdf
```

Need to finish:

- Replace student names and IDs in `docs/report/Project_report_skeleton.md`
- Keep the report under 10 pages after exporting to PDF
- Add 5 Streamlit Cloud screenshots:
  - input page with YouTube URL
  - summary metrics
  - seven-emotion distribution chart
  - comment-level result table
  - marketing recommendation
- Fill experiment tables with real accuracy and runtime numbers
- Add final app testing table using 3 to 5 YouTube videos
- Export the final report as PDF

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

Need to finish:

- Open both notebooks in Google Colab
- Runtime: T4 GPU
- Run `Runtime -> Restart & Run All`
- Keep all outputs visible in the notebooks
- Make sure the reported numbers match the project report

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

### 4. Dataset Files

Folder to submit:

```text
GroupXX_Dataset_files/
```

Include:

```text
data/go_emotions_7class/train.csv
data/go_emotions_7class/validation.csv
data/go_emotions_7class/test.csv
data/sample_comments.csv
fine_tuned_model_files/youtube-emotion-distilbert/
```

### 5. Experimental Results Excel

File to prepare:

```text
GroupXX_documentation/Experimental_results.xlsx
```

Must include:

- Model selection results
- Accuracy
- Runtime with model loading
- Runtime without model loading
- CPU vs GPU comparison
- Streamlit Cloud app testing accuracy
- Reasonable sample size, not just one or two comments

Recommended sheets:

- `Model_Selection`
- `Streamlit_App_Performance`
- `Manual_Comment_Labels`
- `Notes`

### 6. Presentation PPT

File to prepare:

```text
GroupXX_presentation/Presentation_slide.pptx
```

Suggested slide order:

1. Title, team, project objective
2. Business problem: digital marketing agency needs scalable YouTube feedback analysis
3. Dataset and preprocessing
4. Model pipeline: YouTube API -> emotion pipeline -> sentiment pipeline -> dashboard
5. Fine-tuning setup and model selection
6. Streamlit Cloud app demo screenshots
7. Experimental results
8. Business insights and conclusion

Keep the presentation under 10 minutes.

### 7. MP4 Presentation Video

File to prepare:

```text
GroupXX_presentation/grpXX.mp4
```

Requirements:

- MP4 format
- Group members' faces visible
- Include app demo
- Keep within 10 minutes
- File name must use two-digit group number, e.g. `grp01.mp4`

### 8. Final Zip

Final zip name:

```text
GroupXX_STUDENTID.zip
```

Recommended structure:

```text
GroupXX_documentation/
├── Project_report.pdf
└── Experimental_results.xlsx

GroupXX_program/
├── Python_notebooks/
│   ├── Fine_tune_Model.ipynb
│   └── Testing_Experiments.ipynb
└── GitHub_App_Files/
    ├── streamlit_app.py
    ├── requirements.txt
    ├── README.md
    ├── .streamlit/
    ├── data/
    └── youtube_emotion/

GroupXX_Dataset_files/
├── data/
└── fine_tuned_model_files/

GroupXX_presentation/
├── Presentation_slide.pptx
└── grpXX.mp4
```

Also submit or paste:

```text
Streamlit Cloud App URL:
https://youtube-emotion-analyzer.streamlit.app/
```

## Highest Priority Next Steps

1. Run `notebooks/testing_experiments.ipynb` in Colab with T4 GPU and save the outputs.
2. Fill the experiment numbers in `docs/report/Project_report_skeleton.md`.
3. Take the 5 app screenshots from the deployed Streamlit Cloud app.
4. Export the final report to PDF.
5. Build the PPT and record the MP4.
6. Package the Canvas zip with the required folder names.
