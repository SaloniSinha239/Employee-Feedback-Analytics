# Employee Feedback Analytics

Sentiment analysis, LDA topic modelling, and K-Means workforce segmentation over employee
survey responses, with SQLite persistence and Excel/PowerPoint deliverables for
non-technical stakeholders.

## Pipeline

1. **Data** – synthetic HR survey dataset (500 responses, 5 departments) with comments,
   engagement, workload, salary satisfaction, and tenure (`src/employee_feedback/generate_data.py`).
2. **Sentiment** – VADER compound scores mapped to positive/neutral/negative labels,
   aggregated by department (`src/employee_feedback/sentiment.py`).
3. **Topics** – `CountVectorizer` + `LatentDirichletAllocation` (5 topics), dominant topic per
   comment and per-topic sentiment (`src/employee_feedback/topics.py`).
4. **Segmentation** – `StandardScaler` + K-Means over engagement, workload, salary
   satisfaction, tenure, and sentiment; k selected by silhouette score
   (`src/employee_feedback/segmentation.py`).
5. **Storage** – results written to `outputs/feedback.db` (SQLite), joinable via SQL
   (`src/employee_feedback/storage.py`).
6. **Reporting** – multi-sheet Excel workbook (`outputs/feedback_insights.xlsx`) and an
   auto-generated PowerPoint deck with charts (`outputs/feedback_insights.pptx`)
   (`src/employee_feedback/reporting.py`).

## Interactive app (Streamlit)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload your own survey CSV (required columns: `department`, `tenure_years`,
`engagement_score`, `workload_score`, `salary_satisfaction`, `comment`) or use the
synthetic sample, then explore sentiment, topics, and segments in the browser and
download the Excel/PowerPoint deliverables.

### Free deployment

- **Streamlit Community Cloud**: push this repo to GitHub, then at
  [share.streamlit.io](https://share.streamlit.io) deploy `app.py` — main file `app.py`,
  requirements auto-detected from `requirements.txt`.
- **Hugging Face Spaces**: create a Space with SDK `Streamlit`, point it at the same repo
  (add `app.py` + `requirements.txt`).

## Usage

```bash
pip install -r requirements.txt
python3 pipeline.py            # generate data + full run
python3 pipeline.py --skip-generate --n 1000   # re-run on existing data
```

## Outputs

| File | Contents |
|---|---|
| `data/survey.csv` | Raw survey responses |
| `outputs/feedback.db` | SQLite: `feedback`, `sentiment_results`, `topic_results`, `segment_results` |
| `outputs/feedback_insights.xlsx` | Overall, Sentiment by Dept, Topics, Segments sheets |
| `outputs/feedback_insights.pptx` | Stakeholder deck with charts and recommendations |
| `outputs/*.png` | Chart images |

## Project structure

```
pipeline.py                    # end-to-end runner
src/employee_feedback/
    generate_data.py           # synthetic data
    sentiment.py               # VADER sentiment
    topics.py                  # LDA topic modelling
    segmentation.py            # K-Means clustering
    storage.py                 # SQLite layer
    reporting.py               # Excel + PowerPoint exports
```
