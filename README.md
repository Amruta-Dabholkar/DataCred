# DataCred — A "Nutrition Label" for Datasets

**Program:** IBM SkillsBuild Data Analytics with AI Academic Internship (BharatCares × AICTE)
**Author:** Amruta Dabholkar

**Live demo:** https://datacred.streamlit.app/

## Project Description

Before a dataset is used to train a model or drive a business decision, it
needs to be sanity-checked — but in practice this rarely happens
consistently. A dataset can look fine at a glance while being significantly
incomplete, out of date, pulled from an unreliable source, or statistically
different from the data a model was originally trained on.

**DataCred** builds a single 0–100 trust score, backed by four transparent,
explainable sub-checks, that lets an analyst sanity-check a dataset in
seconds — the same idea as a nutrition label: a fast, standardized way to
check something before you consume it.

**The four checks:**
| Check | What it measures | Method |
|---|---|---|
| Completeness | How much of the dataset is populated | % non-null cells, per column |
| Freshness | How current the data is | Age of newest record vs. a decay curve |
| Source reliability | How trusted the data's origin is | Configurable lookup table |
| Drift | Whether the data's distribution has shifted | Two-sample Kolmogorov–Smirnov test vs. a historical baseline |

The four scores combine into one weighted overall trust score with a plain-English explanation.

## Project Structure

```
.
├── Amruta_DataCred.ipynb   # original analysis notebook (source of truth for the scoring logic)
├── Amruta_ProjectReport.docx
├── app.py                  # Streamlit dashboard version of the same logic
├── requirements.txt
└── README.md
```

## Dataset

The notebook generates its own **synthetic retail sales dataset** (see
Section 2 of the notebook), designed to mimic a realistic e-commerce orders
table. Synthetic data was used deliberately so the project could inject
*known* data-quality problems — missing values, stale timestamps, and
distribution drift — and demonstrate DataCred correctly catching each one.
The same code works unchanged on any real CSV or database table; simply
replace the generated DataFrames with `pd.read_csv("your_file.csv")`
(the Streamlit app does exactly this — it lets you upload your own CSVs).

No external dataset download is required to run this project.

## Technologies Used

- **Python 3** — core language
- **pandas / NumPy** — data loading, cleaning, manipulation
- **SciPy** (`stats.ks_2samp`) — statistical drift detection
- **Matplotlib** — exploratory visualizations and score breakdown chart
- **Streamlit** — interactive dashboard / web app
- **Jupyter Notebook** — original project delivery format
- **IBM Bob** (VS Code AI coding agent) — used to scaffold and iterate on the code during development

## Run the notebook

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open the notebook:
   ```bash
   jupyter notebook Amruta_DataCred.ipynb
   ```
3. Run all cells top to bottom (**Kernel → Restart & Run All**). The notebook
   is fully self-contained — it generates its own sample data, runs all four
   checks, computes the overall trust score, and produces visualizations —
   no external files or API keys are required.

## Run the Streamlit app locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

This opens an interactive version of the same scoring pipeline in your
browser: pick the demo data or upload your own CSVs, adjust the check
weights live, and download the JSON result.

## Deploy the Streamlit app for free (Streamlit Community Cloud)

1. Push this repository to GitHub (public repo).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **New app**, select this repo/branch, and set the main file path
   to `app.py`.
4. Click **Deploy**. You'll get a permanent public URL like
   `https://<your-app-name>.streamlit.app`.
5. Paste that URL at the top of this README, in your resume, and on your
   LinkedIn projects section.

## Key Information / Results

Running the notebook (or the app, with demo data selected) produces:

```
OVERALL TRUST SCORE: 55.55 / 100 -- Low trust -- investigate before use

- Completeness is strong (97% of cells populated).
  Worst columns: customer_age (88% filled), region (95% filled), order_value (98% filled)
- Data is stale -- newest record is 385.4 days old. Treat with caution.
- Source 'production_database' has a configured trust rating of 95.0/100.
- Drift detected in 3/4 numeric columns vs. baseline: order_id, customer_age, order_value.
```

This confirms DataCred correctly identifies the exact issues deliberately
injected into the "current" dataset (staleness and distribution drift),
while correctly recognizing that completeness and source reliability were
both strong — demonstrating that the score is driven by real, explainable
signal rather than a single blunt metric.

## Possible Extensions

- Categorical drift detection (chi-squared test) alongside the numeric KS test
- Scheduled scoring with automated alerts when a score drops below a threshold
- A score-history log to track a dataset's trust score over time
- An optional AI-generated narrative summary layered on top of the deterministic score, for non-technical stakeholders
