"""
DataCred — A "Nutrition Label" for Datasets
Streamlit dashboard version.

This app wraps the exact same scoring logic developed in
Amruta_DataCred.ipynb (completeness, freshness, source reliability,
drift-vs-baseline) in an interactive UI. Upload your own baseline +
current CSVs, or click "Use demo data" to see it run on the same
synthetic retail dataset used in the notebook.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy for free:
    Push this repo to GitHub, then connect it at share.streamlit.io
    (Streamlit Community Cloud). Main file path: app.py
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

# ---------------------------------------------------------------------------
# Config (identical to the notebook)
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "completeness": 0.30,
    "freshness": 0.20,
    "source_reliability": 0.20,
    "drift": 0.30,
}

SOURCE_TRUST_TABLE = {
    "production_database": 95,
    "data_warehouse": 90,
    "verified_api": 85,
    "internal_etl_pipeline": 80,
    "partner_feed": 65,
    "manual_upload": 50,
    "third_party_export": 45,
    "unknown": 30,
}

FRESH_GRACE_DAYS = 7
FRESH_MAX_DAYS = 90

SCORE_BANDS = [
    (85, "High trust"),
    (65, "Moderate trust -- review before use"),
    (40, "Low trust -- investigate before use"),
    (0, "Do not trust -- likely unsafe for modeling"),
]


def label_for_score(score: float) -> str:
    for threshold, label in SCORE_BANDS:
        if score >= threshold:
            return label
    return SCORE_BANDS[-1][1]


# ---------------------------------------------------------------------------
# Synthetic demo data (same generators as the notebook)
# ---------------------------------------------------------------------------

def make_baseline(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    base_date = datetime(2025, 6, 1)
    return pd.DataFrame({
        "order_id": range(1, n + 1),
        "customer_age": rng.normal(loc=38, scale=12, size=n).clip(18, 90).round(0),
        "order_value": rng.gamma(shape=2.0, scale=45, size=n).round(2),
        "items_in_order": rng.poisson(lam=3, size=n) + 1,
        "region": rng.choice(["North", "South", "East", "West"], size=n, p=[0.3, 0.25, 0.25, 0.2]),
        "last_updated": [base_date - timedelta(days=int(x)) for x in rng.integers(0, 5, size=n)],
    })


def make_current(n=1800, seed=7):
    rng = np.random.default_rng(seed)
    base_date = datetime(2025, 6, 1) + timedelta(days=90)
    df = pd.DataFrame({
        "order_id": range(2001, 2001 + n),
        "customer_age": rng.normal(loc=47, scale=14, size=n).clip(18, 95).round(0),
        "order_value": rng.gamma(shape=1.3, scale=70, size=n).round(2),
        "items_in_order": rng.poisson(lam=3, size=n) + 1,
        "region": rng.choice(["North", "South", "East", "West"], size=n, p=[0.3, 0.25, 0.25, 0.2]),
        "last_updated": [base_date - timedelta(days=int(x)) for x in rng.integers(0, 5, size=n)],
    })
    for col, frac in [("customer_age", 0.12), ("region", 0.05), ("order_value", 0.02)]:
        mask = rng.random(n) < frac
        df.loc[mask, col] = np.nan
    return df


# ---------------------------------------------------------------------------
# The four checks (unchanged from the notebook)
# ---------------------------------------------------------------------------

@dataclass
class CompletenessResult:
    score: float
    overall_fill_rate: float
    per_column_fill_rate: dict = field(default_factory=dict)
    worst_columns: list = field(default_factory=list)


def check_completeness(df: pd.DataFrame, worst_n: int = 5) -> CompletenessResult:
    if df.empty:
        return CompletenessResult(score=0.0, overall_fill_rate=0.0)
    per_col = (1 - df.isnull().mean()).to_dict()
    overall = 1 - df.isnull().mean().mean()
    worst = sorted(per_col.items(), key=lambda kv: kv[1])[:worst_n]
    return CompletenessResult(
        score=round(overall * 100, 2),
        overall_fill_rate=round(float(overall), 4),
        per_column_fill_rate={k: round(float(v), 4) for k, v in per_col.items()},
        worst_columns=[(k, round(float(v), 4)) for k, v in worst],
    )


@dataclass
class FreshnessResult:
    score: float
    age_days: Optional[float]
    newest_timestamp: Optional[str]
    timestamp_column: Optional[str]
    note: str = ""


def check_freshness(df, timestamp_col=None, as_of=None) -> FreshnessResult:
    as_of = as_of or datetime.utcnow()
    if timestamp_col is None or timestamp_col not in df.columns:
        return FreshnessResult(50.0, None, None, None,
                                "No timestamp column provided; defaulted to neutral score of 50.")
    ts = pd.to_datetime(df[timestamp_col], errors="coerce").dropna()
    if ts.empty:
        return FreshnessResult(0.0, None, None, timestamp_col, f"Column '{timestamp_col}' had no parseable dates.")
    newest = ts.max()
    age_days = max((as_of - newest.to_pydatetime()).total_seconds() / 86400.0, 0.0)
    if age_days <= FRESH_GRACE_DAYS:
        score = 100.0
    elif age_days >= FRESH_MAX_DAYS:
        score = 0.0
    else:
        span = FRESH_MAX_DAYS - FRESH_GRACE_DAYS
        score = 100.0 * (1 - (age_days - FRESH_GRACE_DAYS) / span)
    return FreshnessResult(round(score, 2), round(age_days, 2), str(newest), timestamp_col)


@dataclass
class SourceReliabilityResult:
    score: float
    source: str
    known_source: bool


def check_source_reliability(source: str) -> SourceReliabilityResult:
    key = (source or "unknown").strip().lower().replace(" ", "_")
    known = key in SOURCE_TRUST_TABLE
    score = SOURCE_TRUST_TABLE.get(key, SOURCE_TRUST_TABLE["unknown"])
    return SourceReliabilityResult(float(score), key, known)


@dataclass
class ColumnDrift:
    column: str
    baseline_mean: float
    current_mean: float
    pct_mean_change: float
    ks_pvalue: float
    drifted: bool


@dataclass
class DriftResult:
    score: float
    columns_checked: int
    columns_drifted: int
    details: list = field(default_factory=list)
    note: str = ""


def check_drift(current_df, baseline_df, significance=0.01) -> DriftResult:
    if baseline_df is None or baseline_df.empty:
        return DriftResult(50.0, 0, 0, note="No baseline provided; defaulted to neutral score of 50.")
    numeric_cols = [c for c in current_df.columns
                    if c in baseline_df.columns
                    and pd.api.types.is_numeric_dtype(current_df[c])
                    and pd.api.types.is_numeric_dtype(baseline_df[c])]
    details = []
    for col in numeric_cols:
        cur = current_df[col].dropna().to_numpy()
        base = baseline_df[col].dropna().to_numpy()
        if len(cur) < 2 or len(base) < 2:
            continue
        ks_stat, p_value = stats.ks_2samp(cur, base)
        base_mean, cur_mean = float(np.mean(base)), float(np.mean(cur))
        pct_change = 0.0 if base_mean == 0 else round((cur_mean - base_mean) / abs(base_mean) * 100, 2)
        details.append(ColumnDrift(col, round(base_mean, 4), round(cur_mean, 4),
                                    pct_change, round(float(p_value), 5), bool(p_value < significance)))
    if not details:
        return DriftResult(50.0, 0, 0, note="No comparable numeric columns with enough data.")
    drifted_count = sum(1 for d in details if d.drifted)
    score = 100.0 * (1 - drifted_count / len(details))
    return DriftResult(round(score, 2), len(details), drifted_count, details)


def compute_trust_score(completeness, freshness, source, drift, weights):
    total_w = sum(weights.values()) or 1.0
    overall = (
        completeness.score * weights["completeness"]
        + freshness.score * weights["freshness"]
        + source.score * weights["source_reliability"]
        + drift.score * weights["drift"]
    ) / total_w
    return round(overall, 2), label_for_score(overall)


def explain(dataset_name, overall_score, label, completeness, freshness, source, drift):
    lines = [f"Trust Score for '{dataset_name}': {overall_score}/100 -- {label}", ""]

    if completeness.score >= 90:
        lines.append(f"- Completeness is strong ({completeness.overall_fill_rate:.0%} of cells populated).")
    elif completeness.score >= 70:
        lines.append(f"- Completeness is decent ({completeness.overall_fill_rate:.0%} filled), worth a glance.")
    else:
        lines.append(f"- Completeness is weak ({completeness.overall_fill_rate:.0%} filled) -- significant missing data.")
    if completeness.worst_columns:
        worst = ", ".join(f"{c} ({r:.0%} filled)" for c, r in completeness.worst_columns[:3])
        lines.append(f"  Worst columns: {worst}")

    if freshness.age_days is None:
        lines.append(f"- Freshness could not be assessed ({freshness.note})")
    elif freshness.score >= 90:
        lines.append(f"- Data is fresh -- newest record is {freshness.age_days:.1f} days old.")
    elif freshness.score >= 50:
        lines.append(f"- Data is aging -- newest record is {freshness.age_days:.1f} days old.")
    else:
        lines.append(f"- Data is stale -- newest record is {freshness.age_days:.1f} days old. Treat with caution.")

    if not source.known_source:
        lines.append(f"- Source '{source.source}' is not in the trust table -- defaulted to {source.score}/100.")
    else:
        lines.append(f"- Source '{source.source}' has a configured trust rating of {source.score}/100.")

    if drift.columns_checked == 0:
        lines.append(f"- Drift could not be assessed ({drift.note})")
    elif drift.columns_drifted == 0:
        lines.append(f"- No significant drift detected across {drift.columns_checked} numeric columns.")
    else:
        drifted_cols = ", ".join(d.column for d in drift.details if d.drifted)
        lines.append(f"- Drift detected in {drift.columns_drifted}/{drift.columns_checked} numeric columns: "
                      f"{drifted_cols}. Investigate before trusting downstream results.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="DataCred — Data Trust Score", page_icon="🏷️", layout="wide")

# ---------------------------------------------------------------------------
# Blue theme — custom CSS on top of .streamlit/config.toml
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

.stApp { background: linear-gradient(180deg, #F4F8FF 0%, #FFFFFF 320px); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B2E6F 0%, #123A8C 100%);
}
/* Default sidebar text: light, for readability on the navy background */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #EAF1FF; }

section[data-testid="stSidebar"] h2 {
    font-size: 12px; text-transform: uppercase; letter-spacing: 1.2px;
    color: #9FC1FF; font-weight: 700; margin-top: 4px;
}
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15); }

/* File uploader renders its own white dropzone — keep its text dark so it stays readable */
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
    color: #0B2E6F !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
    background: #EAF1FF !important; color: #0B2E6F !important; border: 1px solid #B7CCF2 !important;
}
/* Text input / selectbox inner boxes also render light — keep their text dark */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #0B2E6F !important;
}

/* Buttons */
.stButton>button, .stDownloadButton>button {
    background: linear-gradient(90deg, #2563EB, #1D4ED8);
    color: #FFFFFF; border: none; font-weight: 700; letter-spacing: 0.2px;
    border-radius: 8px; padding: 0.6em 1em;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    background: linear-gradient(90deg, #1D4ED8, #1E3A8A); color: #FFFFFF;
}

/* Header banner */
.dc-hero {
    background: linear-gradient(120deg, #0B2E6F 0%, #2563EB 100%);
    color: #FFFFFF; padding: 28px 32px; border-radius: 16px; margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(11,46,111,0.18);
}
.dc-hero h1 { margin: 0 0 4px 0; font-size: 30px; font-weight: 800; letter-spacing: -0.5px; }
.dc-hero p { margin: 0; color: #CFE0FF; font-size: 15px; }

/* Score card */
.dc-score-card {
    background: #FFFFFF; border: 1px solid #DCE7FA; border-radius: 16px;
    padding: 22px 26px; box-shadow: 0 6px 18px rgba(37,99,235,0.08);
}
.dc-score-num { font-size: 52px; font-weight: 800; color: #0B2E6F; line-height: 1; }
.dc-score-label { font-size: 15px; font-weight: 700; margin-top: 6px; }

/* Expanders / info boxes */
.streamlit-expanderHeader { font-weight: 600; color: #0B2E6F; }
div[data-testid="stMetricValue"] { color: #0B2E6F; }

/* Code / report block */
pre, code { background: #EEF4FF !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="dc-hero">
  <h1>🏷️ DataCred</h1>
  <p>A nutrition label for datasets — a 0–100 trust score backed by four transparent checks.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("1. Choose your data")
    data_mode = st.radio("Data source", ["Use demo data (synthetic)", "Upload my own CSVs"])

    if data_mode == "Upload my own CSVs":
        current_file = st.file_uploader("Current dataset (required)", type="csv")
        baseline_file = st.file_uploader("Baseline / historical dataset (optional, enables drift check)", type="csv")
        ts_col_hint = st.text_input("Timestamp column name (optional, enables freshness check)", value="")
    else:
        current_file, baseline_file, ts_col_hint = None, None, "last_updated"

    st.header("2. Source")
    source_name = st.selectbox(
        "Where did this data come from?",
        list(SOURCE_TRUST_TABLE.keys()),
        index=0,
        format_func=lambda k: k.replace("_", " ").title(),
    )

    st.header("3. Check weights")
    st.caption("How much each check contributes to the overall score. Auto-normalized.")
    w_completeness = st.slider("Completeness", 0.0, 1.0, DEFAULT_WEIGHTS["completeness"], 0.05)
    w_freshness = st.slider("Freshness", 0.0, 1.0, DEFAULT_WEIGHTS["freshness"], 0.05)
    w_source = st.slider("Source reliability", 0.0, 1.0, DEFAULT_WEIGHTS["source_reliability"], 0.05)
    w_drift = st.slider("Drift", 0.0, 1.0, DEFAULT_WEIGHTS["drift"], 0.05)
    weights = {
        "completeness": w_completeness,
        "freshness": w_freshness,
        "source_reliability": w_source,
        "drift": w_drift,
    }

    run = st.button("Score this dataset", type="primary", use_container_width=True)

if run:
    if data_mode == "Use demo data (synthetic)":
        current_df = make_current()
        baseline_df = make_baseline()
        ts_col = "last_updated"
        dataset_name = "current_sales (demo)"
    else:
        if current_file is None:
            st.error("Please upload a current dataset CSV.")
            st.stop()
        current_df = pd.read_csv(current_file)
        baseline_df = pd.read_csv(baseline_file) if baseline_file is not None else None
        ts_col = ts_col_hint.strip() or None
        dataset_name = getattr(current_file, "name", "uploaded_dataset")

    completeness_result = check_completeness(current_df)
    freshness_result = check_freshness(current_df, timestamp_col=ts_col)
    source_result = check_source_reliability(source_name)
    drift_result = check_drift(current_df, baseline_df)

    overall_score, label = compute_trust_score(completeness_result, freshness_result, source_result, drift_result, weights)
    report_text = explain(dataset_name, overall_score, label, completeness_result, freshness_result, source_result, drift_result)

    # Blue-shade band coloring: deeper, more saturated blue = higher trust
    band_color = "#1D4ED8" if overall_score >= 65 else "#5B8DEF" if overall_score >= 40 else "#9DB8E8"

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div class="dc-score-card">
            <div style="font-size:13px; font-weight:700; color:#5B7BB8; text-transform:uppercase; letter-spacing:1px;">
                Overall Trust Score
            </div>
            <div class="dc-score-num">{overall_score}<span style="font-size:20px; color:#5B7BB8;">/100</span></div>
            <div class="dc-score-label" style="color:{band_color};">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        labels_ = ["Completeness", "Freshness", "Source\nReliability", "Drift"]
        scores_ = [completeness_result.score, freshness_result.score, source_result.score, drift_result.score]
        # Monochromatic blue scale: darker navy = stronger score, pale blue = weaker
        blue_scale = ["#0B2E6F", "#2563EB", "#7FA8E8", "#C7DAF7"]
        colors_ = []
        for s in scores_:
            if s >= 85: colors_.append(blue_scale[0])
            elif s >= 65: colors_.append(blue_scale[1])
            elif s >= 40: colors_.append(blue_scale[2])
            else: colors_.append(blue_scale[3])

        fig, ax = plt.subplots(figsize=(7, 3.2))
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")
        bars = ax.bar(labels_, scores_, color=colors_, width=0.55, zorder=3)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Score (0-100)", color="#0B2E6F", fontweight="bold")
        ax.tick_params(colors="#0B2E6F")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#C7DAF7")
        ax.yaxis.grid(True, color="#E4ECFB", zorder=0)
        for bar, s in zip(bars, scores_):
            ax.text(bar.get_x() + bar.get_width() / 2, s + 2.5, f"{s}", ha="center",
                     fontweight="bold", color="#0B2E6F")
        st.pyplot(fig)

    st.subheader("Plain-English report")
    st.code(report_text, language=None)

    with st.expander("Completeness detail (per column fill rate)"):
        st.dataframe(pd.DataFrame(
            [(k, v) for k, v in completeness_result.per_column_fill_rate.items()],
            columns=["column", "fill_rate"],
        ).sort_values("fill_rate"))

    if drift_result.details:
        with st.expander("Drift detail (vs. baseline, per numeric column)"):
            st.dataframe(pd.DataFrame([d.__dict__ for d in drift_result.details]))

    result_dict = {
        "dataset_name": dataset_name,
        "scored_at": datetime.utcnow().isoformat(),
        "overall_score": overall_score,
        "label": label,
        "weights": weights,
        "completeness": completeness_result.__dict__,
        "freshness": freshness_result.__dict__,
        "source_reliability": source_result.__dict__,
        "drift": {
            "score": drift_result.score,
            "columns_checked": drift_result.columns_checked,
            "columns_drifted": drift_result.columns_drifted,
            "details": [d.__dict__ for d in drift_result.details],
        },
    }
    st.download_button(
        "Download full result as JSON",
        data=json.dumps(result_dict, indent=2, default=str),
        file_name="datacred_result.json",
        mime="application/json",
    )
else:
    st.info("Configure your data source in the sidebar, then click **Score this dataset**.")
