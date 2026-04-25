# ---------- app.py ----------
import os, json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ------------------------------------------------------------
# Page config + light styling
# ------------------------------------------------------------
st.set_page_config(page_title="NGS QC Dashboard", layout="wide")

st.markdown(
    """
    <style>
      .smallcap { font-size: 0.85rem; opacity: 0.8; }
      .kpi-card { padding: 14px; border-radius: 14px; background: rgba(255,255,255,0.03);
                 border: 1px solid rgba(255,255,255,0.08); }
      .kpi-label { font-size: 0.85rem; opacity: 0.75; margin-bottom: 6px; }
      .kpi-value { font-size: 1.6rem; font-weight: 700; line-height: 1.1; }
      .kpi-sub   { font-size: 0.85rem; opacity: 0.75; margin-top: 6px; }
      .section-title { margin-top: 0.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("NGS QC Dashboard — Ion Torrent / NGS QC")

st.caption(
    "Interactive dashboard for longitudinal NGS quality control using robust statistics (Hampel filter with MAD)."
)

# ------------------------------------------------------------
# Optional OpenAI setup (LLM summary only)
# ------------------------------------------------------------
USE_LLM = False
client = None
try:
    from openai import OpenAI  # requires openai>=1.x
    if os.getenv("OPENAI_API_KEY"):
        client = OpenAI()
        USE_LLM = True
except Exception:
    USE_LLM = False


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def safe_json_dumps(obj):
    """JSON-safe serialization (handles timestamps, numpy types)."""
    return json.dumps(obj, indent=2, default=str)


def pick_first_existing(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of these columns exist: {candidates}")
    return None


def kpi(label, value, sub=""):
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _safe_num(s):
    return pd.to_numeric(s, errors="coerce")


def _safe_dt(s):
    return pd.to_datetime(s, errors="coerce")


def _base_dir() -> Path:
    """
    Robust base directory for HF / Docker.
    In Streamlit, __file__ should exist. If not, fall back to cwd.
    """
    try:
        return Path(__file__).resolve().parent
    except Exception:
        return Path.cwd()


APP_DIR = _base_dir()        # folder containing app.py
ROOT_DIR = APP_DIR.parent    # repo root (usually)
SEARCH_DIRS = [APP_DIR, ROOT_DIR, Path.cwd()]


@st.cache_data(show_spinner=False)
def load_csv_from_path(path_str: str) -> pd.DataFrame:
    """Cached loader ONLY for local file paths (stable cache key)."""
    df = pd.read_csv(path_str)
    for dcol in ["Run_Date", "Analysis_Run_Date", "Analysis_Date", "Date"]:
        if dcol in df.columns:
            df[dcol] = _safe_dt(df[dcol])
            df = df.sort_values(dcol, na_position="last")
            break
    return df


def load_csv_from_upload(uploaded_file) -> pd.DataFrame:
    """
    Non-cached loader for Streamlit UploadedFile objects.
    (Caching UploadedFile objects can cause hangs / hashing issues.)
    """
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    df = pd.read_csv(uploaded_file)
    for dcol in ["Run_Date", "Analysis_Run_Date", "Analysis_Date", "Date"]:
        if dcol in df.columns:
            df[dcol] = _safe_dt(df[dcol])
            df = df.sort_values(dcol, na_position="last")
            break
    return df


def find_csv(filename: str) -> Path | None:
    """
    Locate a CSV by exact name in common dirs (fast),
    then light fallback patterns (also fast).
    No recursive scanning.
    """
    for d in SEARCH_DIRS:
        p = d / filename
        if p.exists() and p.is_file():
            return p

    low = filename.lower()
    patterns: list[str] = []
    if "hampel" in low:
        patterns += ["*hampel*mad*.csv", "*hampel*.csv"]
    if "run_metrics" in low or "metrics" in low:
        patterns += ["*run_metrics*.csv", "*metrics*.csv"]

    for d in SEARCH_DIRS:
        for pat in patterns:
            hits = sorted(d.glob(pat))
            if hits:
                hits = sorted(hits, key=lambda x: (len(x.name), x.name))
                return hits[0]
    return None


# ------------------------------------------------------------
# Hampel/MAD multi-file loader helpers
# ------------------------------------------------------------
import re
from typing import Dict

def hampel_set_key_from_filename(name: str) -> str:
    """
    Convert Hampel filename into stable dict key.
    Examples:
      ngs_qc_hampel_mad_results__by_Instrument (1).csv -> by_instrument
      ngs_qc_hampel_mad_results__by_Chip_Lot_Number.csv -> by_chip_lot_number
    """
    base = Path(name).name
    base = re.sub(r"\.csv$", "", base, flags=re.IGNORECASE)

    # remove OS duplicate markers like " (1)"
    base = re.sub(r"\s*\(\d+\)\s*$", "", base)

    m = re.search(r"__by_(.+)$", base, flags=re.IGNORECASE)
    if m:
        suffix = m.group(1)
        suffix = re.sub(r"[^A-Za-z0-9]+", "_", suffix).strip("_").lower()
        return f"by_{suffix}"

    # fallback safe key
    safe = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_").lower()
    return safe


def find_all_hampel_csvs(
    prefix: str = "ngs_qc_hampel_mad_results",
    dirs=None
) -> list[Path]:
    """
    Find ALL Hampel CSVs in SEARCH_DIRS matching:
      ngs_qc_hampel_mad_results.csv
      ngs_qc_hampel_mad_results__by_*.csv
    """
    if dirs is None:
        dirs = SEARCH_DIRS

    hits: list[Path] = []
    patterns = [
        f"{prefix}.csv",
        f"{prefix}__by_*.csv",
    ]

    for d in dirs:
        for pat in patterns:
            for p in d.glob(pat):
                if p.exists() and p.is_file() and p.suffix.lower() == ".csv":
                    hits.append(p)

    # deduplicate
    uniq = {str(p.resolve()): p for p in hits}.values()
    return sorted(uniq, key=lambda p: (len(p.name), p.name.lower()))


def load_hampel_sets_from_paths(paths: list[Path]) -> Dict[str, pd.DataFrame]:
    """
    Load all Hampel CSVs into dict:
        hampel_sets["by_instrument"] = df
        hampel_sets["by_chip_lot"] = df
    """
    hampel_sets: Dict[str, pd.DataFrame] = {}

    for p in paths:
        try:
            df = load_csv_from_path(str(p))
            key = hampel_set_key_from_filename(p.name)
            hampel_sets[key] = df
        except Exception:
            continue

    return hampel_sets



# ------------------------------------------------------------
# Hampel/MAD Option 2 (multi-metric) — stable + keeps Instrument
# ------------------------------------------------------------
def add_hampel_mad_columns_multi(
    df: pd.DataFrame,
    date_col: str = "Run_Date",
    value_cols=("Mean_Read_Length_bp",),
    group_cols=("Instrument",),
    window: int = 15,
    k: float = 3.5,
    min_periods: int = 8,
    drop_dupes: bool = True,
    outlier_na=pd.NA,
) -> pd.DataFrame:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")

    value_cols = list(value_cols)
    group_cols = list(group_cols) if group_cols else []

    for m in value_cols:
        if m not in d.columns:
            raise KeyError(f"Missing metric column: {m}")
        d[m] = pd.to_numeric(d[m], errors="coerce")

    for gc in group_cols:
        if gc not in d.columns:
            raise KeyError(f"Missing group column: {gc}")

    sort_cols = group_cols + [date_col] if group_cols else [date_col]
    d = d.sort_values(sort_cols)

    if drop_dupes and group_cols:
        d = d.drop_duplicates(subset=group_cols + [date_col], keep="last")

    def _compute(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(date_col).copy()
        for m in value_cols:
            rm = g[m].rolling(window=window, min_periods=min_periods).median()
            mad = (g[m] - rm).abs().rolling(window=window, min_periods=min_periods).median()
            robust_sigma = 1.4826 * mad
            robust_z = (g[m] - rm) / robust_sigma

            g[f"robust_median__{m}"] = rm
            g[f"mad__{m}"] = mad
            g[f"robust_sigma__{m}"] = robust_sigma
            g[f"robust_z__{m}"] = robust_z
            g[f"H_outlier__{m}"] = np.where(robust_z.isna(), outlier_na, robust_z.abs() > k)
        return g

    if group_cols:
        parts = []
        for keys, g in d.groupby(group_cols, sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            for gc, kv in zip(group_cols, keys):
                g[gc] = kv
            parts.append(_compute(g))
        out = pd.concat(parts, ignore_index=True)
        out = out.sort_values(group_cols + [date_col]).reset_index(drop=True)
        return out

    return _compute(d).reset_index(drop=True)


# ------------------------------------------------------------
# Sidebar uploaders  (KEEP ONLY ONE COPY OF THIS SECTION)
# ------------------------------------------------------------
st.sidebar.header("Data")

uploaded_metrics = st.sidebar.file_uploader(
    "Upload ngs_run_metrics.csv (required)",
    type=["csv"],
    key="metrics",
)

uploaded_hampel_files = st.sidebar.file_uploader(
    "Upload Hampel/MAD CSV(s) (optional — you can upload multiple)",
    type=["csv"],
    key="hampel_multi",
    accept_multiple_files=True,
)

# Show uploaded Hampel filenames (nice UX)
if uploaded_hampel_files:
    st.sidebar.caption("Uploaded Hampel files:")
    st.sidebar.write([f.name for f in uploaded_hampel_files])

DEFAULT_METRICS_NAME = "ngs_run_metrics.csv"
DEFAULT_HAMPEL_NAME = "ngs_qc_hampel_mad_results.csv"

debug_on = st.sidebar.checkbox("Show debug info", value=False)
if debug_on:
    with st.sidebar.expander("🔎 Debug (paths & CSVs)", expanded=True):
        st.write("APP_DIR:", str(APP_DIR))
        st.write("ROOT_DIR:", str(ROOT_DIR))
        st.write("CWD:", str(Path.cwd()))
        st.write("APP_DIR CSVs:", [p.name for p in APP_DIR.glob("*.csv")])
        st.write("ROOT_DIR CSVs:", [p.name for p in ROOT_DIR.glob("*.csv")])
        st.write("CWD CSVs:", [p.name for p in Path.cwd().glob("*.csv")])


# ------------------------------------------------------------
# Load metrics (required)
# ------------------------------------------------------------
try:
    if uploaded_metrics is not None:
        df_metrics = load_csv_from_upload(uploaded_metrics)
        metrics_source = f"upload:{uploaded_metrics.name}"
    else:
        metrics_path = find_csv(DEFAULT_METRICS_NAME)
        if metrics_path is None:
            st.sidebar.error(f"Missing {DEFAULT_METRICS_NAME}")
            st.error(
                f"Could not find `{DEFAULT_METRICS_NAME}` in the repo. "
                f"Upload it in the sidebar or add it to the repo root."
            )
            st.stop()
        df_metrics = load_csv_from_path(str(metrics_path))
        metrics_source = f"file:{metrics_path}"

    st.sidebar.success(f"Loaded metrics: {len(df_metrics)} rows")
    if debug_on:
        st.sidebar.caption(f"Metrics source: {metrics_source}")

except Exception as e:
    st.sidebar.error("Failed to load metrics CSV")
    st.exception(e)
    st.stop()

# ------------------------------------------------------------
# Load Hampel/MAD CSVs (multiple uploads supported)
# ------------------------------------------------------------
hampel_sets = {}      # e.g., hampel_sets["by_instrument"] = df
hampel_sources = {}   # key -> "upload:filename.csv"

try:
    if uploaded_hampel_files:
        for uf in uploaded_hampel_files:
            df = load_csv_from_upload(uf)

            # Requires the helper you added near find_csv():
            # hampel_set_key_from_filename(name: str) -> str
            key = hampel_set_key_from_filename(uf.name)

            # Optional: warn on key collisions (e.g., "by_instrument" overwritten)
            if key in hampel_sets and debug_on:
                st.sidebar.warning(f"Duplicate Hampel key '{key}' — overwriting with: {uf.name}")

            hampel_sets[key] = df
            hampel_sources[key] = f"upload:{uf.name}"

        st.sidebar.success(f"Loaded Hampel sets (uploads): {len(hampel_sets)} file(s)")

        if debug_on:
            with st.sidebar.expander("Hampel uploads (debug)", expanded=False):
                st.write({k: hampel_sources[k] for k in sorted(hampel_sources)})

    else:
        st.sidebar.info("No Hampel/MAD CSVs uploaded. (Optional)")

except Exception as e:
    st.sidebar.warning("Hampel/MAD uploads failed to load (optional).")
    st.sidebar.exception(e)
    hampel_sets = {}
    hampel_sources = {}

    

# ------------------------------------------------------------
# Column detection (metrics)
# ------------------------------------------------------------
DATE_COL = pick_first_existing(df_metrics, ["Run_Date", "Analysis_Run_Date", "Analysis_Date", "Date"], required=True)

INSTR_COL = pick_first_existing(
    df_metrics,
    ["Instrument", "Instrument_ID", "InstrumentName", "Sequencer"],
    required=False,
)

LOT_COL = pick_first_existing(
    df_metrics,
    [
        "Chip_Lot_Number",
        "ExT_Sequencing_Reagent_Lot",
        "Cleaning_Solution_Lot",
        "ExT_Wash_Solution_Lot",
        "Lot",
        "Lot_Number",
    ],
    required=False,
)

# Ensure numeric for common columns if present
for c in ["Total_Reads", "Usable_Reads_%", "Usable_Reads", "Mean_Read_Length_bp"]:
    if c in df_metrics.columns:
        df_metrics[c] = _safe_num(df_metrics[c])

# Derive usable reads if needed
if "Usable_Reads" not in df_metrics.columns:
    if ("Total_Reads" in df_metrics.columns) and ("Usable_Reads_%" in df_metrics.columns):
        df_metrics["Usable_Reads"] = df_metrics["Total_Reads"] * (df_metrics["Usable_Reads_%"] / 100.0)

PRIMARY_METRIC = (
    "Usable_Reads" if "Usable_Reads" in df_metrics.columns else
    ("Total_Reads" if "Total_Reads" in df_metrics.columns else None)
)

if "Total_Reads" in df_metrics.columns:
    df_metrics["FLAG_FailedRun_NoReads"] = df_metrics["Total_Reads"].fillna(0) <= 0
else:
    df_metrics["FLAG_FailedRun_NoReads"] = False

df_metrics = df_metrics.sort_values(DATE_COL)


# ------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------
st.sidebar.header("Filters")

min_date = df_metrics[DATE_COL].min()
max_date = df_metrics[DATE_COL].max()
if pd.isna(min_date) or pd.isna(max_date):
    st.error("Could not parse Run_Date in metrics file.")
    st.stop()

date_range = st.sidebar.date_input(
    "Run date range",
    value=(min_date.date(), max_date.date()),
)

# Instrument filter
if INSTR_COL and INSTR_COL in df_metrics.columns:
    instr_options = ["(All)"] + sorted(df_metrics[INSTR_COL].dropna().astype(str).unique().tolist())
    instr_pick = st.sidebar.selectbox("Instrument", instr_options, index=0)
else:
    instr_pick = "(All)"

# Chip lot filter (your rename request)
if LOT_COL and LOT_COL in df_metrics.columns:
    lot_options = ["(All)"] + sorted(df_metrics[LOT_COL].dropna().astype(str).unique().tolist())
    lot_pick = st.sidebar.selectbox("Chip Lot Number", lot_options, index=0)
else:
    lot_pick = "(All)"

# -------------------------
# Additional factor filters (expander)
# -------------------------
FACTOR_FILTER_COLS = [
    "Chip_Wafer",
    "Chef_Reagent_Lot",
    "Chef_Solution_Lot",
    "Cleaning_Solution_Lot",
    "ExT_Sequencing_Reagent_Lot",
    "ExT_Wash_Solution_Lot",
]

with st.sidebar.expander("Reagents / Consumables (optional)", expanded=False):
    factor_picks = {}  # {col_name: selected_value}
    for col in FACTOR_FILTER_COLS:
        if col in df_metrics.columns:
            opts = ["(All)"] + sorted(df_metrics[col].dropna().astype(str).unique().tolist())
            factor_picks[col] = st.selectbox(col.replace("_", " "), opts, index=0)
        else:
            # If a column doesn't exist in this dataset, skip it silently
            continue

# Hampel controls
st.sidebar.header("Hampel/MAD")
k_pick = st.sidebar.number_input("Hampel band k (median ± k·(1.4826×MAD))", value=float(3.5), step=0.5)
window_pick = st.sidebar.slider("Rolling window (runs)", min_value=5, max_value=40, value=15)
minp_pick = st.sidebar.slider("Min periods", min_value=3, max_value=window_pick, value=min(8, window_pick))

# ------------------------------------------------------------
# Apply filters to metrics
# ------------------------------------------------------------
df_metrics_f = df_metrics[
    (df_metrics[DATE_COL].dt.date >= date_range[0]) &
    (df_metrics[DATE_COL].dt.date <= date_range[1])
].copy()

if instr_pick != "(All)" and INSTR_COL:
    df_metrics_f = df_metrics_f[df_metrics_f[INSTR_COL].astype(str) == instr_pick].copy()

if lot_pick != "(All)" and LOT_COL:
    df_metrics_f = df_metrics_f[df_metrics_f[LOT_COL].astype(str) == lot_pick].copy()

# Apply optional factor filters
for col, pick in factor_picks.items():
    if pick != "(All)" and col in df_metrics_f.columns:
        df_metrics_f = df_metrics_f[df_metrics_f[col].astype(str) == str(pick)].copy()

failures_df = df_metrics_f[df_metrics_f["FLAG_FailedRun_NoReads"]].copy()

# ------------------------------------------------------------
# KPI Hampel dataset: use loaded Hampel SETS (first set for now),
# else compute later per metric
# ------------------------------------------------------------
df_hampel_f = None
H_DATE = None
H_INSTR = None
H_SET_KEY = None  # which hampel set is currently being used (for debug / display)

if "hampel_sets" in locals() and isinstance(hampel_sets, dict) and len(hampel_sets) > 0:
    # For now: pick the first available set (stable choice).
    # Next step can add a sidebar dropdown to choose which set to use.
    H_SET_KEY = sorted(hampel_sets.keys())[0]
    df_hampel_f = hampel_sets[H_SET_KEY].copy()

    H_DATE = pick_first_existing(
        df_hampel_f,
        ["Run_Date", "Analysis_Run_Date", "Analysis_Date", "Date"],
        required=False,
    )
    H_INSTR = pick_first_existing(
        df_hampel_f,
        ["Instrument", "Instrument_ID", "InstrumentName", "Sequencer"],
        required=False,
    )

    # Date filter
    if H_DATE and H_DATE in df_hampel_f.columns:
        df_hampel_f[H_DATE] = _safe_dt(df_hampel_f[H_DATE])
        df_hampel_f = df_hampel_f[
            (df_hampel_f[H_DATE].dt.date >= date_range[0]) &
            (df_hampel_f[H_DATE].dt.date <= date_range[1])
        ].copy()

    # Instrument filter
    if instr_pick != "(All)" and H_INSTR and H_INSTR in df_hampel_f.columns:
        df_hampel_f = df_hampel_f[
            df_hampel_f[H_INSTR].astype(str) == str(instr_pick)
        ].copy()

    # --------------------------------------------------------
    # Apply optional factor filters (reagents / consumables)
    # --------------------------------------------------------
    if "factor_picks" in locals():
        for col, pick in factor_picks.items():
            if pick != "(All)" and col in df_hampel_f.columns:
                df_hampel_f = df_hampel_f[
                    df_hampel_f[col].astype(str) == str(pick)
                ].copy()

    # Optional debug
    if "debug_on" in locals() and debug_on:
        st.sidebar.caption(f"Hampel set in use (for KPI): {H_SET_KEY}")

# else: df_hampel_f stays None, and the Hampel tab can compute in-app per metric


# ------------------------------------------------------------
# Overview KPIs
# ------------------------------------------------------------
st.markdown("### Overview")

n_runs = len(df_metrics_f)
n_fail = int(len(failures_df))

n_hampel = int(len(df_hampel_f)) if (df_hampel_f is not None) else 0

# Outlier count for KPI: if we have a hampel file loaded, try to count any H_outlier__* cols
n_out = 0
if df_hampel_f is not None and len(df_hampel_f) > 0:
    out_cols = [c for c in df_hampel_f.columns if c.startswith("H_outlier__")]
    if out_cols:
        # count rows that are outliers in ANY metric (or just first; choose "any" for KPI)
        out_any = pd.DataFrame({
            c: df_hampel_f[c].astype("boolean").fillna(False) for c in out_cols
        }).any(axis=1)
        n_out = int(out_any.sum())

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi("Runs (filtered)", f"{n_runs}", "From ngs_run_metrics.csv")
with c2:
    kpi("Failures", f"{n_fail}", "Total_Reads ≤ 0 (if Total_Reads exists)")
with c3:
    kpi("Hampel rows (filtered)", f"{n_hampel}", "From ngs_qc_hampel_mad_results.csv (if provided)")
with c4:
    kpi("Hampel outliers", f"{n_out}", f"Any H_outlier__* == True (k for bands: {k_pick})")


# ------------------------------------------------------------
# Tabs layout (Hampel/MAD-centric)
# ------------------------------------------------------------
tabs = ["📌 Overview", "💥 Failures", "📊 Metrics", "🧪 Hampel/MAD", "⚖️ Instrument Comparison", "🧠 LLM Interpretation"]
tab_overview, tab_fail, tab_metrics, tab_hampel, tab_mw, tab_llm = st.tabs(tabs)



# -------------------------
# Tab: Overview
# -------------------------
with tab_overview:
    st.markdown("#### What you’re looking at")
    st.markdown(
        f"""
- **Primary date column (metrics):** `{DATE_COL}`  
- **Instrument column (metrics):** `{INSTR_COL if INSTR_COL else '(not detected)'}`  
- **Lot column (metrics):** `{LOT_COL if LOT_COL else '(not detected)'}`  
- **Failures:** Total_Reads ≤ 0 (when Total_Reads exists)  
**Hampel / MAD (robust QC) methodology**
For each metric and grouping (e.g., Instrument or Lot), the app computes **rolling robust statistics**:
- **Rolling median (baseline):**  
  `medianₜ = median(xₜ₋w … xₜ)`
- **Rolling MAD (robust scale):**  
  `MADₜ = median(|x − medianₜ|)`
- **MAD → robust scale (normal-equivalent):**  
  `scaleₜ = 1.4826 × MADₜ`
- **Robust z-score:**  
  `robust_z = (x − medianₜ) / scaleₜ`
- **Outlier flag:**  
  `H_outlier = |robust_z| > k`  (default `k = {k_pick}`)
This approach is **non-parametric**, resistant to outliers, and supports longitudinal NGS QC without assuming normality.
        """
    )

    if df_hampel_f is None or len(df_hampel_f) == 0:
        st.info(
            "Hampel/MAD dataset not loaded. That’s OK — the app can compute Hampel/MAD per selected metric in the Hampel/MAD tab."
        )
    else:
        st.markdown("#### Quick peek (last 10 Hampel-evaluated rows)")

        # show a few robust columns if present
        sample_metric = None
        for c in df_hampel_f.columns:
            if isinstance(c, str) and c.startswith("robust_z__"):
                sample_metric = c.replace("robust_z__", "")
                break

        show_cols = []
        H_DATE2 = pick_first_existing(
            df_hampel_f, ["Run_Date", "Analysis_Run_Date", "Analysis_Date", "Date"], required=False
        )
        H_INSTR2 = pick_first_existing(
            df_hampel_f, ["Instrument", "Instrument_ID", "InstrumentName", "Sequencer"], required=False
        )

        if H_DATE2:
            show_cols.append(H_DATE2)
        if H_INSTR2:
            show_cols.append(H_INSTR2)
        if "Run_Name" in df_hampel_f.columns:
            show_cols.append("Run_Name")

        if sample_metric:
            # Prefer showing MAD-based "scale" if present; fall back to robust_sigma__ if that's your column name
            preferred_scale_col = None
            if f"robust_scale__{sample_metric}" in df_hampel_f.columns:
                preferred_scale_col = f"robust_scale__{sample_metric}"
            elif f"robust_sigma__{sample_metric}" in df_hampel_f.columns:
                preferred_scale_col = f"robust_sigma__{sample_metric}"

            candidate_cols = [
                sample_metric,
                f"robust_z__{sample_metric}",
                f"H_outlier__{sample_metric}",
                f"robust_median__{sample_metric}",
            ]
            if preferred_scale_col:
                candidate_cols.append(preferred_scale_col)

            for c in candidate_cols:
                if c in df_hampel_f.columns and c not in show_cols:
                    # Insert the raw metric value right after Run_Name (if present) else after instrument/date
                    if c == sample_metric:
                        insert_at = 2 if H_INSTR2 else 1
                        if "Run_Name" in show_cols:
                            insert_at = show_cols.index("Run_Name") + 1
                        show_cols.insert(insert_at, c)
                    else:
                        show_cols.append(c)

        sort_col = H_DATE2 if H_DATE2 else df_hampel_f.columns[0]
        preview = df_hampel_f.sort_values(sort_col).tail(10)
        st.dataframe(
            preview[show_cols] if show_cols else preview,
            use_container_width=True,
        )
# -------------------------
# Tab: Failures
# -------------------------
with tab_fail:
    st.markdown("#### Failure events (Total_Reads ≤ 0)")
    if len(failures_df):
        display_cols = [DATE_COL]
        if "Run_Name" in failures_df.columns: display_cols.append("Run_Name")
        if INSTR_COL and INSTR_COL in failures_df.columns: display_cols.append(INSTR_COL)
        if LOT_COL and LOT_COL in failures_df.columns: display_cols.append(LOT_COL)

        for c in ["Total_Reads", "Usable_Reads_%", "Usable_Reads", "Source_File"]:
            if c in failures_df.columns:
                display_cols.append(c)

        st.dataframe(
            failures_df[display_cols].sort_values(DATE_COL, ascending=False),
            use_container_width=True,
        )
    else:
        st.success("No failed runs detected in the current filters.")


# -------------------------
# Tab: Metrics
# -------------------------
with tab_metrics:
    st.markdown("#### Metrics overview (from ngs_run_metrics.csv)")

    if PRIMARY_METRIC is None:
        st.info("No Usable_Reads or Total_Reads column found; metrics plots are limited.")
    else:
        # ---- Clarify how the primary metric is defined ----
        primary_note = ""
        if PRIMARY_METRIC == "Usable_Reads":
            if (
                "Total_Reads" in df_metrics.columns
                and "Usable_Reads_%" in df_metrics.columns
            ):
                primary_note = "Computed as: `Usable_Reads = Total_Reads × (Usable_Reads_% / 100)`"
            else:
                primary_note = "Usable_Reads represents a read count."
        elif PRIMARY_METRIC == "Total_Reads":
            primary_note = "Total number of reads generated per run."

        st.caption(
            f"Primary metric: `{PRIMARY_METRIC}`"
            + (f" — {primary_note}" if primary_note else "")
        )
        # ---------------------------------------------------

        if INSTR_COL and INSTR_COL in df_metrics_f.columns:
            st.markdown("##### Distribution by instrument")
            fig = px.box(
                df_metrics_f[df_metrics_f[PRIMARY_METRIC].notna()],
                x=INSTR_COL,
                y=PRIMARY_METRIC,
                points="outliers",
            )
            fig.update_layout(
                height=420,
                title=f"{PRIMARY_METRIC} by Instrument",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("##### Longitudinal trend (Read Length metrics)")

    d = df_metrics_f.sort_values(DATE_COL).copy()


    # Choose the series to plot (only those that exist)
    series_cols = [c for c in ["Mean_Read_Length_bp", "Median_Read_Length_bp", "Mode_bp"] if c in d.columns]
    if not series_cols:
        st.info("No read-length columns found (expected one of Mean_Read_Length_bp, Median_Read_Length_bp, Mode_bp).")
    else:
        # Ensure numeric
        for c in series_cols:
            d[c] = _safe_num(d[c])

        # Long format for Plotly
        plot_df = d[[DATE_COL] + series_cols].melt(
            id_vars=[DATE_COL],
            value_vars=series_cols,
            var_name="Metric",
            value_name="Value"
        ).dropna(subset=["Value"])

        # Optional nicer labels (matches your screenshot vibe)
        label_map = {
            "Mean_Read_Length_bp": "Mean Read Length (bp)",
            "Median_Read_Length_bp": "Median Read Length (bp)",
            "Mode_bp": "Mode (bp)",
        }
        plot_df["Metric"] = plot_df["Metric"].map(label_map).fillna(plot_df["Metric"])

        # Title reflects sidebar instrument filter (if any)
        title_suffix = ""
        if INSTR_COL and instr_pick != "(All)":
            title_suffix = f" — {instr_pick}"

        fig = px.line(
            plot_df,
            x=DATE_COL,
            y="Value",
            color="Metric",
            markers=True,
            height=520,
            title=f"Read Length metrics over time{title_suffix}",
        )
        fig.update_layout(
            xaxis_title="Run Date",
            yaxis_title="Read Length (bp)",
            hovermode="x unified",
            legend_title_text="",
        )

        st.plotly_chart(fig, use_container_width=True)


# -------------------------
# Tab: Hampel/MAD (Option 2)
# -------------------------
with tab_hampel:
    st.markdown("#### Hampel / MAD time-series")

    # ---------------------------------------------
    # Metric choice: ONLY continuous QC metrics
    # (exclude flags, IDs, lots, barcodes, etc.)
    # ---------------------------------------------
    EXCLUDE_PATTERNS = (
        "FLAG_",        # binary flags
        "_Lot", "Lot_",  # lot identifiers
        "Chip_",        # chip-related identifiers (barcode/wafer/etc.)
        "_ID",          # generic identifiers
        "_Barcode",     # barcodes
    )

    CORE_QC_METRICS = {
        "Mean_Read_Length_bp",
        "Median_Read_Length_bp",
        "Mode_bp",
        "ISP_Loading_%",
        "Total_Reads",
        "Usable_Reads",
        "Usable_Reads_%",
    }

    def is_continuous(series: pd.Series) -> bool:
        """Heuristic: exclude binary/enums and near-constant numeric fields."""
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) < 10:
            return False
        if s.nunique() <= 3:
            return False
        return True

    excluded_metrics = {DATE_COL, INSTR_COL, LOT_COL, "Source_File", "Run_Name"}

    numeric_candidates = []
    for c in df_metrics_f.columns:
        if c is None or c in excluded_metrics:
            continue
        if not isinstance(c, str):
            continue

        # Exclude known non-QC patterns
        if any(pat in c for pat in EXCLUDE_PATTERNS):
            continue

        # Include numeric continuous metrics (or explicit core QC metrics)
        if pd.api.types.is_numeric_dtype(df_metrics_f[c]):
            if (c in CORE_QC_METRICS) or is_continuous(df_metrics_f[c]):
                numeric_candidates.append(c)

    # Ensure core QC metrics appear if present in the dataframe
    for c in CORE_QC_METRICS:
        if c in df_metrics_f.columns and c not in numeric_candidates:
            numeric_candidates.append(c)

    numeric_candidates = sorted(set(numeric_candidates))
    if not numeric_candidates:
        st.error("No continuous QC metric columns detected for Hampel/MAD analysis.")
        st.stop()


    # ---------------------------------------------
    # Compact controls (two rows; minimal scrolling)
    # ---------------------------------------------
    st.markdown("##### Controls")

    # Row 1: Metric + Instrument + Hampel source (horizontal radio)
    r1 = st.columns([2.2, 1.6, 2.4], gap="small")

    with r1[0]:
        metric_pick = st.selectbox(
            "Metric",
            numeric_candidates,
            index=(numeric_candidates.index("Mean_Read_Length_bp")
                   if "Mean_Read_Length_bp" in numeric_candidates else 0),
        )

    with r1[1]:
        if INSTR_COL and INSTR_COL in df_metrics_f.columns:
            inst_list = ["(All instruments)"] + sorted(
                df_metrics_f[INSTR_COL].dropna().astype(str).unique().tolist()
            )
            inst_pick = st.selectbox("Instrument", inst_list, index=0)
        else:
            inst_pick = "(All instruments)"

    with r1[2]:
        mode = st.radio(
            "Hampel source",
            ["Compute in-app", "Use uploaded Hampel CSV"],
            index=0,
            horizontal=True,
        )

    # Row 2: Display toggles (compact)
    r2 = st.columns([1.2, 1.4, 3.4], gap="small")
    with r2[0]:
        share_y = st.checkbox("Same Y-axis", value=True)
    with r2[1]:
        show_out = st.checkbox("Highlight outliers", value=True)
    with r2[2]:
        st.caption(
            "Bands use the sidebar Hampel settings (k, window, min periods). "
            "Compute-in-app guarantees Option 2 columns for the selected metric."
        )

    st.divider()

    # ------------------------------------------------------------
    # Choose dataframe to plot (prefer loaded Hampel CSV, else compute)
    # ------------------------------------------------------------
    if mode.startswith("Use") and (df_hampel_f is not None) and len(df_hampel_f) > 0:
        d = df_hampel_f.copy()
        H_DATE = pick_first_existing(d, ["Run_Date", "Analysis_Run_Date", "Analysis_Date", "Date"], required=True)
        H_INSTR = pick_first_existing(d, ["Instrument", "Instrument_ID", "InstrumentName", "Sequencer"], required=False)
        d[H_DATE] = _safe_dt(d[H_DATE])

        if H_INSTR is None:
            st.warning("Uploaded Hampel CSV has no Instrument column; small multiples disabled.")

        # Filter instrument
        if inst_pick != "(All instruments)" and H_INSTR and H_INSTR in d.columns:
            d = d[d[H_INSTR].astype(str) == inst_pick].copy()

    else:
        # Compute Hampel in-app from df_metrics_f (Option 2)
        if INSTR_COL is None:
            st.error("Instrument column not detected in metrics; cannot compute per-instrument Hampel/MAD.")
            st.stop()

        base = df_metrics_f.copy()
        base[DATE_COL] = _safe_dt(base[DATE_COL])
        base[metric_pick] = _safe_num(base[metric_pick])

        if inst_pick != "(All instruments)":
            base = base[base[INSTR_COL].astype(str) == inst_pick].copy()

        with st.spinner("Computing Hampel/MAD…"):
            d = add_hampel_mad_columns_multi(
                base,
                date_col=DATE_COL,
                value_cols=[metric_pick],
                group_cols=(INSTR_COL,),
                window=int(window_pick),
                k=float(k_pick),
                min_periods=int(minp_pick),
            )

        H_DATE = DATE_COL
        H_INSTR = INSTR_COL

    # ------------------------------------------------------------
    # Metric-specific robust columns (Option 2)
    # ------------------------------------------------------------
    med_col = f"robust_median__{metric_pick}"
    sig_col = f"robust_sigma__{metric_pick}"
    z_col   = f"robust_z__{metric_pick}"
    out_col = f"H_outlier__{metric_pick}"

    missing_needed = [c for c in [med_col, sig_col] if c not in d.columns]
    if missing_needed:
        st.error(
            "Hampel data does not contain the required Option 2 columns for the selected metric.\n\n"
            f"Missing: {missing_needed}\n\n"
            "Tip: pick **Compute in-app** mode, or ensure your Hampel CSV includes Option 2 fields."
        )
        st.stop()

    d = d.sort_values(H_DATE).copy()
    d[metric_pick] = _safe_num(d[metric_pick])

    # Build bands using Option 2 columns
    d["band_lo"] = _safe_num(d[med_col]) - float(k_pick) * _safe_num(d[sig_col])
    d["band_hi"] = _safe_num(d[med_col]) + float(k_pick) * _safe_num(d[sig_col])

    # Shared y-range (optional)
    y_range = None
    if share_y:
        y_min = np.nanmin([d["band_lo"].min(), d[metric_pick].min()])
        y_max = np.nanmax([d["band_hi"].max(), d[metric_pick].max()])
        if np.isfinite(y_min) and np.isfinite(y_max):
            pad = 0.03 * (y_max - y_min) if y_max > y_min else 1.0
            y_range = [float(y_min - pad), float(y_max + pad)]

    def plot_one(gi: pd.DataFrame, title: str):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=gi[H_DATE], y=gi["band_lo"],
            mode="lines", line=dict(width=0),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=gi[H_DATE], y=gi["band_hi"],
            mode="lines", line=dict(width=0),
            fill="tonexty",
            name=f"Robust band (±{k_pick}·robust σ)",
            hoverinfo="skip",
            opacity=0.25,
        ))
        fig.add_trace(go.Scatter(
            x=gi[H_DATE], y=_safe_num(gi[med_col]),
            mode="lines",
            name="Rolling robust median",
        ))

        # Outlier mask (warning-free)
        if out_col in gi.columns:
            out_mask = gi[out_col].astype("boolean").fillna(False).to_numpy(dtype=bool)
        else:
            out_mask = np.zeros(len(gi), dtype=bool)

        normal = gi[~out_mask].copy()
        outl = gi[out_mask].copy()

        if z_col in normal.columns:
            normal_z = _safe_num(normal[z_col])
            fig.add_trace(go.Scatter(
                x=normal[H_DATE], y=_safe_num(normal[metric_pick]),
                mode="markers", name="Runs (normal)",
                customdata=normal_z,
                hovertemplate=(
                    f"{H_DATE}: %{{x}}<br>{metric_pick}: %{{y}}<br>"
                    "robust_z: %{customdata:.2f}<extra></extra>"
                ),
            ))
        else:
            fig.add_trace(go.Scatter(
                x=normal[H_DATE], y=_safe_num(normal[metric_pick]),
                mode="markers", name="Runs (normal)",
            ))

        if show_out and len(outl):
            if z_col in outl.columns:
                outl_z = _safe_num(outl[z_col])
                fig.add_trace(go.Scatter(
                    x=outl[H_DATE], y=_safe_num(outl[metric_pick]),
                    mode="markers", name="Runs (outlier)",
                    marker=dict(size=10, symbol="x"),
                    customdata=outl_z,
                    hovertemplate=(
                        f"{H_DATE}: %{{x}}<br>{metric_pick}: %{{y}}<br>"
                        "robust_z: %{customdata:.2f}<extra></extra>"
                    ),
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=outl[H_DATE], y=_safe_num(outl[metric_pick]),
                    mode="markers", name="Runs (outlier)",
                    marker=dict(size=10, symbol="x"),
                ))

        fig.update_layout(
            title=title,
            xaxis_title="Run date",
            yaxis_title=metric_pick,
            hovermode="x unified",
            height=520,
        )
        if y_range is not None:
            fig.update_yaxes(range=y_range)
        return fig

    # Plot: small multiples by instrument
    if inst_pick == "(All instruments)" and H_INSTR and H_INSTR in d.columns:
        st.caption(f"Showing all instruments as small multiples for **{metric_pick}**.")
        for inst in sorted(d[H_INSTR].dropna().astype(str).unique().tolist()):
            gi = d[d[H_INSTR].astype(str) == inst].copy()
            fig = plot_one(gi, title=f"{metric_pick} over time — {inst} (Hampel/MAD ±{k_pick}·robust σ)")
            st.plotly_chart(fig, use_container_width=True)
    else:
        title_inst = f" — {inst_pick}" if inst_pick != "(All instruments)" else ""
        fig = plot_one(d, title=f"{metric_pick} over time{title_inst} (Hampel/MAD ±{k_pick}·robust σ)")
        st.plotly_chart(fig, use_container_width=True)

    # Outliers table (make it compact/collapsible)
    with st.expander("Outliers (filtered)", expanded=False):
        if out_col in d.columns and d[out_col].astype("boolean").fillna(False).any():
            out_df = d[d[out_col].astype("boolean").fillna(False)].copy()
            out_df = out_df.sort_values(H_DATE, ascending=False)

            keep = [H_DATE]
            if H_INSTR and H_INSTR in out_df.columns: keep.append(H_INSTR)
            if "Run_Name" in out_df.columns: keep.append("Run_Name")
            if metric_pick in out_df.columns: keep.append(metric_pick)
            for c in [z_col, med_col, sig_col]:
                if c in out_df.columns: keep.append(c)
            if LOT_COL and LOT_COL in out_df.columns: keep.append(LOT_COL)

            st.dataframe(out_df[keep], use_container_width=True)
        else:
            st.success("No Hampel outliers detected in the current filters.")

# -------------------------
# Tab: Instrument Comparison (Mann–Whitney U)
# -------------------------
with tab_mw:
    st.markdown("#### Instrument comparison — Mann–Whitney U test")
    st.caption(
        "Non-parametric comparison of **Usable Reads** between two instruments. "
        "Tests whether their distributions differ without assuming normality."
    )

    # ---- SciPy availability (fail-safe) ----
    try:
        from scipy.stats import mannwhitneyu
        SCIPY_AVAILABLE = True
    except Exception:
        SCIPY_AVAILABLE = False

    if not SCIPY_AVAILABLE:
        st.error(
            "SciPy is required for the Mann–Whitney U test but is not available.\n\n"
            "✅ Fix: add this line to `requirements.txt` and redeploy:\n"
            "`scipy>=1.9`\n\n"
            "Then restart the Space."
        )
        st.stop()

    # ---- Required columns ----
    if INSTR_COL is None or INSTR_COL not in df_metrics_f.columns:
        st.error("Instrument column not available in the loaded metrics data.")
        st.stop()

    if "Usable_Reads" not in df_metrics_f.columns:
        st.error(
            "Usable_Reads column not available. Cannot perform comparison.\n\n"
            "Tip: If you only have Usable_Reads_% and Total_Reads, ensure the app derives Usable_Reads."
        )
        st.stop()

    # ---- Instrument selection ----
    instruments = sorted(df_metrics_f[INSTR_COL].dropna().astype(str).unique().tolist())
    if len(instruments) < 2:
        st.warning("At least two instruments are required for comparison.")
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        inst_a = st.selectbox("Instrument A", instruments, index=0)
    with c2:
        inst_b = st.selectbox("Instrument B", instruments, index=1 if len(instruments) > 1 else 0)

    if inst_a == inst_b:
        st.info("Please select two different instruments.")
        st.stop()

    # ---- Prepare data (respect global filters already applied in df_metrics_f) ----
    d = df_metrics_f.copy()
    d["Usable_Reads"] = _safe_num(d["Usable_Reads"])

    a_vals = d[d[INSTR_COL].astype(str) == inst_a]["Usable_Reads"].dropna()
    b_vals = d[d[INSTR_COL].astype(str) == inst_b]["Usable_Reads"].dropna()

    if len(a_vals) == 0 or len(b_vals) == 0:
        st.error(
            "No usable data after filters for one or both instruments.\n\n"
            f"{inst_a}: {len(a_vals)} values, {inst_b}: {len(b_vals)} values"
        )
        st.stop()

    if len(a_vals) < 8 or len(b_vals) < 8:
        st.warning(
            "Each instrument should ideally have at least ~8 runs for a stable comparison.\n\n"
            f"Current counts — {inst_a}: {len(a_vals)}, {inst_b}: {len(b_vals)}"
        )

    # ---- Mann–Whitney U test (two-sided; auto method picks exact/asymptotic safely) ----
    try:
        u_stat, p_val = mannwhitneyu(
            a_vals.to_numpy(dtype=float),
            b_vals.to_numpy(dtype=float),
            alternative="two-sided",
            method="auto",
        )
    except TypeError:
        # For older SciPy where "method" may not exist
        u_stat, p_val = mannwhitneyu(
            a_vals.to_numpy(dtype=float),
            b_vals.to_numpy(dtype=float),
            alternative="two-sided",
        )
    except Exception as e:
        st.error("Failed to compute Mann–Whitney U test.")
        st.exception(e)
        st.stop()

    # ---- Summary stats ----
    med_a = float(np.nanmedian(a_vals))
    med_b = float(np.nanmedian(b_vals))

    # ---- Display results ----
    st.markdown("##### Results")

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        kpi("Runs — A", f"{len(a_vals)}", inst_a)
    with r2:
        kpi("Runs — B", f"{len(b_vals)}", inst_b)
    with r3:
        kpi("Median A", f"{med_a:,.0f}", "Usable Reads")
    with r4:
        kpi("Median B", f"{med_b:,.0f}", "Usable Reads")

    st.markdown("---")
    st.markdown(
        f"""
**Mann–Whitney U statistic:** `{u_stat:.2f}`  
**p-value (two-sided):** `{p_val:.4g}`
"""
    )

    # ---- Interpretation ----
    alpha = 0.05
    if p_val < alpha:
        st.warning(
            f"""
**Statistically significant difference detected (p < {alpha}).**
This suggests the **distribution of usable reads differs** between  
**{inst_a}** and **{inst_b}** over the selected date range.
⚠️ This does **not** imply causation or instrument failure —  
it indicates a systematic shift worth investigating (e.g., lot mix, workflow changes, time effects).
"""
        )
    else:
        st.success(
            f"""
**No statistically significant difference detected (p ≥ {alpha}).**
The usable read distributions for **{inst_a}** and **{inst_b}**  
are statistically indistinguishable over this period.
"""
        )

    # ---- Optional raw data peek ----
    with st.expander("Show underlying data (Usable Reads)", expanded=False):
        cols = [DATE_COL, INSTR_COL, "Usable_Reads"]
        if LOT_COL and LOT_COL in d.columns:
            cols.append(LOT_COL)
        if "Run_Name" in d.columns:
            cols.insert(1, "Run_Name")

        show_df = d[d[INSTR_COL].astype(str).isin([inst_a, inst_b])][cols].copy()
        show_df = show_df.sort_values(DATE_COL)
        st.dataframe(show_df, use_container_width=True)

    # ---- Store MW result for LLM tab ----
    st.session_state["mw_result"] = {
        "metric": "Usable_Reads",
        "instrument_a": str(inst_a),
        "instrument_b": str(inst_b),
        "n_a": int(len(a_vals)),
        "n_b": int(len(b_vals)),
        "median_a": float(med_a),
        "median_b": float(med_b),
        "u_stat": float(u_stat),
        "p_value": float(p_val),
        "alpha": float(alpha),
        "significant": bool(p_val < alpha),
    }


# -------------------------
# Tab: LLM (QC report style + QC Status banner + MW integration)
# -------------------------
with tab_llm:
    st.markdown("#### LLM interpretation (optional)")
    st.caption("Requires OPENAI_API_KEY in Space Secrets. Output is advisory only.")

    # ---- Session state init ----
    if "llm_text" not in st.session_state:
        st.session_state["llm_text"] = ""

    # ---- Helper: QC status logic (simple + user-friendly) ----
    def compute_qc_status(fail_count: int, mw_result: dict | None, hampel_outliers_any: int | None = None):
        bullets = []

        # Actionable: failures
        if fail_count and fail_count > 0:
            bullets.append(("actionable", f"Failures detected (Total_Reads ≤ 0): {fail_count}"))

        # MW interpretation
        if isinstance(mw_result, dict) and "p_value" in mw_result:
            p = float(mw_result["p_value"])
            a = str(mw_result.get("instrument_a", "A"))
            b = str(mw_result.get("instrument_b", "B"))

            if p < 0.05:
                bullets.append(("actionable", f"Mann–Whitney significant: {a} vs {b} (p={p:.4g}). Distributions differ."))
            elif p < 0.10:
                bullets.append(("informational", f"Mann–Whitney borderline: {a} vs {b} (p={p:.4g}). Consider watching."))
            else:
                bullets.append(("informational", f"Mann–Whitney not significant: {a} vs {b} (p={p:.4g})."))

        # Hampel outliers KPI (if available)
        if hampel_outliers_any is not None:
            if hampel_outliers_any > 0:
                bullets.append(("informational", f"Hampel outliers (any metric, filtered view): {int(hampel_outliers_any)}"))
            else:
                bullets.append(("informational", "No Hampel outliers detected in the filtered view."))

        # Decide status
        level = "Stable"
        reason = "No high-confidence signals detected."
        if any(t == "actionable" for t, _ in bullets):
            level = "Investigate"
            reason = "High-confidence signals that merit follow-up checks."
        elif any("borderline" in txt for _, txt in bullets):
            level = "Watch"
            reason = "Some signals suggest monitoring, but evidence is not strong."

        return level, reason, bullets

    # ---- Helper: compact metrics snapshot (supports Colab-like narrative) ----
    def build_metrics_snapshot(df: pd.DataFrame, date_col: str, instr_col: str | None, metrics: list[str]) -> dict:
        d = df.copy()
        if date_col in d.columns:
            d[date_col] = _safe_dt(d[date_col])
            d = d.sort_values(date_col)

        n = len(d)
        q = max(8, int(np.ceil(0.20 * n))) if n else 0
        recent = d.tail(q) if q else d

        out = {"n_rows": int(n), "recent_n": int(len(recent)), "metrics": {}}

        def _summ(s: pd.Series) -> dict:
            x = _safe_num(s).dropna()
            if len(x) == 0:
                return {"n": 0}
            return {
                "n": int(len(x)),
                "median": float(np.nanmedian(x)),
                "min": float(np.nanmin(x)),
                "max": float(np.nanmax(x)),
            }

        for m in metrics:
            if m not in d.columns:
                continue
            out["metrics"][m] = {
                "overall": _summ(d[m]),
                "recent": _summ(recent[m]),
            }
            if instr_col and instr_col in d.columns:
                per_inst = {}
                for inst, g in d.groupby(instr_col):
                    per_inst[str(inst)] = _summ(g[m])
                out["metrics"][m]["per_instrument"] = per_inst

        return out

# ---- Summarize Hampel Option2 (one metric at a time) ----
def summarize_hampel_option2(
    df_h: pd.DataFrame,
    date_col: str,
    instr_col: str,
    metric: str,
    k: float,
    min_periods: int | None = None,   # <-- NEW: evaluable threshold
):
    med_col = f"robust_median__{metric}"
    sig_col = f"robust_sigma__{metric}"
    z_col   = f"robust_z__{metric}"
    out_col = f"H_outlier__{metric}"

    d = df_h.copy()
    d[date_col] = _safe_dt(d[date_col])
    d = d.sort_values(date_col)

    # numeric coercion
    for c in [metric, med_col, sig_col, z_col]:
        if c in d.columns:
            d[c] = _safe_num(d[c])

    # evaluable counts
    n_rows = int(len(d))
    n_numeric = int(d[metric].notna().sum()) if metric in d.columns else 0
    n_valid_z = int(d[z_col].notna().sum()) if z_col in d.columns else 0

    insufficient_data = False
    if min_periods is not None:
        insufficient_data = (n_valid_z < int(min_periods))

    # outlier mask
    if out_col in d.columns:
        out_mask = d[out_col].astype("boolean").fillna(False)
    elif z_col in d.columns:
        out_mask = d[z_col].abs().gt(k).fillna(False)
    else:
        out_mask = pd.Series(False, index=d.index)

    d["_out"] = out_mask
    d["_absz"] = d[z_col].abs() if z_col in d.columns else np.nan

    # clustering (2 of last 3)
    d["_out_int"] = d["_out"].astype(int)
    d["_out_run"] = (d["_out_int"].rolling(window=3, min_periods=1).sum() >= 2)

    # max consecutive outliers
    max_consec = 0
    try:
        if len(d):
            runs = d["_out_int"]
            max_consec = int((runs.groupby((runs != runs.shift()).cumsum()).cumsum()).max())
    except Exception:
        max_consec = 0

    # worst points (by |z|), even if not flagged
    worst = (
        d[d["_absz"].notna()]
        .sort_values("_absz", ascending=False)
        .head(12)
    )
    worst_records = []
    for _, r in worst.iterrows():
        worst_records.append({
            "date": r.get(date_col),
            "instrument": r.get(instr_col) if (instr_col in d.columns) else None,
            "value": r.get(metric),
            "robust_z": r.get(z_col),
            "robust_median": r.get(med_col),
            "robust_sigma": r.get(sig_col),
            "is_outlier": bool(r.get("_out")),
        })

    # per-instrument
    inst_summary = []
    if instr_col and instr_col in d.columns:
        for inst, g in d.groupby(instr_col):
            g = g.sort_values(date_col)
            n = int(len(g))
            n_out = int(g["_out"].sum())
            rate = float(n_out / n) if n else 0.0

            n_numeric_i = int(g[metric].notna().sum()) if metric in g.columns else 0
            n_valid_z_i = int(g[z_col].notna().sum()) if z_col in g.columns else 0
            insuff_i = (min_periods is not None and n_valid_z_i < int(min_periods))

            q = max(8, n // 4) if n else 0
            recent = g.tail(q) if q else g
            prior = g.iloc[-2*q:-q] if (q and n >= 2*q) else g.head(min(q, n))

            recent_med = float(np.nanmedian(recent[metric])) if (metric in recent.columns and len(recent)) else np.nan
            prior_med = float(np.nanmedian(prior[metric])) if (metric in prior.columns and len(prior)) else np.nan
            delta = float(recent_med - prior_med) if np.isfinite(recent_med) and np.isfinite(prior_med) else np.nan

            zmax = float(np.nanmax(g["_absz"])) if n else np.nan

            inst_summary.append({
                "instrument": str(inst),
                "n_runs": n,
                "n_numeric": n_numeric_i,
                "n_valid_z": n_valid_z_i,
                "insufficient_data": bool(insuff_i),
                "n_outliers": n_out,
                "outlier_rate": rate,
                "recent_vs_prior_median_delta": delta,
                "max_abs_robust_z": zmax,
            })

        inst_summary = sorted(inst_summary, key=lambda x: (x["outlier_rate"], x["max_abs_robust_z"]), reverse=True)

    return {
        "metric": metric,
        "n_rows": n_rows,
        "n_numeric": n_numeric,
        "n_valid_z": n_valid_z,
        "min_periods_required": int(min_periods) if min_periods is not None else None,
        "insufficient_data": bool(insufficient_data),

        "n_outliers": int(d["_out"].sum()) if len(d) else 0,
        "outlier_rate": float(d["_out"].mean()) if len(d) else 0.0,
        "outlier_cluster_flag_last3_2of3": bool(d["_out_run"].iloc[-1]) if len(d) else False,
        "max_consecutive_outliers": max_consec,

        "worst_outliers": worst_records,
        "per_instrument": inst_summary[:12],
    }

    
# ---- LLM call ----
def run_llm_summary(payload: dict) -> str:
    system = (
        "You are a senior NGS QC analyst for Ion Torrent runs. "
        "Write a practical QC report for lab technologists and supervisors.\n\n"

        "Strict rules:\n"
        "- Use ONLY the provided data.\n"
        "- Do NOT invent causes, trends, or explanations.\n"
        "- Quantify all statements (counts, medians, ranges, p-values, dates).\n"
        "- Separate ACTIONABLE vs INFORMATIONAL.\n"
        "- Do NOT speculate about filtering or subsets unless explicitly stated in the data.\n\n"

        "Hampel/MAD interpretation rules (MANDATORY):\n"
        "- Each metric includes n_numeric, n_valid_z, min_periods_required, and insufficient_data.\n"
        "- If insufficient_data == True OR n_valid_z < min_periods_required, "
        "you MUST write: 'Insufficient data for Hampel on <metric> (n_valid_z=X < min_periods=Y)'.\n"
        "- In that case, DO NOT say 'stable' and DO NOT say 'no outliers'.\n"
        "- Only state 'No Hampel outliers detected' for a metric if:\n"
        "    insufficient_data == False AND n_outliers == 0.\n"
        "- If n_outliers > 0, report the count and identify the worst cases.\n\n"

        "- If data is insufficient to support ANY conclusion, you MUST explicitly say "
        "'Insufficient evidence' and specify exactly what is missing.\n"
        "- Never assume outliers are driven by unfiltered data unless such comparison is explicitly provided.\n\n"

        "Output format:\n"
        "A) QC STATUS (1 line) + 1–2 sentence justification\n"
        "B) METRICS SUMMARY (bullet points)\n"
        "C) FAILURES (bullet points)\n"
        "D) INSTRUMENT COMPARISON (Mann–Whitney)\n"
        "E) HAMPEL/MAD (multi-metric; one bullet per metric)\n"
        "F) ACTIONABLE CHECKS (verification-style)\n"
        "G) INFORMATIONAL NOTES (max 4 bullets)\n"
    )

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": safe_json_dumps(payload)},
        ],
    )
    return resp.choices[0].message.content


# ---- Layout ----
cA, cB = st.columns([1, 2])

with cA:
    mw_result = st.session_state.get("mw_result", None)

    try:
        hampel_outliers_any = int(n_out)
    except Exception:
        hampel_outliers_any = None

    qc_level, qc_reason, qc_bullets = compute_qc_status(
        fail_count=int(len(failures_df)),
        mw_result=mw_result if isinstance(mw_result, dict) else None,
        hampel_outliers_any=hampel_outliers_any,
    )

    if qc_level == "Investigate":
        st.error(f"🚨 **QC Status: {qc_level}**\n\n{qc_reason}")
    elif qc_level == "Watch":
        st.warning(f"⚠️ **QC Status: {qc_level}**\n\n{qc_reason}")
    else:
        st.success(f"✅ **QC Status: {qc_level}**\n\n{qc_reason}")

    with st.expander("Why this status?", expanded=False):
        if not qc_bullets:
            st.write("No notable signals in current filters.")
        else:
            for tag, txt in qc_bullets:
                if tag == "actionable":
                    st.markdown(f"🔴 **Actionable** — {txt}")
                else:
                    st.markdown(f"🟢 **Informational** — {txt}")

    if not USE_LLM:
        st.info("LLM is not available. Add OPENAI_API_KEY to Space Secrets and ensure `openai` is in requirements.txt.")
        use_llm = st.checkbox("Enable LLM interpretation", value=False, disabled=True)
        btn_llm = st.button("Generate interpretation", disabled=True)
    else:
        use_llm = st.checkbox("Enable LLM interpretation", value=False)
        btn_llm = st.button("Generate interpretation", disabled=not use_llm)

    st.markdown(
        f"<div class='smallcap'>LLM available: <b>{'Yes' if USE_LLM else 'No'}</b></div>",
        unsafe_allow_html=True,
    )

    if isinstance(mw_result, dict) and "p_value" in mw_result:
        st.info(
            f"📄 **MW included:** {mw_result.get('instrument_a')} vs {mw_result.get('instrument_b')} "
            f"(p={float(mw_result.get('p_value')):.5g})"
        )

if btn_llm and use_llm and USE_LLM:
    with st.spinner("Generating LLM interpretation…"):

        # ---- Multi-metric Hampel (computed from df_metrics_f) ----
        if INSTR_COL is None:
            hampel_option2_summaries = {
                "error": "Missing Instrument column; cannot compute Hampel/MAD per instrument."
            }
        else:
            base = df_metrics_f.copy()
            base[DATE_COL] = _safe_dt(base[DATE_COL])

            llm_hampel_metrics = [
                c for c in [
                    "Total_Reads",
                    "Usable_Reads_%",
                    "Usable_Reads",
                    "Mean_Read_Length_bp",
                    "Median_Read_Length_bp",
                    "Mode_bp",
                    "ISP_Loading_%",
                ]
                if c in base.columns and pd.api.types.is_numeric_dtype(base[c])
            ]

            if not llm_hampel_metrics:
                hampel_option2_summaries = {"error": "No numeric QC metrics available for Hampel/MAD."}
            else:
                for m in llm_hampel_metrics:
                    base[m] = _safe_num(base[m])

                df_h_for_llm = add_hampel_mad_columns_multi(
                    base,
                    date_col=DATE_COL,
                    value_cols=llm_hampel_metrics,   # <-- multi-metric
                    group_cols=(INSTR_COL,),
                    window=int(window_pick),
                    k=float(k_pick),
                    min_periods=int(minp_pick),
                )

                hampel_option2_summaries = {}
                for m in llm_hampel_metrics:
                    needed = [f"robust_median__{m}", f"robust_sigma__{m}", f"robust_z__{m}"]
                    if all(col in df_h_for_llm.columns for col in needed):
                        hampel_option2_summaries[m] = summarize_hampel_option2(
                            df_h_for_llm,
                            date_col=DATE_COL,
                            instr_col=INSTR_COL,
                            metric=m,
                            k=float(k_pick),
                            min_periods=int(minp_pick),   # <-- IMPORTANT (enables "insufficient data" logic)
                        )
                    else:
                        hampel_option2_summaries[m] = {"error": "Option2 columns missing for this metric."}

        qc_metrics = [c for c in [
            "Total_Reads", "Usable_Reads_%", "Usable_Reads",
            "Mean_Read_Length_bp", "Median_Read_Length_bp", "Mode_bp"
        ] if c in df_metrics_f.columns]

        metrics_snapshot = build_metrics_snapshot(
            df_metrics_f, date_col=DATE_COL, instr_col=INSTR_COL, metrics=qc_metrics
        )

        payload = {
            "qc_status_banner": {
                "status": qc_level,
                "reason": qc_reason,
                "signals": [{"tag": t, "text": txt} for t, txt in qc_bullets],
            },
            "filters": {
                "date_range": [str(date_range[0]), str(date_range[1])],
                "instrument_filter": instr_pick,
                "lot_filter": lot_pick,
                "k": float(k_pick),
                "window": int(window_pick),
                "min_periods": int(minp_pick),
            },
            "metrics_counts": {
                "runs_filtered": int(len(df_metrics_f)),
                "failures_total_reads_le_0": int(len(failures_df)),
            },
            "metrics_snapshot": metrics_snapshot,
            "hampel_option2_summaries": hampel_option2_summaries,  # <-- multi-metric Hampel summaries
            "instrument_comparison_mannwhitney": mw_result if isinstance(mw_result, dict) else None,
        }

        st.session_state["llm_text"] = run_llm_summary(payload)

with cB:
    if USE_LLM and st.session_state["llm_text"]:
        st.markdown("##### Interpretation")
        st.markdown(st.session_state["llm_text"])

        st.download_button(
            "Download interpretation (.txt)",
            st.session_state["llm_text"],
            file_name="ngs_qc_llm_interpretation.txt",
        )
        st.text_area("Copy-friendly text", st.session_state["llm_text"], height=340)
    elif USE_LLM:
        st.info("Enable LLM and click **Generate interpretation**.")

