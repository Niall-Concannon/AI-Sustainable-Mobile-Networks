"""
Mobile network energy dashboard (prototype).

Data comes from the sidebar upload, then data/sample_dataset.csv, then generated demo data.
If the dataset uses different column names or a different time format, edit COLUMN_MAP and
TIME_FORMAT below.
"""
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Mobile Network Energy Dashboard", layout="wide")

DEFAULT_CSV = Path(__file__).parent.parent / "data" / "sample_dataset.csv"
REQUIRED = ["Time", "BS", "Energy", "load", "ESMODE", "TXpower"]

# Map the dataset's own column names to the names used in this app,
# for example {"energy_kwh": "Energy", "station": "BS"}
COLUMN_MAP = {}
TIME_FORMAT = "%Y%m%d %H%M%S"


# Data
@st.cache_data
def make_demo_data(n_bs=300):
    """Generate placeholder data in the same format. Not real data."""
    rng = np.random.default_rng(42)
    times = pd.date_range("2023-01-01 01:00", "2023-01-08 00:00", freq="h")
    daily = 0.5 + 0.4 * np.sin((times.hour.values - 9) / 24 * 2 * np.pi)
    frames = []
    for i in range(n_bs):
        base, tx, scale = rng.choice([14, 20, 55, 62]), rng.uniform(5.5, 7.3), rng.uniform(0.5, 1.2)
        load = np.clip(daily * scale + rng.normal(0, 0.08, len(times)), 0.01, 1)
        saving_mode = (load < 0.2) & (rng.random(len(times)) < 0.05)
        esmode = np.where(saving_mode, rng.uniform(0.05, 4, len(times)), 0)
        energy = base + 18 * load * tx / 7 - 3 * np.minimum(esmode, 1) + rng.normal(0, 1.2, len(times))
        frames.append(pd.DataFrame({"Time": times.strftime("%Y%m%d %H%M%S"), "BS": f"B_{i}",
                                    "Energy": energy, "load": load, "ESMODE": esmode, "TXpower": tx}))
    return pd.concat(frames, ignore_index=True)


def clean(raw):
    """Convert types, drop incomplete rows and add time features."""
    df = raw.copy()
    df["Time"] = pd.to_datetime(df["Time"], format=TIME_FORMAT, errors="coerce")
    for col in ["Energy", "load", "ESMODE", "TXpower"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=REQUIRED).sort_values("Time").reset_index(drop=True)
    df["hour"] = df["Time"].dt.hour
    df["dow"] = df["Time"].dt.dayofweek
    for name, period in [("hour", 24), ("dow", 7)]:
        df[f"{name}_sin"] = np.sin(2 * np.pi * df[name] / period)
        df[f"{name}_cos"] = np.cos(2 * np.pi * df[name] / period)
    return df


@st.cache_data(show_spinner="Loading data...")
def load_data(uploaded_bytes):
    if uploaded_bytes:
        raw, source = pd.read_csv(BytesIO(uploaded_bytes)), "Uploaded file"
    elif DEFAULT_CSV.exists():
        raw, source = pd.read_csv(DEFAULT_CSV), "data/sample_dataset.csv"
    else:
        raw, source = make_demo_data(), "Demo data (generated)"
    raw = raw.rename(columns=COLUMN_MAP)
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return clean(raw), source


# Helpers
def sorted_stations(series):
    return sorted(series.unique(), key=lambda s: (len(s), s))


def kpis(items):
    """Show a row of metrics from a {label: value} dict."""
    for col, (label, value) in zip(st.columns(len(items)), items.items()):
        col.metric(label, value)


# Pages
def overview_page(df):
    st.title("Mobile Network Energy Overview")
    kpis({"Base stations": f"{df['BS'].nunique():,}", "Readings": f"{len(df):,}",
          "Avg energy / reading": f"{df['Energy'].mean():.1f}", "Avg load": f"{df['load'].mean():.2f}"})
    with st.expander("Preview raw data"):
        st.dataframe(df[REQUIRED].head(200), width="stretch")


# App
PAGES = {
    "Overview": overview_page,
}


def main():
    st.sidebar.title("Energy Dashboard")
    uploaded = st.sidebar.file_uploader("Load a dataset (CSV)", type="csv",
                                        help="CSV file with the columns listed in REQUIRED")
    try:
        df, source = load_data(uploaded.getvalue() if uploaded else None)
    except Exception as err:
        st.error(f"Could not load data: {err}")
        st.stop()

    if source.startswith("Demo"):
        st.sidebar.warning("Showing generated demo data. Put your file at data/sample_dataset.csv "
                           "or upload it above.")
    else:
        st.sidebar.success(f"Source: {source}")

    page = st.sidebar.radio("Page", list(PAGES))
    st.sidebar.caption(f"{len(df):,} readings, {df['BS'].nunique():,} base stations")
    st.sidebar.caption(f"{df['Time'].min():%d %b %Y %H:%M} to {df['Time'].max():%d %b %Y %H:%M}")
    PAGES[page](df)


main()
