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
import plotly.express as px
import streamlit as st
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Mobile Network Energy Dashboard", layout="wide")

DEFAULT_CSV = Path(__file__).parent.parent / "data" / "sample_dataset.csv"
# Columns the app needs. Any other columns in the dataset are ignored.
REQUIRED = ["Time", "BS", "Energy", "load", "ESMODE", "TXpower"]

# Map the dataset's own column names to the names used in this app,
# for example {"energy_kwh": "Energy", "station": "BS"}
COLUMN_MAP = {}
TIME_FORMAT = "%Y%m%d %H%M%S"
# Used to label the heatmap (pandas numbers Monday as 0).
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
# Labels for the five equal-sized groups used in the band charts.
LOAD_LEVELS = ["Very quiet", "Quiet", "Medium", "Busy", "Very busy"]
POWER_LEVELS = ["Lowest", "Low", "Medium", "High", "Highest"]


# Data
@st.cache_data
def make_demo_data(n_bs=300):
    """Generate placeholder data in the same format. Not real data."""
    # Fixed seed so the demo data is identical on every run.
    rng = np.random.default_rng(42)
    times = pd.date_range("2023-01-01 01:00", "2023-01-08 00:00", freq="h")
    # Shared daily pattern: quiet overnight, busiest in the afternoon.
    daily = 0.5 + 0.4 * np.sin((times.hour.values - 9) / 24 * 2 * np.pi)
    frames = []
    for i in range(n_bs):
        # Give each station its own base energy, signal strength and traffic level.
        base, tx, scale = rng.choice([14, 20, 55, 62]), rng.uniform(5.5, 7.3), rng.uniform(0.5, 1.2)
        load = np.clip(daily * scale + rng.normal(0, 0.08, len(times)), 0.01, 1)
        # Power-saving mode is only switched on occasionally, and only when load is low.
        saving_mode = (load < 0.2) & (rng.random(len(times)) < 0.05)
        esmode = np.where(saving_mode, rng.uniform(0.05, 4, len(times)), 0)
        # Energy rises with load and signal strength, with a small drop in power-saving mode.
        energy = base + 18 * load * tx / 7 - 3 * np.minimum(esmode, 1) + rng.normal(0, 1.2, len(times))
        frames.append(pd.DataFrame({"Time": times.strftime("%Y%m%d %H%M%S"), "BS": f"B_{i}",
                                    "Energy": energy, "load": load, "ESMODE": esmode, "TXpower": tx}))
    return pd.concat(frames, ignore_index=True)


def clean(raw):
    """Convert types, drop incomplete rows and add time features."""
    df = raw.copy()
    # errors='coerce' turns invalid values into NaT/NaN instead of raising an error,
    # so those rows can be removed in one step below.
    df["Time"] = pd.to_datetime(df["Time"], format=TIME_FORMAT, errors="coerce")
    for col in ["Energy", "load", "ESMODE", "TXpower"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Drop rows with missing or invalid values and sort by time so the charts line up.
    df = df.dropna(subset=REQUIRED).sort_values("Time").reset_index(drop=True)
    df["hour"] = df["Time"].dt.hour
    df["dow"] = df["Time"].dt.dayofweek
    # Hour and day of week repeat in a cycle (23:00 is right next to 00:00). Sine and cosine
    # encoding lets the model see that, which plain numbers such as 0 to 23 would not.
    for name, period in [("hour", 24), ("dow", 7)]:
        df[f"{name}_sin"] = np.sin(2 * np.pi * df[name] / period)
        df[f"{name}_cos"] = np.cos(2 * np.pi * df[name] / period)
    return df


@st.cache_data(show_spinner="Loading data...")
def load_data(uploaded_bytes):
    # Priority: uploaded file, then the data folder, then demo data. The bytes are passed in
    # (not the file object) so Streamlit can cache the result.
    if uploaded_bytes:
        raw, source = pd.read_csv(BytesIO(uploaded_bytes)), "Uploaded file"
    elif DEFAULT_CSV.exists():
        raw, source = pd.read_csv(DEFAULT_CSV), "data/sample_dataset.csv"
    else:
        raw, source = make_demo_data(), "Demo data (generated)"
    # Rename first so COLUMN_MAP can fix different column names before they are checked.
    raw = raw.rename(columns=COLUMN_MAP)
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return clean(raw), source


# Helpers
def sorted_stations(series):
    # Sort by name length first so B_2 comes before B_10 (alphabetical order would put B_10 first).
    return sorted(series.unique(), key=lambda s: (len(s), s))


def kpis(items):
    """Show a row of metrics from a {label: value} dict."""
    for col, (label, value) in zip(st.columns(len(items)), items.items()):
        col.metric(label, value)


def show(fig, where=st):
    where.plotly_chart(fig, width="stretch")


def time_chart(data, column, title, ylabel):
    """A simple line chart of one value over time."""
    fig = px.line(data, x="Time", y=column, title=title, labels={column: ylabel, "Time": ""})
    fig.update_layout(height=320)
    return fig


def band_chart(df, column, labels, title, xlabel):
    """Group readings into equal-sized bands (for example Very quiet to Very busy)
    and show the average energy used in each band."""
    # Rank first so repeated values (many readings share the same load) cannot make
    # the bands uneven or break the bin edges.
    bands = pd.qcut(df[column].rank(method="first"), len(labels), labels=labels)
    avg = df.groupby(bands, observed=True)["Energy"].mean()
    fig = px.bar(x=avg.index.astype(str), y=avg.values, title=title,
                 labels={"x": xlabel, "y": "Average energy used"})
    return fig


# Model
# Model inputs. BS_avg_energy is added in train_model and stands in for which station
# a reading came from.
FEATURES = ["load", "ESMODE", "TXpower", "BS_avg_energy", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]


def score(actual, predicted):
    # RMSE punishes large mistakes more than MAE. R2 is how much of the variation in
    # energy the model explains.
    return {"MAE": mean_absolute_error(actual, predicted),
            "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
            "R2": r2_score(actual, predicted)}


@st.cache_data(show_spinner="Training prediction model...")
def train_model(df):
    """Train on the first 80% of the time range and test on the rest.

    Each base station is encoded as its average energy in the training data, which works
    for any number of stations.
    """
    # Split by time, not randomly. A random split would let the model train on readings
    # from the same hours it is tested on and look better than it really is. The cut-off comes
    # from the unique timestamps so every station is split at the same moment.
    split_time = np.sort(df["Time"].unique())[int(df["Time"].nunique() * 0.8)]
    train, test = df[df["Time"] < split_time], df[df["Time"] >= split_time]

    # Station averages come from the training data only. Using the test data here would
    # leak the answers into the model.
    station_mean = train.groupby("BS")["Energy"].mean()
    def encode(part):
        return part.assign(BS_avg_energy=part["BS"].map(station_mean).fillna(train["Energy"].mean()))
    train, test = encode(train), encode(test)

    model = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, max_leaf_nodes=31, random_state=42)
    model.fit(train[FEATURES], train["Energy"])
    test = test.assign(Predicted=model.predict(test[FEATURES]))
    # Baseline that always predicts the training average, to show the model adds something.
    baseline = DummyRegressor(strategy="mean").fit(train[FEATURES], train["Energy"])

    return {"test": test, "split_time": split_time, "n_train": len(train),
            "model": score(test["Energy"], test["Predicted"]),
            "baseline": score(test["Energy"], baseline.predict(test[FEATURES]))}


def actual_vs_predicted(data, title):
    data = data.rename(columns={"Energy": "Actual"})
    fig = px.line(data, x="Time", y=["Actual", "Predicted"], title=title,
                  labels={"value": "Energy used", "Time": ""})
    fig.update_traces(line_dash="dash", selector=dict(name="Predicted"))
    fig.update_layout(height=380, legend=dict(orientation="h", y=-0.2, title=""))
    return fig


# Pages
def overview_page(df):
    st.title("Mobile Network Energy Overview")
    st.write("This dashboard looks at how much electricity the base stations in a mobile network use, "
             "when they use it, and where energy could be saved without affecting customers.")

    with st.expander("What do the terms mean?"):
        st.markdown(
            "- **Base station**: a mobile mast that connects phones to the network.\n"
            "- **Reading**: one measurement of one base station at one point in time.\n"
            "- **Energy used**: the electricity a base station used in that reading, in the units of the dataset.\n"
            "- **Busyness (load)**: how much traffic a base station was handling. Higher means busier.\n"
            "- **Signal strength (transmit power)**: how powerfully the station sends its radio signal.\n"
            "- **Power-saving mode**: a setting that lets a station use less electricity, for example when it is quiet."
        )

    kpis({"Base stations": f"{df['BS'].nunique():,}", "Readings": f"{len(df):,}",
          "Average energy per reading": f"{df['Energy'].mean():.1f}", "Average busyness": f"{df['load'].mean():.2f}"})

    # Average energy for each hour of the day, used in the summary sentence below.
    by_hour = df.groupby("hour")["Energy"].mean()
    st.info(f"On average the network uses the most energy around {by_hour.idxmax():02d}:00 and the least around "
            f"{by_hour.idxmin():02d}:00. Even at its quietest hour, a base station still uses "
            f"{by_hour.min() / by_hour.max():.0%} of the energy it uses at its busiest hour.")

    # Energy is added up across stations, but load is averaged because adding up
    # busyness values would not mean anything.
    network = df.groupby("Time").agg(Energy=("Energy", "sum"), Load=("load", "mean")).reset_index()
    left, right = st.columns(2)
    show(time_chart(network, "Energy", "Total energy used by all base stations", "Energy used"), left)
    show(time_chart(network, "Load", "How busy the base stations were (average)", "Busyness"), right)
    st.caption("Each line shows one value over time. When the two lines rise and fall together, "
               "energy use is closely following how busy the network is.")

    with st.expander("Preview raw data"):
        st.dataframe(df[REQUIRED].head(200), width="stretch")


def patterns_page(df):
    st.title("Energy Patterns")
    st.write("This page looks for patterns in when and why base stations use energy.")

    hourly = df.groupby("hour")["Energy"].mean().reset_index()
    fig = px.bar(hourly, x="hour", y="Energy", title="Average energy used at each hour of the day",
                 labels={"hour": "Hour of the day (0 = midnight)", "Energy": "Average energy used"})
    show(fig.update_xaxes(dtick=1))
    st.caption("Taller bars mean more energy used in that hour.")

    left, right = st.columns(2)
    show(band_chart(df, "load", LOAD_LEVELS, "Do busier stations use more energy?", "How busy the station was"), left)
    show(band_chart(df, "TXpower", POWER_LEVELS, "Does a stronger signal use more energy?",
                    "Signal strength (transmit power)"), right)
    st.caption("Readings are sorted into five equal groups and the average energy is shown for each group. "
               "If the bars are nearly the same height, that factor makes little difference to energy use. "
               "If energy hardly drops when a station is quiet, there is room to save energy.")

    heat = df.pivot_table(index="dow", columns="hour", values="Energy")
    heat.index = [DAYS[i] for i in heat.index]
    show(px.imshow(heat, aspect="auto", color_continuous_scale="Blues",
                   title="Energy used by day of the week and hour of the day",
                   labels=dict(x="Hour of the day", y="Day", color="Energy used")))
    st.caption("Darker squares mean more energy. Look for patterns that repeat from day to day.")

    # Correlation with energy turned into a 0 to 100% strength. The sign is dropped because the
    # chart only shows how strong the link is, and fillna(0) covers columns that never change.
    links = df[["load", "TXpower", "ESMODE"]].corrwith(df["Energy"]).abs().fillna(0) * 100
    links = links.rename({"load": "How busy the station is", "TXpower": "Signal strength",
                          "ESMODE": "Power-saving mode"}).sort_values()
    show(px.bar(x=links.values, y=links.index, orientation="h", range_x=[0, 100],
                title="What is most closely linked to energy use?",
                labels={"x": "Strength of link (%)", "y": ""}))
    st.caption("A longer bar means that factor moves more closely with energy use. "
               "A strong link does not always mean it is the cause.")


def predictions_page(df):
    st.title("Energy Consumption Predictions")
    st.write("A computer model learns from the first 80% of the data, then predicts energy use for the "
             "remaining 20%, which it has not seen. Comparing its predictions with what really happened "
             "shows how far the predictions can be trusted.")
    res = train_model(df)
    test, model, baseline = res["test"], res["model"], res["baseline"]
    st.caption(f"Trained on {res['n_train']:,} readings and tested on {len(test):,} readings "
               f"from {pd.Timestamp(res['split_time']):%d %b %H:%M} onward.")

    # Show the error as a share of a typical reading so it makes sense without knowing the units.
    st.info(f"On average, a prediction is off by {model['MAE']:.1f} energy units, "
            f"which is {model['MAE'] / test['Energy'].mean():.1%} of a typical reading.")

    # Deltas compare against the simple baseline. Lower is better for the error metrics, so
    # their colour is inverted. The R2 difference is shown in percentage points.
    a, b, c = st.columns(3)
    a.metric("Average miss", f"{model['MAE']:.2f}", f"{model['MAE'] - baseline['MAE']:+.2f} vs simple guess",
             delta_color="inverse", help="How far off the prediction is on average. Lower is better.")
    b.metric("Typical big miss", f"{model['RMSE']:.2f}", f"{model['RMSE'] - baseline['RMSE']:+.2f} vs simple guess",
             delta_color="inverse", help="Like the average miss, but it counts large mistakes more heavily. Lower is better.")
    c.metric("Accuracy score", f"{model['R2']:.0%}", f"{(model['R2'] - baseline['R2']) * 100:+.0f} points vs simple guess",
             help="How much of the ups and downs in energy use the model explains. 100% is perfect, 0% is no better than guessing the average.")
    st.caption("The simple guess always predicts the average energy use. A useful model should beat it clearly.")

    network = test.groupby("Time")[["Energy", "Predicted"]].sum().reset_index()
    show(actual_vs_predicted(network, "Total energy for the whole network: real vs predicted"))
    st.caption("The closer the dashed line follows the solid line, the better the predictions.")

    station = st.selectbox("Choose a base station", sorted_stations(test["BS"]))
    show(actual_vs_predicted(test[test["BS"] == station], f"{station}: real vs predicted energy used"))

    # Percentage error for each reading. Zero values become NaN to avoid dividing by zero.
    pct_error = (test["Predicted"] - test["Energy"]).abs() / test["Energy"].abs().replace(0, np.nan)
    within = pd.Series({f"Within {p}%": (pct_error <= p / 100).mean() * 100 for p in (5, 10, 20)})
    # Average error per station, to show where the model struggles most.
    worst = ((test["Predicted"] - test["Energy"]).abs().groupby(test["BS"]).mean()
             .sort_values(ascending=False).head(10).round(2).rename("Average miss"))

    left, right = st.columns(2)
    show(px.bar(x=within.index, y=within.values, range_y=[0, 100],
                title="How close were the predictions?",
                labels={"x": "", "y": "Share of predictions (%)"}), left)
    left.caption("For example, the 'Within 10%' bar is the share of predictions that were no more than "
                 "10% away from the real value.")
    right.write("Base stations with the largest average miss")
    right.dataframe(worst.rename_axis("Base station"), width="stretch")


def saving_page(df):
    st.title("Energy-Saving Opportunities")
    st.write("Base stations use a lot of energy even when few people are using the network. This page finds "
             "quiet moments where a station could be switched to a power-saving mode, and estimates how much "
             "energy that might save.")
    st.caption("A saving opportunity is a quiet reading where no power-saving mode was already on. "
               "The saving percentage is an assumption you can adjust, not a measured value.")

    left, right = st.columns(2)
    percentile = left.slider("What counts as quiet? (the quietest % of readings)", 5, 50, 25)
    saving_pct = right.slider("Energy saved if a quiet station uses power-saving mode (%)", 5, 50, 20)

    # 'Quiet' is defined as a percentile of load, so the threshold adapts to whatever data is loaded.
    threshold = df["load"].quantile(percentile / 100)
    low = df["load"] <= threshold
    # An opportunity is a quiet reading where no power-saving mode was already on, since a
    # station that is already saving energy has less to gain.
    candidates = df[low & (df["ESMODE"] <= 0)]
    # The saving is an assumption: the chosen percentage of the energy used in those readings.
    estimated = candidates["Energy"].sum() * saving_pct / 100

    kpis({"Quiet means busyness below": f"{threshold:.3f}", "Readings counted as quiet": f"{low.mean():.1%}",
          "Saving opportunities": f"{len(candidates):,}",
          "Estimated energy saved": f"{estimated:,.0f} ({estimated / df['Energy'].sum():.1%} of total)"})

    # Compare quiet readings with the rest to show how little energy drops when stations are quiet.
    quiet, other = df[low]["Energy"].mean(), df[~low]["Energy"].mean()
    # Guard against dividing by zero (or an empty comparison group).
    if other > 0:
        st.info(f"In quiet periods a base station still uses {quiet / other:.0%} of the energy it uses "
                f"at other times, even though it is carrying much less traffic.")

    # Reindex so hours with no opportunities still appear as empty bars.
    by_hour = candidates.groupby("hour").size().reindex(range(24), fill_value=0)
    left, right = st.columns(2)
    show(px.bar(x=by_hour.index, y=by_hour.values, title="When do the saving opportunities happen?",
                labels={"x": "Hour of the day (0 = midnight)", "y": "Number of opportunities"}), left)
    show(px.bar(x=["Quiet periods", "Other periods"], y=[quiet, other],
                title="Average energy used: quiet vs other periods",
                labels={"x": "", "y": "Average energy used"}), right)
    st.caption("The left chart shows the times of day when stations are quiet enough to save energy. "
               "The right chart shows how much energy is still used when quiet.")

    # One row per station, ranked by estimated saving. Only the top 20 are shown here; the
    # CSV download below contains every opportunity.
    top = (candidates.groupby("BS")
           .agg(**{"Saving opportunities": ("Energy", "size"), "Average energy": ("Energy", "mean"),
                   "Average busyness": ("load", "mean"), "Energy in these readings": ("Energy", "sum")})
           .assign(**{"Estimated saving": lambda t: t["Energy in these readings"] * saving_pct / 100})
           .sort_values("Estimated saving", ascending=False).head(20).round(2)
           .rename_axis("Base station"))
    st.subheader("Top 20 base stations by estimated saving")
    st.dataframe(top, width="stretch")
    st.download_button("Download full list of saving opportunities (CSV)", candidates[REQUIRED].to_csv(index=False),
                       "saving_candidates.csv", "text/csv")


def explorer_page(df):
    st.title("Base Station Explorer")
    st.write("Pick one base station to see how much energy it uses and how busy it is over time.")
    station = st.selectbox("Choose a base station", sorted_stations(df["BS"]))
    one = df[df["BS"] == station]
    kpis({"Average energy": f"{one['Energy'].mean():.1f}", "Average busyness": f"{one['load'].mean():.2f}",
          "Signal strength": f"{one['TXpower'].mean():.2f}",
          "Readings in power-saving mode": f"{(one['ESMODE'] > 0).sum():,}"})

    left, right = st.columns(2)
    show(time_chart(one, "Energy", f"{station}: energy used", "Energy used"), left)
    show(time_chart(one, "load", f"{station}: how busy it was", "Busyness"), right)

    ranking = df.groupby("BS")["Energy"].mean().sort_values(ascending=False)
    st.subheader("How does this compare with other base stations?")
    left, right = st.columns(2)
    show(px.bar(ranking.head(15), title="Base stations that use the most energy",
                labels={"value": "Average energy used", "BS": "Base station"}), left)
    show(px.bar(ranking.tail(15), title="Base stations that use the least energy",
                labels={"value": "Average energy used", "BS": "Base station"}), right)


# App
# Maps sidebar names to page functions, so adding a page only needs one new entry here.
PAGES = {
    "Overview": overview_page,
    "Energy patterns": patterns_page,
    "Predictions": predictions_page,
    "Saving opportunities": saving_page,
    "Base station explorer": explorer_page,
}


def main():
    st.sidebar.title("Energy Dashboard")
    uploaded = st.sidebar.file_uploader("Load a dataset (CSV)", type="csv",
                                        help="CSV file with the columns listed in REQUIRED")
    # Show a readable message if the file has the wrong columns or format, instead of a traceback.
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