
"""
app.py - Energy Clustering Prediction App
Run: python -m streamlit run app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------- Step 1: Page configuration ----------

st.set_page_config(
    page_title="Energy Cluster Predictor",
    page_icon="⚡",
    layout="wide"
)

# Find files relative to app.py
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "kmeans_notebook.joblib"
SCALER_PATH = BASE_DIR / "scaler_notebook.joblib"
DATA_PATH = BASE_DIR / "energy_data.csv"

FEATURES = ["energy_consumption", "temperature"]

COLORS = [
    "#00B894",
    "#0984E3",
    "#FDCB6E",
    "#E17055",
    "#6C5CE7",
    "#FD79A8"
]


# ---------- Step 2: Load model and scaler ----------

@st.cache_resource
def load_model_and_scaler():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}"
        )

    saved = joblib.load(MODEL_PATH)

    # If the saved file contains a dictionary
    if isinstance(saved, dict):

        if "model" not in saved:
            raise KeyError(
                "The saved model dictionary does not contain 'model'."
            )

        model = saved["model"]

        scaler = saved.get("scaler")

        features = saved.get("features", FEATURES)

    # If the saved file contains the K-Means model directly
    else:

        model = saved

        features = FEATURES

        scaler = None

    # Load scaler separately if it was not included
    if scaler is None:

        if not SCALER_PATH.exists():
            raise FileNotFoundError(
                f"Scaler file not found: {SCALER_PATH.name}"
            )

        scaler = joblib.load(SCALER_PATH)

    return model, scaler, features


try:

    model, scaler, FEATURES = load_model_and_scaler()

except Exception as e:

    st.error(f"Error loading model or scaler: {e}")

    st.stop()


# ---------- Step 3: Load energy dataset ----------

@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.name}"
        )

    data = pd.read_csv(DATA_PATH)

    missing_columns = [
        col for col in FEATURES
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {missing_columns}"
        )

    return data


try:

    data = load_data()

except Exception as e:

    st.error(f"Error loading dataset: {e}")

    st.stop()


# ---------- Step 4: Generate cluster labels ----------

# Make sure the dataset contains valid numeric values
data = data.dropna(subset=FEATURES).copy()

data[FEATURES] = data[FEATURES].apply(
    pd.to_numeric,
    errors="coerce"
)

data = data.dropna(subset=FEATURES)

if data.empty:

    st.error(
        "The dataset does not contain valid energy and temperature values."
    )

    st.stop()


# Scale dataset using the same scaler used during training
scaled_data = scaler.transform(data[FEATURES])

# Predict cluster numbers
data["cluster"] = model.predict(scaled_data)


# ---------- Step 5: Calculate cluster centres ----------

centres = pd.DataFrame(
    scaler.inverse_transform(model.cluster_centers_),
    columns=FEATURES
)

avg_energy = data["energy_consumption"].mean()

avg_temp = data["temperature"].mean()


# ---------- Step 6: Give clusters easy names ----------

names = {}
advice = {}

for i, row in centres.iterrows():

    energy_level = (
        "High"
        if row["energy_consumption"] > avg_energy
        else "Low"
    )

    weather = (
        "Hot"
        if row["temperature"] > avg_temp
        else "Cool"
    )

    names[i] = f"{energy_level} usage · {weather} weather"

    if energy_level == "High":

        advice[i] = (
            "Heavy load: shift usage away from peak hours "
            "and use efficient equipment."
        )

    else:

        advice[i] = (
            "Light load: good example of efficient energy usage."
        )


data["segment"] = data["cluster"].map(names)


# ---------- Step 7: Page style ----------

st.markdown(
    """
    <style>

    .hero {
        padding: 26px 30px;
        border-radius: 18px;
        color: white;
        margin-bottom: 20px;
        background: linear-gradient(
            120deg,
            #00B894,
            #0984E3 60%,
            #6C5CE7
        );
    }

    .hero h1 {
        margin: 0;
    }

    .hero p {
        margin: 6px 0 0;
        opacity: 0.95;
    }

    .result {
        padding: 22px;
        border-radius: 16px;
        background: #1A1F2B;
        border-left: 7px solid var(--c);
    }

    .result h2 {
        margin: 0;
        color: var(--c);
    }

    .result p {
        margin: 8px 0 0;
        color: #D0D6E0;
    }

    </style>

    <div class="hero">

        <h1>⚡ Energy Cluster Predictor</h1>

        <p>
        Enter an energy reading and temperature.
        The K-Means model tells you which group it belongs to.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ---------- Step 8: User input ----------

st.sidebar.header("🔧 Enter values")

energy = st.sidebar.slider(
    "Energy consumption (kWh)",
    50.0,
    500.0,
    275.0,
    1.0
)

temperature = st.sidebar.slider(
    "Temperature (°C)",
    15.0,
    35.0,
    25.0,
    0.1
)

st.sidebar.caption(
    "Ranges match the training data."
)


# ---------- Step 9: Predict cluster ----------

new_data = pd.DataFrame(
    [[energy, temperature]],
    columns=FEATURES
)

new_scaled = scaler.transform(new_data)

cluster = int(model.predict(new_scaled)[0])

color = COLORS[cluster % len(COLORS)]


# ---------- Step 10: Display prediction ----------

left, right = st.columns([1, 1.4])


with left:

    st.subheader("Prediction")

    st.markdown(
        f"""
        <div class="result" style="--c:{color}">

            <h2>Cluster {cluster}</h2>

            <b>{names[cluster]}</b>

            <p>{advice[cluster]}</p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    c1, c2 = st.columns(2)

    c1.metric(
        "Your energy",
        f"{energy:.0f} kWh",
        f"{energy - avg_energy:+.0f} vs average"
    )

    c2.metric(
        "Your temperature",
        f"{temperature:.1f} °C",
        f"{temperature - avg_temp:+.1f} vs average"
    )

    # Distance from input to cluster centres
    distances = (
        (new_scaled - model.cluster_centers_) ** 2
    ).sum(axis=1) ** 0.5

    st.caption(
        "Distance to each cluster centre (smaller = closer)"
    )

    st.bar_chart(
        pd.Series(
            distances,
            index=[
                f"Cluster {i}"
                for i in range(len(distances))
            ]
        )
    )


# ---------- Step 11: Visualize clusters ----------

with right:

    st.subheader("Where your reading falls")

    # Avoid sampling more rows than the dataset contains
    plot_data = data.sample(
        n=min(2500, len(data)),
        random_state=1
    )

    fig = px.scatter(
        plot_data,
        x="temperature",
        y="energy_consumption",
        color="segment",
        opacity=0.4,
        template="plotly_dark",
        labels={
            "temperature": "Temperature (°C)",
            "energy_consumption": "Energy (kWh)"
        }
    )

    # User's input
    fig.add_trace(
        go.Scatter(
            x=[temperature],
            y=[energy],
            mode="markers",
            name="Your input",
            marker=dict(
                size=20,
                color="white",
                symbol="star",
                line=dict(
                    width=2,
                    color="black"
                )
            )
        )
    )

    # Cluster centres
    fig.add_trace(
        go.Scatter(
            x=centres["temperature"],
            y=centres["energy_consumption"],
            mode="markers",
            name="Cluster centres",
            marker=dict(
                size=14,
                color="red",
                symbol="x"
            )
        )
    )

    fig.update_layout(
        height=470,
        legend_title_text="",
        margin=dict(
            l=0,
            r=0,
            t=10,
            b=0
        )
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


# ---------- Step 12: Predict from CSV ----------

st.divider()

st.subheader("📁 Predict from a CSV file")

st.caption(
    "Columns needed: energy_consumption, temperature"
)

file = st.file_uploader(
    "Upload CSV",
    type="csv"
)

if file:

    df_in = pd.read_csv(file)

    if set(FEATURES).issubset(df_in.columns):

        df_in[FEATURES] = df_in[FEATURES].apply(
            pd.to_numeric,
            errors="coerce"
        )

        valid_rows = df_in[FEATURES].notna().all(axis=1)

        if valid_rows.any():

            df_in.loc[valid_rows, "cluster"] = model.predict(
                scaler.transform(
                    df_in.loc[valid_rows, FEATURES]
                )
            )

            df_in["cluster"] = df_in["cluster"].astype("Int64")

            df_in["segment"] = df_in["cluster"].map(names)

            st.dataframe(
                df_in,
                width="stretch"
            )

            st.download_button(
                "⬇️ Download predictions",
                df_in.to_csv(index=False),
                "predictions.csv",
                mime="text/csv"
            )

        else:

            st.error(
                "The uploaded CSV contains no valid numeric readings."
            )

    else:

        st.error(
            f"Your CSV must contain these columns: {FEATURES}"
        )


# ---------- Step 13: Cluster summary ----------

with st.expander("📋 Cluster summary"):

    summary = data.groupby("segment").agg(
        readings=("cluster", "size"),
        avg_energy=("energy_consumption", "mean"),
        avg_temp=("temperature", "mean")
    ).round(1)

    st.dataframe(
        summary,
        width="stretch"
    )
