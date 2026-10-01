"""
Streamlit front end for the Wine Quality API
Run: streamlit run src/wine_api/ui.py
"""

import pandas as pd
import requests
import streamlit as st

from wine_api.config import settings
from wine_api.data import FEATURES

BATCH_SIZE = 100

# label, min, max, step, display format
FIELDS = {
    "fixed_acidity": ("Fixed acidity (g/dm3)", 4.0, 16.0, 0.1, "%.1f"),
    "volatile_acidity": ("Volatile acidity (g/dm3)", 0.1, 1.6, 0.01, "%.2f"),
    "citric_acid": ("Citric acid (g/dm3)", 0.0, 1.0, 0.01, "%.2f"),
    "residual_sugar": ("Residual sugar (g/dm3)", 0.5, 16.0, 0.1, "%.1f"),
    "chlorides": ("Chlorides (g/dm3)", 0.01, 0.62, 0.001, "%.3f"),
    "free_sulfur_dioxide": ("Free sulfur dioxide (mg/dm3)", 1.0, 72.0, 1.0, "%.0f"),
    "total_sulfur_dioxide": ("Total sulfur dioxide (mg/dm3)", 6.0, 290.0, 1.0, "%.0f"),
    "density": ("Density (g/cm3)", 0.990, 1.004, 0.0001, "%.4f"),
    "pH": ("pH", 2.7, 4.1, 0.01, "%.2f"),
    "sulphates": ("Sulphates (g/dm3)", 0.3, 2.0, 0.01, "%.2f"),
    "alcohol": ("Alcohol (% vol)", 8.0, 15.0, 0.1, "%.1f"),
}

# Rows taken from the UCI red wine dataset
PRESETS = {
    "Scored 3": [11.6, 0.58, 0.66, 2.2, 0.074, 10.0, 47.0, 1.0008, 3.25, 0.57, 9.0],
    "Scored 5": [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4],
    "Scored 8": [7.9, 0.35, 0.46, 3.6, 0.078, 15.0, 37.0, 0.9973, 3.35, 0.86, 12.8],
}

CATEGORY_TEXT = {"low": "Low (4 or less)", "average": "Average (5 to 6)", "high": "High (7 or more)"}

st.set_page_config(page_title="Wine Quality", layout="wide")


def api(method: str, path: str, **kwargs) -> dict:
    response = requests.request(method, f"{st.session_state.api_url}{path}", timeout=15, **kwargs)
    if response.status_code == 422:
        detail = response.json().get("detail", [])
        raise ValueError("; ".join(f"{'.'.join(map(str, d['loc'][1:]))}: {d['msg']}" for d in detail))
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_model_info(api_url: str) -> dict:
    return requests.get(f"{api_url}/model-info", timeout=10).json()


def apply_preset(values: list[float]) -> None:
    for name, value in zip(FEATURES, values):
        st.session_state[name] = value


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    return df


def predict_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = df[FEATURES].astype(float).to_dict("records")
    predictions = []
    for i in range(0, len(rows), BATCH_SIZE):
        predictions += api("POST", "/predict/batch", json={"samples": rows[i : i + BATCH_SIZE]})["predictions"]
    out = pd.DataFrame(predictions).add_prefix("predicted_")
    out["predicted_warnings"] = out["predicted_warnings"].map("; ".join)
    return pd.concat([df.reset_index(drop=True), out], axis=1)


def sidebar() -> dict | None:
    st.sidebar.title("Wine Quality")
    st.sidebar.text_input("API URL", value=settings.api_url, key="api_url")
    try:
        info = fetch_model_info(st.session_state.api_url)
    except (requests.RequestException, ValueError):
        st.sidebar.error("API is not reachable")
        return None
    st.sidebar.success(f"Connected, model {info['model_version']}")
    return info


def predict_tab() -> None:
    st.subheader("Single wine")
    st.caption("Load a real example or move the sliders. The prediction updates on every change.")

    cols = st.columns(len(PRESETS))
    for col, (label, values) in zip(cols, PRESETS.items()):
        col.button(label, on_click=apply_preset, args=(values,), use_container_width=True)

    for name, value in zip(FEATURES, PRESETS["Scored 5"]):
        st.session_state.setdefault(name, value)

    left, right = st.columns([3, 2], gap="large")
    with left:
        grid = st.columns(2)
        for i, name in enumerate(FEATURES):
            label, lo, hi, step, fmt = FIELDS[name]
            grid[i % 2].slider(label, lo, hi, step=step, format=fmt, key=name)

    payload = {name: float(st.session_state[name]) for name in FEATURES}
    with right:
        try:
            result = api("POST", "/predict", json=payload)
        except ValueError as exc:
            st.error(f"Invalid input: {exc}")
            return
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")
            return

        st.metric("Predicted quality", f"{result['quality']:.2f}")
        st.progress(min(result["quality"] / 10, 1.0))
        st.write(f"Category: **{CATEGORY_TEXT[result['category']]}**")
        st.write(f"Likely range: {result['low']:.2f} to {result['high']:.2f}")
        st.caption("Range is the 10th to 90th percentile of the individual trees in the forest.")
        for message in result["warnings"]:
            st.warning(message)


def batch_tab() -> None:
    st.subheader("Batch prediction")
    st.caption(f"Upload a CSV with these columns: {', '.join(FEATURES)}. Spaces in names and ; separators both work.")
    upload = st.file_uploader("CSV file", type="csv")
    if upload is None:
        return

    df = normalize_columns(pd.read_csv(upload, sep=None, engine="python"))
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        return
    if df[FEATURES].isna().any().any():
        st.error("The feature columns contain empty values. Fill or drop those rows first.")
        return

    try:
        with st.spinner(f"Predicting {len(df)} rows"):
            result = predict_frame(df)
    except (ValueError, requests.RequestException) as exc:
        st.error(f"Prediction failed: {exc}")
        return

    st.dataframe(result, use_container_width=True)
    left, right = st.columns([1, 3])
    left.download_button("Download CSV", result.to_csv(index=False), "predictions.csv", "text/csv")
    right.caption(f"{len(result)} rows, {(result['predicted_warnings'] != '').sum()} with out-of-range inputs")


def model_tab(info: dict) -> None:
    st.subheader("Model")
    metrics = info["metrics"]
    cols = st.columns(4)
    cols[0].metric("MAE", metrics.get("mae"), help="Mean absolute error on the held-out test set")
    cols[1].metric("Baseline MAE", metrics.get("baseline_mae"), help="Error of always predicting the training mean")
    cols[2].metric("R2", metrics.get("r2"))
    cols[3].metric("Within 0.5", f"{metrics.get('within_half_point', 0):.0%}")
    st.caption(f"Trained on {metrics.get('n_train')} rows, tested on {metrics.get('n_test')}.")

    st.markdown("**Feature importance**")
    importances = pd.Series(info["feature_importances"]).sort_values()
    st.bar_chart(importances, horizontal=True)


def main() -> None:
    info = sidebar()
    st.title("Red wine quality predictor")
    if info is None:
        st.info("Start the API first: `uvicorn wine_api.api.main:app --port 8000`")
        return
    single, batch, model = st.tabs(["Predict", "Batch", "Model"])
    with single:
        predict_tab()
    with batch:
        batch_tab()
    with model:
        model_tab(info)


main()
