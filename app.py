import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_PATH = "fetal_health.csv"
HEALTH_LABELS = {1: "Normal", 2: "Suspect", 3: "Pathological"}
FEATURE_DEFAULT_RANGES = {
    "baseline_value": (80.0, 160.0),
    "accelerations": (0.0, 30.0),
    "fetal_movement": (0.0, 10.0),
    "uterine_contractions": (0.0, 10.0),
    "light_decelerations": (0.0, 10.0),
    "severe_decelerations": (0.0, 10.0),
    "prolongued_decelerations": (0.0, 10.0),
    "change_in_baseline": (0.0, 10.0),
    "abnormal_short_term_variability": (0.0, 10.0),
    "mean_short_term_variability": (0.1, 5.0),
    "percentage_of_time_with_abnormal_long_term_variability": (0.0, 100.0),
    "mean_long_term_variability": (0.5, 5.0),
    "histogram_width": (0.0, 50.0),
    "histogram_min": (0.0, 100.0),
    "histogram_max": (100.0, 200.0),
    "histogram_number_of_peaks": (0.0, 20.0),
    "histogram_number_of_zeroes": (0.0, 20.0),
    "histogram_mode": (0.0, 100.0),
    "histogram_mean": (0.0, 100.0),
    "histogram_median": (0.0, 100.0),
    "histogram_variance": (0.0, 50.0),
}


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def get_default_value(feature: str, df: pd.DataFrame) -> float:
    return float(df[feature].median())


@st.cache_data
def load_and_prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=["fetal_health"])
    y = df["fetal_health"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


@st.cache_resource
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def format_feature_name(name: str) -> str:
    return name.replace("_", " ").capitalize()


def main() -> None:
    st.set_page_config(page_title="Fetal Health Classifier", layout="wide")
    st.title("Fetal Health Classification Web App")
    st.write(
        "Enter values for the fetal cardiotocography features below. "
        "The AI model trained on real fetal health data will predict the fetal health status."
    )

    df = load_data()
    st.sidebar.header("Model Information")
    st.sidebar.write(f"Dataset: {len(df)} records")
    X_train, X_test, y_train, y_test = load_and_prepare_data(df)
    model = train_model(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    st.sidebar.metric("Model Accuracy", f"{accuracy:.2%}")

    st.subheader("Dataset Overview")
    st.write(f"Total records: {df.shape[0]} | Features: {df.shape[1]}")
    st.dataframe(df.head(5), use_container_width=True)

    st.markdown("---")
    st.subheader("Fetal Health Prediction")
    st.write("Fill in the fetal cardiotocography feature values and click Predict to get the health status.")

    with st.form("prediction_form"):
        cols = st.columns(2)
        input_data = {}
        feature_columns = [col for col in df.columns if col != "fetal_health"]

        for idx, feature in enumerate(feature_columns):
            default_value = get_default_value(feature, df)
            label = format_feature_name(feature)
            input_data[feature] = cols[idx % 2].number_input(
                label,
                value=default_value,
                format="%.3f",
                step=0.1,
                key=feature,
            )

        submitted = st.form_submit_button("Predict Fetal Health")

    if submitted:
        input_df = pd.DataFrame([input_data])
        prediction = int(model.predict(input_df)[0])
        proba = model.predict_proba(input_df)[0]
        st.success(f"Predicted Fetal Health Status: **{HEALTH_LABELS[prediction]}**")

        col1, col2, col3 = st.columns(3)
        for i, label in enumerate(["Normal", "Suspect", "Pathological"]):
            with col1 if i == 0 else (col2 if i == 1 else col3):
                st.metric(label, f"{proba[i]:.1%}")

    st.markdown("---")
    st.subheader("Health Status Distribution in Dataset")
    dist_data = df["fetal_health"].map(HEALTH_LABELS).value_counts()
    st.bar_chart(dist_data)


if __name__ == "__main__":
    main()
