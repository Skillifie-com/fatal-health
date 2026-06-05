import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_PATH = "fetal_health.csv"
HEALTH_LABELS = {1: "Normal", 2: "Suspect", 3: "Pathological"}
FEATURE_NAMES = [
    "baseline_value",
    "accelerations",
    "fetal_movement",
    "uterine_contractions",
    "light_decelerations",
    "severe_decelerations",
    "prolongued_decelerations",
    "change_in_baseline",
    "abnormal_short_term_variability",
    "mean_short_term_variability",
    "percentage_of_time_with_abnormal_long_term_variability",
    "mean_long_term_variability",
    "histogram_width",
    "histogram_min",
    "histogram_max",
    "histogram_number_of_peaks",
    "histogram_number_of_zeroes",
    "histogram_mode",
    "histogram_mean",
    "histogram_median",
    "histogram_variance",
]
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


def load_data(path: str = DATA_PATH) -> pd.DataFrame | None:
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def get_default_value(feature: str, df: pd.DataFrame | None) -> float:
    if df is not None and feature in df.columns:
        return float(df[feature].median())

    low, high = FEATURE_DEFAULT_RANGES.get(feature, (0.0, 1.0))
    return float((low + high) / 2)


def create_demo_dataset(n_samples: int = 300, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    data = {}

    for feature, (low, high) in FEATURE_DEFAULT_RANGES.items():
        if feature in {
            "fetal_movement",
            "uterine_contractions",
            "light_decelerations",
            "severe_decelerations",
            "prolongued_decelerations",
            "abnormal_short_term_variability",
            "histogram_number_of_peaks",
            "histogram_number_of_zeroes",
        }:
            data[feature] = rng.integers(int(low), int(high) + 1, size=n_samples).astype(float)
        else:
            data[feature] = rng.uniform(low, high, size=n_samples)

    X = pd.DataFrame(data)
    y = rng.choice([1, 2, 3], size=n_samples, p=[0.6, 0.25, 0.15])
    return train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)


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
        "This interactive app lets users upload a fetal health dataset or enter feature values manually. "
        "A RandomForest model is trained and used to predict fetal health from the entered values."
    )

    uploaded_file = st.file_uploader("Upload fetal_health.csv", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        demo_mode = False
    else:
        df = load_data()
        demo_mode = df is None

    if not demo_mode:
        st.subheader("Dataset Overview")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("---")
        st.subheader("Feature Summary")
        st.write(df.describe())

        st.sidebar.header("Model Training")
        st.sidebar.write("A RandomForest classifier is trained on the selected dataset.")
        X_train, X_test, y_train, y_test = load_and_prepare_data(df)
        model = train_model(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        st.sidebar.metric("Test accuracy", f"{accuracy:.2%}")
    else:
        st.warning(
            "No dataset found. The app is running in demo mode with synthetic example data. "
            "Upload a real fetal_health.csv file to train the model on actual fetal health data."
        )
        st.sidebar.header("Demo Mode")
        st.sidebar.write("A synthetic example dataset is used for interactive prediction.")
        X_train, X_test, y_train, y_test = create_demo_dataset()
        model = train_model(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        st.sidebar.metric("Demo accuracy", f"{accuracy:.2%}")

    st.markdown("---")
    st.subheader("Prediction")
    st.write(
        "Enter values for the fetal cardiotocography features below and hit Predict to get an output."
    )

    with st.form("prediction_form"):
        cols = st.columns(2)
        input_data = {}

        for idx, feature in enumerate(FEATURE_NAMES):
            default_value = get_default_value(feature, df if not demo_mode else None)
            label = format_feature_name(feature)
            input_data[feature] = cols[idx % 2].number_input(
                label,
                value=default_value,
                format="%.3f",
                step=0.1,
                key=feature,
            )

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([input_data])
        prediction = int(model.predict(input_df)[0])
        proba = model.predict_proba(input_df)[0]
        st.success(f"Predicted fetal health: **{HEALTH_LABELS[prediction]}**")

        prob_text = "\n".join(
            f"{HEALTH_LABELS[i+1]}: {prob:.2%}" for i, prob in enumerate(proba)
        )
        st.info(prob_text)

    st.markdown("---")
    st.subheader("Target Distribution")
    if not demo_mode:
        st.bar_chart(df["fetal_health"].value_counts().sort_index())
    else:
        st.bar_chart(pd.Series(y_train).value_counts().sort_index())


if __name__ == "__main__":
    main()
