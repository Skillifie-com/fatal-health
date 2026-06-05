import os

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_PATH = "fetal_health.csv"
HEALTH_LABELS = {1: "Normal", 2: "Suspect", 3: "Pathological"}


def load_data(path: str = DATA_PATH) -> pd.DataFrame | None:
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_and_prepare_data(path: str = DATA_PATH) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] | tuple[None, None, None, None]:
    df = load_data(path)
    if df is None:
        return None, None, None, None

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
        "Upload the `fetal_health.csv` dataset or use the local file. "
        "The app trains a classification model and provides a prediction form."
    )

    uploaded_file = st.file_uploader("Upload fetal_health.csv", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = load_data()

    if df is None:
        st.warning(
            "No dataset found. Upload `fetal_health.csv` or place it in the app folder."
        )
        st.stop()

    st.subheader("Dataset Overview")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")
    st.subheader("Feature Summary")
    st.write(df.describe())

    st.sidebar.header("Model Training")
    st.sidebar.write("A RandomForest classifier is trained on the dataset.")

    X_train, X_test, y_train, y_test = load_and_prepare_data()
    if X_train is None:
        st.error("Unable to load training data. Please upload a valid fetal health CSV.")
        st.stop()

    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.sidebar.metric("Test accuracy", f"{accuracy:.2%}")

    st.markdown("---")
    st.subheader("Prediction")
    st.write(
        "Enter values for the fetal cardiotocography features and get a predicted fetal health class."
    )

    with st.form("prediction_form"):
        cols = st.columns(2)
        input_data = {}

        for idx, feature in enumerate(X_train.columns):
            default_value = float(df[feature].median())
            label = format_feature_name(feature)
            input_data[feature] = cols[idx % 2].number_input(
                label,
                value=default_value,
                format="%.3f",
                step=0.1,
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
    st.bar_chart(df["fetal_health"].value_counts().sort_index())


if __name__ == "__main__":
    main()
