# Fetal Health Web App

A simple Streamlit app for the `fetal_health.csv` dataset.

## What it does
- Loads the fetal health dataset
- Trains a Random Forest classifier
- Shows dataset preview and summary statistics
- Provides an interactive prediction form

## Setup
1. Move `fetal_health.csv` into this folder (`C:\Users\Pc\fetal_health_app`) or upload it in the app.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

## Notes
- The app supports uploading your own copy of `fetal_health.csv`.
- If the local file is present, it will load automatically.
