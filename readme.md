# AI-Doctor — Early Diabetes Risk Screening

Brief: AI-Doctor is a lightweight Python project that trains and ships a simple, interpretable early diabetes risk screening system. It provides a desktop GUI (`gui_app.py`), a Streamlit web UI (`web_app.py`), model training utilities and model explainability tools in `src/`.

**Key features:**
- Fast logistic model training with evaluation and calibration
- Preprocessing pipeline for clinical features
- Explainability using SHAP (`src/explainability.py`)
- Basic subgroup (bias) checks (`src/bias_checks.py`)
- Exportable models and feature schema (saved to `models/`)
- Two user interfaces: Tkinter desktop GUI and Streamlit web app

## Project structure

- `data/` — dataset(s). Primary CSV: `data/dataset.csv`.
- `models/` — saved model artifacts (populated after running training/save).
- `src/` — core code:
	- `preprocessing.py` — data cleaning and feature engineering
	- `model.py` — training routine (logistic regression)
	- `calibration.py` — probability calibration utilities
	- `explainability.py` — SHAP-based explanation helper
	- `bias_checks.py` — subgroup AUC checks
	- `save_model.py` / `save_schema.py` — persist models and schema
- `gui_app.py` — Tkinter desktop interface for quick screening
- `web_app.py` — Streamlit web interface
- `test.py` — example script that runs training, explains a sample and saves artifacts
- `requirements.txt` — Python dependencies

## Requirements

Install dependencies (recommended inside a virtual environment):

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Minimum dependencies (from `requirements.txt`): `streamlit`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `shap`.

## Quick start

1) Run the Streamlit web UI (recommended for quick evaluation):

```bash
streamlit run web_app.py
```

2) Or run the desktop GUI:

```bash
python gui_app.py
```

3) Train and save models / reproduce results (example):

```bash
python test.py
```

This will:
- load `data/dataset.csv`;
- run preprocessing (`src/preprocessing.py`);
- train a logistic model (`src/model.py`);
- calibrate it (`src/calibration.py`);
- calculate explainability for example patients (`src/explainability.py`);
- save models and schema to `models/` via `src/save_model.py` and `src/save_schema.py`.

## Usage notes

- Before using either UI, ensure the `models/` folder contains the saved artifacts: `logistic_base.pkl`, `logistic_calibrated.pkl`, `background.pkl`, and `feature_schema.pkl`. Running `python test.py` will produce these.
- The UIs align input features with the saved `feature_schema` before prediction.
- `explain_patient` in `src/explainability.py` uses a Linear SHAP explainer to return the top drivers for an individual risk estimate.

## Data

The sample dataset is `data/dataset.csv`. It contains demographic and clinical features (columns include `gender`, `age`, `hypertension`, `heart_disease`, `smoking_history`, `bmi`, `HbA1c_level`, `blood_glucose_level`, `diabetes`). Use this format for model training and evaluation.

## Development

- To iterate on preprocessing or model behaviour, modify the corresponding module in `src/` and re-run `python test.py`.
- Add unit tests or small notebooks as needed; `test.py` is a runnable example harness.

## Limitations & Disclaimer

This code provides an early risk estimation prototype for research and demonstration only. It is NOT a medical device and must not be used for diagnosis or clinical decision making. Real-world deployment requires medical validation, regulatory review, privacy safeguards and robust testing.

## Contributing

Feel free to open issues, suggest improvements, or submit PRs. Suggested improvements:
- add unit tests, CI, and packaging
- add dataset validation and richer schema checks
- integrate more robust explainability and fairness pipelines

## License & Credit

Add a license file as appropriate (MIT, Apache-2.0, etc.).

---

If you'd like, I can:
- run `python test.py` to generate model artifacts and confirm the UIs work locally, or
- expand sections (examples, API snippets, badges), or
- add a short CONTRIBUTING.md and LICENSE.
