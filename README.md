# Heart Disease Risk Prediction System

A college mini-project: a machine-learning classifier on the UCI Heart Disease (Cleveland) dataset with a Streamlit dashboard.

> **Disclaimer:** This is an educational prototype, not a clinically validated medical device. Its output is not a diagnosis or a calibrated personal risk estimate and must not guide treatment decisions.

## Folder structure

```
heart_disease_project/
├── app.py                  # Streamlit dashboard
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py           # paths, feature schema, encodings, valid ranges
│   ├── data_loader.py      # dataset loading (cache -> ucimlrepo -> UCI URL) + binary target
│   ├── eda.py              # exploratory analysis, saves figures
│   └── train.py            # training, CV model selection, evaluation, saves pipeline
├── data/                   # cached dataset CSV (created on first run)
├── models/                 # heart_disease_pipeline.joblib, metrics.json, model_comparison.csv
├── reports/figures/        # EDA and evaluation plots
└── docs/
    ├── project_report.md
    └── viva_questions.md
```

## Dataset

UCI Heart Disease, dataset ID 45 (Cleveland subset, 303 rows, 13 features).

| Column | Meaning |
|---|---|
| age | Age in years |
| sex | 1 = male, 0 = female |
| cp | Chest pain type: 1 typical angina, 2 atypical angina, 3 non-anginal, 4 asymptomatic |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (1 = true) |
| restecg | Resting ECG: 0 normal, 1 ST-T abnormality, 2 LV hypertrophy |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina (1 = yes) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak-exercise ST segment: 1 up, 2 flat, 3 down |
| ca | Major vessels (0-3) coloured by fluoroscopy (has missing values) |
| thal | 3 normal, 6 fixed defect, 7 reversible defect (has missing values) |
| num | Original target, 0-4 |

**Target encoding:** `target = 0` if `num == 0`, else `target = 1` (`num > 0`). These are dataset labels, not a confirmed clinical diagnosis.

## Run on Windows PowerShell

```powershell
cd heart_disease_project

# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 2. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. (Optional) Exploratory data analysis -> reports/figures
python src/eda.py

# 4. Train, evaluate and save the model
python src/train.py

# 5. Launch the dashboard
streamlit run app.py
```

Always run commands from the `heart_disease_project` folder.

## Method summary

- Stratified 80/20 train/test split (`random_state=42`).
- One scikit-learn `Pipeline` per model: imputation + scaling/one-hot encoding + classifier. Because preprocessing is inside the pipeline, cross-validation refits it on each training fold only (no leakage).
- Logistic Regression and Random Forest compared by 5-fold stratified CV ROC-AUC (small grid search) on training data only.
- The selected model is evaluated once on the held-out test set: accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
- Whole fitted pipeline saved with joblib.

Results are produced by `python src/train.py` and written to `models/metrics.json`; none are hard-coded here.

## Troubleshooting

| Problem | Fix |
|---|---|
| **Dataset download error** (`Could not load the dataset`) | Check internet/proxy. Run `python src/data_loader.py` to see which source failed. Manual fix: download `processed.cleveland.data` from the UCI archive link printed in the error, add a header row `age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,num`, replace `?` with blanks, and save as `data/heart_disease_raw.csv`. Once that file exists, no download is needed. |
| **Missing columns** (`Dataset is missing required columns`) | The CSV header must match the 13 feature names plus `num` exactly (lower-case). Delete `data/heart_disease_raw.csv` to force a fresh download. |
| **Model file not found** | Run `python src/train.py` from the project folder; confirm `models/heart_disease_pipeline.joblib` exists, then restart Streamlit. |
| **`ModuleNotFoundError`** | Activate the venv, then `pip install -r requirements.txt`. Run scripts from the project root (e.g. `python src/train.py`). |
| **PowerShell cannot activate venv** | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. |
| **Version errors after upgrading scikit-learn** | A model saved with one version may not load in another. Re-run `python src/train.py`. |
| **`streamlit` not recognised** | Use `python -m streamlit run app.py`. |
| **Port already in use** | `streamlit run app.py --server.port 8502` |
