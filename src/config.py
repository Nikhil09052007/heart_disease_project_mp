"""Central configuration: paths, feature schema, encodings and valid ranges.

Every other file (data loader, EDA, training, Streamlit app) imports from here,
so the feature names, order and encodings are defined in exactly ONE place.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA_PATH = DATA_DIR / "heart_disease_raw.csv"      # cached copy of the dataset
MODEL_PATH = MODELS_DIR / "heart_disease_pipeline.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"

# ---------------------------------------------------------------- dataset
UCI_DATASET_ID = 45
UCI_FALLBACK_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
SOURCE_TARGET_COL = "num"   # original UCI target: 0 = no disease, 1-4 = disease present
TARGET_COL = "target"       # our binary label: 0 = source target 0, 1 = source target > 0

# Column order of the raw Cleveland file (used by the URL fallback)
RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", SOURCE_TARGET_COL,
]

# ---------------------------------------------------------------- feature schema
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
# Exact column order the model is trained on and the app must supply.
FEATURE_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                   "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# ---------------------------------------------------------------- encodings (human-readable)
FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type (1-4)",
    "trestbps": "Resting blood pressure (mm Hg) on admission",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
    "restecg": "Resting ECG result (0-2)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina (1 = yes, 0 = no)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of the peak-exercise ST segment (1-3)",
    "ca": "Number of major vessels (0-3) coloured by fluoroscopy",
    "thal": "Thalassemia test (3 = normal, 6 = fixed defect, 7 = reversible defect)",
}

SEX_LABELS = {0: "Female", 1: "Male"}
CP_LABELS = {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"}
FBS_LABELS = {0: "No (<= 120 mg/dl)", 1: "Yes (> 120 mg/dl)"}
RESTECG_LABELS = {0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"}
EXANG_LABELS = {0: "No", 1: "Yes"}
SLOPE_LABELS = {1: "Upsloping", 2: "Flat", 3: "Downsloping"}
THAL_LABELS = {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}

CATEGORY_LABELS = {
    "sex": SEX_LABELS, "cp": CP_LABELS, "fbs": FBS_LABELS, "restecg": RESTECG_LABELS,
    "exang": EXANG_LABELS, "slope": SLOPE_LABELS, "thal": THAL_LABELS,
}

# Allowed values for categorical features / plausible ranges for numeric ones.
# Used by the app to validate inputs before they reach the model.
VALID_CATEGORIES = {col: sorted(labels) for col, labels in CATEGORY_LABELS.items()}
NUMERIC_RANGES = {              # (min, max, default)
    "age": (20, 100, 54),
    "trestbps": (80, 220, 130),
    "chol": (100, 600, 240),
    "thalach": (60, 220, 150),
    "oldpeak": (0.0, 7.0, 1.0),
    "ca": (0, 3, 0),
}

# ---------------------------------------------------------------- modelling
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

DISCLAIMER = (
    "This is an educational prototype, not a clinically validated medical device. "
    "Its output is not a diagnosis or a calibrated personal risk estimate and must "
    "not be used to guide treatment decisions. Consult a qualified clinician for "
    "any health concern."
)
