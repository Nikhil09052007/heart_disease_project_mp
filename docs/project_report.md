# Heart Disease Risk Prediction System: Mini-Project Report

**Student name:** ____________________  **Roll no.:** ____________  
**Course:** B.Tech Information Technology  **Semester:** ______  
**Guide:** ____________________  **Institution:** ____________________  **Date:** ____________

## Abstract
This project builds a machine-learning application that classifies records from the UCI Heart Disease (Cleveland) dataset into "no disease" and "disease present" classes using 13 clinical attributes. Logistic Regression and Random Forest are compared with stratified cross-validation, the selected model is evaluated on a held-out test set, and the fitted pipeline is deployed in a Streamlit dashboard. The system is an educational prototype and is not a medical device.

## 1. Introduction
Cardiovascular disease is a major cause of mortality worldwide. Machine-learning methods can find patterns in clinical measurements, which makes the topic a useful learning exercise. This project demonstrates a complete, leakage-safe ML workflow from data loading to a user interface.

## 2. Objectives
1. Load and understand the UCI Heart Disease dataset.
2. Perform exploratory data analysis.
3. Build a preprocessing-and-model pipeline without data leakage.
4. Compare Logistic Regression and Random Forest.
5. Evaluate honestly on a held-out test set.
6. Deploy the model in an interactive dashboard with a clear disclaimer.

## 3. Tools and technologies
Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Streamlit, joblib, ucimlrepo.

## 4. Dataset
- Source: UCI Machine Learning Repository, Heart Disease (ID 45), Cleveland subset.
- 303 records, 13 features, one target `num` (0-4).
- Missing values are present in `ca` and `thal`.
- Binary label: `target = 0` if `num = 0`, else `1`. These are dataset labels, not a confirmed clinical diagnosis.
- Feature dictionary: see README.md.

## 5. System design
Data loader (`src/data_loader.py`) -> EDA (`src/eda.py`) -> Training pipeline (`src/train.py`) -> Saved pipeline (`models/*.joblib`) -> Streamlit app (`app.py`). All feature names and encodings live in `src/config.py`.

## 6. Methodology
1. **Split:** stratified 80/20 train/test, `random_state = 42`.
2. **Preprocessing (inside the Pipeline):** numeric features: median imputation + standard scaling; categorical features: most-frequent imputation + one-hot encoding (`handle_unknown="ignore"`).
3. **Models:** Logistic Regression; Random Forest.
4. **Model selection:** 5-fold stratified cross-validation with a small grid search on the training set, scoring ROC-AUC.
5. **Final evaluation:** selected model evaluated once on the test set.
6. **Leakage prevention:** preprocessing is fitted only on training folds because it is part of the pipeline.
7. **Persistence:** full pipeline saved with joblib.

## 7. Exploratory data analysis
Insert figures from `reports/figures/`:
- Missing values (01), class distribution (02), numeric distributions (03), boxplots (04), categorical counts (05), correlation heatmap (06).

Observations (fill in after running `python src/eda.py`): ______________________________

## 8. Results
*Fill in only after running `python src/train.py` (values are in `models/metrics.json` and `models/model_comparison.csv`).*

**Cross-validation (training data)**

| Model | CV ROC-AUC (mean ± std) | CV accuracy | CV F1 | Best parameters |
|---|---|---|---|---|
| Logistic Regression | | | | |
| Random Forest | | | | |

**Selected model:** ____________________

**Held-out test results**

| Metric | Value |
|---|---|
| Accuracy | |
| Precision | |
| Recall | |
| F1-score | |
| ROC-AUC | |

**Confusion matrix:** TN = ____  FP = ____  FN = ____  TP = ____

Insert figures 07 (confusion matrix), 08 (ROC curve), 09 (feature importance).

**Discussion (write after seeing results):** ______________________________

## 9. Application
Screenshots of the Streamlit app: prediction tab, performance tab, project info tab (insert here).

## 10. Limitations
- Only 303 records from one hospital; test set is small, so metrics are noisy.
- Labels are angiographic dataset labels, not clinical diagnoses.
- Model scores are not calibrated probabilities and are not personal risk estimates.
- No external validation; possible demographic bias.
- Historical data; not validated for current clinical populations.

## 11. Ethical considerations and disclaimer
This is an educational prototype, not a clinically validated medical device. Its output is not a diagnosis or a calibrated personal risk estimate and must not guide treatment decisions.

## 12. Future scope
External validation, calibration, additional UCI subsets, more models, explainability (SHAP), and clinical review.

## 13. Conclusion
(Write after running: summarise the selected model, its test performance, and the main limitation.)

## 14. References
1. Detrano, R. et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology*, 64(5), 304-310.
2. UCI Machine Learning Repository: Heart Disease dataset, https://archive.ics.uci.edu/dataset/45/heart+disease
3. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825-2830.
4. Streamlit documentation, https://docs.streamlit.io
