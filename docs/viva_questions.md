# Viva Questions and Short Answers

1. **What does your project do?** It classifies records from the UCI Heart Disease dataset into no-disease (0) and disease-present (1) using clinical features, and serves the model through a Streamlit app.
2. **Which dataset did you use?** UCI Heart Disease, ID 45, Cleveland subset: 303 records, 13 features.
3. **How did you encode the target?** The original `num` is 0-4; label 0 if `num == 0`, label 1 if `num > 0`.
4. **Is the output a diagnosis?** No. It is a dataset-label prediction from an educational prototype, not a diagnosis or a calibrated risk estimate.
5. **Which columns had missing values?** `ca` and `thal`.
6. **How did you handle missing values?** Median imputation for numeric features and most-frequent imputation for categorical ones, inside the pipeline.
7. **What is data leakage?** Information from the test (or validation) data influencing training, e.g. fitting a scaler on all data before splitting.
8. **How did you avoid leakage?** All preprocessing is part of a scikit-learn Pipeline, so cross-validation refits it on each training fold only; the test set is used once.
9. **Why a stratified split?** It keeps the class proportions the same in train and test sets.
10. **Why standardise numeric features?** Logistic Regression is sensitive to feature scale; scaling helps optimisation and makes coefficients comparable.
11. **Why one-hot encode categorical features?** Codes like chest-pain type are labels, not ordered quantities; one-hot avoids implying a false order.
12. **What does `handle_unknown="ignore"` do?** Unseen categories at prediction time become all-zero rows instead of raising an error.
13. **Why Logistic Regression?** It is simple, fast, interpretable, and a strong baseline for tabular data.
14. **Why Random Forest?** It captures non-linear interactions and is robust, at the cost of interpretability.
15. **How did you select the model?** By mean 5-fold stratified cross-validated ROC-AUC on the training data only.
16. **What is cross-validation?** Splitting training data into k folds, training on k-1 and validating on the remaining fold, rotating through all folds.
17. **What is ROC-AUC?** The area under the ROC curve; the probability that a random positive is scored higher than a random negative. 0.5 is chance.
18. **Define precision and recall.** Precision = TP/(TP+FP); recall = TP/(TP+FN).
19. **Why is recall important here?** A false negative (missing a disease-present case) is usually costlier than a false positive.
20. **What is F1-score?** The harmonic mean of precision and recall.
21. **What does the confusion matrix show?** Counts of TN, FP, FN and TP, showing the kinds of errors the model makes.
22. **Is accuracy enough?** No; with imbalance or unequal error costs you also need precision, recall, F1 and ROC-AUC.
23. **What does the model score mean?** The predicted probability for class 1, uncalibrated; not a personal risk.
24. **What is calibration?** Whether predicted probabilities match observed frequencies (e.g. 70% scores are positive about 70% of the time).
25. **Why save the whole pipeline with joblib?** So prediction applies exactly the same preprocessing as training; no mismatch.
26. **How does the app ensure the correct features?** `config.py` defines the columns and encodings; the app builds a one-row DataFrame in that order and checks the model's `feature_names_in_`.
27. **What are the main limitations?** Small single-hospital dataset, noisy test metrics, no external validation, labels are not clinical diagnoses.
28. **How could you improve it?** More data, external validation, calibration, other models, explainability such as SHAP, clinical review.
29. **Why does `random_state` matter?** It makes the split and model training reproducible.
30. **What is overfitting and how did you reduce it?** Fitting noise in training data; reduced by cross-validation, regularisation (C in LR) and limiting tree depth/leaf size in RF.
