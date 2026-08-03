"""
modeling.py
-----------
Trains three models under TWO imbalance-handling strategies (class weighting
and SMOTE) = 6 model variants, so we can compare both the algorithm choice
and the imbalance strategy choice side by side.

WHY THESE THREE MODELS, AND WHY THEY BEHAVE DIFFERENTLY ON THIS DATA:

Logistic Regression
  Assumes a LINEAR relationship (in log-odds) between each feature and
  default risk. Credit risk data has real nonlinearities (e.g. very young
  AND very old applicants can both be risky in different ways; a
  credit-to-income ratio's effect isn't constant across income levels) and
  the label-encoded categoricals impose a fake ordinal structure. Expect
  LR to be the weakest of the three on AUC, but it remains valuable:
  coefficients are directly interpretable ("each unit increase in X moves
  the log-odds of default by exactly B"), which matters a lot in banking,
  where model risk / fair-lending regulators (e.g. ECOA, Reg B in the US)
  often require an explainable baseline model.

Random Forest
  An ensemble of de-correlated decision trees (bagging + random feature
  subsets). It captures nonlinearities and interactions LR can't, and is
  fairly robust to outliers and unscaled features. Its weakness here:
  individual trees are grown deep and can overfit noise, and RF's variance
  reduction (averaging many trees) is good but generally has HIGHER BIAS
  than a well-tuned boosting model on structured/tabular data like this
  because trees are grown independently rather than sequentially
  correcting each other's mistakes.

XGBoost
  Gradient-boosted trees grown SEQUENTIALLY, each new tree explicitly
  fitting the residual errors of the ensemble so far, plus L1/L2
  regularization on tree weights. On structured/tabular data (exactly the
  format of Home Credit) gradient boosting is very consistently the top
  performer in real Kaggle leaderboards for this reason, and that's what
  we expect to see here too. We also use XGBoost's own scale_pos_weight
  parameter as its native version of "class weighting."

We evaluate ONLY on AUC-ROC / precision / recall / PR-curve (see
evaluate.py) -- never accuracy, given the 91:9 imbalance established in
eda.py.
"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from preprocessing import get_class_weight_dict


def load_processed():
    X_train = pd.read_parquet("data/processed/X_train.parquet")
    X_test = pd.read_parquet("data/processed/X_test.parquet")
    y_train = pd.read_parquet("data/processed/y_train.parquet").iloc[:, 0]
    y_test = pd.read_parquet("data/processed/y_test.parquet").iloc[:, 0]
    X_train_smote = pd.read_parquet("data/processed/X_train_smote.parquet")
    y_train_smote = pd.read_parquet("data/processed/y_train_smote.parquet").iloc[:, 0]
    return X_train, X_test, y_train, y_test, X_train_smote, y_train_smote


def train_all_models(X_train, y_train, X_train_smote, y_train_smote):
    """Returns a dict of {model_name: fitted_model} for 6 variants."""
    models = {}
    class_weights = get_class_weight_dict(y_train)
    scale_pos_weight = class_weights[1] / class_weights[0]  # XGBoost's own convention

    # Logistic Regression needs scaled features (its coefficients / convergence
    # are sensitive to feature scale, unlike tree models which split on raw
    # thresholds and are scale-invariant). Fit scaler on train only.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_smote_scaled = scaler.transform(X_train_smote)  # same scaler, no re-fit (no leakage)

    # ---------- Class-weighted variants (weights applied to ORIGINAL data) ----------
    models["LogReg_ClassWeight"] = LogisticRegression(
        max_iter=1000, class_weight=class_weights, random_state=42
    ).fit(X_train_scaled, y_train)

    models["RandomForest_ClassWeight"] = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        class_weight=class_weights, random_state=42, n_jobs=-1
    ).fit(X_train, y_train)

    models["XGBoost_ClassWeight"] = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, subsample=0.8, colsample_bytree=0.8,
        eval_metric="auc", random_state=42, n_jobs=-1
    ).fit(X_train, y_train)

    # ---------- SMOTE variants (trained on oversampled data, NO class weight) --------
    models["LogReg_SMOTE"] = LogisticRegression(
        max_iter=1000, random_state=42
    ).fit(X_train_smote_scaled, y_train_smote)

    models["RandomForest_SMOTE"] = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        random_state=42, n_jobs=-1
    ).fit(X_train_smote, y_train_smote)

    models["XGBoost_SMOTE"] = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="auc", random_state=42, n_jobs=-1
    ).fit(X_train_smote, y_train_smote)

    return models, scaler


if __name__ == "__main__":
    import sys
    import time
    import os

    X_train, X_test, y_train, y_test, X_train_smote, y_train_smote = load_processed()
    class_weights = get_class_weight_dict(y_train)
    scale_pos_weight = class_weights[1] / class_weights[0]

    scaler_path = "models/scaler.joblib"
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler().fit(X_train)
        joblib.dump(scaler, scaler_path)
    X_train_scaled = scaler.transform(X_train)
    X_train_smote_scaled = scaler.transform(X_train_smote)

    single = sys.argv[1] if len(sys.argv) > 1 else None

    builders = {
        "LogReg_ClassWeight": lambda: LogisticRegression(
            max_iter=1000, class_weight=class_weights, random_state=42
        ).fit(X_train_scaled, y_train),
        "RandomForest_ClassWeight": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=20,
            class_weight=class_weights, random_state=42, n_jobs=-1
        ).fit(X_train, y_train),
        "XGBoost_ClassWeight": lambda: xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, subsample=0.8, colsample_bytree=0.8,
            eval_metric="auc", random_state=42, n_jobs=-1
        ).fit(X_train, y_train),
        "LogReg_SMOTE": lambda: LogisticRegression(
            max_iter=1000, random_state=42
        ).fit(X_train_smote_scaled, y_train_smote),
        "RandomForest_SMOTE": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=20,
            random_state=42, n_jobs=-1
        ).fit(X_train_smote, y_train_smote),
        "XGBoost_SMOTE": lambda: xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="auc", random_state=42, n_jobs=-1
        ).fit(X_train_smote, y_train_smote),
    }

    if single:
        names = [single]
    else:
        names = list(builders.keys())

    out_dir = "models/pieces"
    os.makedirs(out_dir, exist_ok=True)
    for name in names:
        t0 = time.time()
        model = builders[name]()
        joblib.dump(model, f"{out_dir}/{name}.joblib")
        print(f"{name}: trained in {time.time()-t0:.1f}s, saved to {out_dir}/{name}.joblib")

