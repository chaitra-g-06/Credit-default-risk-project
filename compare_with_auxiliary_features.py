"""
compare_with_auxiliary_features.py
------------------------------------
The key comparison this project is building toward: does engineering
features from bureau/previous-application/POS/credit-card history actually
improve the model, on the SAME train/test split, holding the algorithm and
imbalance strategy fixed (XGBoost, class-weighted -- the best-justified
variant from the baseline comparison)?

This is the single most convincing plot/number for an interview: it
directly measures the value of the "future work" this project originally
scoped out, instead of just asserting auxiliary tables would probably help.
"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
import xgboost as xgb

from preprocessing import load_and_clean, encode_and_impute, split_data, get_class_weight_dict

FIG_DIR = "reports/figures"


def train_and_eval_xgb(X_train, X_test, y_train, y_test, label):
    class_weights = get_class_weight_dict(y_train)
    scale_pos_weight = class_weights[1] / class_weights[0]
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, subsample=0.8, colsample_bytree=0.8,
        eval_metric="auc", random_state=42, n_jobs=-1
    ).fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    print(f"{label}: AUC-ROC = {auc:.4f}, Avg Precision = {avg_prec:.4f}, "
          f"n_features = {X_train.shape[1]}")
    return model, auc, avg_prec, (fpr, tpr)


if __name__ == "__main__":
    # ---------- Baseline: application_train.csv only ----------
    print("Building BASELINE (application-only) dataset...")
    df_base = load_and_clean(path="data/raw/application_train.csv")
    df_base_enc, _ = encode_and_impute(df_base)
    Xb_train, Xb_test, yb_train, yb_test = split_data(df_base_enc, seed=42)

    # ---------- Enhanced: application_train + auxiliary features ----------
    print("Building ENHANCED (application + auxiliary) dataset...")
    df_merged = pd.read_parquet("data/processed/application_train_with_aux.parquet")
    df_aux = load_and_clean(df=df_merged)   # applies same DAYS_EMPLOYED fix, ratio features, etc.
    df_aux_enc, _ = encode_and_impute(df_aux)
    Xa_train, Xa_test, ya_train, ya_test = split_data(df_aux_enc, seed=42)

    # Same random_state/seed on the same underlying row order -> the two
    # splits contain the SAME applicants in train and the SAME applicants in
    # test, so the comparison isn't confounded by different train/test rows.
    assert Xb_train.shape[0] == Xa_train.shape[0]
    assert (yb_test.values == ya_test.values).all()

    print("\nTraining baseline XGBoost (class-weighted)...")
    model_base, auc_base, ap_base, roc_base = train_and_eval_xgb(
        Xb_train, Xb_test, yb_train, yb_test, "Baseline (application-only)")

    print("Training enhanced XGBoost (class-weighted, + auxiliary features)...")
    model_aux, auc_aux, ap_aux, roc_aux = train_and_eval_xgb(
        Xa_train, Xa_test, ya_train, ya_test, "Enhanced (+ bureau/prev/POS/CC features)")

    lift = auc_aux - auc_base
    print(f"\nAUC-ROC lift from auxiliary features: {lift:+.4f} "
          f"({auc_base:.4f} -> {auc_aux:.4f})")

    # ---------- Plot ----------
    plt.figure(figsize=(7, 6))
    plt.plot(roc_base[0], roc_base[1], label=f"Baseline (AUC={auc_base:.3f})", linewidth=1.8)
    plt.plot(roc_aux[0], roc_aux[1], label=f"+ Auxiliary features (AUC={auc_aux:.3f})", linewidth=1.8)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Impact of Bureau/Previous-Application/POS/CC Features on AUC-ROC")
    plt.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/08_auxiliary_feature_lift.png", dpi=120)
    plt.close()

    # ---------- Which new features mattered most? ----------
    importances = pd.DataFrame({
        "feature": Xa_train.columns, "importance": model_aux.feature_importances_
    }).sort_values("importance", ascending=False)
    aux_feature_names = set(Xa_train.columns) - set(Xb_train.columns)
    new_feature_ranks = importances[importances["feature"].isin(aux_feature_names)].head(10)
    print("\nTop new (auxiliary) features by XGBoost importance:")
    print(new_feature_ranks.to_string(index=False))

    with open("reports/auxiliary_feature_lift.json", "w") as f:
        json.dump({
            "baseline_auc": auc_base, "enhanced_auc": auc_aux, "lift": lift,
            "baseline_avg_precision": ap_base, "enhanced_avg_precision": ap_aux,
            "baseline_n_features": int(Xb_train.shape[1]),
            "enhanced_n_features": int(Xa_train.shape[1]),
        }, f, indent=2)
    importances.to_csv("reports/enhanced_model_feature_importance.csv", index=False)
    print("\nSaved reports/auxiliary_feature_lift.json, "
          "reports/enhanced_model_feature_importance.csv, "
          f"{FIG_DIR}/08_auxiliary_feature_lift.png")
