"""
shap_analysis.py
-----------------
SHAP (SHapley Additive exPlanations) analysis.

MODEL CHOICE FOR THIS SECTION: our metrics table (see evaluate.py output)
actually shows LogReg_ClassWeight with the top AUC on this dataset, with
XGBoost_ClassWeight a close second. We use XGBoost_ClassWeight for the SHAP
walkthrough anyway, for a reason worth stating out loud in an interview:
LLogistic regression's coefficients ARE its explanation already (that's
precisely why simple linear/logistic models are still required in some
regulated credit contexts) -- there's little for SHAP to add there. SHAP is
most valuable for models like XGBoost that don't hand you an explanation for
free: it decomposes each individual prediction into additive per-feature
contributions, consistent with cooperative game theory (each feature is
treated as a "player" and its Shapley value is its average marginal
contribution across all possible feature orderings). That's the tool a
Risk/Decision Science team actually reaches for when they need to explain
an XGBoost line-item decision to a regulator or to an applicant who was
declined (adverse action reasons, in US lending law).

We report TWO views because they answer different questions:
  - Global importance (mean |SHAP value|): "which features matter most
    across the whole portfolio?" -- useful for model documentation/model
    risk management review.
  - Summary beeswarm: "which features push risk UP vs DOWN, and does the
    effect depend on the feature's value?" -- e.g. does low EXT_SOURCE
    always increase risk, or does its effect depend on income level too?
"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = "reports/figures"

if __name__ == "__main__":
    models = joblib.load("models/all_models.joblib")
    X_test = pd.read_parquet("data/processed/X_test.parquet")

    model_name = "XGBoost_ClassWeight"
    model = models[model_name]

    # TreeExplainer is exact and fast for tree ensembles (no sampling
    # approximation needed, unlike KernelExplainer for arbitrary models).
    explainer = shap.TreeExplainer(model)

    # Use a sample for speed/plot legibility -- 2000 rows is plenty to get
    # stable global importance estimates without waiting minutes.
    X_sample = X_test.sample(n=min(2000, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(X_sample)

    # --- Global importance bar chart -------------------------------------
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": X_sample.columns, "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)
    print("=" * 70)
    print(f"TOP 15 FEATURES DRIVING DEFAULT RISK ({model_name})")
    print("=" * 70)
    print(importance_df.head(15).to_string(index=False))
    importance_df.to_csv("reports/shap_feature_importance.csv", index=False)

    plt.figure(figsize=(8, 6))
    top15 = importance_df.head(15).iloc[::-1]
    plt.barh(top15["feature"], top15["mean_abs_shap"], color="#2A9D8F")
    plt.xlabel("Mean |SHAP value| (average impact on model output)")
    plt.title(f"Global Feature Importance -- {model_name}")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/06_shap_global_importance.png", dpi=120)
    plt.close()

    # --- Beeswarm summary plot -------------------------------------------
    plt.figure(figsize=(8, 7))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/07_shap_summary_beeswarm.png", dpi=120, bbox_inches="tight")
    plt.close()

    # --- One individual explanation, for concreteness --------------------
    # Picking a single high-risk applicant to show a "why was THIS applicant
    # flagged" explanation -- the kind of thing an adverse-action letter
    # needs to be able to justify.
    probs = model.predict_proba(X_sample)[:, 1]
    idx = np.argmax(probs)
    print(f"\nExample high-risk applicant (predicted default prob = {probs[idx]:.3f}):")
    single_shap = pd.DataFrame({
        "feature": X_sample.columns,
        "value": X_sample.iloc[idx].values,
        "shap_value": shap_values[idx]
    }).sort_values("shap_value", key=abs, ascending=False)
    print(single_shap.head(8).to_string(index=False))
    single_shap.to_csv("reports/shap_example_applicant.csv", index=False)

    print(f"\nSaved SHAP figures to {FIG_DIR}/ and CSVs to reports/")
