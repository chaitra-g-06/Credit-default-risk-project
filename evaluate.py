"""
evaluate.py
-----------
Evaluates all 6 model variants on AUC-ROC, precision, recall, and F1 --
explicitly NOT accuracy (see eda.py for why: 91:9 imbalance means a
"predict everyone repays" model gets ~91% accuracy while catching zero
defaulters -- accuracy is meaningless here and would be a red flag if
reported alone in an interview).

WHAT PRECISION AND RECALL MEAN IN A LENDING CONTEXT (say this explicitly
in an interview -- it's the single most important conceptual point of
this whole project):

  Precision = of the loans we FLAG as high-risk, what fraction actually
  default?  Low precision means we're rejecting/flagging a lot of GOOD
  applicants who would have repaid -- this is a cost of LOST BUSINESS:
  foregone interest income, applicants who go to a competitor instead.

  Recall = of the loans that ACTUALLY default, what fraction did we catch?
  Low recall means we're APPROVING loans that go bad -- this is a cost of
  BAD DEBT: the full principal (or a large fraction, after recovery) is
  lost, which is typically a much larger dollar loss per incident than
  the missed interest from one rejected good applicant.

  In consumer lending, a single default typically costs far more than a
  single foregone approval (loss given default on an unsecured loan can be
  60-90% of principal, versus a few hundred dollars of margin missed on a
  rejected good applicant). This asymmetry is WHY credit risk models are
  usually tuned to prioritize RECALL on the default class, even at the
  cost of precision -- i.e. we intentionally decline some good applicants
  to catch more bad ones. The right operating point isn't "50% probability
  threshold" -- it should be chosen by walking the precision-recall curve
  against the bank's actual $ cost of a bad loan vs $ margin on a good one,
  which is exactly what a Risk/Decision Science role does when setting
  approval cutoffs.

AUC-ROC is reported as a THRESHOLD-INDEPENDENT summary of ranking quality
(can the model rank a random defaulter above a random non-defaulter?) --
useful for comparing models, but it does NOT tell you where to set your
approval cutoff. That's what the PR curve and a cost-based threshold
analysis are for.
"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    precision_score, recall_score, f1_score, average_precision_score,
    classification_report
)

FIG_DIR = "reports/figures"


def get_probs(name, model, X_test, X_test_scaled):
    """LogReg needs scaled features; tree models don't."""
    X = X_test_scaled if name.startswith("LogReg") else X_test
    return model.predict_proba(X)[:, 1]


def evaluate_all(models, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    results = []
    roc_data, pr_data = {}, {}

    for name, model in models.items():
        y_prob = get_probs(name, model, X_test, X_test_scaled)
        y_pred = (y_prob >= 0.5).astype(int)  # default 0.5 cutoff, revisited below

        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        avg_prec = average_precision_score(y_test, y_prob)

        results.append({
            "model": name, "auc_roc": auc, "precision": prec,
            "recall": rec, "f1": f1, "avg_precision": avg_prec
        })

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, auc)
        p, r, _ = precision_recall_curve(y_test, y_prob)
        pr_data[name] = (p, r, avg_prec)

    results_df = pd.DataFrame(results).sort_values("auc_roc", ascending=False).reset_index(drop=True)
    return results_df, roc_data, pr_data


def plot_roc_curves(roc_data, path=f"{FIG_DIR}/04_roc_curves.png"):
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=1.8)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves -- All Model Variants")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_pr_curves(pr_data, baseline_rate, path=f"{FIG_DIR}/05_precision_recall_curves.png"):
    plt.figure(figsize=(7, 6))
    for name, (p, r, avg_prec) in pr_data.items():
        plt.plot(r, p, label=f"{name} (AP={avg_prec:.3f})", linewidth=1.8)
    plt.axhline(baseline_rate, color="gray", linestyle="--", linewidth=1,
                label=f"No-skill baseline ({baseline_rate:.3f})")
    plt.xlabel("Recall (of actual defaulters, % caught)")
    plt.ylabel("Precision (of flagged loans, % that actually default)")
    plt.title("Precision-Recall Curves -- All Model Variants")
    plt.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def threshold_tradeoff_table(best_model_name, models, scaler, X_test, y_test):
    """
    Shows precision/recall at several candidate thresholds for the best
    model -- this is the table you'd actually bring to a Risk meeting.
    A LOWER threshold flags more people as risky -> higher recall (catch
    more bad loans), lower precision (reject more good applicants too).
    """
    model = models[best_model_name]
    X_test_scaled = scaler.transform(X_test)
    y_prob = get_probs(best_model_name, model, X_test, X_test_scaled)

    rows = []
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred = (y_prob >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "flagged_pct": y_pred.mean(),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    models = joblib.load("models/all_models.joblib")
    scaler = joblib.load("models/scaler.joblib")
    X_test = pd.read_parquet("data/processed/X_test.parquet")
    y_test = pd.read_parquet("data/processed/y_test.parquet").iloc[:, 0]

    results_df, roc_data, pr_data = evaluate_all(models, scaler, X_test, y_test)
    print("=" * 80)
    print("MODEL COMPARISON (sorted by AUC-ROC, NOT accuracy)")
    print("=" * 80)
    print(results_df.to_string(index=False))

    plot_roc_curves(roc_data)
    plot_pr_curves(pr_data, baseline_rate=y_test.mean())

    best_model_name = results_df.iloc[0]["model"]
    print(f"\nBest model by AUC-ROC: {best_model_name}")

    print("\n" + "=" * 80)
    print(f"THRESHOLD TRADEOFF TABLE -- {best_model_name}")
    print("(this is what you bring to a Risk/policy meeting, not a single accuracy number)")
    print("=" * 80)
    tradeoff_df = threshold_tradeoff_table(best_model_name, models, scaler, X_test, y_test)
    print(tradeoff_df.to_string(index=False))

    results_df.to_csv("reports/model_comparison.csv", index=False)
    tradeoff_df.to_csv("reports/threshold_tradeoff.csv", index=False)
    with open("reports/best_model.json", "w") as f:
        json.dump({"best_model_name": best_model_name}, f)
    print("\nSaved reports/model_comparison.csv, reports/threshold_tradeoff.csv, "
          "and figures to reports/figures/")
