"""
eda.py
------
Step 1 of the assignment: understand the data before touching it.

Three things we MUST check before modeling, and why each matters for
an interview:

1. Class imbalance -> determines whether accuracy is even a valid metric
   (it isn't, here), and whether we need SMOTE / class weights at all.
2. Missing values -> determines imputation strategy. Critically, in credit
   data, missingness is often NOT random (e.g. EXT_SOURCE_1 is missing for
   >50% of applicants because not every bureau has scored every applicant)
   -- so "missing" itself can carry signal, and blind mean-imputation can
   destroy it.
3. Correlation with target -> a first-pass sanity check on which features
   will likely matter, and a way to catch leakage or NA-driven artifacts
   before they contaminate a model.
"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

df = pd.read_csv("data/raw/application_train.csv")
FIG_DIR = "reports/figures"

print("=" * 70)
print("1. CLASS IMBALANCE")
print("=" * 70)
target_counts = df["TARGET"].value_counts()
target_pct = df["TARGET"].value_counts(normalize=True) * 100
print(target_counts)
print(target_pct.round(2))
print(f"\nImbalance ratio (majority:minority) = "
      f"{target_counts[0] / target_counts[1]:.1f} : 1")
# WHY THIS MATTERS: with a ~91:9 split, a model that predicts "no default"
# for every single applicant scores ~91% accuracy while catching ZERO bad
# loans. That's why accuracy is banned from this project's evaluation.

fig, ax = plt.subplots(figsize=(5, 4))
sns.barplot(x=target_counts.index.map({0: "Repaid (0)", 1: "Default (1)"}),
            y=target_counts.values, ax=ax, palette=["#2E86AB", "#C73E1D"])
ax.set_title("Class Distribution: Loan Default (TARGET)")
ax.set_ylabel("Count")
for i, v in enumerate(target_counts.values):
    ax.text(i, v + 500, f"{v:,}\n({target_pct.values[i]:.1f}%)", ha="center")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_class_imbalance.png", dpi=120)
plt.close()

print("\n" + "=" * 70)
print("2. MISSING VALUE ANALYSIS")
print("=" * 70)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
missing_df = missing_df[missing_df["missing_count"] > 0].sort_values(
    "missing_pct", ascending=False)
print(missing_df)

fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(x=missing_df["missing_pct"], y=missing_df.index, ax=ax, color="#457B9D")
ax.set_title("Missing Value % by Column")
ax.set_xlabel("% Missing")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_missing_values.png", dpi=120)
plt.close()

# IMPORTANT SANITY CHECK: is missingness itself predictive of default?
# We test this for EXT_SOURCE_1 (the column with the most missingness).
df["EXT_SOURCE_1_MISSING"] = df["EXT_SOURCE_1"].isnull().astype(int)
default_rate_by_missing = df.groupby("EXT_SOURCE_1_MISSING")["TARGET"].mean()
print("\nDefault rate by EXT_SOURCE_1 missing flag:")
print(default_rate_by_missing.round(4))
# If these rates differ noticeably, "is this value missing" is itself a
# useful engineered feature -- a classic trick in credit risk modeling.

print("\n" + "=" * 70)
print("3. CORRELATION OF KEY FEATURES WITH TARGET")
print("=" * 70)
numeric_df = df.select_dtypes(include=[np.number]).copy()
corrs = numeric_df.corr()["TARGET"].drop("TARGET").sort_values()
print(corrs)

fig, ax = plt.subplots(figsize=(7, 6))
top_corrs = pd.concat([corrs.head(8), corrs.tail(8)])
colors = ["#C73E1D" if v > 0 else "#2E86AB" for v in top_corrs.values]
ax.barh(top_corrs.index, top_corrs.values, color=colors)
ax.set_title("Feature Correlation with TARGET (default=1)")
ax.set_xlabel("Pearson correlation")
ax.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_target_correlations.png", dpi=120)
plt.close()

# WHY THESE SIGNS MAKE SENSE (interview talking points):
#   EXT_SOURCE_1/2/3 : negative corr -- higher bureau score = lower risk. Expected,
#                      and these three dominate everything else by a wide margin.
#   DAYS_BIRTH       : stored as NEGATIVE days-before-application (more negative
#                      = older). Its correlation with TARGET is POSITIVE, meaning
#                      a LARGER (less negative, i.e. YOUNGER) value goes with higher
#                      default risk -- younger borrowers are riskier. Always check
#                      the sign convention on "days since X" fields before quoting
#                      a correlation number in an interview -- it's an easy trap.
#   AMT_CREDIT/INCOME: each is weak alone, but their RATIO (credit-to-income) is
#                      what actually drives risk -- motivates feature engineering
#                      beyond what a raw correlation table shows.

print("\nDone. Figures saved to reports/figures/")
