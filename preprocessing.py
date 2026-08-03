"""
preprocessing.py
-----------------
Handles missing values, categorical encoding, and gives you TWO strategies
for class imbalance so you can compare them (as requested):

  (A) Class weighting  -- reweight the loss function, don't touch the data
  (B) SMOTE             -- synthesize new minority-class rows

TRADEOFF (say this in the interview):
  Class weighting is applied only inside the loss function during training.
  It doesn't change what data the model sees, doesn't risk creating
  unrealistic synthetic borrowers, and it's essentially free (one parameter).
  Its downside: for models that aren't loss-based in a simple way, or when
  the imbalance is very extreme, reweighting alone sometimes isn't enough
  to move the decision boundary.

  SMOTE (Synthetic Minority Oversampling) interpolates between real
  minority-class points in feature space to manufacture new synthetic
  defaulters. It can help tree models see more minority-class *density*
  to split on. Its downsides are real: (1) it can generate synthetic
  applicants that don't correspond to any plausible real person, especially
  in a high-dimensional mixed categorical/numeric space like this one, (2)
  it must ONLY be applied to the training fold, never before the train/test
  split or you leak information (synthetic points derived from a test-set
  neighbor "leak" that applicant's outcome into training), and (3) it changes
  the training class balance, which shifts the model's predicted
  probabilities away from being properly calibrated to the real-world 9%
  base rate -- you'd need to recalibrate (e.g. Platt scaling) before using
  raw predicted probabilities for something like expected-loss pricing.

  Given (3), for a credit risk model whose OUTPUT probabilities feed into
  business decisions (approve/reject thresholds, pricing), class weighting
  is usually the safer default in industry. SMOTE is worth showing you know
  it and its risks -- which is exactly why both are implemented here.
"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


def load_and_clean(path="data/raw/application_train.csv", df=None):
    if df is None:
        df = pd.read_csv(path)
    else:
        df = df.copy()

    # --- Fix the DAYS_EMPLOYED anomaly BEFORE imputing ------------------
    # 365243 is a sentinel Home Credit uses for "not currently employed"
    # (pensioners/unemployed). Left as-is, it would be treated as a huge
    # positive number of days employed -- exactly backwards. We flag it
    # as its own binary feature (often predictive) and then null it out
    # so the imputer doesn't pollute the employed-years distribution.
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df.loc[df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan

    # --- Feature engineering (cheap, high-value ratios) -----------------
    # Raw AMT_CREDIT / AMT_INCOME_TOTAL had near-zero correlation with
    # TARGET individually (see EDA); their RATIO is the actual risk driver
    # underwriters use in practice (debt-to-income style logic).
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25  # NaN where anomalous, handled by imputer

    # Missingness-as-signal flags for the two most-missing bureau scores
    # (EDA showed missingness itself carries a small but real signal).
    df["EXT_SOURCE_1_MISSING"] = df["EXT_SOURCE_1"].isnull().astype(int)
    df["EXT_SOURCE_3_MISSING"] = df["EXT_SOURCE_3"].isnull().astype(int)

    return df


def encode_and_impute(df):
    df = df.copy()
    target = df["TARGET"]
    df = df.drop(columns=["TARGET", "SK_ID_CURR"])

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Categorical: fill missing with an explicit "Missing" category (informative
    # for tree models) then label-encode. We use label encoding rather than
    # one-hot here because XGBoost/RandomForest handle integer-coded categoricals
    # fine via splits, and it keeps dimensionality low -- for Logistic Regression
    # specifically this is a real limitation (label encoding implies a false
    # ordinal relationship), which we call out explicitly and accept as a
    # documented tradeoff for a project of this scope. In production you'd use
    # one-hot or target encoding for the linear model.
    encoders = {}
    for c in cat_cols:
        df[c] = df[c].fillna("Missing")
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c])
        encoders[c] = le

    # Numeric: median imputation. Median (not mean) because several of these
    # distributions (AMT_INCOME_TOTAL, AMT_CREDIT) are heavily right-skewed,
    # and mean imputation would be pulled by outliers.
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    df["TARGET"] = target
    return df, encoders


def split_data(df, test_size=0.2, seed=42):
    X = df.drop(columns=["TARGET"])
    y = df["TARGET"]
    # stratify=y is mandatory here -- with a 91:9 split, a non-stratified
    # split could easily shift the test-set default rate by a percentage
    # point or more just from sampling noise, making metrics incomparable
    # across runs.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train, seed=42):
    """Oversample the minority class -- TRAIN FOLD ONLY, never touch X_test."""
    smote = SMOTE(random_state=seed)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res


def get_class_weight_dict(y_train):
    """Inverse-frequency class weights for use in class_weight= params."""
    counts = y_train.value_counts()
    total = len(y_train)
    return {cls: total / (2 * count) for cls, count in counts.items()}


if __name__ == "__main__":
    df = load_and_clean()
    df, encoders = encode_and_impute(df)
    X_train, X_test, y_train, y_test = split_data(df)

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train default rate: {y_train.mean():.4f}, Test default rate: {y_test.mean():.4f}")
    print(f"\nClass weights (for class_weight= param): {get_class_weight_dict(y_train)}")

    X_train_smote, y_train_smote = apply_smote(X_train, y_train)
    print(f"\nAfter SMOTE -- Train shape: {X_train_smote.shape}, "
          f"new default rate: {y_train_smote.mean():.4f}")

    # Persist processed splits for the modeling script
    X_train.to_parquet("data/processed/X_train.parquet")
    X_test.to_parquet("data/processed/X_test.parquet")
    y_train.to_frame().to_parquet("data/processed/y_train.parquet")
    y_test.to_frame().to_parquet("data/processed/y_test.parquet")
    X_train_smote.to_parquet("data/processed/X_train_smote.parquet")
    y_train_smote.to_frame().to_parquet("data/processed/y_train_smote.parquet")
    print("\nSaved processed splits to data/processed/")
