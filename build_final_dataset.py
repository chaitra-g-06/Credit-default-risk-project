"""
build_final_dataset.py
------------------------
Merges application_train.csv (the file with SK_ID_CURR + TARGET) with the
auxiliary feature table produced by aggregate_auxiliary.py, and handles the
missing-history case correctly.

WHY MISSING-VALUE HANDLING HERE IS DIFFERENT FROM src/preprocessing.py:
A NaN in, say, BUREAU_LOAN_COUNT after this left-join doesn't mean "data
quality problem" the way a NaN in EXT_SOURCE_1 might -- it means "this
applicant genuinely has zero bureau-reported credit history." That's a
real, meaningful state (a "thin-file" applicant), not missingness to be
imputed away with a median. So:
  - COUNT-type columns (BUREAU_LOAN_COUNT, PREV_APP_COUNT, POS_RECORD_COUNT,
    CC_RECORD_COUNT, and the DPD/overdue/prolonged counts derived from them)
    are filled with 0 -- "zero prior loans" is the correct literal value,
    not a missing one.
  - RATE/RATIO/MEAN columns computed FROM those counts (PREV_APPROVAL_RATE,
    BUREAU_DEBT_CREDIT_RATIO, CC_UTILIZATION_MEAN, etc.) are left as NaN and
    handled by the standard median-imputation step in preprocessing.py,
    because "0% approval rate" and "no history to compute a rate from" are
    NOT the same thing, and silently coding them identically would create
    a false signal.
  - We ALSO keep an explicit HAS_BUREAU_HISTORY / HAS_PREV_APP_HISTORY /
    HAS_POS_HISTORY / HAS_CC_HISTORY flag for each table, in case "having
    no history at all" is itself predictive (a very plausible story: a
    thin-file applicant is neither obviously good nor bad, just unscored,
    and models often learn a specific risk premium for that group).
"""
import pandas as pd
import numpy as np

COUNT_COLS = [
    "BUREAU_LOAN_COUNT", "BUREAU_ACTIVE_COUNT", "BUREAU_CREDIT_TYPES_NUNIQUE",
    "BUREAU_CNT_PROLONGED_TOTAL", "BUREAU_BB_DPD_MONTHS_TOTAL",
    "BUREAU_AMT_CREDIT_SUM_TOTAL", "BUREAU_AMT_CREDIT_SUM_DEBT_TOTAL",
    "BUREAU_AMT_CREDIT_SUM_OVERDUE_TOTAL",
    "PREV_APP_COUNT", "PREV_APPROVED_COUNT", "PREV_REFUSED_COUNT",
    "POS_RECORD_COUNT", "POS_DPD_MONTHS", "POS_COMPLETED_COUNT",
    "CC_RECORD_COUNT", "CC_DPD_MONTHS", "CC_DRAWINGS_TOTAL",
]

HISTORY_FLAG_SOURCE_COLS = {
    "HAS_BUREAU_HISTORY": "BUREAU_LOAN_COUNT",
    "HAS_PREV_APP_HISTORY": "PREV_APP_COUNT",
    "HAS_POS_HISTORY": "POS_RECORD_COUNT",
    "HAS_CC_HISTORY": "CC_RECORD_COUNT",
}


def build_final_dataset(
    application_path="data/raw/application_train.csv",
    auxiliary_path="data/processed/auxiliary_features.parquet",
    out_path="data/processed/application_train_with_aux.parquet",
):
    app = pd.read_csv(application_path)
    aux = pd.read_parquet(auxiliary_path)

    print(f"application_train: {app.shape[0]:,} rows")
    print(f"auxiliary features: {aux.shape[0]:,} applicants with some history")

    # Presence flags computed BEFORE filling counts with 0, so we don't lose
    # the "never had one of these" signal.
    for flag_col, source_col in HISTORY_FLAG_SOURCE_COLS.items():
        aux[flag_col] = aux[source_col].notnull().astype("int8")

    merged = app.merge(aux, on="SK_ID_CURR", how="left")

    n_matched = merged["HAS_BUREAU_HISTORY"].notnull().sum()
    match_rate = n_matched / len(merged) * 100
    print(f"Matched auxiliary data for {n_matched:,} / {len(merged):,} "
          f"applicants ({match_rate:.1f}%)")

    # An applicant present in application_train but absent from EVERY
    # auxiliary table (never appeared in bureau/prev/POS/CC at all) gets
    # all-NaN history flags from the merge -- fix those to 0 explicitly.
    for flag_col in HISTORY_FLAG_SOURCE_COLS:
        merged[flag_col] = merged[flag_col].fillna(0).astype("int8")

    for col in COUNT_COLS:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    merged.to_parquet(out_path, index=False)
    print(f"Saved merged dataset to {out_path}  "
          f"({merged.shape[0]:,} rows x {merged.shape[1]} cols)")
    return merged


if __name__ == "__main__":
    build_final_dataset()
