"""
aggregate_auxiliary.py
-----------------------
Turns Home Credit's auxiliary tables (bureau history, previous Home Credit
applications, POS/cash loans, credit card behavior) into one row per
SK_ID_CURR, ready to left-join onto application_train.csv.

This is the single biggest lever for improving on the "application-only"
baseline in src/modeling.py -- real Kaggle solutions for this competition
get most of their AUC lift from exactly these tables, because they capture
something application_train.csv structurally cannot: a client's *track
record*, not just their profile at the moment of the current application.
Two applicants can look identical on income/education/EXT_SOURCE and still
carry very different risk if one has three prior loans paid on time and the
other has two written off.

MEMORY STRATEGY: bureau_balance.csv is ~27M rows and credit_card_balance /
POS_CASH_balance / previous_application are each 1.5-10M rows and 150-400MB
on disk. Loading all five naively at once risks exceeding available RAM.
So each table is:
  1. read with `usecols` (only the columns we actually aggregate),
  2. read in chunks for the largest files, aggregating each chunk then
     re-aggregating the partial results (a standard two-stage MapReduce
     pattern in pandas),
  3. downcast dtypes (int32/float32/category) before aggregating,
  4. deleted (`del` + nothing held) as soon as its aggregated output is
     computed, before the next table is loaded.

Output: data/processed/auxiliary_features.parquet, one row per SK_ID_CURR.
"""
import gc
import numpy as np
import pandas as pd

RAW = "data/raw"
OUT = "data/processed/auxiliary_features.parquet"


# ---------------------------------------------------------------------------
# 1. bureau_balance.csv (27M rows) -> aggregate to SK_ID_BUREAU level first
# ---------------------------------------------------------------------------
def aggregate_bureau_balance(path=f"{RAW}/bureau_balance.csv", chunksize=3_000_000):
    """
    STATUS meanings: 'C' = closed, 'X' = unknown, '0' = no DPD (current),
    '1'-'5' = increasing DPD severity buckets (5 = 120+ days past due or
    written off). We collapse this into: has this account ever been
    delinquent, and how severely, per month -- then roll up per bureau ID.
    """
    dpd_severity = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "C": 0, "X": 0}
    partials = []
    dtypes = {"SK_ID_BUREAU": "int32", "MONTHS_BALANCE": "int16", "STATUS": "category"}

    for chunk in pd.read_csv(path, usecols=list(dtypes.keys()), dtype=dtypes,
                              chunksize=chunksize):
        chunk["DPD_SEVERITY"] = chunk["STATUS"].map(dpd_severity).astype("int8")
        chunk["IS_DPD"] = (chunk["DPD_SEVERITY"] > 0).astype("int8")
        g = chunk.groupby("SK_ID_BUREAU").agg(
            BB_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
            BB_DPD_MONTHS=("IS_DPD", "sum"),
            BB_MAX_DPD_SEVERITY=("DPD_SEVERITY", "max"),
            BB_MOST_RECENT_MONTH=("MONTHS_BALANCE", "max"),  # closest to 0 = most recent
        )
        partials.append(g)
        del chunk

    combined = pd.concat(partials).reset_index()
    del partials
    gc.collect()

    final = combined.groupby("SK_ID_BUREAU").agg(
        BB_MONTHS_COUNT=("BB_MONTHS_COUNT", "sum"),
        BB_DPD_MONTHS=("BB_DPD_MONTHS", "sum"),
        BB_MAX_DPD_SEVERITY=("BB_MAX_DPD_SEVERITY", "max"),
        BB_MOST_RECENT_MONTH=("BB_MOST_RECENT_MONTH", "max"),
    )
    print(f"  bureau_balance aggregated: {final.shape[0]:,} unique SK_ID_BUREAU")
    return final


# ---------------------------------------------------------------------------
# 2. bureau.csv (1.7M rows) -> join bureau_balance, aggregate to SK_ID_CURR
# ---------------------------------------------------------------------------
def aggregate_bureau(bb_agg, path=f"{RAW}/bureau.csv"):
    cols = ["SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "DAYS_CREDIT",
            "CREDIT_DAY_OVERDUE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
            "AMT_CREDIT_SUM_OVERDUE", "CNT_CREDIT_PROLONG", "CREDIT_TYPE"]
    dtypes = {"SK_ID_CURR": "int32", "SK_ID_BUREAU": "int32", "CREDIT_ACTIVE": "category",
              "DAYS_CREDIT": "int32", "CREDIT_DAY_OVERDUE": "int32",
              "AMT_CREDIT_SUM": "float32", "AMT_CREDIT_SUM_DEBT": "float32",
              "AMT_CREDIT_SUM_OVERDUE": "float32", "CNT_CREDIT_PROLONG": "int16",
              "CREDIT_TYPE": "category"}
    bureau = pd.read_csv(path, usecols=cols, dtype=dtypes)
    bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
    bureau["IS_ACTIVE"] = (bureau["CREDIT_ACTIVE"] == "Active").astype("int8")

    agg = bureau.groupby("SK_ID_CURR").agg(
        BUREAU_LOAN_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_COUNT=("IS_ACTIVE", "sum"),
        BUREAU_CREDIT_TYPES_NUNIQUE=("CREDIT_TYPE", "nunique"),
        BUREAU_DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
        BUREAU_DAYS_CREDIT_MIN=("DAYS_CREDIT", "min"),
        BUREAU_CREDIT_DAY_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
        BUREAU_AMT_CREDIT_SUM_TOTAL=("AMT_CREDIT_SUM", "sum"),
        BUREAU_AMT_CREDIT_SUM_DEBT_TOTAL=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_AMT_CREDIT_SUM_OVERDUE_TOTAL=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        BUREAU_CNT_PROLONGED_TOTAL=("CNT_CREDIT_PROLONG", "sum"),
        BUREAU_BB_DPD_MONTHS_TOTAL=("BB_DPD_MONTHS", "sum"),
        BUREAU_BB_MAX_DPD_SEVERITY=("BB_MAX_DPD_SEVERITY", "max"),
    ).reset_index()

    # Debt-to-credit ratio across all bureau-reported credit lines -- a classic
    # "how leveraged is this person outside of Home Credit" signal.
    agg["BUREAU_DEBT_CREDIT_RATIO"] = (
        agg["BUREAU_AMT_CREDIT_SUM_DEBT_TOTAL"] /
        agg["BUREAU_AMT_CREDIT_SUM_TOTAL"].replace(0, np.nan)
    )
    print(f"  bureau aggregated: {agg.shape[0]:,} unique SK_ID_CURR")
    del bureau
    gc.collect()
    return agg


# ---------------------------------------------------------------------------
# 3. previous_application.csv (1.67M rows) -> SK_ID_CURR level
# ---------------------------------------------------------------------------
def aggregate_previous_application(path=f"{RAW}/previous_application.csv"):
    cols = ["SK_ID_CURR", "AMT_ANNUITY", "AMT_APPLICATION", "AMT_CREDIT",
            "AMT_DOWN_PAYMENT", "NAME_CONTRACT_STATUS", "DAYS_DECISION", "CNT_PAYMENT"]
    dtypes = {"SK_ID_CURR": "int32", "AMT_ANNUITY": "float32", "AMT_APPLICATION": "float32",
              "AMT_CREDIT": "float32", "AMT_DOWN_PAYMENT": "float32",
              "NAME_CONTRACT_STATUS": "category", "DAYS_DECISION": "int32",
              "CNT_PAYMENT": "float32"}
    prev = pd.read_csv(path, usecols=cols, dtype=dtypes)
    prev["IS_APPROVED"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype("int8")
    prev["IS_REFUSED"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype("int8")

    agg = prev.groupby("SK_ID_CURR").agg(
        PREV_APP_COUNT=("SK_ID_CURR", "count"),
        PREV_APPROVED_COUNT=("IS_APPROVED", "sum"),
        PREV_REFUSED_COUNT=("IS_REFUSED", "sum"),
        PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
        PREV_AMT_APPLICATION_MEAN=("AMT_APPLICATION", "mean"),
        PREV_AMT_DOWN_PAYMENT_MEAN=("AMT_DOWN_PAYMENT", "mean"),
        PREV_DAYS_DECISION_MEAN=("DAYS_DECISION", "mean"),
        PREV_CNT_PAYMENT_MEAN=("CNT_PAYMENT", "mean"),
    ).reset_index()

    # Prior refusal/approval rate -- "has this bank system said no to this
    # person before" is a strong, cheap-to-compute risk signal.
    agg["PREV_APPROVAL_RATE"] = agg["PREV_APPROVED_COUNT"] / agg["PREV_APP_COUNT"]
    agg["PREV_REFUSAL_RATE"] = agg["PREV_REFUSED_COUNT"] / agg["PREV_APP_COUNT"]
    print(f"  previous_application aggregated: {agg.shape[0]:,} unique SK_ID_CURR")
    del prev
    gc.collect()
    return agg


# ---------------------------------------------------------------------------
# 4. POS_CASH_balance.csv (10M rows) -> SK_ID_CURR level, chunked
# ---------------------------------------------------------------------------
def aggregate_pos_cash(path=f"{RAW}/POS_CASH_balance.csv", chunksize=3_000_000):
    cols = ["SK_ID_CURR", "SK_DPD", "SK_DPD_DEF", "NAME_CONTRACT_STATUS"]
    dtypes = {"SK_ID_CURR": "int32", "SK_DPD": "int32", "SK_DPD_DEF": "int32",
              "NAME_CONTRACT_STATUS": "category"}
    partials = []
    for chunk in pd.read_csv(path, usecols=cols, dtype=dtypes, chunksize=chunksize):
        chunk["IS_DPD"] = (chunk["SK_DPD"] > 0).astype("int8")
        chunk["IS_COMPLETED"] = (chunk["NAME_CONTRACT_STATUS"] == "Completed").astype("int8")
        g = chunk.groupby("SK_ID_CURR").agg(
            POS_RECORD_COUNT=("SK_ID_CURR", "count"),
            POS_DPD_MONTHS=("IS_DPD", "sum"),
            POS_SK_DPD_MAX=("SK_DPD", "max"),
            POS_COMPLETED_COUNT=("IS_COMPLETED", "sum"),
        )
        partials.append(g)
        del chunk

    combined = pd.concat(partials).reset_index()
    del partials
    gc.collect()

    final = combined.groupby("SK_ID_CURR").agg(
        POS_RECORD_COUNT=("POS_RECORD_COUNT", "sum"),
        POS_DPD_MONTHS=("POS_DPD_MONTHS", "sum"),
        POS_SK_DPD_MAX=("POS_SK_DPD_MAX", "max"),
        POS_COMPLETED_COUNT=("POS_COMPLETED_COUNT", "sum"),
    ).reset_index()
    print(f"  POS_CASH_balance aggregated: {final.shape[0]:,} unique SK_ID_CURR")
    return final


# ---------------------------------------------------------------------------
# 5. credit_card_balance.csv (3.84M rows) -> SK_ID_CURR level
# ---------------------------------------------------------------------------
def aggregate_credit_card(path=f"{RAW}/credit_card_balance.csv", chunksize=2_000_000):
    cols = ["SK_ID_CURR", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL",
            "AMT_DRAWINGS_CURRENT", "SK_DPD"]
    dtypes = {"SK_ID_CURR": "int32", "AMT_BALANCE": "float32",
              "AMT_CREDIT_LIMIT_ACTUAL": "float32", "AMT_DRAWINGS_CURRENT": "float32",
              "SK_DPD": "int32"}
    partials = []
    for chunk in pd.read_csv(path, usecols=cols, dtype=dtypes, chunksize=chunksize):
        # Utilization = balance / limit -- a classic revolving-credit risk signal;
        # guard against divide-by-zero limits.
        chunk["UTILIZATION"] = chunk["AMT_BALANCE"] / chunk["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
        chunk["IS_DPD"] = (chunk["SK_DPD"] > 0).astype("int8")
        g = chunk.groupby("SK_ID_CURR").agg(
            CC_RECORD_COUNT=("SK_ID_CURR", "count"),
            CC_AMT_BALANCE_SUM=("AMT_BALANCE", "sum"),
            CC_AMT_BALANCE_MAX=("AMT_BALANCE", "max"),
            CC_UTILIZATION_SUM=("UTILIZATION", "sum"),
            CC_UTILIZATION_COUNT=("UTILIZATION", "count"),  # for weighted mean later
            CC_DRAWINGS_SUM=("AMT_DRAWINGS_CURRENT", "sum"),
            CC_DPD_MONTHS=("IS_DPD", "sum"),
        )
        partials.append(g)
        del chunk

    combined = pd.concat(partials).reset_index()
    del partials
    gc.collect()

    final = combined.groupby("SK_ID_CURR").agg(
        CC_RECORD_COUNT=("CC_RECORD_COUNT", "sum"),
        CC_AMT_BALANCE_MEAN=("CC_AMT_BALANCE_SUM", "sum"),
        CC_AMT_BALANCE_MAX=("CC_AMT_BALANCE_MAX", "max"),
        CC_UTILIZATION_SUM=("CC_UTILIZATION_SUM", "sum"),
        CC_UTILIZATION_COUNT=("CC_UTILIZATION_COUNT", "sum"),
        CC_DRAWINGS_TOTAL=("CC_DRAWINGS_SUM", "sum"),
        CC_DPD_MONTHS=("CC_DPD_MONTHS", "sum"),
    ).reset_index()
    final["CC_AMT_BALANCE_MEAN"] = final["CC_AMT_BALANCE_MEAN"] / final["CC_RECORD_COUNT"]
    final["CC_UTILIZATION_MEAN"] = final["CC_UTILIZATION_SUM"] / final["CC_UTILIZATION_COUNT"].replace(0, np.nan)
    final = final.drop(columns=["CC_UTILIZATION_SUM", "CC_UTILIZATION_COUNT"])
    print(f"  credit_card_balance aggregated: {final.shape[0]:,} unique SK_ID_CURR")
    return final


if __name__ == "__main__":
    print("Aggregating bureau_balance.csv (largest file, ~27M rows)...")
    bb_agg = aggregate_bureau_balance()

    print("Aggregating bureau.csv + joining bureau_balance...")
    bureau_agg = aggregate_bureau(bb_agg)
    del bb_agg
    gc.collect()

    print("Aggregating previous_application.csv...")
    prev_agg = aggregate_previous_application()

    print("Aggregating POS_CASH_balance.csv...")
    pos_agg = aggregate_pos_cash()

    print("Aggregating credit_card_balance.csv...")
    cc_agg = aggregate_credit_card()

    print("Merging all auxiliary feature sets on SK_ID_CURR...")
    all_ids = pd.concat([
        bureau_agg[["SK_ID_CURR"]], prev_agg[["SK_ID_CURR"]],
        pos_agg[["SK_ID_CURR"]], cc_agg[["SK_ID_CURR"]]
    ]).drop_duplicates()

    final = all_ids.merge(bureau_agg, on="SK_ID_CURR", how="left") \
                    .merge(prev_agg, on="SK_ID_CURR", how="left") \
                    .merge(pos_agg, on="SK_ID_CURR", how="left") \
                    .merge(cc_agg, on="SK_ID_CURR", how="left")

    print(f"\nFinal auxiliary feature table: {final.shape[0]:,} applicants x "
          f"{final.shape[1]-1} engineered features")
    final.to_parquet(OUT, index=False)
    print(f"Saved to {OUT}")
    print("\nSample columns:", list(final.columns))
