"""
generate_data.py
-----------------
Builds a SYNTHETIC replica of Home Credit Default Risk's application_train.csv.

WHY THIS EXISTS: the real file needs a Kaggle login to download (see
data/raw/DOWNLOAD_INSTRUCTIONS.md). This generator reproduces the real
dataset's *statistical fingerprint* so the rest of the pipeline is genuine,
runnable, and produces honest metrics:
  - same column names / dtypes as the real Home Credit schema
  - same class imbalance (~8% default rate)
  - same missingness pattern (EXT_SOURCE_3 missing ~20%, OCCUPATION_TYPE
    missing ~31%, the DAYS_EMPLOYED "365243" anomaly for pensioners, etc.)
  - same *direction* of signal Home Credit is famous for: EXT_SOURCE_1/2/3
    are by far the most predictive features in the real competition, so
    we bake in that relationship rather than pure noise. This is what
    lets SHAP later "rediscover" a genuinely known pattern -- a good
    sanity check to mention in an interview.

Swap this out for `pd.read_csv("data/raw/application_train.csv")` once you
have the real file; every downstream script is schema-compatible.
"""
import numpy as np
import pandas as pd


def generate_home_credit_like(n=60000, seed=42):
    rng = np.random.default_rng(seed)

    # --- Demographics -------------------------------------------------
    age_years = rng.normal(43, 11, n).clip(21, 69)
    days_birth = -(age_years * 365.25).astype(int)

    gender = rng.choice(["F", "M"], size=n, p=[0.65, 0.35])
    own_car = rng.choice(["Y", "N"], size=n, p=[0.34, 0.66])
    own_realty = rng.choice(["Y", "N"], size=n, p=[0.69, 0.31])
    cnt_children = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.7, 0.19, 0.08, 0.02, 0.01])

    education = rng.choice(
        ["Secondary / secondary special", "Higher education",
         "Incomplete higher", "Lower secondary", "Academic degree"],
        size=n, p=[0.71, 0.24, 0.03, 0.015, 0.005]
    )
    family_status = rng.choice(
        ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
        size=n, p=[0.64, 0.15, 0.1, 0.065, 0.045]
    )
    housing_type = rng.choice(
        ["House / apartment", "With parents", "Municipal apartment",
         "Rented apartment", "Office apartment", "Co-op apartment"],
        size=n, p=[0.88, 0.045, 0.035, 0.025, 0.01, 0.005]
    )

    # --- Employment / income -------------------------------------------
    income_type = rng.choice(
        ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed"],
        size=n, p=[0.52, 0.23, 0.18, 0.065, 0.005]
    )
    is_pensioner = income_type == "Pensioner"

    days_employed_raw = -(rng.exponential(2000, n)).astype(int)
    # famous Home Credit data quirk: pensioners/unemployed get a sentinel 365243
    days_employed = np.where(is_pensioner, 365243, days_employed_raw)

    occupation_type = rng.choice(
        ["Laborers", "Sales staff", "Core staff", "Managers", "Drivers",
         "High skill tech staff", "Accountants", "Medicine staff", None],
        size=n, p=[0.18, 0.12, 0.13, 0.08, 0.09, 0.05, 0.04, 0.04, 0.27]
    )

    amt_income_total = rng.lognormal(mean=11.9, sigma=0.45, size=n).round(-2).clip(25000, 2_000_000)
    amt_credit = (amt_income_total * rng.uniform(1.5, 6.0, n)).round(-3)
    amt_annuity = (amt_credit / rng.uniform(8, 30, n)).round(-1)
    amt_goods_price = (amt_credit * rng.uniform(0.85, 1.0, n)).round(-3)

    contract_type = rng.choice(["Cash loans", "Revolving loans"], size=n, p=[0.9, 0.1])

    # --- External bureau scores (the real dataset's strongest signal) --
    # EXT_SOURCE_1/2/3 are normalized external credit-bureau scores.
    # We correlate them negatively with default risk and with each other,
    # then apply Home Credit's real missingness rates.
    latent_risk = rng.normal(0, 1, n)  # unobserved "true" creditworthiness
    ext_source_1 = np.clip(0.5 - 0.15 * latent_risk + rng.normal(0, 0.15, n), 0, 1)
    ext_source_2 = np.clip(0.5 - 0.18 * latent_risk + rng.normal(0, 0.13, n), 0, 1)
    ext_source_3 = np.clip(0.5 - 0.20 * latent_risk + rng.normal(0, 0.14, n), 0, 1)

    ext_source_1[rng.random(n) < 0.56] = np.nan   # ~56% missing in real data
    ext_source_2[rng.random(n) < 0.002] = np.nan  # ~0.2% missing
    ext_source_3[rng.random(n) < 0.20] = np.nan   # ~20% missing

    region_rating = rng.choice([1, 2, 3], size=n, p=[0.13, 0.65, 0.22])
    days_last_phone_change = -rng.exponential(600, n).astype(int)
    flag_own_car_age = np.where(own_car == "Y", rng.integers(0, 25, n), np.nan)

    # a handful of the FLAG_DOCUMENT_x binary columns (mostly near-constant
    # in the real data, included here to show how EDA correctly discards them)
    flag_document_3 = rng.choice([0, 1], size=n, p=[0.29, 0.71])
    flag_document_6 = rng.choice([0, 1], size=n, p=[0.94, 0.06])

    # --- TARGET ---------------------------------------------------------
    # Build default probability from a logistic combination of realistic
    # risk drivers, mirroring known Home Credit relationships:
    #   - lower EXT_SOURCE scores -> higher risk (dominant effect)
    #   - younger age -> higher risk
    #   - unemployed/very short employment -> higher risk
    #   - higher credit-to-income ratio -> higher risk
    #   - lower education -> higher risk
    credit_income_ratio = amt_credit / amt_income_total
    employed_years = np.where(days_employed == 365243, 0, -days_employed / 365.25)

    stacked = np.vstack([ext_source_1, ext_source_2, ext_source_3])
    with np.errstate(invalid="ignore"):
        ext_avg = np.where(np.all(np.isnan(stacked), axis=0), 0.5, np.nanmean(stacked, axis=0))

    education_risk = pd.Series(education).map({
        "Lower secondary": 0.6, "Secondary / secondary special": 0.3,
        "Incomplete higher": 0.15, "Higher education": -0.3, "Academic degree": -0.5
    }).values

    logit = (
        -3.15
        - 3.2 * (ext_avg - 0.5)
        - 0.02 * (age_years - 43)
        - 0.015 * np.minimum(employed_years, 20)
        + 0.35 * np.log1p(credit_income_ratio)
        + 0.4 * education_risk
        + rng.normal(0, 0.6, n)
    )
    prob_default = 1 / (1 + np.exp(-logit))
    target = (rng.random(n) < prob_default).astype(int)

    df = pd.DataFrame({
        "SK_ID_CURR": np.arange(100001, 100001 + n),
        "TARGET": target,
        "NAME_CONTRACT_TYPE": contract_type,
        "CODE_GENDER": gender,
        "FLAG_OWN_CAR": own_car,
        "FLAG_OWN_REALTY": own_realty,
        "CNT_CHILDREN": cnt_children,
        "AMT_INCOME_TOTAL": amt_income_total,
        "AMT_CREDIT": amt_credit,
        "AMT_ANNUITY": amt_annuity,
        "AMT_GOODS_PRICE": amt_goods_price,
        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": education,
        "NAME_FAMILY_STATUS": family_status,
        "NAME_HOUSING_TYPE": housing_type,
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "OCCUPATION_TYPE": occupation_type,
        "REGION_RATING_CLIENT": region_rating,
        "EXT_SOURCE_1": ext_source_1,
        "EXT_SOURCE_2": ext_source_2,
        "EXT_SOURCE_3": ext_source_3,
        "DAYS_LAST_PHONE_CHANGE": days_last_phone_change,
        "OWN_CAR_AGE": flag_own_car_age,
        "FLAG_DOCUMENT_3": flag_document_3,
        "FLAG_DOCUMENT_6": flag_document_6,
    })
    return df


if __name__ == "__main__":
    df = generate_home_credit_like()
    out_path = "data/raw/application_train.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {df.shape[0]} rows x {df.shape[1]} cols to {out_path}")
    print(f"Default rate: {df['TARGET'].mean():.4f}")
