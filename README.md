# Credit Default Risk Prediction

A resume/interview project for fintech Risk & Decision Science internships
(Amex-style), predicting probability of consumer loan default from
application-time data plus credit-bureau/prior-application history, and
doing it the way a real risk team would defend it: imbalance-aware
evaluation, an explicit and justified imbalance-handling strategy, model
comparison across three algorithm families, a measured (not assumed) lift
from history-based feature engineering, and SHAP-based explainability
suitable for a model-risk review.

**All results below are on the real Kaggle Home Credit Default Risk
dataset** (307,511 applicants, 122 base columns, plus 5 auxiliary tables).

## Problem

Given an applicant's demographic, income, employment, and credit-bureau
data at the moment of a loan application — plus their history of prior
bureau-reported credit, prior Home Credit applications, and POS/credit-card
repayment behavior — predict the probability they default (`TARGET = 1`).
**8.07% of applicants default** in this dataset (11.4:1 imbalance), a
realistic base rate that shapes almost every methodology decision below.

## Data

Kaggle's **"Home Credit Default Risk"** competition:
- `application_train.csv` — 307,511 rows, 122 columns, the main table
  (`SK_ID_CURR` + `TARGET` + applicant profile at time of application)
- `bureau.csv` (1.72M rows) + `bureau_balance.csv` (27.3M rows) — each
  applicant's credit history reported to external credit bureaus
- `previous_application.csv` (1.67M rows) — the applicant's own prior
  applications to Home Credit
- `POS_CASH_balance.csv` (10.0M rows) — monthly snapshots of prior
  point-of-sale/cash loan repayment behavior
- `credit_card_balance.csv` (3.84M rows) — monthly snapshots of prior
  Home Credit credit card balance/repayment behavior

See `data/raw/DOWNLOAD_INSTRUCTIONS.md` for exact commands to pull these
yourself via the Kaggle CLI or website.

## Project structure

```
credit-default-risk/
├── data/
│   ├── raw/                          # application_train.csv + 5 auxiliary tables + column dictionary
│   └── processed/                    # train/test splits, SMOTE-resampled set, aggregated aux features (parquet)
├── notebooks/
│   └── credit_risk_walkthrough.ipynb # full narrated walkthrough, pre-executed on real data
├── src/
│   ├── eda.py                        # class imbalance, missingness, correlation analysis
│   ├── preprocessing.py              # cleaning, feature engineering, encoding, SMOTE vs class weights
│   ├── aggregate_auxiliary.py        # memory-efficient aggregation of bureau/prev-app/POS/CC tables to SK_ID_CURR level
│   ├── build_final_dataset.py        # merges application_train with aggregated auxiliary features
│   ├── modeling.py                   # trains 6 model variants (3 algorithms x 2 imbalance strategies)
│   ├── evaluate.py                   # AUC-ROC / precision / recall / PR curves / threshold tradeoff table
│   ├── compare_with_auxiliary_features.py  # head-to-head: application-only vs +auxiliary-features AUC lift
│   ├── shap_analysis.py              # SHAP global importance + beeswarm + single-applicant explanation
│   └── generate_data.py              # synthetic data generator (kept for offline/no-download demos only)
├── models/                           # trained model artifacts (joblib)
├── reports/
│   ├── figures/                      # all plots (imbalance, missingness, correlations, ROC, PR, SHAP, aux-feature lift)
│   ├── model_comparison.csv
│   ├── threshold_tradeoff.csv
│   ├── auxiliary_feature_lift.json
│   ├── enhanced_model_feature_importance.csv
│   └── shap_feature_importance.csv
└── README.md
```

Run order: `eda.py` → `preprocessing.py` → `modeling.py` → `evaluate.py` →
`aggregate_auxiliary.py` → `build_final_dataset.py` →
`compare_with_auxiliary_features.py` → `shap_analysis.py`. Or open the
notebook — it re-imports from `src/` and does not duplicate the logic.

## 1. EDA — what we found and why it matters

| Check | Finding | Why it matters |
|---|---|---|
| Class imbalance | 91.93% repaid / 8.07% default (11.4:1) | A "predict everyone repays" model scores ~92% accuracy while catching **zero** defaulters. This is why accuracy is banned from this project's evaluation — see Section 4. |
| Missing values | Building-quality fields (`COMMONAREA_*`, `NONLIVINGAPARTMENTS_*`) missing ~70%; `EXT_SOURCE_1` missing a large share | Not missing-at-random — most applicants don't live in a building with recorded quality data, and not every applicant has been scored by every bureau. Missingness itself can be a feature. |
| Correlation with target | `EXT_SOURCE_3/2/1` strongest (r ≈ −0.179, −0.160, −0.155); `DAYS_BIRTH` r ≈ +0.078 | Bureau scores dominate everything else — consistent with published Home Credit analyses. `DAYS_BIRTH` is stored as negative days-before-application, so its *positive* correlation with `TARGET` means younger applicants are riskier — an easy sign-convention trap to misstate out loud. |

Figures: `reports/figures/01_class_imbalance.png`, `02_missing_values.png`,
`03_target_correlations.png`.

## 2. Preprocessing

- Fixed the `DAYS_EMPLOYED == 365243` sentinel (Home Credit's code for
  "not currently employed") — flagged as its own binary feature, then
  nulled before imputation.
- Engineered `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `CREDIT_TERM`,
  `AGE_YEARS` — debt-service-style ratios, motivated by the EDA finding
  that raw `AMT_CREDIT`/`AMT_INCOME_TOTAL` are individually weak.
- Median imputation for numeric (skew-robust), explicit `"Missing"`
  category for categoricals.
- Label encoding for categoricals — a known and explicitly flagged
  limitation for logistic regression (implies false ordinality).
- **Stratified** train/test split (80/20) — mandatory at 11.4:1 imbalance.

### Class-weighting vs. SMOTE

Both implemented and compared head-to-head, not just discussed.

| | Class weighting | SMOTE |
|---|---|---|
| What it does | Reweights the loss function; training rows untouched | Interpolates between real minority-class points to synthesize new rows |
| Cost | Essentially free (one parameter) | Adds synthetic data that may not correspond to a plausible real applicant |
| Leakage risk | None | Must be fit on the training fold only — pre-split SMOTE leaks test-set neighbors into training |
| Calibration | Predicted probabilities stay close to the real ~8% base rate | Training on a synthetically balanced 50/50 set decalibrates predicted probabilities |

**Result on real data: class-weighting wins decisively across all three
algorithms** (full numbers in Section 4). Most striking case: XGBoost
trained on SMOTE-resampled data collapses to **1.6% recall** at the default
threshold, despite a respectable-looking 0.736 AUC — a concrete, measured
illustration of the calibration-shift risk described above, not just a
theoretical concern.

## 3. Modeling

Three algorithms × two imbalance strategies = 6 variants, trained in
`src/modeling.py`.

- **Logistic Regression** — linear in log-odds; directly interpretable
  coefficients, valuable where regulators (ECOA/Reg B) expect an
  explainable baseline.
- **Random Forest** — bagged, de-correlated trees; captures nonlinearities,
  robust to outliers/unscaled features.
- **XGBoost** — sequential boosting, each tree fits the ensemble's residual
  errors, with L1/L2 regularization. Consistently the top performer on
  structured/tabular Kaggle leaderboards including this exact competition.

## 4. Evaluation — never accuracy

**Precision vs. recall in a lending context (the key concept):**
- **Precision** = of loans flagged high-risk, what fraction actually
  default. Low precision → rejecting good applicants → cost = **lost
  business**.
- **Recall** = of loans that actually default, what fraction we caught.
  Low recall → approving loans that go bad → cost = **bad debt** (loss
  given default on unsecured consumer credit is often 60–90% of principal).

Because a missed default typically costs far more per incident than one
foregone good approval, credit models are usually tuned toward **recall on
the default class**, and the right operating point comes from walking the
precision-recall curve against actual dollar costs, not a default 0.5 cutoff.

### Results (real data, sorted by AUC-ROC)

| Model | AUC-ROC | Precision | Recall | F1 | Avg. Precision (PR-AUC) |
|---|---|---|---|---|---|
| **XGBoost (class-weighted)** | **0.7679** | 0.176 | 0.675 | 0.279 | 0.258 |
| LogReg (class-weighted) | 0.7504 | 0.161 | 0.686 | 0.261 | 0.233 |
| XGBoost (SMOTE) | 0.7362 | 0.388 | **0.016** | 0.031 | 0.206 |
| RandomForest (class-weighted) | 0.7357 | 0.161 | 0.641 | 0.258 | 0.214 |
| RandomForest (SMOTE) | 0.6434 | 0.159 | 0.122 | 0.138 | 0.128 |
| LogReg (SMOTE) | 0.6381 | 0.159 | 0.260 | 0.198 | 0.134 |

*(precision/recall/F1 at the default 0.5 threshold; see threshold table below)*

**XGBoost (class-weighted) wins outright** — matching the well-documented
pattern on this exact competition, where gradient boosting consistently
tops public leaderboards because real applicant data has genuine nonlinear
interactions (e.g. the effect of age depends on employment type) a linear
model can't represent.

Note the **XGBoost_SMOTE row**: a deceptively high 0.388 precision paired
with 1.6% recall means it's essentially only flagging the handful of
highest-confidence defaults and calling everyone else safe — nearly useless
for actually catching bad loans, despite a respectable AUC. This is exactly
why AUC alone is an incomplete picture and precision/recall have to be read
together.

Figures: `reports/figures/04_roc_curves.png`, `05_precision_recall_curves.png`.

### Threshold tradeoff (best model: XGBoost, class-weighted)

| Threshold | Precision | Recall | % of applicants flagged |
|---|---|---|---|
| 0.1 | 0.084 | 0.995 | 95.8% |
| 0.2 | 0.097 | 0.963 | 80.4% |
| 0.3 | 0.117 | 0.896 | 62.0% |
| 0.4 | 0.144 | 0.807 | 45.2% |
| **0.5** | **0.176** | **0.675** | **31.0%** |
| 0.6 | 0.219 | 0.529 | 19.5% |
| 0.7 | 0.281 | 0.354 | 10.2% |

This is the table to bring to a Risk/policy meeting, not a single accuracy
number: at 0.5 the model flags 31% of applicants and catches 67.5% of
actual defaulters at 17.6% precision. Moving to 0.3 catches 90% of
defaulters but means flagging 62% of the entire pool — the real tradeoff a
bank's risk appetite, not a data scientist, should ultimately decide.

## 5. Does credit-bureau / prior-application history help? (measured, not assumed)

Everything above uses only `application_train.csv` — the applicant's
profile *at the moment of this specific application*. Real Kaggle solutions
for this competition get most of their additional lift from the auxiliary
tables, because those capture something a single application snapshot
structurally cannot: a client's **track record**.

`src/aggregate_auxiliary.py` builds ~35 aggregate features per applicant —
processed with chunked, dtype-optimized pandas (343MB peak memory against
1.5GB of raw CSVs) rather than loading everything naively:
- **Bureau history**: number of prior bureau-reported loans, how many are
  currently active, total debt-to-credit ratio, max days overdue, months
  spent delinquent (via `bureau_balance.csv`'s monthly status codes)
- **Prior Home Credit applications**: count, approval rate, **refusal
  rate**, average credit amount, average term
- **POS/cash loan history**: months in delinquency, max days-past-due,
  completed-loan count
- **Credit card history**: average balance, **utilization rate**, total
  drawings, months delinquent

We retrained the best model (XGBoost, class-weighted) with these added,
holding the train/test split, algorithm, and hyperparameters fixed for a
clean, apples-to-apples comparison:

| | AUC-ROC | Avg. Precision | # Features |
|---|---|---|---|
| Baseline (application-only) | 0.7679 | 0.258 | 128 |
| **+ Auxiliary features** | **0.7807** | **0.275** | 165 |
| **Lift** | **+0.0127** | **+0.017** | +37 |

This is squarely in line with what published Home Credit solutions report
for this class of feature engineering, and it's a **measured** result on a
fixed split, not an assumption. Figure:
`reports/figures/08_auxiliary_feature_lift.png`.

**Top new features by importance:** `PREV_REFUSAL_RATE` (has this
applicant been turned down by Home Credit before), `CC_UTILIZATION_MEAN`
(revolving credit utilization — a classic risk signal), and
`BUREAU_DEBT_CREDIT_RATIO` (how leveraged is this person outside Home
Credit). All three are exactly the "track record" signal a single
application snapshot can't capture — two applicants can look identical on
income/education/EXT_SOURCE and still carry very different risk depending
on their credit history.

## 6. Explainability — SHAP

Run on **XGBoost (class-weighted)**, the genuine best model — deliberately
not on logistic regression, since its coefficients are already its own
explanation. SHAP decomposes each prediction into additive per-feature
contributions, the tool a Risk team uses to justify an individual
declined-applicant decision (adverse action reasons under US lending law).

### Top features driving predicted default risk

| Rank | Feature | Mean \|SHAP value\| |
|---|---|---|
| 1 | `EXT_SOURCE_3` | 0.370 |
| 2 | `EXT_SOURCE_2` | 0.348 |
| 3 | `CREDIT_TERM` | 0.167 |
| 4 | `AMT_GOODS_PRICE` | 0.142 |
| 5 | `EXT_SOURCE_1` | 0.141 |
| 6 | `CODE_GENDER` | 0.113 |
| 7 | `NAME_EDUCATION_TYPE` | 0.096 |
| 8 | `AMT_ANNUITY` | 0.092 |

External bureau scores dominate by a wide margin, followed by loan-term
structure (`CREDIT_TERM` = annuity/credit — a proxy for loan length) and
demographic/employment features. This matches both the Section 1
correlation table and published Home Credit Kaggle kernels closely —
consistency across three independent views (correlation, SHAP, and public
benchmarks) is a strong sanity check that this model has learned genuine
signal rather than noise or a data leak.

Figures: `reports/figures/06_shap_global_importance.png`,
`07_shap_summary_beeswarm.png`.

## Key talking points for an interview

1. **Why not accuracy** — an 11.4:1 imbalance makes it a vanity metric; a
   trivial all-repaid model would score ~92%.
2. **Class-weighting vs SMOTE, measured head-to-head** — class-weighting
   won decisively; XGBoost_SMOTE's recall collapse to 1.6% despite a
   respectable AUC is a concrete illustration of *why*, not just a
   theoretical claim.
3. **Precision/recall asymmetry in lending** — bad-debt loss (missed
   default) is typically far larger per incident than foregone-margin loss
   (rejected good applicant), which is why risk models are tuned toward
   recall, and why the approval threshold is a business decision.
4. **Feature engineering value, measured not assumed** — bureau/prior-app/
   POS/credit-card history lifted AUC by +0.0127 on a fixed train/test
   split; can name the specific top new features and explain intuitively
   why each matters (refusal rate, utilization, leverage).
5. **SHAP model choice** — used on XGBoost specifically because linear
   models are already self-explaining; cross-checked against the
   correlation table and published benchmarks rather than taken at face
   value.
6. **Memory-conscious engineering** — 27M-row `bureau_balance.csv`
   aggregated in 343MB peak RAM via chunked, dtype-optimized pandas rather
   than loading 1.5GB of raw CSVs naively — a real production concern when
   these tables are 10-100x larger in an actual bank's data warehouse.

## Reproducing this project

```bash
pip install -r requirements.txt

# Get the real data (see data/raw/DOWNLOAD_INSTRUCTIONS.md for exact commands),
# or run src/generate_data.py for an offline synthetic stand-in.

python src/eda.py
python src/preprocessing.py
python src/modeling.py                        # trains all 6 variants
python src/evaluate.py
python src/aggregate_auxiliary.py             # bureau/prev-app/POS/CC -> per-applicant features
python src/build_final_dataset.py             # merge onto application_train
python src/compare_with_auxiliary_features.py # baseline vs +auxiliary AUC lift
python src/shap_analysis.py
```

Or open `notebooks/credit_risk_walkthrough.ipynb` for the full narrated,
pre-executed version.

## Honest limitations / natural next steps to mention if asked

- Hyperparameters are reasonable defaults, not tuned via grid/Bayesian
  search — a natural "if I had more time" answer.
- Auxiliary features are aggregate statistics (mean/sum/max per applicant);
  richer time-series features (e.g. trend in DPD over the last 6 months)
  would likely add further lift.
- Label encoding for Logistic Regression's categoricals is a known,
  explicitly-flagged shortcut; a production linear model would use one-hot
  or target encoding instead.
- No probability calibration step (e.g. Platt scaling/isotonic regression)
  was applied on top of the class-weighted model — worth adding before
  predicted probabilities are used directly in expected-loss pricing.
