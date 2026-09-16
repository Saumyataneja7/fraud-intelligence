# Classical ML Report — Fraud Intelligence

**Project:** Fraud Intelligence — Graph ML + Explainable AI  
**Phase:** 5 — Classical Machine Learning  
**Status:** Complete and frozen  
**Dataset:** fraud-intelligence-synthetic v1.0.0  
**Dataset size:** 100,000 transactions  
**Target fraud rate:** 1%  
**Random seed:** 42  

---

## 1. Objective

The objective of Phase 5 is to establish a classical machine-learning baseline for transaction-level fraud detection before introducing heterogeneous graph machine learning.

The classical ML stage answers:

> Can transaction-level behavioral, temporal, historical, velocity, entity, novelty, and deviation features identify suspicious transactions without using future information or graph-derived information?

The models evaluated are:

1. Logistic Regression
2. Random Forest
3. XGBoost

The models are evaluated using a chronological train/validation/test split.

---

## 2. Dataset

The project uses a custom synthetic fintech transaction dataset.

The dataset is explicitly synthetic and does not represent real customer or financial activity.

The development dataset contains:

- 100,000 transactions
- 99,000 legitimate transactions
- 1,000 fraudulent transactions
- 1% synthetic fraud rate
- UTC transaction timestamps
- 2025 transaction period
- USD transactions

Fraud scenarios include:

- ACCOUNT_TAKEOVER
- SHARED_DEVICE
- SHARED_IP
- MERCHANT_ABUSE
- VELOCITY_ATTACK
- GEO_ANOMALY
- FRAUD_RING

The dataset is reproducible using random seed 42.

---

## 3. Temporal Split

The dataset was divided chronologically rather than randomly.

The split configuration is:

| Split | Proportion | Rows |
|---|---:|---:|
| Train | 70% | 70,000 |
| Validation | 15% | 15,000 |
| Test | 15% | 15,000 |

The temporal ordering prevents future transactions from being used to train models evaluated on earlier or later data.

The test set remains isolated from threshold optimization.

---

## 4. Model Features

The model dataset contains **53 approved engineered features**.

The features originate from the Phase 4 feature-engineering pipeline.

Feature families include:

- transaction features
- temporal features
- customer historical features
- velocity features
- entity historical features
- novelty features
- behavioral deviation features

The following are excluded from model input:

- `is_fraud`
- `fraud_scenario`
- transaction identifiers
- customer/account/card identifiers
- merchant/device/IP identifiers
- raw timestamp
- raw transaction fields

The target is maintained separately as:

```text
is_fraud
'''

---

## 5. Class Imbalance

The training data contains substantially more legitimate transactions than fraudulent transactions.

The Phase 5.3 imbalance strategy uses:

-balanced class weights for Logistic Regression
-balanced class weights for Random Forest
-scale_pos_weight for XGBoost

No synthetic oversampling was introduced.

This keeps the baseline pipeline deterministic and avoids introducing an additional sampling layer before graph-based modeling.

---

## 6. Models
### 6.1 Logistic Regression

Configuration:

-C = 1.0
-max_iter = 1000
-solver = lbfgs
-random_state = 42
-balanced class weights
-StandardScaler preprocessing

Logistic Regression provides a linear baseline and establishes how much predictive signal can be obtained without nonlinear decision boundaries.

### 6.2 Random Forest

Configuration:

-n_estimators = 200
-max_features = sqrt
-random_state = 42
-n_jobs = -1
-balanced class weights

Random Forest provides a nonlinear tree-based baseline without feature scaling.

## 6.3 XGBoost

Configuration:

-n_estimators = 200
-max_depth = 6
-learning_rate = 0.05
-subsample = 0.8
-colsample_bytree = 0.8
-min_child_weight = 1.0
-reg_alpha = 0.0
-reg_lambda = 1.0
-random_state = 42
-n_jobs = -1
-eval_metric = logloss
-class imbalance handled using scale_pos_weight

---

## 7. Baseline Evaluation

The baseline classification threshold is 0.5.

### Validation

Model	Precision	Recall	F1	PR-AUC	ROC-AUC
Logistic Regression	5.90%	72.50%	10.91%	0.331	0.899
Random Forest	94.74%	22.50%	36.36%	0.398	0.877
XGBoost	21.05%	42.50%	28.16%	0.366	0.877

### Test
Model	Precision	Recall	F1	PR-AUC	ROC-AUC
Logistic Regression	5.60%	78.26%	10.46%	0.318	0.912
Random Forest	88.00%	15.94%	26.99%	0.363	0.874
XGBoost	23.28%	39.13%	29.19%	0.333	0.904

---

## 8. Interpretation of Baseline Results

The models show substantially different behavior at the default threshold of 0.5.

Logistic Regression identifies a large number of transactions as suspicious. This produces relatively high recall but also a large number of false positives.

Random Forest is much more conservative at the default threshold. Its flagged transactions have high precision, but it misses a large proportion of actual fraud cases.

XGBoost produces an intermediate operating point between the two models at the default threshold.

The results demonstrate why a fraud detection system should not be evaluated using a single metric.

In particular, ROC-AUC measures ranking/discrimination across thresholds, whereas precision, recall, and F1 at threshold 0.5 describe one specific operating point.

Because the synthetic dataset contains a 1% positive class, precision-recall metrics and false-positive/false-negative counts are important diagnostic measures.

---

## 9. Threshold Optimization

Phase 5.8 performs threshold optimization using the validation set.

The optimization objective is F1.

The test set is not used to select the threshold.

This separation preserves the test set as an unbiased final evaluation set.

The threshold is treated as a decision-policy parameter rather than as part of the trained model itself.

---

## 10. Ranking Evaluation

Phase 5.9 evaluates the models using:

-Precision@K
-Recall@K

This is important for fraud investigation workflows where analysts may only be able to manually review a limited number of alerts.

The ranking evaluation considers the highest-scoring transactions rather than relying exclusively on a binary classification threshold.

Recommended K values for the 100,000-transaction development dataset include:

-K = 100
-K = 500
-K = 1,000
-K = 5,000

---

## 11. Error Analysis

Phase 5.10 separates classification errors into:

-True Positives
-True Negatives
-False Positives
-False Negatives

False positives represent legitimate transactions incorrectly flagged as suspicious.

False negatives represent fraudulent transactions that were not detected.

Error analysis can additionally examine error concentration across transaction attributes and fraud scenarios.

fraud_scenario is permitted for post-hoc diagnostic analysis but is never used as a model feature.

## 12. Leakage Controls

The Phase 4 leakage audit and Phase 5 temporal split provide the primary leakage controls.

The pipeline enforces:

-target columns excluded from model features
-identifier columns excluded from model features
-historical features use only earlier transactions
-velocity features use only strictly previous transactions
-same-timestamp transactions cannot influence one another
-future-transaction invariance
-chronological train/validation/test splitting
-validation-only threshold selection
-test-set isolation during threshold optimization

Label-derived fraud-ring information remains restricted to diagnostic analysis and is not used as a predictive feature.

---

## 13. Reproducibility

The classical ML stage is designed to be reproducible.

Key controls include:

-fixed dataset version
-fixed random seed
-deterministic temporal split
-deterministic feature selection
-deterministic ranking ordering
-fixed model configurations
-explicit class-imbalance strategy
-persisted evaluation artifacts

Dataset version:

fraud-intelligence-synthetic v1.0.0

Random seed:

42

---

## 14. Phase 5 Artifacts

The following artifacts are generated or maintained during Phase 5:

reports/
├── model_evaluation/
│   ├── classical_model_metrics.csv
│   └── classical_model_metrics.json
│
└── classical_ml/
    └── CLASSICAL_ML_REPORT.md

The feature dataset remains:

data/processed/features/feature_dataset.parquet

---

## 15. Limitations

The dataset is synthetic and therefore does not represent real-world fraud prevalence, customer behavior, transaction distributions, or operational alert volumes.

The 1% fraud rate is a project design choice.

The classical models operate primarily on transaction-level engineered features and do not yet exploit the full relational structure between customers, accounts, cards, merchants, devices, and IP addresses.

The baseline threshold of 0.5 is not assumed to be operationally optimal.

Model metrics are specific to the current synthetic dataset, feature pipeline, temporal split, and model configurations.

Therefore, these results should be interpreted as a controlled project baseline rather than as evidence of production fraud-detection performance.

---

## 16. Transition to Graph ML

The classical ML baseline establishes a transaction-level benchmark.

The next modeling stage introduces relational information.

The graph representation will model relationships such as:

Customer
   ↓
Account
   ↓
Card
   ↓
Transaction
   ↓
Merchant

Customer
   ↓
Device
   ↓
Transaction

Customer
   ↓
IP
   ↓
Transaction

This allows the project to investigate patterns that are difficult to represent using independent transaction rows.

Examples include:

-shared devices
-shared IP addresses
-repeated merchant relationships
-dense customer-device connections
-coordinated transaction behavior
-potential fraud rings

The graph stage will therefore be evaluated against the classical ML baseline rather than replacing it without comparison.

---

## 17. Phase 5 Freeze

Phase 5 establishes the classical machine-learning baseline for Fraud Intelligence.

The implementation includes:

-temporal splitting
-model dataset preparation
-class imbalance handling
-Logistic Regression
-Random Forest
-XGBoost
-baseline evaluation
-threshold optimization
-Precision@K / Recall@K
-error analysis
-model comparison
-leakage controls
-reproducible evaluation artifacts

The classical ML stage is now frozen.

Future changes to the feature pipeline, split strategy, model configurations, or target definition should create a new experiment/version rather than silently modifying the frozen baseline.

