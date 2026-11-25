# FraudShield

FraudShield is a machine learning project for detecting fraudulent credit card transactions. It compares five classifiers and examines a practical trade-off: how many frauds each model catches and how many legitimate transactions it flags along the way.

Developed for **COE 546/746 · Fall 2026**.

## Dataset

The [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) contains 284,807 transactions, including 492 labeled as fraud. Its inputs are the anonymized features `V1`–`V28`, elapsed time, and transaction amount. The target column is `Class` (`0` for legitimate, `1` for fraud).

After removing 1,081 exact duplicate rows, the notebook uses 283,726 transactions, including 473 frauds. With fraud making up only 0.17% of the cleaned data, accuracy alone would hide how many frauds a model misses.

## How to run

Use Python 3.12 to match the notebook’s documented environment. From the repository folder, create and activate a virtual environment. These commands are for Git Bash on Windows:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install numpy pandas matplotlib seaborn scikit-learn==1.8.0 notebook
```

Download and extract `creditcard.csv` from Kaggle and place it at `data/creditcard.csv` relative to the repository root. Keep the CSV local. Then start Jupyter:

```bash
jupyter notebook
```

Open the final FraudShield notebook and run the cells from top to bottom. The model searches can take several minutes.

## Method

- **Part A — Data preparation:** Check data quality, inspect class imbalance and feature distributions, remove exact duplicates, and create a stratified split. Fit scaling inside training folds for models that need it.
- **Part B — Model comparison:** Tune Logistic Regression, Decision Tree, Random Forest, Histogram Gradient Boosting, and k-Nearest Neighbors on the same five stratified folds. Select settings by mean validation average precision (AP).
- **Part C — Evaluation:** Compare all five models on unseen transactions. Report AP for ranking quality and use counts, precision, recall, and F1 to explain the effect of the alert threshold. Part D uses a separate holdout.

## Part C results

The evaluation set contained 28,373 transactions, including 48 frauds. Alert counts below use a score threshold of 0.50.

| Model | Average precision | Frauds caught (out of 48) | False alerts |
| --- | ---: | ---: | ---: |
| Histogram Gradient Boosting | 0.760 | 35 | 1 |
| k-Nearest Neighbors | 0.753 | 30 | 1 |
| Random Forest | 0.744 | 31 | 2 |
| Decision Tree | 0.672 | 31 | 5 |
| Logistic Regression | 0.659 | 40 | 659 |

Logistic Regression caught five more frauds than Gradient Boosting and generated 658 additional false alerts. Lowering the Gradient Boosting threshold to 0.232, based on predictions from the training folds, caught the same 35 frauds and increased false alerts from one to four.

The evaluation covers transactions from the same short period. With just 48 fraud cases, one additional detection shifts recall by about 2.1 percentage points.

## Part D: Amount-aware fraud alerts

Part D tests whether using both the fraud score and the transaction amount lowers the modeled cost of fraud decisions. The amount-aware policy flags a transaction when its fraud score multiplied by its amount reaches the assumed review cost. The experiment charges €5 for each review and counts the full transaction amount as the loss for a missed fraud. These are modeling assumptions. The notebook also tests isotonic calibration on training predictions and varies the assumed review cost.

The final evaluation used a separate set of 28,373 transactions with 47 frauds. The three policies below use the same Gradient Boosting model.

| Policy at €5 per review | Alerts | Frauds caught (out of 47) | False alerts | Fraud amount flagged | Modeled cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default threshold (0.50) | 37 | 35 | 2 | €3,108.56 | €1,523.11 |
| Global threshold tuned for cost | 59 | 37 | 22 | €3,145.60 | €1,596.07 |
| Amount-aware policy | 37 | 19 | 18 | €3,110.86 | €1,520.81 |

With the same 37 reviews, the amount-aware rule flagged roughly the same fraud value as the default while catching 16 fewer fraudulent transactions. Its modeled cost was €2.30 lower, with a paired 95% bootstrap interval of −€63.31 to +€55.14 for the cost difference. The sample therefore leaves the direction of the cost difference uncertain.

The evaluation covers a short collection period and a small number of frauds. The cost model assumes a flagged fraud avoids its full transaction loss and omits downstream effects, such as the value of catching small transactions that may signal later fraud. See the notebook for the sensitivity analysis, detailed results, and references.
