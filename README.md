# Online Shoppers Purchasing Intention — Classification

M.Tech (AIML/DSE) — Machine Learning — Assignment 2

## a. Problem Statement

Online retailers want to know, in real time, whether a visitor browsing
their site is likely to complete a purchase during that session. Being
able to predict this "purchasing intention" lets a business trigger
targeted interventions (discounts, chat pop-ups, retargeting) for
visitors who are unlikely to buy, and avoid annoying visitors who are
already likely to convert.

This project frames it as a **binary classification problem**: given a
set of session-level behavioural and technical features, predict
whether the session ends in a purchase (`Revenue = True`) or not
(`Revenue = False`).

## b. Dataset Description

**Source:** UCI Machine Learning Repository — *Online Shoppers
Purchasing Intention Dataset* (Sakar & Kastro, 2018).

- **Instances:** 12,330 sessions (one row per user session)
- **Features:** 17 (exceeds the minimum required 12)
- **Target:** `Revenue` (Boolean — purchase / no purchase)
- **Class balance:** 10,422 negative (no purchase) vs. 1,908 positive
  (purchase) — an imbalanced ~85% / 15% split, which is realistic for
  e-commerce conversion data.

**Feature groups:**
| Type | Features |
|---|---|
| Page-visit counts & durations | `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration` |
| Google-Analytics style metrics | `BounceRates`, `ExitRates`, `PageValues` |
| Calendar context | `SpecialDay`, `Month`, `Weekend` |
| Technical / demographic | `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType` |

No missing values were present in the raw data. Numeric features were
standardized (`StandardScaler`) and categorical features (`Month`,
`VisitorType`) were one-hot encoded inside a single `sklearn` `Pipeline`
so that the exact same preprocessing is applied at training and at
inference time in the Streamlit app.

A stratified 80/20 train/test split was used (`random_state=42`) to
preserve the class ratio in both splits. The 20% hold-out split
(2,466 rows) is saved as `test_data.csv` and is what gets uploaded to
the Streamlit app for scoring.

## c. GitHub Repository Link

`https://github.com/2025ac05870/2025ac05870_ML_assignment_2`

## d. Models Used

All 5 models were trained on identical preprocessed data
(`train.py`) and evaluated on the same hold-out test split.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8812 | 0.8877 | 0.7432 | 0.3560 | 0.4814 | 0.4603 |
| Decision Tree | 0.8556 | 0.7307 | 0.5330 | 0.5497 | 0.5412 | 0.4557 |
| kNN | 0.8739 | 0.7993 | 0.6715 | 0.3639 | 0.4720 | 0.4322 |
| Naive Bayes | 0.6736 | 0.7939 | 0.2941 | 0.7906 | 0.4287 | 0.3249 |
| Random Forest (Ensemble) | **0.8994** | **0.9236** | 0.7410 | 0.5393 | **0.6242** | **0.5774** |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong AUC and precision, but recall is low (0.36) — the linear decision boundary misses many buyers, biased toward the majority "no purchase" class. |
| Decision Tree | Balanced precision/recall (best recall among simple models) but the lowest AUC — a single unpruned tree overfits to noisy splits and ranks probabilities poorly. |
| kNN | Similar pattern to Logistic Regression: decent precision, weak recall. Distance-based similarity struggles with the one-hot encoded categorical features and class imbalance. |
| Naive Bayes | Highest recall by far (0.79) because the independence assumption pushes many borderline cases into the positive class, but this tanks precision/accuracy — lots of false alarms. |
| Random Forest (Ensemble) | **Best overall** on every metric except recall vs. Naive Bayes/Decision Tree. Averaging many trees reduces variance and handles the mixed numeric/categorical feature set and class imbalance far better than any single model. |
| **Overall Winner** | **Random Forest (Ensemble)** — highest Accuracy, AUC, F1, and MCC, making it the most reliable model for this imbalanced purchasing-intention dataset. |

## Live App

`https://2025ac05870mlassignment2-n3635gp7pte5gbxrgaeepn.streamlit.app/`
