# Model Selection for Logistic Regression

Assignment 5 for the ML Life Cycle course: Evaluation and Deployment. This project trains and tunes a logistic regression model to predict whether an Airbnb host is a "superhost."

## Problem

- **Task:** Binary classification
- **Label:** `host_is_superhost` (True / False)
- **Features:** All remaining columns in the Airbnb listings dataset (already one-hot encoded, scaled, and imputed)
- **Data split:** 90% train / 10% test, `random_state=42`

## What This Notebook Does

1. Loads and prepares the Airbnb dataset (`data_LR/airbnbData_train.csv`)
2. Trains a baseline logistic regression model using scikit-learn's default `C=1.0`
3. Runs `GridSearchCV` (5-fold CV) over 10 values of `C` (`10^-5` to `10^4`) to find the optimal hyperparameter
4. Trains a second model, `model_best`, using the optimal `C`
5. Compares both models using:
   - Confusion matrices
   - Precision-recall curves
   - ROC curves and AUC scores
6. Explores feature selection with `SelectKBest` to identify the top 5 predictive features
7. Saves the best-performing model to a `.pkl` file using `pickle`

## Results

| Model | AUC |
|---|---|
| Default (`C=1.0`) | 0.8206 |
| Best (tuned `C`) | 0.8209 |

- Best hyperparameter found via grid search: `C = 1000`
- Top 5 features by `SelectKBest` (ANOVA F-test): `host_response_rate`, `number_of_reviews`, `number_of_reviews_ltm`, `number_of_reviews_l30d`, `review_scores_cleanliness`
- AUC using only the top 5 features: 0.7926 (lower than the full-feature model, so the additional features do contribute meaningfully)

## Repo Structure

```
model-selection-logistic-regression/
├── README.md
├── data/
│   └── airbnbData_train.csv
├── models/
│   └── airbnb_model.pkl
├── notebooks/
│   └── ModelSelectionForLogisticRegression.ipynb
└── .gitignore
```

## Tools Used

- Python, pandas, NumPy
- scikit-learn (`LogisticRegression`, `GridSearchCV`, `SelectKBest`, ROC/AUC and confusion matrix utilities)
- matplotlib, seaborn
- pickle (model persistence)

## How to Reproduce

1. Clone the repo
2. Open `notebooks/ModelSelectionForLogisticRegression.ipynb`
3. Make sure `data/airbnbData_train.csv` is accessible at the path referenced in the notebook (`data_LR/airbnbData_train.csv`, adjust path if needed)
4. Run all cells top to bottom
