# Insurance Claim Severity for Decision Support

This repository contains my MSDS 696 Practicum II project at Regis University. The project looks at whether historical claim information can be used to predict final claim cost well enough to help claims staff decide which files may need earlier reserve review.

The model is not intended to set reserves automatically or decide which adjuster should handle a claim. I treated the prediction as one more signal that could help a person decide what to review first.

## Project question

Can machine learning predict insurance claim severity from the available claim information well enough to support reserve review and claim triage?

In this project, claim severity means the final cost of a claim to the insurer.

## Data

The analysis uses the Allstate Claims Severity dataset.

The files used by the notebook are:

- `train.csv` — 188,318 rows and 132 columns
- `test.csv` — 125,546 rows and 131 columns
- `sample_submission.csv` — 125,546 rows and 2 columns

The training data contains 116 categorical features along with continuous features, an ID field, and the final claim loss. The feature names are anonymized, so the analysis cannot confirm the business meaning or exact timing of every predictor.

Place the three CSV files in the `data/` folder. The notebook also supports the `ALLSTATE_DATA_DIR` environment variable if the files are stored somewhere else.

## Approach

I first checked the files, row counts, IDs, duplicate records, and train/test separation. I then split the training data into 80% development data and 20% holdout validation data.

The project compares simple mean and median baselines with two ridge-regression pipelines:

1. Ridge regression with ordinal encoding and a log-transformed loss target.
2. Ridge regression with out-of-fold target encoding and a log-transformed loss target.

I kept the regression model family the same so the comparison mainly tested how the categorical variables were encoded.

Ridge regression was a practical choice because the dataset has many predictors and the model stays stable when features are correlated. The log transformation was used because claim loss is strongly right-skewed and a small number of very large claims can dominate a model trained directly on dollar values.

Out-of-fold target encoding performed slightly better than ordinal encoding. It also avoids using a training row's own loss to create that row's encoded feature values, which reduces target-leakage risk.

## Main results

The selected model was ridge regression with out-of-fold target encoding and a log-transformed target.

| Measure | Result |
|---|---:|
| Validation claims | 37,664 |
| MAE | $1,254 |
| Bootstrap 95% MAE interval | $1,237 to $1,270 |
| RMSE | $2,142 |
| Median absolute error | $720 |
| R² | 0.417 |
| MAE improvement over median baseline | 30.2% |
| Spearman rank correlation | 0.707 |

The ranking result was especially useful for the business question. When the top 10% of model-ranked claims were sent for additional review, the queue captured 55.5% of severe claims. The lift over random review was 5.55x.

The model still missed 1,676 severe claims at that review level, so the score should not replace current rules or human judgment.

## Upper-tail limitation

The most expensive claims were also the hardest cases.

For claims above the training-set 95th-percentile loss threshold:

- MAE was $5,513.
- 85.4% were underpredicted.
- When underprediction occurred, the average shortfall was about $5,707.

I also tested a Duan-smearing correction because the model was trained on log-transformed loss. The correction reduced some underprediction in the highest-cost group, but overall validation error became worse, so I kept the uncorrected inverse transformation as the final prediction method.

## What the result means

The project supports using the model as a review-priority signal. A higher score can help identify claims that may deserve earlier attention.

The project does not show that the model should:

- set a reserve amount by itself,
- approve or settle a claim,
- replace claims staff,
- or assign a claim to a specific adjuster.

The dataset does not contain adjuster workload, adjuster skill, reserve history, policy limits, claim notes, or enough timing information to support those decisions.

## Repository files

```text
Insurance_Claim_Severity_GitHub_Repo/
├── README.md
├── requirements.txt
├── .gitignore
├── insurance_claim_severity_practicum.ipynb
├── data/
│   └── README.md
├── outputs/
│   └── .gitkeep
└── presentation/
    └── Insurance_Claim_Severity_Final_Presentation.pptx
```

## How to run the notebook

1. Clone or download this repository.
2. Create a Python environment.
3. Install the packages:

```bash
pip install -r requirements.txt
```

4. Put `train.csv`, `test.csv`, and `sample_submission.csv` in the `data/` folder.
5. Start Jupyter:

```bash
jupyter notebook
```

6. Open `insurance_claim_severity_practicum.ipynb` and run the cells from top to bottom.

The notebook uses a fixed random seed of 42. When the full analysis finishes, it creates:

```text
Allstate_Claim_Severity_Predictions.csv
```

## Next step

The most important next step would be to test the model on newer claims using only fields that are confirmed to be available when the claim is first reported. I would only move to an operational pilot if accuracy, ranking, calibration, and severe-claim capture remain stable.

## Author

Vasantha Perugu  
MSDS 696 – Practicum II  
Regis University
