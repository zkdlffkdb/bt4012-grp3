# BT4012 Group 3: IEEE-CIS Fraud Detection
## Overview
This repository contains the end-to-end data processing, feature engineering, exploratory data analysis, and machine learning benchmarking for the IEEE-CIS Fraud Detection project. The objective is to identify fraudulent e-commerce transactions while addressing severe class imbalance across high-dimensional tabular data.

## EDA
- Data Sources: Merged transaction records (train_transaction.csv) and identity attributes (train_identity.csv) linked by TransactionID.
- Transaction Dataset: Contains 590,540 rows and 394 features covering transaction amounts, card attributes, address codes, and anonymized V-series features.
- Identity Dataset: Contains 144,233 rows and 41 features detailing device types, network information, and browser metadata.
- Merged Dataset Dimensions: Merging transactions and identity records produces a combined dataset of 590,540 rows and 434 columns.
- Target Class Imbalance: Demonstrates extreme target skewness with 569,877 non-fraudulent cases (97%) and 20,663 fraudulent cases (3%).
- Data Quality: Contains 0 duplicate rows across all 590,540 records (0.00% duplicates).
- Temporal Range: TransactionDT values span from 86,400 to 15,811,131 seconds.
- Missing Value Profile: Features display significant missingness, with specific identity attributes (id_24, id_25, id_07, id_08, id_21, id_26) exceeding 99% missing values.   

## Preprocessing & Modeling Pipeline
- Dataset Splits:
  - Train Set: 440,018 samples, 488 columns.
  - Validation Set: 61,196 samples, 488 columns.
  - Test Set: 89,326 samples, 488 columns.
- Excluded Features: Non-predictive metadata fields excluded prior to modeling include LABEL (isFraud), TransactionID, TransactionDT, TransactionDateTime, Transaction_Day, and uid.
- Cleaning & Categorical Encoding:Replaced infinite values (+inf, -inf) with NaN across training and validation sets.
- Identified 37 categorical features (e.g., ProductCD, card4, card6, P_emaildomain, R_emaildomain, M1–M9, DeviceType, DeviceInfo, id_12–id_38) and encoded them via OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).
- Evaluation Metric: Primary optimization metric is Average Precision (PR-AUC / average_precision_score) to evaluate minority class retrieval under heavy imbalance.
- Imbalance Scaling Factor: Calculated positive class weight (POS_WEIGHT) as 27.56 based on training class proportions.

## Fine Tuning
The selected XGBoost model was tuned using three chronological cross-validation folds, with Average Precision (AP) as the selection metric. A decision threshold of approximately 0.3574 was then selected by maximizing validation F1.

On the test set:
- Hyperparameter tuning improved AP from **0.5942 to 0.6261** and F1 from **0.5826 to 0.6090** at the default 0.5 threshold.
- Lowering the threshold increased recall from **52.83% to 58.48%**, detecting **176 additional fraud cases** but producing **584 additional false alarms**.
- The selected threshold achieved test F1 of **0.5910**; its validation F1 improvement did not generalize to test.
- See the README in `Fine_Tuning/` for the full methodology, evaluation limitations, results, and reproduction instructions.

## Folder Directory

```plaintext
bt4012-grp3/
├── data/
│   ├── train_transaction.csv   # Transaction features
│   ├── train_identity.csv      # Identity metadata
│   ├── train.csv               # Train split (440,018 samples)
│   ├── val.csv                 # Validation split (61,196 samples)
│   └── test.csv                # Test split (89,326 samples)
├── Data_Transformation_Feature_Engineering.ipynb     # Exploratory Data Analysis & missing value profiling
├── Modelling.ipynb       # Initial baseline modeling (XGBoost, LightGBM, Logistic Regression) and imbalance expts
├── saved_models/                   # trained model artifacts
├── .gitignore                      # Git ignore file for data files and heavy artifacts
├── .venv                           # virtual environment
├── requirements.txt                # Python environment dependencies
├── Fine_Tuning/fine_tuning
│   ├── Step_1_Baseline.ipynb
│   ├── Step_2_Hyperparameter_Tuning.ipynb
│   ├── Step_3_Threshold_Tuning.ipynb
│   ├── Step_4_Final_Test_Evaluation.ipynb
│   ├── baseline_outputs/                    # Intermediate artifacts
│   ├── step_2_outputs/                      # Hyperparameter search results
│   ├── step_3_outputs/                      # Threshold selection results
│   ├── step_4_outputs/                      # Final model, metrics and plots
│   ├── requirements-tuning.txt
│   ├── .gitignore
│   └── README.md                            # Detailed tuning documentation
└── README.md                       # Project documentation and summary
```
