# BT4012 Group 3: IEEE-CIS Fraud Detection
## Overview
This repository contains the end-to-end data processing, feature engineering, exploratory data analysis, and machine learning benchmarking for the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) project. The objective is to identify fraudulent e-commerce transactions while addressing severe class imbalance across high-dimensional tabular data.

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

## Evaluation and Validation (k-fold)
The notebooks above judge every model on one split of the data (train → validation → test). One split gives one number per metric. It cannot tell us how much that number would change if we scored a different month. `Evaluation.ipynb` adds k-fold cross-validation on top of the pipeline to answer three questions:

| Question | What the notebook does |
| --- | --- |
| How well does the model do on weeks it has never seen, and how much does that vary? | **Chronological 5-fold CV** on the development data (train + validation, Dec 2017 – Apr 2018). The first 40% of days are only used for training. The remaining days are cut into five blocks of about 18 days. Each fold trains on all rows at least 7 days before its block, then scores the block. This is how a fraud model is used in practice, so this is the estimate we trust. |
| How much would a classic shuffled k-fold overstate performance? | **Shuffled `StratifiedKFold` (5 folds)** on the same rows, reported for comparison only. In a shuffled split, rows from the future end up in the training folds. |
| How certain are the test numbers? | **Bootstrap (1,000 resamples) of the test split** with the fixed final model: 95% confidence intervals for every metric and for the differences between configurations. |

Two models are compared, both trained with random undersampling (fraud : non-fraud = 1 : 10): the **baseline XGBoost** (library defaults, as in `Modelling.ipynb`) and the **tuned XGBoost** (Fine_Tuning candidate 15). To avoid leakage, every fold rebuilds the 18 frequency encodings, the two label-history columns (`prev_uid_fraud_count`, `prev_uid_has_fraud`, with the 7-day label delay) and the category codes from its training rows only. The test split is never used in cross-validation.

The notebook reads the tuned hyperparameters and the decision threshold from the Fine_Tuning output folders. Those folders now hold a full re-run of the Fine_Tuning notebooks made on 2026-10-23 (candidate 15 was selected again; the F1-best validation threshold came out as 0.4594 instead of 0.3574). The original run is kept unchanged in `Fine_Tuning/fine_tuning/recorded_run/`, and `Fine_Tuning/fine_tuning/README.md` compares the two runs.

Key results:

| Estimate | Baseline XGBoost AP | Tuned XGBoost AP |
| --- | ---: | ---: |
| Chronological 5-fold CV (mean ± sd) | 0.654 ± 0.031 | **0.675 ± 0.020** — tuned better in 5 of 5 folds |
| Shuffled 5-fold CV (mean ± sd) | 0.780 ± 0.002 | 0.765 ± 0.003 — ranking reversed |
| Single validation split | 0.706 | 0.707 |
| Test split, 95% bootstrap interval | 0.594 [0.576, 0.612] | **0.626 [0.609, 0.642]** |

What this tells us:
- **Tuning helps, and the gain holds over time.** The tuned model wins every chronological fold (+0.021 AP on average). On the test split the gain is clearly larger than sampling noise: AP +0.032 (95% CI +0.026 to +0.039) and F1 at 0.5 +0.026 (CI +0.017 to +0.035).
- **Shuffled k-fold is the wrong tool for this data.** It overstates AP by about 0.09 and ranks the two models the wrong way round (the default baseline wins all five shuffled folds), because future rows and repeat cards leak into training. Only the chronological estimate should be used for decisions.
- **Performance depends on the period.** Tuned-model AP ranges from 0.647 to 0.697 across the five scoring blocks, and the May test split (0.626) is harder than any development block. A fair summary is "AP ≈ 0.67 ± 0.02 on unseen future weeks, 0.63 on the May test month".
- **The F1-best threshold is unstable.** It lands anywhere between 0.39 and 0.58 depending on the fold. On the test split the validation-chosen threshold changes F1 by only −0.0006 (CI −0.005 to +0.004) compared with 0.5; it trades precision for recall. The working threshold should be set from review capacity or from the cost of a missed fraud versus a false alarm, not from one month's F1 peak.

All tables and plots are written to `evaluation_outputs/` (`cv_fold_results.csv`, `cv_summary.csv`, `cv_paired_comparison.csv`, `test_bootstrap_ci.csv`, `test_bootstrap_differences.csv`, `consolidated_summary.csv`, three PNG figures and `evaluation_run.json` with the settings, package versions and data checksums of the run). The settings are constants at the top of the notebook (`N_FOLDS = 5`, `FIRST_SCORE_FRAC = 0.40`, `GAP_DAYS = 7`, `LABEL_DELAY_DAYS = 7`, `SAMPLING_RATIO = 0.1`, `N_BOOTSTRAP = 1000`, `SEED = 42`). The whole notebook runs in about 2 minutes on an Apple M4 Pro.

## How to Run the Notebooks End to End

**Step 1 – Get the data.** Download `train_transaction.csv` and `train_identity.csv` from the [Kaggle competition page](https://www.kaggle.com/competitions/ieee-fraud-detection/data) and put them in the repository root. All CSV files are git-ignored, so they are never committed.

**Step 2 – Create the Python environment.** The Fine_Tuning notebooks were recorded with pinned package versions, so the whole pipeline runs in one environment built from those pins (Python 3.13). From the repository root:

```bash
python3.13 -m venv .venv-tuning
.venv-tuning/bin/python -m pip install numpy==2.1.3 pandas==2.2.3 scikit-learn==1.6.1 imbalanced-learn==0.14.2 \
    xgboost==3.4.1 joblib==1.6.0 matplotlib==3.11.2 seaborn==0.13.2 lightgbm==4.7.0 ipykernel nbconvert
```

The core pins are the ones listed in `Fine_Tuning/fine_tuning/requirements-tuning.txt`. On macOS, LightGBM and XGBoost also need the OpenMP runtime: `brew install libomp`.

**Step 3 – Run the notebooks in this order.** Every notebook runs top to bottom without edits.

| Order | Notebook | Needs | Writes | Time (Apple M4 Pro) |
| --- | --- | --- | --- | --- |
| 1 | `Data_Transformation_Feature_Engineering.ipynb` | the two raw Kaggle files | `train.csv`, `val.csv`, `test.csv`, `split_assignments.csv`, `split_summary.csv` | ~4 min |
| 2 | `Modelling.ipynb` | the three split CSVs | comparison tables inside the notebook | ~2 min |
| 3 | `Fine_Tuning/fine_tuning/Step_1_Baseline.ipynb`, then `Step_2_Hyperparameter_Tuning.ipynb`, `Step_3_Threshold_Tuning.ipynb`, `Step_4_Final_Test_Evaluation.ipynb` | the three split CSVs, reached through `Fine_Tuning/fine_tuning/data/` (see the notes below) | `baseline_outputs/`, `step_2_outputs/`, `step_3_outputs/`, `step_4_outputs/` | ~5 min |
| 4 | `Evaluation.ipynb` | the three split CSVs and the Fine_Tuning output folders | `evaluation_outputs/` | ~2 min |

*To run a notebook interactively:* open it in VS Code, Antigravity or JupyterLab, pick the `.venv-tuning` interpreter as the kernel, and choose *Run All*. Notebooks 1, 2 and 4 must be opened from the repository root; the four Fine_Tuning notebooks from `Fine_Tuning/fine_tuning/` (this is the default when you open the file where it sits).

*To run everything from the terminal* (each command executes one notebook in place and saves its outputs; run from the repository root):

```bash
NB=.venv-tuning/bin/jupyter
$NB nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 Data_Transformation_Feature_Engineering.ipynb
$NB nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 Modelling.ipynb
for step in Step_1_Baseline Step_2_Hyperparameter_Tuning Step_3_Threshold_Tuning Step_4_Final_Test_Evaluation; do
    $NB nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 Fine_Tuning/fine_tuning/$step.ipynb
done
$NB nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 Evaluation.ipynb
```

On a Mac laptop, put `caffeinate -i` in front of a command so the machine does not go to sleep while a notebook is running.

**Step 4 – Make predictions (`result.csv`).** The last section of `Evaluation.ipynb` (section 10) defines `write_result`, which scores an engineered CSV with the final model and writes one row per transaction: `TransactionID`, `fraud_score` (model probability), `is_fraud_pred` (1 if the score is above the saved threshold) and, if the file has labels, the true `isFraud`. Running that cell writes `result.csv` for `test.csv` in about 5 seconds; it only needs the notebook's Setup cell to have run first. To score another file or use another threshold, call it again in the notebook:

```python
write_result(PATHS['val'], 'val_result.csv')          # another engineered CSV
write_result(PATHS['test'], 'result.csv', threshold=0.5)
```

The model, encoder, feature order and threshold are read from `Fine_Tuning/fine_tuning/step_4_outputs/final_model_bundle.joblib`, and the input is read in chunks, so large files work too. No separate script is needed; everything runs inside the notebook.

**Notes**
- **Only want `Evaluation.ipynb`?** It needs the three split CSVs (notebook 1) and the Fine_Tuning output folders. Those folders are in git, so after notebook 1 you can run `Evaluation.ipynb` straight away.
- **Fine_Tuning data folder.** The Fine_Tuning notebooks look for the CSVs in a `data/` folder next to them. That folder holds git-ignored symlinks to the root CSVs. Recreate them if they are missing:
  ```bash
  mkdir -p Fine_Tuning/fine_tuning/data
  ln -sf ../../../train.csv Fine_Tuning/fine_tuning/data/train.csv
  ln -sf ../../../val.csv   Fine_Tuning/fine_tuning/data/val.csv
  ln -sf ../../../test.csv  Fine_Tuning/fine_tuning/data/test.csv
  ```
- **Re-running the Fine_Tuning steps.** Steps 2–4 refuse to write into output folders that came from different inputs. Move `baseline_outputs/`, `step_2_outputs/`, `step_3_outputs/` and `step_4_outputs/` out of the way (for example into a `previous_run/` folder) before starting a fresh Step 1.
- **Different machine, slightly different numbers.** XGBoost results can differ in the third decimal between CPU types. The conclusions do not change, but the F1-best threshold chosen in Step 3 can move noticeably, because validation F1 is flat around its peak.

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
├── Evaluation.ipynb      # k-fold evaluation and validation (chronological + shuffled CV, test bootstrap intervals)
├── evaluation_outputs/             # tables, plots and run record written by Evaluation.ipynb
├── result.csv                      # predictions for test.csv written by Evaluation.ipynb (TransactionID, fraud_score, is_fraud_pred, isFraud)
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
