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
- Cleaning & Categorical Encoding: Replaced infinite values (+inf, -inf) with NaN across training and validation sets.
- Identified 37 categorical features (e.g., ProductCD, card4, card6, P_emaildomain, R_emaildomain, M1–M9, DeviceType, DeviceInfo, id_12–id_38) and encoded them via OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).
- Evaluation Metric: Primary optimization metric is Average Precision (AP, computed with average_precision_score) to evaluate minority class retrieval under heavy imbalance.
- Imbalance Scaling Factor: Calculated positive class weight (POS_WEIGHT) as 27.56 based on training class proportions for the weighted-loss experiments. The selected model instead uses random undersampling ratio 0.1 and scale_pos_weight=1.

## Fine Tuning

XGBoost hyperparameters were selected using three chronological training
folds, with mean Average Precision (AP) as the selection metric. Twenty
sampled configurations were compared with the baseline. Candidate 15
achieved mean AP **0.679552**, compared with **0.653863** for the untuned
XGBoost with random undersampling.

The selected model uses 400 trees, a learning rate of 0.05 and maximum
depth of 6. Random undersampling remained at a minority-to-majority ratio
of 0.1. A threshold of **0.459392** was selected by maximising validation
F1, using the decision rule `score > threshold`.

### Step 1 Baseline Rerun

Step 1 refits the untuned XGBoost with random undersampling selected in
`Modelling.ipynb`. It verifies the baseline and exports the fitted model,
categorical encoder and feature order for subsequent steps; it is not
a separate model-selection experiment.

`Modelling.ipynb` runs in `.venv` (`requirements.txt`, `pandas==3.0.6`, where `.astype(str)` keeps missing categories as `NaN`) and achieves validation AP **0.701045** (**1,267 true positives**, **418 false positives**). Step 1 runs in `.venv-tuning` (`Fine_Tuning/fine_tuning/requirements-tuning.txt`, `pandas==2.2.3`, where `.astype(str)` turns missing categories into `"nan"`) and achieves validation AP **0.705831** (**1,277 true positives**, **448 false positives**), so the mismatch flags in `baseline_comparison.csv` reflect that `pandas` version difference rather than a change in model settings.

### Test Results

| Configuration | AP | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Untuned XGBoost with RUS at 0.5 | 0.594245 | 66.61% | 51.77% | 0.582580 |
| Tuned XGBoost at 0.5 | 0.626285 | 71.92% | 52.73% | 0.608486 |
| Tuned XGBoost at selected threshold | 0.626285 | 68.78% | 54.46% | 0.607885 |

Hyperparameter tuning improved test AP and F1 at the same 0.5 threshold.
Lowering the threshold detected **54 additional fraud cases** but produced
**129 additional false alarms** compared with the tuned model at 0.5.
The threshold's validation F1 improvement did not generalise to test.

### Run Reference and Reproduction

The results above describe the current rerun used by `Evaluation.ipynb`
and the final prediction file. The original experiment, which selected
threshold **0.357439**, is archived under
`Fine_Tuning/fine_tuning/recorded_run/` and should be treated separately.

Saved outputs can be reviewed without retraining. For a fresh
reproduction, set `BT4012_TUNING_RUN` to a new name before running
Steps 1–4. Evaluation uses the same named run, preserving the committed
artifacts.

See [the tuning README](Fine_Tuning/fine_tuning/README.md) for complete
settings, saved artifacts, reproduction instructions and limitations.

## Evaluation and Validation (k-fold)
The sections above judge each model on a single split (train → validation → test). That gives one number per metric and no sense of how much it would change for a different month. `Evaluation.ipynb` repeats the comparison across several time periods and adds confidence intervals to the test results.

What the notebook does:
- Compares the **baseline XGBoost** (library defaults, as in `Modelling.ipynb`) with the **tuned XGBoost** (Fine_Tuning candidate 15). Both use random undersampling (fraud : non-fraud = 1 : 10).
- **Chronological 5-fold CV** on train + validation (Dec 2017 – Apr 2018): each fold trains on earlier weeks, skips 7 days, and scores the next block of about 18 days. This mirrors real use (train on the past, score the future), so it is the estimate we trust.
- **Shuffled 5-fold CV** (`StratifiedKFold`) on the same rows, for comparison only. It shows how much a classic k-fold overstates performance when future rows end up in training.
- **Bootstrap of the test split** (1,000 resamples) with the fixed final model: 95% confidence intervals for every metric and for the differences between models.
- Inside every fold, the frequency encodings, the label-history features (`prev_uid_*`, 7-day label delay) and the category codes are rebuilt from that fold's training rows only, so nothing leaks from the future. The test split is never used in cross-validation.

Results (Average Precision):

| Estimate | Baseline XGBoost | Tuned XGBoost |
| --- | ---: | ---: |
| Chronological 5-fold CV (mean ± sd) | 0.654 ± 0.031 | **0.675 ± 0.020** (better in 5 of 5 folds) |
| Shuffled 5-fold CV (mean ± sd) | 0.780 ± 0.002 | 0.765 ± 0.003 (ranking reversed) |
| Single validation split | 0.706 | 0.707 |
| Test split (95% bootstrap interval) | 0.594 [0.576, 0.612] | **0.626 [0.609, 0.642]** |

Key takeaways:
- **Tuning helps, and the gain holds over time.** The tuned model wins every chronological fold (**+0.021** AP on average), and on the test split the gain is clearly larger than sampling noise (AP **+0.032**, 95% CI +0.0255 to +0.0385).
- **Shuffled k-fold is misleading here.** It overstates AP by about **0.09** and even ranks the two models the wrong way round, because future rows and repeat cards leak into training. Only the chronological estimate should guide decisions.
- **Results vary by period.** Tuned-model AP ranges from **0.647 to 0.697** across the five scoring blocks, and the May test split (**0.626**) is harder than any of them. A fair summary: about 0.67 ± 0.02 on unseen future weeks, 0.63 on the test month.
- **The F1-best threshold is unstable.** It lands anywhere between **0.39 and 0.58** depending on the fold, and on the test split the chosen threshold changes F1 by only −0.0006 compared with 0.5. The operating threshold should come from review capacity or the cost of a missed fraud versus a false alarm, not from one month's F1 peak.

Notes:
- `Modelling.ipynb` uses `requirements.txt` (`pandas==3.0.6`, validation AP **0.7010**), while Fine_Tuning and `Evaluation.ipynb` use `Fine_Tuning/fine_tuning/requirements-tuning.txt` (`pandas==2.2.3`, validation AP **0.7058**). The lower CV (**0.654**) and test (**0.594**) baseline numbers reflect scoring across multiple time windows and the harder May test month rather than the single April validation month.
- The notebook reads the tuned hyperparameters and threshold from the Fine_Tuning output folders. Those folders now hold a full re-run made on 2026-10-23 (candidate 15 again; F1-best validation threshold **0.4594** instead of 0.3574). The original run is archived unchanged in `Fine_Tuning/fine_tuning/recorded_run/`, and `Fine_Tuning/fine_tuning/README.md` compares the two.
- All tables and plots are saved in `evaluation_outputs/`. The settings (`N_FOLDS`, `GAP_DAYS`, `N_BOOTSTRAP`, …) are constants at the top of the notebook, and the whole notebook runs in about 2 minutes.

## How to Run the Notebooks End to End

**Step 1 – Get the data.** Download `train_transaction.csv` and `train_identity.csv` from the [Kaggle competition page](https://www.kaggle.com/competitions/ieee-fraud-detection/data) and put them in the repository root. Raw datasets and engineered splits are git-ignored. Aggregate result CSVs and the committed prediction export are retained.

**Step 2 – Create the Python environments.** Two pinned dependency files are kept so both the original baseline (`0.7010` validation AP) and the Fine_Tuning / Evaluation runs (`0.7058` validation AP) reproduce cleanly (Python 3.12 or 3.13). From the repository root:

```bash
# Baseline & feature engineering environment (pandas 3.0.6, scikit-learn 1.9.1)
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt nbconvert

# Fine-tuning & evaluation environment (pandas 2.2.3, scikit-learn 1.6.1)
python3 -m venv .venv-tuning
.venv-tuning/bin/python -m pip install -r Fine_Tuning/fine_tuning/requirements-tuning.txt
```

On macOS, LightGBM and XGBoost also need the OpenMP runtime: `brew install libomp`.

**Step 3 – Choose saved-result review or a fresh reproduction.** Saved notebook outputs and artifacts can be reviewed without running training. For a new run, use a unique output name as explained below; do not execute Step 1 over an existing downstream run.

| Order | Notebook | Environment | Needs | Writes | Time (Apple M4 Pro) |
| --- | --- | --- | --- | --- | --- |
| 1 | `Data_Transformation_Feature_Engineering.ipynb` | `.venv` | the two raw Kaggle files | `train.csv`, `val.csv`, `test.csv`, `split_assignments.csv`, `split_summary.csv` | ~4 min |
| 2 | `Modelling.ipynb` | `.venv` | the three split CSVs | comparison tables inside the notebook (`0.7010` validation AP) | ~2 min |
| 3 | `Fine_Tuning/fine_tuning/Step_1_Baseline.ipynb`, then `Step_2_Hyperparameter_Tuning.ipynb`, `Step_3_Threshold_Tuning.ipynb`, `Step_4_Final_Test_Evaluation.ipynb` | `.venv-tuning` | the three split CSVs in the repository root or root `data/` (no symlinks required) | `baseline_outputs/`, `step_2_outputs/`, `step_3_outputs/`, `step_4_outputs/` | ~5 min |
| 4 | `Evaluation.ipynb` | `.venv-tuning` | the three split CSVs and the Fine_Tuning output folders | `evaluation_outputs/`, `result.csv` | ~2 min |

*To run a notebook interactively:* open it in VS Code, Antigravity or JupyterLab, select `.venv` (for notebooks 1–2) or `.venv-tuning` (for notebooks 3–4) as the kernel, and choose *Run All*. Run feature engineering and initial modelling from the repository root. Tuning and Evaluation resolve the repository automatically from the working directory or its parents. The three engineered CSVs can be in the repository root or root `data/`; root files take precedence. No local symlinks are required.

*To run everything from the terminal* (each command executes one notebook in place and saves its outputs; run from the repository root):

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 Data_Transformation_Feature_Engineering.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 Modelling.ipynb
# Set a NEW name for each complete reproduction; keep it set through Evaluation.
export BT4012_TUNING_RUN=reproduction_01
for step in Step_1_Baseline Step_2_Hyperparameter_Tuning Step_3_Threshold_Tuning Step_4_Final_Test_Evaluation; do
    .venv-tuning/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 Fine_Tuning/fine_tuning/$step.ipynb
done
.venv-tuning/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 Evaluation.ipynb
```

The named run writes under `Fine_Tuning/fine_tuning/reproductions/reproduction_01/`, including its own evaluation outputs and prediction CSV. Choose a different name for a new complete rerun. A Step 2 search can resume its existing checkpoints only when the upstream bundle and manifest are unchanged; do not rerun Step 1 first when resuming. To return to the committed reference artifacts, unset `BT4012_TUNING_RUN` and restart the notebook kernels. In PowerShell, set it with `$env:BT4012_TUNING_RUN="reproduction_01"`. In Colab, set `os.environ["BT4012_REPO_DIR"]` to the extracted repository root and `os.environ["BT4012_TUNING_RUN"]` before the setup cells.

On a Mac laptop, put `caffeinate -i` in front of a command so the machine does not go to sleep while a notebook is running.

**Step 4 – Make predictions (`result.csv`).** The last section of `Evaluation.ipynb` (section 10) defines `write_result`, which scores an engineered CSV with the final model and writes one row per transaction: `TransactionID`, `fraud_score` (model probability), `is_fraud_pred` (1 if the score is above the saved threshold) and, if the file has labels, the true `isFraud`. Running that cell writes `result.csv` for `test.csv` in about 5 seconds; it only needs the notebook's Setup cell to have run first. To score another file or use another threshold, call it again in the notebook:

```python
write_result(PATHS['val'], 'val_result.csv')          # another engineered CSV
write_result(PATHS['test'], 'result.csv', threshold=0.5)
```

By default, the model, encoder, feature order and threshold are read from `Fine_Tuning/fine_tuning/step_4_outputs/final_model_bundle.joblib`. When `BT4012_TUNING_RUN` is set, Evaluation reads that named reproduction instead and writes predictions inside its evaluation output directory. The input must contain the engineered features expected by the bundle.


## Folder Directory

```plaintext
bt4012-grp3/
├── data/                      # Optional local data directory; root CSVs also supported
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
├── .venv                           # Baseline & feature-engineering virtual environment
├── .venv-tuning                    # Fine-tuning & evaluation virtual environment
├── requirements.txt                # Baseline Python dependencies (pandas 3.0.6, scikit-learn 1.9.1)
├── Fine_Tuning/fine_tuning
│   ├── Step_1_Baseline.ipynb
│   ├── Step_2_Hyperparameter_Tuning.ipynb
│   ├── Step_3_Threshold_Tuning.ipynb
│   ├── Step_4_Final_Test_Evaluation.ipynb
│   ├── baseline_outputs/                    # Intermediate artifacts
│   ├── step_2_outputs/                      # Hyperparameter search results
│   ├── step_3_outputs/                      # Threshold selection results
│   ├── step_4_outputs/                      # Final model, metrics and plots
│   ├── requirements-tuning.txt              # Fine-tuning & evaluation dependencies (pandas 2.2.3, scikit-learn 1.6.1)
│   └── README.md                            # Detailed tuning documentation
└── README.md                       # Project documentation and summary
```
