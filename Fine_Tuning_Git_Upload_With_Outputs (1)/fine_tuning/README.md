# Fraud model tuning

XGBoost hyperparameter tuning, validation threshold selection and fixed-model test evaluation for the IEEE-CIS fraud-detection project.

## Repository placement

Add this `fine_tuning` directory to the existing project repository. Its notebook and output-folder layout is preserved so dependencies remain in their expected locations. The package does not replace the original `Modelling.ipynb`, feature-engineering notebook, project README or root dependency file.

## Environment and data

The completed experiment used Python 3.13.16. Core package versions are recorded in `requirements-tuning.txt` and the run metadata. Install the recorded versions in a separate environment; do not silently replace them with different versions when reproducing saved model results. Plotting and notebook-tool versions were not recorded.

From this directory:

```bash
python -m pip install -r requirements-tuning.txt
jupyter lab
```

Place the original engineered `train.csv`, `val.csv` and `test.csv` files in `data/`, or directly beside the notebooks. The training alias `train (2).csv` and test alias `test (3).csv` are also accepted. Retain only one copy of each split. These datasets are excluded from the package and Git. The notebooks verify the original file hashes, so regenerating or re-exporting the CSVs may require a separate reproduction run.

Each notebook defines `PROJECT_DIR = Path.cwd()`. Launch Jupyter from this directory, or set `PROJECT_DIR` to its absolute path. In Colab, use the commented Drive-mount configuration and set `PROJECT_DIR` to the directory containing these notebooks and output folders. `DATA_DIR` can be set independently when the CSVs are elsewhere.

## Notebook sequence

| Notebook | Purpose |
| --- | --- |
| `Step_1_Baseline.ipynb` | Dataset checks, baseline fitting and saved-model verification |
| `Step_2_Hyperparameter_Tuning.ipynb` | Three-fold chronological random search, parameter selection and validation comparison |
| `Step_3_Threshold_Tuning.ipynb` | Validation threshold sweep and export of the fixed model-threshold bundle |
| `Step_4_Final_Test_Evaluation.ipynb` | Fixed-model test inference, comparison and final artifact export |

The completed results and fitted artifacts are included. Inline tables, printed results and plots are retained from the recorded notebook runs. Step 2 was saved during an interrupted rerun; its completed results are displayed by a separate reporting cell that reads the exported CSV and JSON files. Original execution counts are retained where available; missing counts are not invented. Model fitting was not rerun during packaging. Numerical results and serialized model artifacts have not been altered during packaging.

For a full reproduction, copy the notebooks to a separate experiment directory, provide the same CSVs and run Steps 1 through 4. Keep the committed output folders as the original experiment record. Re-fitting a model can change its serialized hash even when metrics agree, so new reproduction outputs should not be mixed with old downstream manifests. Step 1's imports and helper definitions precede its path configuration.

## Selected configuration

- Model: XGBoost candidate 15.
- Training imbalance treatment: random undersampling with minority/majority ratio 0.1; positive-class weight 1.
- Hyperparameters: `n_estimators=400`, `learning_rate=0.05`, `max_depth=6`, `min_child_weight=3`, `subsample=1.0`, `colsample_bytree=0.85`, `reg_alpha=0.0`, `reg_lambda=3.0`.
- Selection metric: mean average precision across three chronological training-only folds.
- Threshold objective: maximum validation F1.
- Final decision rule: **score > 0.3574390113353729**.

## Test results

| Configuration | Average precision | Precision | Recall | F1 | True positives | False positives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline at 0.5 | 0.594245 | 0.666116 | 0.517662 | 0.582580 | 1612 | 808 |
| Tuned model at 0.5 | 0.626093 | 0.718969 | 0.528259 | 0.609034 | 1645 | 643 |
| Final model at 0.357439 | 0.626093 | 0.597441 | 0.584778 | 0.591042 | 1821 | 1227 |

Hyperparameter tuning improved test ranking and F1 at the same 0.5 threshold. Lowering the threshold detected 176 additional fraud cases but produced 584 additional false alarms relative to the tuned model at 0.5. The validation-selected threshold's F1 benefit did not carry over to test. The preselected threshold is retained; test outcomes were not used to select a replacement.

## Artifacts and dependencies

The primary handoff is `step_4_outputs/final_model_bundle.joblib`, with `final_config.json` and `final_test_results.json`. The bundle contains the fitted model, categorical encoder, feature order and threshold. It expects already-engineered features from the original project pipeline. `final_xgboost.json` contains only the model and is insufficient by itself to reproduce the complete prediction procedure.

Use `predict_proba` followed by the full-precision saved threshold and strict `>` comparison. `model.predict()` applies the default decision rule. The final notebook contains the complete prediction helper. Load serialized model files only from a trusted project source.

Intermediate bundles, manifests and the approximately 2 MB `step_2_outputs/validation_scores.csv` are retained because later notebooks depend on them and verify their hashes. Dataset CSVs, redundant row-level prediction exports and the full threshold-candidate table are omitted; the latter outputs are regenerated by the notebooks. Existing plots and aggregate result tables are included.

## Evaluation limitations

Frequency encodings and historical fraud features were rebuilt within each tuning fold. The validation set previously influenced model-family selection and subsequently threshold selection; its scores are development estimates. The original upstream split boundaries were selected using fraud rates across periods, including test. Final evaluation retains that limitation. The test results do not establish statistical significance, calibrated fraud probabilities or deployment readiness.

Sources: the four notebooks, `step_2_outputs/cv_summary.csv`, `step_3_outputs/threshold_selection.json`, and `step_4_outputs/test_comparison.csv`.
