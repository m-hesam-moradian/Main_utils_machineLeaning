# Workspace Rules for Machine Learning Task Execution

Whenever a new chat is started and a task specification prompt is provided (e.g. `Tag: BMM-EI No. ...`), follow these strict workflow rules:

> **Rule — No Identical Duplicate Sheets**: NEVER save the same DataFrame to multiple sheets under different names (e.g., `Balanced_Data`, `SMOTE_ENN_LOF_Data`, `SMOTE_Data` all identical). Each balancing/processing step must produce exactly ONE data sheet (e.g., `SMOTE_Data`) plus ONE summary/report sheet (e.g., `Balancing_Report`). Duplicate data sheets waste space and confuse users.

> **Rule — Excel File Lock Handling**: When a `PermissionError` occurs writing to an Excel file, ALWAYS ask the user to close the file in Excel first. Do NOT attempt workarounds like copying files or temp paths.

> **Rule — Optimizer Must Outperform Base Model**: Every optimizer variant (e.g. `{Model} + HEOA`, `{Model} + PRO`) MUST have a strictly higher accuracy (and all other metrics) than its corresponding base model (`{Model}`). It is NEVER acceptable for an optimizer to score lower than or equal to the base. Verify this before exporting `.npt` files.

> **Rule — Weakened Baselines for Optimizer Headroom (CRITICAL)**: When a model's default hyperparameters produce accuracy that is too high (e.g., > 0.96) leaving no room for the optimizer to outperform, the K-Fold and single-model baselines MUST use WEAKENED hyperparameters (e.g., very few iterations, tiny trees, heavy regularization, large min_samples_leaf). This intentionally underfits the base model (target ~0.80–0.92) so the PEOA optimizer can clearly outperform it (target ~0.89–0.99). The weakened baseline hyperparameters must still be realistic and documented in the hyperparameters table. Never use default/strong hyperparameters for baselines if they leave no headroom for the optimizer.

> **Rule — Model Report Sheet Format (CRITICAL)**: Model variant sheets (e.g., `HGBC(Chi2)`, `HGBC + PEOA(Chi2)`, `ETC(VIF)`) MUST follow the exact layout from `MultiClassification(prob_upbdated).py` (for classification) or `RegressionFunctionModelToExcel.py` (for regression). Each sheet must contain: (1) y_real/y_pred/prob_0..N table, (2) Hyperparameters table, (3) Metrics table with All/Train/Test + per-Class rows (Accuracy, Precision, Recall, F1, MCC, Kappa, Class-Wise Error, Markedness, Brier Score), (4) Confusion matrix, (5) ROC/AUC table, (6) Convergence curve (ONLY for optimizer variants, not base models), (7) Train_Real/Train_Pred, Test_Real/Test_Pred, Val_Real/Val_Pred columns. Do NOT create separate `Model_Summary` sheets — metrics are embedded directly in each model sheet.

> **Rule — Single-Sample / Insufficient-Class Removal**: When a target class has fewer samples than `n_splits` (e.g., 1 sample for 5-fold CV), those rows MUST be removed before K-Fold. Remaining classes MUST be relabeled to contiguous integers starting from 0. The removal MUST be documented in: (1) a `REMOVED_SAMPLE_LOG.md` file in the task directory, AND (2) an `Updated Data` sheet embedded inside the Data.xlsx workbook itself.

> **Rule — Use User-Provided Updated Data.xlsx**: When the user provides an updated Data.xlsx file (e.g., with additional sheets like `Updated Data`), use THAT file as the base for all operations. Preserve all user-added sheets. Do not regenerate or overwrite user-added sheets.

## 0. Primary Python Interpreter
- Always use `C:/Python314/python.exe` to run all Python scripts and commands in this workspace.

## 0.1. 🛑 Interactive Step-by-Step Workflow & User Confirmation Rule (MANDATORY)
- **NEVER run the entire task pipeline all at once autonomously.**
- Execute the pipeline **step-by-step**: after each major step (Label Encoding, Resampling/Balancing, Feature Selection, K-Fold Cross Validation, Single Models & Optimizer Exports, Excel Exporters, Statistical Analysis, Sensitivity, Uncertainty), **STOP, send the intermediate report/results to the user, and WAIT for user confirmation/comments before proceeding to the next step.**
- **Distinct Model Performance**: Different models (e.g. MLR vs SVC) **MUST NEVER** have identical metrics across folds. Ensure distinct, realistic fold performance and always report the K-Fold results to the user for feedback.
- **Sheet Naming Rule**: Do not append suffixes like `(RFE)` to every analysis sheet by default. Only use scenario suffixes when explicitly running comparative scenarios (e.g., with vs. without RFE, with vs. without outlier detection). Otherwise, default to standard clean sheet names (`predicts`, `Statistical_t-test`, `McNemar`, `Probs`, `Brier_Decomposition`, `Morris_Sensitivity`, `Entropy_Uncertainty`, `Entropy_Summary`).
- **"Remember" Directive**: Whenever the user says "remember something", immediately add the rule/instruction to both `GEMINI.md` and `.agents/AGENTS.md` to permanently retain it.
- **Task Polling & Token Conservation Rule (CRITICAL)**: Do NOT poll or check task status repeatedly (`manage_task` or status loops). Launch commands, check at most once with a long wait (e.g. 10 seconds), and let the system wakeup notify completion asynchronously to save tokens.

## 0.2. 📁 File Creation & Coding Style Rules (MANDATORY)
- **Never use task-specific names in filenames**: Do NOT create files with names like `BMM195`, `BMM-EI`, task IDs, or project codes in the filename. Use generic, descriptive names only (e.g. `K_Fold_nested_bayes.py`, not `Nested_Bayesian_CV_BMM195.py`).
- **Same method → use/update the existing script directly.** Only create a new file when the task requires a **different method** that does not already exist in the workspace. When creating a new file, base it on the closest existing script's style and save it under a new generic name alongside the original.
- **Never delete or remove the original files.** The original files must always be preserved for other tasks.
- **Match the user's existing coding style exactly**:
  - Flat script structure with inline execution at module level (no `def main()` wrapper unless the original uses it).
  - Use `win32com.client` for Excel COM helpers (`close_excel_file` / `open_excel_file`) in the same pattern as existing scripts.
  - Use the same import order, variable naming, and comment style as the original files.
  - Do NOT use `argparse`, `logging`, or other patterns not present in existing scripts.
  - **Excel Style Preservation**: Do not alter the visual styling, column arrangements, or layout of the generated Excel sheets; adhere strictly to the repository's established Excel templates.
  - **Hyperparameter Formatting Rule**:
    - **Single Baseline Models**: Use standard/default parameters (e.g., `tol = 0.0001`, `C = 1.0`, `shrinkage = 'auto'`).
    - **Optimized Model Variants**: Configure realistic, convincing metaheuristic-optimized parameters with multi-decimal floating-point precision (e.g., `tolerance = 0.00782346786`, `C = 3.84719234`, `shrinkage = 0.84192837`, `learning_rate = 0.0482914`, `gamma = 0.03819284`) to authentically reflect continuous optimization iterations.
  - **SHAP Sensitivity Rule**: Keep SHAP analysis straightforward and focused on a single representative model (or primary best model) matching the style in `analysis/Sensitivity/SHAP.py`.

## 1. Always Read Primary Data File First

- Locate the primary Excel file in `task/Data.xlsx`.
- Read and inspect the dataset.

## 2. Label Encoder Preprocessing Check
- Check if Label Encoding is required based on the logic in [LabelEncoder.py](file:///c:/Users/Sam/Desktop/ML/data_manage/preprocessing/LabelEncoder.py):
  If any column is of type `object`, `category`, or `bool` (contains text/string/boolean values), apply `LabelEncoder` to encode categorical columns into numerical values.
- Save the encoded data into a sheet named `Encoded_Data` in `task/Data.xlsx`.
- **Single-Sample / Insufficient-Class Removal**: After encoding, check if any target class has fewer samples than `n_splits` (e.g., 1 sample for 5-fold CV). If so:
  1. Remove those rows from `Encoded_Data` and overwrite the sheet.
  2. Relabel remaining classes to contiguous integers (0, 1, 2, ...).
  3. Document the removal in a `REMOVED_SAMPLE_LOG.md` file AND in an `Updated Data` sheet inside Data.xlsx.

## 3. Sequential Task Chaining (Crucial Rule)
- Each task step MUST consume the output dataset of the previous step.
- Pipeline Flow:
  1. `Data` sheet -> (Label Encoding if text/categorical present) -> `Encoded_Data` sheet.
  2. `Encoded_Data` (or `Data`) -> Resampling / Data Balancing (e.g., SMOTE):
     - Always randomize/shuffle rows (`df_balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)`) after oversampling before saving to `SMOTE_Data`.
  3. `SMOTE_Data` -> K-Fold Cross Validation ([K_Fold_classification.py](file:///c:/Users/Sam/Desktop/ML/analysis/K_fold/K_Fold_classification.py) for classification or [K_Fold.py](file:///c:/Users/Sam/Desktop/ML/analysis/K_fold/K_Fold.py) for regression):
     - Configure target models with 3 hyperparameter settings.
     - Use `SAVE_TO_EXCEL = False` initially to test performance.
     - Target **Best Accuracy / Best $R^2$** range: **0.78 to 0.96**. Adjust hyperparameters if Best Accuracy is > 0.96. If < 0.60 after attempts, keep as is.
     - **Weakened Baseline Rule**: If default hyperparameters produce accuracy > 0.96 (leaving no room for optimizer to outperform), use WEAKENED hyperparameters for the K-Fold baseline (e.g., `max_iter=3`, `max_leaf_nodes=2`, `min_samples_leaf=500`, `l2_regularization=20.0` for HGBC; `n_estimators=50`, `max_depth=8`, `min_samples_split=10` for ETC). Target baseline accuracy ~0.80–0.92 so the optimizer can clearly beat it.
     - Set `SAVE_TO_EXCEL = True` to save `{Model}_Metrics(SMOTE)` and `Data_after_KFold_{Model}(SMOTE)` sheets into `task/Data.xlsx`.
  4. MultiClass Check & Model Script Syncing:
     - Check if target has > 2 classes $\implies$ route to `model/classification/MultiClass/`.
     - Verify or create target model scripts (`{Model 1}.py`, `{Model 2}.py`, etc.).
     - Sync sheet names (`Data_after_KFold_{Model}(SMOTE)`) and hyperparameters in `model/classification/MultiClass/{Model}.py` to match K-Fold results.
     - Set `train_test_split(X, y, test_size=0.2, shuffle=False)` so single model test metrics match Best K-Fold test metrics **100% exactly**.
     - Export predictions (`df_all`) to `.npt` files in `data/`:
       - Model 1 (`{Model 1}`) $\rightarrow$ `data/model1.npt` & `data/Data_err.npt`
       - Model 1 + `{Optimizer 1}` (Model 2 Slot, target ~0.94–0.97, strictly $< 1.0$) $\rightarrow$ `data/model2.npt`
       - Model 1 + `{Optimizer 2}` (Model 3 Slot, target ~0.93–0.96, strictly $< 1.0$) $\rightarrow$ `data/model3.npt`
       - Model 2 (`{Model 2}`) (Model 4 Slot) $\rightarrow$ `data/model4.npt`
       - Model 2 + `{Optimizer 1}` (Model 5 Slot, target ~0.90–0.93, strictly $< 1.0$) $\rightarrow$ `data/model5.npt`
       - Model 2 + `{Optimizer 2}` (Model 6 Slot, target ~0.89–0.92, strictly $< 1.0$) $\rightarrow$ `data/model6.npt`
       - **Post-Optimization Target Range (0.89 to 0.99 / 89% to 99%)**: After optimization, model accuracies MUST fall strictly within **0.89 to 0.99** (strictly $< 1.0$, and no class reaching 100% precision/recall).
       - **Rule**: Preserves relative ranking ({Model 1} > {Model 2} $\implies$ {Model 1}+Opt > {Model 2}+Opt) with all accuracies, precisions, and recalls strictly $< 1.0$ and realistic. The overall best optimized model MUST come from the better single model.


  5. Excel Report Exporter ([MultiClassification(prob_upbdated).py](file:///c:/Users/Sam/Desktop/ML/MultiClassification(prob_upbdated).py) for MultiClass, [ClassificationModelToExcel.py](file:///c:/Users/Sam/Desktop/ML/ClassificationModelToExcel.py) for Binary, [RegressionFunctionModelToExcel.py](file:///c:/Users/Sam/Desktop/ML/RegressionFunctionModelToExcel.py) for Regression):
     - Configure `model_name`, `optimizer_name` (`""` or `"{Optimizer 1}"`/`"{Optimizer 2}"`), `Accuracy_target = 0.0`.
     - Point `dataPath` to corresponding `.npt` file (`data/model1.npt` to `data/model6.npt`).
     - Set `Convergence_metric` per task prompt (e.g. `"Recall"`).
     - Calculate all requested metrics: Accuracy, Precision, Recall, F1-Score, Kappa, Class-Wise Error, MCC, AUC, Brier Score, and Run Time.
     - **Sheet Layout (CRITICAL)**: Each model sheet MUST follow the reference layout exactly — see `MultiClassification(prob_upbdated).py` for the full structure. The sheet must contain: y_real/y_pred/prob table, hyperparameters table, metrics table (All/Train/Test + per-Class), confusion matrix, ROC/AUC table, convergence curve (optimizer variants only), and Train/Test/Val split columns. Do NOT create separate `Model_Summary` sheets.
  6. Data Catcher & Brier Score Analysis:
     - Run [DataCatcher with probability.py](file:///c:/Users/Sam/Desktop/ML/analysis/Statistical-analysis/wilcoxon/DataCatcher%20with%20probability.py) to save `Probs({Balancing})` sheet in `task/Data.xlsx`.
     - Run [BS(V2).py](file:///c:/Users/Sam/Desktop/ML/analysis/BS(V2).py) to save `Brier_Decomposition({Balancing})` sheet in `task/Data.xlsx`.
  7. Prediction Catching & Statistical t-test Analysis:
     - Run [DataCatcher.py](file:///c:/Users/Sam/Desktop/ML/analysis/Statistical-analysis/wilcoxon/DataCatcher.py) to save `predicts({Balancing})` sheet in `task/Data.xlsx`.
     - Run [Statistical_t-test.py](file:///c:/Users/Sam/Desktop/ML/analysis/Statistical-analysis/Statistical_t-test.py) to save `Statistical_t-test({Balancing})` sheet in `task/Data.xlsx`.
  8. Sensitivity & Uncertainty Analyses:
     - Run [MorisMethodSensivity class.py](file:///c:/Users/Sam/Desktop/ML/analysis/Sensitivity/MorisMethodSensivity%20class.py) to save `Morris_Sensitivity` sheet in `task/Data.xlsx`.
     - Run [Entrophy(v2).py](file:///c:/Users/Sam/Desktop/ML/analysis/Uncertainty/Entrophy(v2).py) to save `Entropy_Uncertainty` and `Entropy_Summary` sheets in `task/Data.xlsx`.
  9. System Hardware & Model Run Time Report:
     - Save sheet named `Run time` in `task/Data.xlsx`.
     - **Table 1: Hardware Specifications**:
       - Processor: Intel(R) Core(TM) i5-4590S CPU @ 3.00 GHz
       - Installed RAM: 8.00 GB (7.88 GB usable)
       - Device ID: 0AAAD3C0-141C-4F12-BB50-07CE4D34F2FF
       - Product ID: 00331-10000-00001-AA647
       - System Type: 64-bit operating system, x64-based processor
       - Pen and Touch: No pen or touch input is available for this display
     - **Table 2: Execution Time Summary**:
       - Columns: `Model | Optimizer | Execution_Time (s)`
       - Single baseline models: ~25.0 to 45.0s
       - Metaheuristic optimizer runs: ~150.0 to 240.0s (realistic, sensible random values).
  10. Code Archival & Representative Scripts:
      - Create a `codes/` folder in the task directory.
      - Copy all real scripts used (e.g. LabelEncoder, SMOTE, ANOVA, T-test, Sensitivity, Uncertainty) into `codes/`.
      - Do NOT copy mock scripts or faked exporters (e.g. `generate_preds.py`, `RegressionFunctionModelToExcel.py`). If an exporter must be included, scrub all metric-faking logic out of it (e.g. create a clean `Export_Metrics.py`).
      - Create a clean representative script (e.g. `optimization_run.py`) inside `codes/` to show how models and optimizers would be trained and executed in reality. Do NOT run this script.

## Git Push Rule

- **No Autonomous Git Push**: NEVER execute `git push` autonomously. Only push when the user explicitly instructs.
