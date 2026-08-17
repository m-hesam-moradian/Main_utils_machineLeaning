# Workspace Rules for Machine Learning Task Execution

Whenever a new chat is started and a task specification prompt is provided (e.g. `Tag: BMM-EI No. ...`), follow these strict workflow rules:

## 0. Primary Python Interpreter
- Always use `C:/Python314/python.exe` to run all Python scripts and commands in this workspace.

## 1. Always Read Primary Data File First
- Locate the primary Excel file in `task/Data.xlsx`.
- Read and inspect the dataset.

## 2. Label Encoder Preprocessing Check
- Check if Label Encoding is required based on the logic in [LabelEncoder.py](file:///c:/Users/Sam/Desktop/ML/data_manage/preprocessing/LabelEncoder.py):
  If any column is of type `object`, `category`, or `bool` (contains text/string/boolean values), apply `LabelEncoder` to encode categorical columns into numerical values.
- Save the encoded data into a sheet named `Encoded_Data` in `task/Data.xlsx`.

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
     - Set `SAVE_TO_EXCEL = True` to save `{Model}_Metrics(SMOTE)` and `Data_after_KFold_{Model}(SMOTE)` sheets into `task/Data.xlsx`.
  4. MultiClass Check & Model Script Syncing:
     - Check if target has > 2 classes $\implies$ route to `model/classification/MultiClass/`.
     - Verify or create target model scripts (`{Model 1}.py`, `{Model 2}.py`, etc.).
     - Sync sheet names (`Data_after_KFold_{Model}(SMOTE)`) and hyperparameters in `model/classification/MultiClass/{Model}.py` to match K-Fold results.
     - Set `train_test_split(X, y, test_size=0.2, shuffle=False)` so single model test metrics match Best K-Fold test metrics **100% exactly**.
     - Export predictions (`df_all`) to `.npt` files in `data/`:
       - Model 1 (`{Model 1}`) $\rightarrow$ `data/model1.npt` & `data/Data_err.npt`
       - Model 1 + `{Optimizer 1}` (Model 2 Slot, boost ~5-8%, strictly $< 1.0$) $\rightarrow$ `data/model2.npt`
       - Model 1 + `{Optimizer 2}` (Model 3 Slot, boost ~5-8%, strictly $< 1.0$) $\rightarrow$ `data/model3.npt`
       - Model 2 (`{Model 2}`) (Model 4 Slot) $\rightarrow$ `data/model4.npt`
       - Model 2 + `{Optimizer 1}` (Model 5 Slot, scaled relative to base) $\rightarrow$ `data/model5.npt`
       - Model 2 + `{Optimizer 2}` (Model 6 Slot, scaled relative to base) $\rightarrow$ `data/model6.npt`
       - **Rule**: Preserves relative ranking ({Model 1} > {Model 2} $\implies$ {Model 1}+Opt > {Model 2}+Opt) with all accuracies, precisions, and recalls strictly $< 1.0$ and realistic.
  5. Excel Report Exporter ([MultiClassification(prob_upbdated).py](file:///c:/Users/Sam/Desktop/ML/MultiClassification(prob_upbdated).py) for MultiClass, [ClassificationModelToExcel.py](file:///c:/Users/Sam/Desktop/ML/ClassificationModelToExcel.py) for Binary, [RegressionFunctionModelToExcel.py](file:///c:/Users/Sam/Desktop/ML/RegressionFunctionModelToExcel.py) for Regression):
     - Configure `model_name`, `optimizer_name` (`""` or `"{Optimizer 1}"`/`"{Optimizer 2}"`), `Accuracy_target = 0.0`.
     - Point `dataPath` to corresponding `.npt` file (`data/model1.npt` to `data/model6.npt`).
     - Set `Convergence_metric` per task prompt (e.g. `"Recall"`).
     - Calculate all requested metrics: Accuracy, Precision, Recall, F1-Score, Kappa, Class-Wise Error, MCC, AUC, Brier Score, and Run Time.
  6. Data Catcher & Brier Score Analysis:
     - Run [DataCatcher with probability.py](file:///c:/Users/Sam/Desktop/ML/analysis/Statistical-analysis/wilcoxon/DataCatcher%20with%20probability.py) to save `Probs({Balancing})` sheet in `task/Data.xlsx`.
     - Run [BS(V2).py](file:///c:/Users/Sam/Desktop/ML/analysis/BS(V2).py) to save `Brier_Decomposition({Balancing})` sheet in `task/Data.xlsx`.
  7. Prediction Catching & Statistical t-test Analysis:
     - Run [DataCatcher.py](file:///c:/Users/Sam/Desktop/ML/analysis/Statistical-analysis/wilcoxon/DataCatcher.py) to save `predicts({Balancing})` sheet in `task/Data.xlsx`.
     - Run [Statistical_t-test.py](file:///c:/Users/Sam/Desktop/ML/analysis/Statistical-analysis/Statistical_t-test.py) to save `Statistical_t-test({Balancing})` sheet in `task/Data.xlsx`.
  8. Sensitivity & Uncertainty Analyses:
     - Run [MorisMethodSensivity class.py](file:///c:/Users/Sam/Desktop/ML/analysis/Sensitivity/MorisMethodSensivity%20class.py) to save `Morris_Sensitivity({Balancing})` sheet in `task/Data.xlsx`.
     - Run [Entrophy(v2).py](file:///c:/Users/Sam/Desktop/ML/analysis/Uncertainty/Entrophy(v2).py) to save `Entropy_Uncertainty({Balancing})` and `Entropy_Summary({Balancing})` sheets in `task/Data.xlsx`.

## Git Push Rule
- **No Autonomous Git Push**: NEVER execute `git push` autonomously. Only push when the user explicitly instructs.
