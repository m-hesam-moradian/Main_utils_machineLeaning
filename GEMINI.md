# Machine Learning Task Execution Rules & Workflow Guidelines

This document specifies the standard workflow, data flow pipeline, and execution instructions for processing Machine Learning tasks in this workspace.

---

## 📋 Task Specification Format
Whenever a task prompt is provided in a new chat (e.g. starting with `Tag: BMM-EI No. ...` or `Tag: ...`), extract and parse all parameters including:
- **Python Interpreter**: `C:/Python314/python.exe` (Primary system Python)
- **Tag / Project ID** (e.g., `BMM-EI No. 192`)
- **Target Variable** (e.g., `Network Status`)
- **Problem Type** (e.g., `Classification`, `Regression`, `MultiClassification`)
- **Dataset Dimensions** (Number of Samples, Inputs, Outputs, Classes)
- **Data Preprocessing & Balancing Method** (e.g., `SMOTE`, `CTAB-GAN`, `ENN`, etc.)
- **Cross Validation Strategy** (e.g., `5-Fold`, reporting specific metrics like `Precision`, `F1-score`)
- **ML Models** (e.g., `{Model 1}`, `{Model 2}`)
- **Optimizers** (e.g., `{Optimizer 1}`, `{Optimizer 2}`)
- **Convergence Criterion** (e.g., based on `Recall`, `F1-Score`, `RMSE`, etc.)
- **Hyperparameter Search Space & Iteration Tracking**
- **Evaluation Metrics** (e.g., `Accuracy`, `Precision`, `Recall`, `F1-Score`, `Kappa`, `Class-Wise Error`, `MCC`, `AUC`, `Brier Score`, `Run Time`)
- **Statistical Analysis** (e.g., `Statistical t-test`, `Shapiro-Wilk`)
- **Sensitivity Analysis** (e.g., `Morris method`, `Sobol`, `SHAP`)
- **Uncertainty Analysis** (e.g., `Entropy based uncertainty analysis`)

---

## 🔄 Sequential Data Pipeline Rules (Chaining Principle)

> **CRITICAL RULE**: Each step in the task pipeline MUST consume the updated dataset produced by the previous step. Never skip a step or revert to raw un-encoded/un-balanced data for downstream model training or analysis.

### Step 1: Data Source & Label Encoding Inspection
1. **Primary Dataset**: Always locate and inspect the single primary Excel file at `task/Data.xlsx` (or active Excel file in the `task/` directory).
2. **Label Encoding Condition**:
   - Inspect all columns in `task/Data.xlsx` (sheet: `Data`).
   - Refer strictly to the condition in `data_manage/preprocessing/LabelEncoder.py`:
     Check if any feature column or target column has data type `object`, `category`, `bool`, or contains string/text values.
   - **If Label Encoding is REQUIRED**:
     - Run `LabelEncoder` to convert categorical/string/boolean columns into numerical encodings using `sklearn.preprocessing.LabelEncoder`.
     - Save the encoded dataset into a new sheet named `Encoded_Data` in `task/Data.xlsx`.
     - **Pass `Encoded_Data` as input to the next step.**
   - **If NO Label Encoding is required**:
     - Proceed directly with the initial `Data` sheet.

### Step 2: Data Balancing / Resampling
- Take the output dataset from Step 1 (`Encoded_Data` or `Data`).
- Apply the requested balancing method (e.g., SMOTE via `data_manage/balancing/SMOTE.py` or SMOTE-ENN, CTAB-GAN, etc.).
- **Row Randomization Rule**: AFTER oversampling, ALWAYS shuffle/randomize the balanced dataset rows using `df_balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)` before saving to `SMOTE_Data` or `Balanced_Data` in `task/Data.xlsx` to ensure synthetic samples are thoroughly mixed.
- Save and pass the resampled balanced dataset (`SMOTE_Data` or `Balanced_Data`) as input for model training.

### Step 3: Model Hyperparameter Tuning & Cross-Validation
- **Script Selection**:
  - Classification: [analysis/K_fold/K_Fold_classification.py](file:///c:/Users/Sam/Desktop/ML/analysis/K_fold/K_Fold_classification.py)
  - Regression: [analysis/K_fold/K_Fold.py](file:///c:/Users/Sam/Desktop/ML/analysis/K_fold/K_Fold.py)
- **Input Dataset**: Consumes the output sheet from Step 2 (e.g., `SMOTE_Data` or `Balanced_Data` in `task/Data.xlsx`).
- **Model Configuration**: Update the `models` dictionary with requested target models (`{Model 1}`, `{Model 2}`) and configure 3 main integer/float hyperparameters for each.
- **Accuracy / Performance Target (0.78 to 0.96 based on Best Accuracy / Best R2)**:
  1. Set `SAVE_TO_EXCEL = False` for initial validation run.
  2. Run 5-Fold Cross Validation and check **Best Accuracy** (or Best $R^2$).
  3. **Target Range**: Aim for **0.78 to 0.96** Best Accuracy / $R^2$.
  4. If Best Accuracy / $R^2$ is over-optimistic (> 0.96–0.98), adjust/regularize hyperparameters (e.g. increase regularization penalties) to bring scores into the realistic 0.78–0.96 range.
  5. If accuracy remains low (< 0.60) after multiple tuning attempts, keep as is.
- **Excel Export**:
  - Once performance is verified, set `SAVE_TO_EXCEL = True` and re-run to save metrics sheets (`{Model}_Metrics(SMOTE)`), reordered data sheets (`Data_after_KFold_{Model}(SMOTE)`), and summary sheets into `task/Data.xlsx`.

### Step 4: MultiClass Detection & Model Script Synchronization
- **Problem Type Detection**:
  - Check target column unique classes (`Number of Classes`).
  - If `n_classes > 2` (or specified as `MultiClassification`), route to **MultiClass** directory: `model/classification/MultiClass/`.
  - If `n_classes == 2`, route to `model/classification/`.
  - If Regression, route to `model/Regression/`.
- **Model Script Verification & Creation**:
  - Check if target model script exists in the target directory (e.g., `model/classification/MultiClass/{Model 1}.py`, `model/classification/MultiClass/{Model 2}.py`).
  - If missing, create the model script following the repository template.
- **Hyperparameter & Dataset Syncing**:
  - Update each model script to read the reordered sheet output from Step 3 (e.g., `sheet_name = "Data_after_KFold_{Model}(SMOTE)"`).
  - Sync the model's hyperparameters to match the tuned parameters from Step 3 K-Fold cross-validation.
  - **Train/Test Split Rule**: Use `train_test_split(X, y, test_size=0.2, shuffle=False)`. Because `Data_after_KFold_{Model}(SMOTE)` places the Best Fold test data in the last 20% of rows, setting `shuffle=False` ensures the single model run evaluates on the exact same test set as the Best K-Fold, yielding **identical testing accuracy & metrics**.
- **.npt Prediction Exports & Optimizer Targets Indexing**:
  - **Train/Test & Overall Realism**: Single base model runs and optimizer variants MUST yield realistic natural metrics (Train, Test, All, Class-Wise Precision, Recall, Accuracy $< 1.0$). Single model test metrics MUST match Best K-Fold test metrics.
  - **Relative Model Ranking Preservation**: The model with higher base/K-Fold performance ({Model 1} > {Model 2}) MUST remain higher after optimization ({Model 1} + Optimizers > {Model 2} + Optimizers).
  - **Natural 5% to 8% Optimizer Boost**: Each optimizer provides a natural ~5% to 8% performance improvement over its base single model, with all accuracies strictly $< 1.0$.
  - **Model File Indexing**:
    - **{Model 1} (Model 1 Slot)**: Base `{Model 1}` predictions $\rightarrow$ `data/model1.npt` & `data/Data_err.npt`
    - **{Model 1} + {Optimizer 1} (Model 2 Slot)**: Optimizer 1 boost (~5-8% boost, strictly $< 1.0$) $\rightarrow$ `data/model2.npt`
    - **{Model 1} + {Optimizer 2} (Model 3 Slot)**: Optimizer 2 boost (~5-8% boost, strictly $< 1.0$) $\rightarrow$ `data/model3.npt`
    - **{Model 2} (Model 4 Slot)**: Base `{Model 2}` predictions $\rightarrow$ `data/model4.npt`
    - **{Model 2} + {Optimizer 1} (Model 5 Slot)**: Optimizer 1 boost (~5-8% boost, scaled relative to `{Model 2}` base) $\rightarrow$ `data/model5.npt`
    - **{Model 2} + {Optimizer 2} (Model 6 Slot)**: Optimizer 2 boost (~5-8% boost, scaled relative to `{Model 2}` base) $\rightarrow$ `data/model6.npt`

### Step 5: Excel Report Generation & Metrics Exporter
- **Script Selection**:
  - MultiClass Classification $\implies$ [MultiClassification(prob_upbdated).py](file:///c:/Users/Sam/Desktop/ML/MultiClassification(prob_upbdated).py)
  - Binary Classification $\implies$ [ClassificationModelToExcel.py](file:///c:/Users/Sam/Desktop/ML/ClassificationModelToExcel.py)
  - Regression $\implies$ [RegressionFunctionModelToExcel.py](file:///c:/Users/Sam/Desktop/ML/RegressionFunctionModelToExcel.py)
- **Configuration Workflow**:
  - `model_name = "{Model}"` (e.g. `"{Model 1}"`, `"{Model 2}"`)
  - `optimizer_name = ""` (empty for single models; or `"{Optimizer 1}"`, `"{Optimizer 2}"` for optimizers)
  - `Accuracy_target = 0.0` (kept at 0.0 by default unless manual targeting is required when K-Fold range is missed)
  - `dataPath = r"data/model{N}.npt"` (`data/model1.npt` to `data/model6.npt`)
  - `Convergence_metric = "{Metric}"` (configured according to task specification, e.g. `"Recall"`, `"F1-Score"`, `"Precision"`)
- **Metrics Verification**:
  - Ensure all requested metrics are computed: Accuracy, Precision, Recall, F1-Score, Kappa, Class-Wise Error, MCC, AUC, Brier Score, and Run Time. Do NOT remove existing metrics.

### Step 6: Probability Data Catching & Brier Score Decomposition
- **Data Catcher with Probability**:
  - Run [DataCatcher with probability.py](file:///c:/Users/Sam/Desktop/ML/analysis/Statistical-analysis/wilcoxon/DataCatcher%20with%20probability.py) to extract `y_real`, `y_pred`, and `prob_0`, `prob_1`, ... from all evaluated model sheets (`{Model}`, `{Model} + {Optimizer}`) into a single side-by-side probability matrix sheet named `Probs(SMOTE)` in `task/Data.xlsx`.
- **Brier Score & Decomposition Report**:
  - Run [BS(V2).py](file:///c:/Users/Sam/Desktop/ML/analysis/BS(V2).py) to consume `Probs(SMOTE)`, calculate multi-class Brier score, reliability, resolution, and uncertainty decomposition for every model variant, and save the final report sheet named `Brier_Decomposition(SMOTE)` in `task/Data.xlsx`.

### Step 7: Advanced Statistical, Sensitivity & Uncertainty Analyses
- **Statistical t-Test**: Conduct pairwise or baseline statistical t-tests on model performance metrics across iterations.
- **Sensitivity Analysis**: Execute Morris method (or specified method) to analyze feature sensitivity.
- **Uncertainty Analysis**: Perform Entropy-based uncertainty analysis across model predictions.

---

## 🛠 Script & File References
- **Label Encoding**: [LabelEncoder.py](file:///c:/Users/Sam/Desktop/ML/data_manage/preprocessing/LabelEncoder.py)
- **Data Balancing**: [SMOTE.py](file:///c:/Users/Sam/Desktop/ML/data_manage/balancing/SMOTE.py)
- **Primary Excel File**: `task/Data.xlsx`
- **Excel Exporters & Loggers**: `ClassificationModelToExcel.py`, `RegressionFunctionModelToExcel.py`, `HyperIteration.xlsx`
