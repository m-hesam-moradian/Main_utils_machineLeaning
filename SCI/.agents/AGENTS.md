# Workspace Rules for SCI Machine Learning Tasks

Whenever an SCI task prompt is provided (e.g., `New SCI task-BMM-SCI .No.XX` or `BMM-SCI. No.XX`), strictly follow these guidelines:

## 1. Role & Environment
- **Role**: Machine Learning Engineer.
- **Environment**: Google Colab / Python modular pipeline.

## 2. Scope & Directory Exclusions
- **Completely Ignore `Tasks/` Folder**: The `Tasks/` directory has nothing to do with our research pipeline. Completely ignore and never access, read, or process files inside `Tasks/`.

## 3. Project Initialization Protocol (On "Start" / Task Trigger)
Whenever the user says **"start"** or initiates a new task:
1. **Inspect Inputs First**: Immediately read and parse the task definition document (`.docx` / specification) and inspect the dataset (`.csv` / `.xlsx`) before writing any code.
2. **Task List & Deliverables Checklist**:
   - Create a structured **Task List** based on the task definition document.
   - Create a comprehensive **Deliverables Checklist** for recording, verifying, and saving/downloading all required CSV files, metric tables, and generated artifacts with all specific details.
3. **Notebook Creation & Colab Setup**: Create a dedicated Jupyter Notebook file (`.ipynb`) ready and configured for **Google Colab** execution (modular cells, environment setup, reproducible pipeline).

## 4. Step-by-Step Execution & Verification
- Implement all tasks sequentially step by step.
- At each step, print all relevant intermediate states, shapes, metrics, and logs to verify correctness before proceeding.
- If any metric or result is improper (e.g. $R^2 = 0.0003$, metric collapse, unrealistic accuracy), halt, diagnose, rectify the issue, and re-verify before advancing.

## 5. Reporting & Artifacts
- Save all required outputs, predictions, fold scores, optimization iterations, and statistical/sensitivity tables as **CSV files**.
- Export any requested plots (e.g., SHAP, ROC, convergence) in high quality.

## 6. Dynamic Rule Tracking & Persistence
- When the user introduces a new rule, requirement, or workflow constraint:
  1. Immediately adhere to the instruction.
  2. Persist the rule by updating `GEMINI.md`, `AGENTS.md`, and `.agents/AGENTS.md` so it remains active across all future conversations without forgetting.
