# SCI Machine Learning Project Guidelines & Workflow Rules

This document outlines the core guidelines, execution rules, and workflow standards for SCI research tasks (e.g., `BMM-SCI. No. XX`) within this workspace.

---

## 🎯 Role & Objective
- **Role**: Senior Machine Learning Engineer.
- **Goal**: Deliver rigorous, reproducible, high-performance, and verifiable machine learning pipelines and scientific benchmarks for SCI research projects.

---

## 📋 Task Identification & Inputs
- **Task Identification**: Tasks follow the standard format: `New SCI task-BMM-SCI .No.XX` or `BMM-SCI. No.XX`.
- **Task Definition & Requirements**: Refer to the task definition document (`.docx` / specification prompt) and associated dataset (`.csv` / `.xlsx`).

---

## 🚫 Directory Exclusions & Scope Restrictions
- **Completely Ignore `Tasks/` Folder**: The `Tasks/` directory must be completely ignored. It has nothing to do with our research pipeline. Never inspect, read, execute, or reference anything inside `Tasks/`.

---

## 🚀 Project Initialization Protocol (On "Start" / Task Trigger)
Whenever the user says **"start"** or initiates a new task:
1. **Inspect Inputs First**: Immediately read and parse the task definition document (`.docx` / specification) and inspect the dataset (`.csv` / `.xlsx`) before writing any code.
2. **Task List & Deliverables Checklist**:
   - Construct a structured **Task List** breaking down all phases defined in the task document.
   - Build a detailed **Deliverables Checklist** tracking every required CSV report, metric table, model artifact, and plot/figure to be recorded and downloaded.
3. **Notebook Creation & Colab Setup**: Create a dedicated Jupyter Notebook file (`.ipynb`) ready and connected/configured for execution in **Google Colab** (including environment setup, drive mounting if needed, modular structure, and step-by-step cells).

---

## 🛠️ Execution & Code Pipeline Rules

### 1. Step-by-Step Modular Execution
- Codebase is structured for execution in **Google Colab** and standalone Python environments.
- Always implement and execute code **step by step** in logical, distinct blocks (Data Loading & Preprocessing $\rightarrow$ Model Construction $\rightarrow$ Cross-Validation / Training $\rightarrow$ Optimization $\rightarrow$ Evaluation & Analysis).

### 2. Comprehensive Logging & Step Verification (Print Everything)
- **Print at Every Step**: At each phase, explicitly print intermediate outputs, data shapes, class distributions, parameter settings, convergence logs, and validation scores.
- **Visual & Metric Sanity Check**: Verify that every step is functioning correctly before progressing. 

### 3. Immediate Error Correction & Metric Sanity
- **Sanity Thresholds**: If an intermediate or final metric is improper, unrealistic, or unphysical (e.g., $R^2 = 0.0003$, collapsed accuracy, inverted loss, extreme over-fitting or near-zero variance), **do not move forward**.
- **Self-Correction**: Immediately debug the root cause (data scaling, loss function, learning rate, feature encoding, hyperparameter bounds), fix the issue, re-run, and verify realistic performance before proceeding.

### 4. CSV Report Generation & Output Management
- All required project outputs, fold metrics, optimization histories, model comparisons, anomaly scores, and sensitivity/uncertainty results must be saved as structured **CSV files** (e.g., `Report_<Task>_<Component>.csv`).
- Ensure high-resolution figures/plots (e.g. SHAP dependency plots, convergence curves) are saved when specified.

---

## 📌 Dynamic Rule Persistence (Continuous Learning Rule)
- **Rule Capture**: Whenever the user specifies a new rule, instruction, constraint, or preference in any interaction:
  1. **Strict Adherence**: Immediately follow and apply the new rule.
  2. **Auto-Save & Persist**: Automatically update this file (`GEMINI.md`), `AGENTS.md`, and `.agents/AGENTS.md` with the new rule so that it is permanently retained and never forgotten across sessions.
