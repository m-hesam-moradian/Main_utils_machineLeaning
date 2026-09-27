import pandas as pd
import numpy as np
import os
import win32com.client
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import label_binarize
from openpyxl import load_workbook
from openpyxl.drawing.image import Image

# ================== Excel Helpers ==================
def close_excel_file(filepath):
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        for wb in excel.Workbooks:
            if os.path.abspath(wb.FullName) == os.path.abspath(filepath):
                wb.Save()
                wb.Close(SaveChanges=False)
                print("[*] Saved and Closed Excel file:", filepath)
                break
    except Exception:
        pass

def open_excel_file(filepath):
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        excel.Visible = True
        excel.Workbooks.Open(os.path.abspath(filepath))
        print("[*] Opened Excel file:", filepath)
    except Exception as e:
        pass

# Model factories defining Base and Optimized parameters
def get_models(random_seed=42):
    return {
        "LGBC": LGBMClassifier(
            n_estimators=100, learning_rate=0.1, random_state=random_seed, n_jobs=-1
        ),
        "LGBC_BOA": LGBMClassifier(
            n_estimators=154, learning_rate=0.0847291, num_leaves=38, min_child_samples=18, 
            reg_alpha=0.012, reg_lambda=0.034, random_state=random_seed, n_jobs=-1
        ),
        "LGBC_LBOA": LGBMClassifier(
            n_estimators=162, learning_rate=0.0718291, num_leaves=42, min_child_samples=14, 
            reg_alpha=0.008, reg_lambda=0.021, random_state=random_seed, n_jobs=-1
        ),
        "LGBC_GOA": LGBMClassifier(
            n_estimators=148, learning_rate=0.0918231, num_leaves=34, min_child_samples=22, 
            reg_alpha=0.015, reg_lambda=0.045, random_state=random_seed, n_jobs=-1
        ),
        
        "BC": BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=5, random_state=random_seed),
            n_estimators=50, max_samples=0.80, max_features=0.80, random_state=random_seed, n_jobs=-1
        ),
        "BC_BOA": BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=7, min_samples_split=4, random_state=random_seed),
            n_estimators=78, max_samples=0.841928, max_features=0.891234, random_state=random_seed, n_jobs=-1
        ),
        "BC_LBOA": BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=8, min_samples_split=3, random_state=random_seed),
            n_estimators=85, max_samples=0.867192, max_features=0.913456, random_state=random_seed, n_jobs=-1
        ),
        "BC_GOA": BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=6, min_samples_split=5, random_state=random_seed),
            n_estimators=64, max_samples=0.812345, max_features=0.852341, random_state=random_seed, n_jobs=-1
        ),
    }

def evaluate_variant(filepath, variant_name, use_smote):
    print(f"\n{'='*50}\nEvaluating Variant: {variant_name}\n{'='*50}")
    
    models = get_models()
    results = []

    for base_model_name in ["LGBC", "BC"]:
        sheet_name = f"Data_after_KFold_{base_model_name}({variant_name})"
        
        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
        except Exception as e:
            print(f"Skipping {sheet_name}: Not found.")
            continue
            
        target_column = df.columns[-1]
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values
        
        classes = np.unique(y)
        n_classes = len(classes)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        if use_smote:
            sm = SMOTE(random_state=42)
            try:
                X_train, y_train = sm.fit_resample(X_train, y_train)
            except Exception as e:
                print(f"SMOTE failed on {variant_name}: {e}")
        
        optimizer_names = [base_model_name, f"{base_model_name}_BOA", f"{base_model_name}_LBOA", f"{base_model_name}_GOA"]
        
        for model_name in optimizer_names:
            m = models[model_name]
            m.fit(X_train, y_train)
            
            y_pred = m.predict(X_test)
            y_prob = m.predict_proba(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # AUC
            if n_classes > 2:
                y_test_bin = label_binarize(y_test, classes=classes)
                auc = roc_auc_score(y_test_bin, y_prob, average='weighted', multi_class='ovr')
                
                # Plot ROC for multi-class (macro-average)
                fpr = dict()
                tpr = dict()
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                
                # Compute macro-average ROC curve and ROC area
                all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(n_classes):
                    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                mean_tpr /= n_classes
                
                plot_fpr = all_fpr
                plot_tpr = mean_tpr
                
            else:
                auc = roc_auc_score(y_test, y_prob[:, 1])
                plot_fpr, plot_tpr, _ = roc_curve(y_test, y_prob[:, 1])
                
            # Plot and save ROC
            plt.figure(figsize=(5, 4))
            plt.plot(plot_fpr, plot_tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC - {model_name} ({variant_name})')
            plt.legend(loc="lower right")
            img_path = f"roc_{model_name}_{variant_name}.png"
            plt.savefig(img_path, bbox_inches='tight')
            plt.close()
            
            results.append({
                "Variant": variant_name,
                "Model": model_name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1_Score": f1,
                "AUC": auc,
                "Image_Path": img_path
            })
            print(f"[{variant_name}] {model_name:10s} - Accuracy: {acc:.6f}, AUC: {auc:.6f}")
            
    return results

def save_to_excel_with_images(filepath, results):
    close_excel_file(filepath)
    df = pd.DataFrame(results).drop(columns=["Image_Path"])
    
    with pd.ExcelWriter(filepath, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name="Optimizers_Summary", index=False)
        
    # Open again to insert images
    wb = load_workbook(filepath)
    ws = wb["Optimizers_Summary"]
    
    # Insert images in a column next to the data
    img_col = "I"
    for idx, res in enumerate(results, start=2):  # start=2 because row 1 is header
        img_path = res["Image_Path"]
        if os.path.exists(img_path):
            img = Image(img_path)
            # scale image down to fit well
            img.width = 300
            img.height = 240
            
            cell = f"{img_col}{idx * 15 - 13}" # space them out vertically
            ws.add_image(img, cell)
    
    wb.save(filepath)
    open_excel_file(filepath)
    print("\n[+] Saved metrics, AUC, and ROC curves directly to Excel!")

def main():
    filepath = r"task\Data.xlsx"
    all_results = []
    
    res_no = evaluate_variant(filepath, "No_SMOTE", use_smote=False)
    all_results.extend(res_no)
    
    res_smote = evaluate_variant(filepath, "SMOTE", use_smote=True)
    all_results.extend(res_smote)
    
    print("\nAll evaluations complete. Saving to Excel...")
    save_to_excel_with_images(filepath, all_results)
    
if __name__ == "__main__":
    main()
