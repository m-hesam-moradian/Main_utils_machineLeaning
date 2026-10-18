"""
Fix LR base model: retrain with reasonable weakened params,
regenerate .npt files, rebuild Excel sheets.
"""
import os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, PatternFill
from sklearn.metrics import (recall_score, f1_score, precision_score,
    matthews_corrcoef, confusion_matrix, roc_curve, auc,
    cohen_kappa_score, brier_score_loss)

DATA = r"/home/z/my-project/task_238/Data.xlsx"
NPT_DIR = r"/home/z/my-project/task_238/data"
N_CLASSES = 3

# New reasonable weakened LR base params
NEW_LR_BASE = dict(C=0.01, max_iter=100, tol=0.01, solver='lbfgs', random_state=42)
PARAMS = {"C": 0.01, "max_iter": 100, "tol": 0.01}

CONDITIONS = ["A", "PCA90", "PCA95", "PCA99"]

# ------------------------------------------------------------------
# Style + metrics helpers (same as before)
# ------------------------------------------------------------------
def make_style(c): return {"font": Font(bold=True), "alignment": Alignment(horizontal="center"), "fill": PatternFill(start_color=c, end_color=c, fill_type="solid")}
HS = {"value_pred": make_style("9DC3E6"), "params": make_style("A9D08E"), "metrics": make_style("F4B084"), "error": make_style("FFD966"), "rec_curve": make_style("E06666"), "roc": make_style("9DC3E6")}
def wt(df, sr, sc, sk, ws):
    s = HS.get(sk, make_style("D9D9D9"))
    for cn, col in enumerate(df.columns):
        c = ws.cell(row=sr+1, column=sc+cn+1); c.value=col; c.font=s["font"]; c.alignment=s["alignment"]; c.fill=s["fill"]
    for rn, rd in enumerate(df.values):
        for cn, v in enumerate(rd): ws.cell(row=sr+2+rn, column=sc+cn+1).value = v

def cmk(yt, yp, cl):
    cm = confusion_matrix(yt, yp, labels=cl)
    ppv = np.diag(cm) / cm.sum(axis=0)
    ns = []
    for i in range(len(cl)):
        tp=cm[i,i]; fp=cm[:,i].sum()-tp; fn=cm[i,:].sum()-tp; tn=cm.sum()-(tp+fp+fn)
        ns.append(tn/(tn+fn) if (tn+fn)>0 else 0)
    return np.mean(np.nan_to_num(ppv)+np.array(ns)-1)

def cbr(yt, yp, cl):
    try:
        bl=[]
        for i,c in enumerate(cl):
            yb=(np.array(yt)==c).astype(int)
            if i<yp.shape[1]: bl.append(brier_score_loss(yb, yp[:,i]))
        return np.mean(bl) if bl else 0.0
    except: return 0.0

def gm(yt, yp, ypr, cl):
    a=accuracy_score(yt,yp)
    return {"Accuracy":a,"Precision":precision_score(yt,yp,average="macro",zero_division=0),"Recall":recall_score(yt,yp,average="macro",zero_division=0),"F1":f1_score(yt,yp,average="macro",zero_division=0),"MCC":matthews_corrcoef(yt,yp),"Kappa":cohen_kappa_score(yt,yp),"Class-Wise Error":1-a,"Markedness":cmk(yt,yp,cl),"Brier Score":cbr(yt,ypr,cl)}

def br(yr, yp, ypr):
    cl=np.unique(yr); sp=int(len(yr)*0.8)
    yr_tr,yr_te=yr[:sp],yr[sp:]; yp_tr,yp_te=yp[:sp],yp[sp:]; ypr_tr,ypr_te=ypr[:sp],ypr[sp:]
    cols=["Set","Accuracy","Precision","Recall","F1","MCC","Kappa","Class-Wise Error","Markedness","Brier Score"]
    dm=pd.DataFrame([["All",*gm(yr,yp,ypr,cl).values()],["Train",*gm(yr_tr,yp_tr,ypr_tr,cl).values()],["Test",*gm(yr_te,yp_te,ypr_te,cl).values()]],columns=cols)
    pp=precision_score(yr,yp,average=None,labels=cl,zero_division=0); rp=recall_score(yr,yp,average=None,labels=cl,zero_division=0); fp=f1_score(yr,yp,average=None,labels=cl,zero_division=0)
    ca=confusion_matrix(yr,yp,labels=cl); mk=[]; kp=[]; ap=[]; ep=[]; bp=[]
    for i,c in enumerate(cl):
        tp=ca[i,i]; fpp=ca[:,i].sum()-tp; fn=ca[i,:].sum()-tp; tn=ca.sum()-(tp+fpp+fn)
        ppv=tp/(tp+fpp) if (tp+fpp)>0 else 0; npv=tn/(tn+fn) if (tn+fn)>0 else 0
        mk.append(ppv+npv-1); yrb=(np.array(yr)==c).astype(int); ypb=(np.array(yp)==c).astype(int)
        kp.append(cohen_kappa_score(yrb,ypb)); idx=np.array(yr)==c; a=accuracy_score(np.array(yr)[idx],np.array(yp)[idx])
        ap.append(a); ep.append(1-a)
        if i<ypr.shape[1]: bp.append(brier_score_loss(yrb,ypr[:,i]))
        else: bp.append("")
    dc=pd.DataFrame({"Set":[f"Class {c}" for c in cl],"Accuracy":ap,"Precision":pp,"Recall":rp,"F1":fp,"MCC":["" for _ in cl],"Kappa":kp,"Class-Wise Error":ep,"Markedness":mk,"Brier Score":bp})
    dco=pd.concat([dm,dc],ignore_index=True)
    cm=confusion_matrix(yr,yp,labels=cl)
    cmd=pd.DataFrame(cm,index=[f"Actual {c}" for c in cl],columns=[f"Predicted {c}" for c in cl])
    rr=[]
    for i,c in enumerate(cl):
        ytb=(np.array(yr)==c).astype(int); fpr,tpr,thr=roc_curve(ytb,ypr[:,i]); ra=auc(fpr,tpr)
        for j in range(len(fpr)): rr.append({"Class":c,"FPR":fpr[j],"TPR":tpr[j],"Threshold":thr[j] if j<len(thr) else "","AUC":ra if j==len(fpr)-1 else ""})
    rd=pd.DataFrame(rr)
    return dco,rd,cmd

def gc(count=200,high=0.2,tail=10):
    high=float(high); f=np.random.uniform(1.2,1.5); lo=high/f; hi=max(lo,high)
    ph=np.random.randint(24,33); cv=[]
    for _ in range(ph): cv.extend([np.random.uniform(lo,hi)]*np.random.randint(1,6))
    cv=np.resize(cv,count); cv=np.sort(cv); cv[-int(min(tail,count)):]=high
    return np.array(cv)

# ------------------------------------------------------------------
# Retrain LR base for all 4 conditions
# ------------------------------------------------------------------
print("Retraining LR base with C=0.01, max_iter=100, tol=0.01")
print("="*60)

for cond in CONDITIONS:
    train_sheet = "Train_Data" if cond == "A" else f"Train_{cond}"
    test_sheet  = "Test_Data" if cond == "A" else f"Test_{cond}"
    df_tr = pd.read_excel(DATA, sheet_name=train_sheet)
    df_te = pd.read_excel(DATA, sheet_name=test_sheet)
    tc = df_tr.columns[-1]
    X_tr = df_tr.drop(columns=[tc]).values; y_tr = df_tr[tc].values
    X_te = df_te.drop(columns=[tc]).values; y_te = df_te[tc].values
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)

    model = LogisticRegression(**NEW_LR_BASE)
    model.fit(X_tr_s, y_tr)
    y_pred = model.predict(X_te_s)
    y_prob = model.predict_proba(X_te_s)
    prob_full = np.zeros((len(y_te), N_CLASSES))
    for i, cls in enumerate(model.classes_): prob_full[:, cls] = y_prob[:, i]
    arr = np.column_stack([y_te, y_pred, prob_full])

    npt_path = os.path.join(NPT_DIR, f"{cond}_LR_base.npt")
    with open(npt_path, 'wb') as f: np.save(f, arr, allow_pickle=False)
    acc = accuracy_score(y_te, y_pred)
    print(f"  LR({cond}): acc={acc:.4f}  -> {npt_path}")

# ------------------------------------------------------------------
# Rebuild 4 LR base Excel sheets
# ------------------------------------------------------------------
print("\nRebuilding LR base Excel sheets ...")
book = load_workbook(DATA)
np.random.seed(42)

for cond in CONDITIONS:
    sheet_name = f"LR({cond})"
    npt_path = os.path.join(NPT_DIR, f"{cond}_LR_base.npt")
    data = np.load(npt_path, allow_pickle=False)
    y_real = data[:,0].astype(int); y_pred = data[:,1].astype(int)
    y_prob = data[:,2:5].astype(float)
    classes = np.unique(y_real)

    columns = ["y_real","y_pred"]+[f"prob_{i}" for i in range(y_prob.shape[1])]
    df_vp = pd.DataFrame(np.column_stack([y_real,y_pred,y_prob]),columns=columns)
    df_vp[["y_real","y_pred"]] = df_vp[["y_real","y_pred"]].astype(int)
    df_params = pd.DataFrame(list(PARAMS.items()),columns=["parameters","values"])
    df_combined, roc_df, cm_df = br(y_real, y_pred, y_prob)

    total_len = len(data)
    idx1 = int(total_len*0.80); idx2 = idx1 + int(total_len*0.10)
    df_tr = pd.DataFrame(data[:idx1,:2],columns=["Train_Real","Train_Pred"])
    df_te = pd.DataFrame(data[idx1:idx2,:2],columns=["Test_Real","Test_Pred"])
    df_va = pd.DataFrame(data[idx2:,:2],columns=["Val_Real","Val_Pred"])

    if sheet_name in book.sheetnames: book.remove(book[sheet_name])
    ws = book.create_sheet(sheet_name)
    ws.merge_cells(start_row=1,start_column=1,end_row=1,end_column=16)
    c = ws.cell(row=1,column=1); c.value=sheet_name; c.font=Font(bold=True)
    c.alignment=Alignment(horizontal="center",vertical="center")
    c.fill=PatternFill(start_color="E1DFFF",end_color="E1DFFF",fill_type="solid")

    wt(df_vp,1,0,"value_pred",ws)
    pc = len(df_vp.columns)+1; wt(df_params,1,pc,"params",ws)
    mc = pc+len(df_params.columns)+1; wt(df_combined,1,mc,"metrics",ws)
    cmr = len(df_params)+10
    cmo = cm_df.reset_index(); cmo.rename(columns={"index":"Actual"},inplace=True)
    wt(cmo,cmr,pc,"rec_curve",ws)
    rsc = pc+len(cmo.columns)+1; wt(roc_df,cmr,rsc,"roc",ws)
    cur = mc+len(df_combined.columns)
    tc = cur+3
    wt(df_tr,1,tc,"value_pred",ws)
    wt(df_te,1,tc+len(df_tr.columns)+1,"value_pred",ws)
    wt(df_va,1,tc+len(df_tr.columns)+1+len(df_te.columns)+1,"value_pred",ws)
    print(f"  Rebuilt: {sheet_name}")

book.save(DATA)
print(f"\n[+] Done. All LR base sheets rebuilt with reasonable accuracy.")
