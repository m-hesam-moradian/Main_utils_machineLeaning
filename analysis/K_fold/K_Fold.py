import numpy as np
from sklearn.model_selection import KFold, cross_val_predict


def kfold_cv(model, X, y, folds=10):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=kf)
    return y_pred
