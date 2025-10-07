from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

data = pd.read_excel("./Dataset_Voids_Marshall.xlsx", sheet_name="numeriacl")

# Separate features (X) and target variable (y)
X = data.drop(["Percentage of Voids in the Marshall Sample"], axis=1, errors="ignore")
y = data["Percentage of Voids in the Marshall Sample"]


col = X.columns

X = np.array(X)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False, random_state=42
)


desired_features = 4

estimator = RandomForestRegressor()
selector = RFE(estimator=estimator, n_features_to_select=desired_features)
selector.fit(x_train, y_train)


selected_features = selector.support_
feature_ranking = selector.ranking_

y_preds = []

new_x_train = np.array([])
for i in range(X.shape[1] - 1):
    global input_and_ranks
    global new_sort_X
    global listDataset
    global r2s
    global rmses
    global listCol
    listCol = []
    listDataset = []
    r2s = []
    rmses = []

    new_sort_X = X
    input_and_ranks = []
    input_and_ranks_array = [[], []]
    for j in range(len(feature_ranking)):
        input_and_ranks.append({"input": j, "rank": feature_ranking[j]})
        input_and_ranks_array[0].append(col[j])
        input_and_ranks_array[1].append(feature_ranking[j])
    input_and_ranks.sort(key=lambda x: x["rank"])

    new_sort_X = X[:, [x["input"] for x in input_and_ranks]]
    for input in range(new_sort_X.shape[1] + 1):
        model = XGBRegressor(subsample=0.8)
        print(input)
        X_deleted = new_sort_X[:, :input]
        ranks = np.array(input_and_ranks)
        input_and_ranks_deleted = [rank["input"] + 1 for rank in ranks[:input]]

        if X_deleted.shape[1] > 0:  # Check if X_deleted is not empty
            listDataset.append(X_deleted)
            listCol.append(input_and_ranks_deleted)
            X_train, X_test, Y_train, Y_test = train_test_split(
                X_deleted, y, test_size=0.2, shuffle=True, random_state=42
            )
            model.fit(X_train, Y_train)
            pred = model.predict(X_test)
            r2 = r2_score(Y_test, pred)
            rmse = np.sqrt(mean_squared_error(Y_test, pred))
            rmses.append(rmse)
            r2s.append(r2)
            y_preds.append(pred)
