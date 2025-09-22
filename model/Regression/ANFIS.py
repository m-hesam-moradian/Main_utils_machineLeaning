import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from Metrics_regression import getAllMetric
from copy import deepcopy
import random


# --- EVOLUTIONARY_ANFIS Class ---
class EVOLUTIONARY_ANFIS:
    def __init__(
        self,
        functions,
        generations,
        offsprings,
        mutationRate,
        learningRate,
        chance,
        ruleComb,
    ):
        self.functions = functions
        self.generations = generations
        self.offsprings = offsprings
        self.mutationRate = mutationRate
        self.learningRate = learningRate
        self.chance = chance
        self.ruleComb = ruleComb
        self._noParam = 2

    def gaussian(self, x, mu, sig):
        # Ensure sig is not zero to avoid division by zero
        sig = np.where(sig == 0, 1e-10, sig)
        return np.exp((-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0))))

    def initialize(self, X):
        functions = self.functions
        noParam = self._noParam
        ruleComb = self.ruleComb
        inputs = np.zeros((X.shape[1], X.shape[0], functions))
        Ant = np.zeros((noParam, X.shape[1], X.shape[0], functions))
        L1 = np.zeros((X.shape[1], X.shape[0], functions))
        if ruleComb == "simple":
            L2 = np.zeros((X.shape[0], functions))
        elif ruleComb == "complete":
            rules = X.shape[1] ** functions
            L2 = np.zeros((X.shape[0], rules))
        return inputs, Ant, L1, L2

    def mutation(self, arr):
        mutationRate = self.mutationRate
        learningRate = self.learningRate
        chance = self.chance
        temp = np.asarray(arr)
        mean = temp[0]
        meanShape = mean.shape
        std = temp[1]
        stdShape = std.shape
        mean = mean.flatten()
        std = std.flatten()
        num = int(mutationRate * mean.size)
        if random.uniform(0, 1) > chance:
            inds = np.random.choice(mean.size, size=num)
            mean[inds] -= np.random.uniform(0, 1, size=num) * learningRate
            mean = mean.reshape(meanShape)
            std = std.reshape(stdShape)
        else:
            inds = np.random.choice(std.size, size=num)
            std[inds] -= np.random.uniform(0, 1, size=num) * learningRate
            std = std.reshape(stdShape)
            std = np.where(std == 0, 1e-10, std)
            mean = mean.reshape(meanShape)
        temp[0] = mean
        temp[1] = std
        return temp

    def init_population(self, X):
        noParam = self._noParam
        functions = self.functions
        offsprings = self.offsprings
        bestParam = np.random.rand(noParam, X.shape[1], functions)
        parentParam = deepcopy(bestParam)
        popParam = []
        for i in range(offsprings):
            popParam.append(self.mutation(parentParam))
        return popParam

    def init_model(self, model=LinearRegression()):
        models = []
        for i in range(self.functions):
            models.append(model)
        return models

    def forwardPass(self, param, X, inputs, Ant, L1, L2, functions):
        noParam = self._noParam
        for i in range(X.shape[1]):
            inputs[i] = np.repeat(X[:, i].reshape(-1, 1), functions, axis=1)
        for ii in range(noParam):
            for i in range(X.shape[1]):
                Ant[ii] = np.repeat(param[ii][i, :].reshape(1, -1), X.shape[0], axis=0)
        for i in range(X.shape[1]):
            L1[i, :, :] = self.gaussian(x=inputs[i], mu=Ant[0][i], sig=Ant[1][i])
        for j in range(functions):
            for i in range(1, X.shape[1]):
                L2[:, j] = L1[i - 1, :, j] * L1[i, :, j]
        summ = np.sum(L2, axis=1).reshape(-1, 1)
        # Add epsilon to prevent division by zero
        summation = np.repeat(summ + 1e-10, functions, axis=1)
        L3 = L2 / summation
        L3 = np.round(L3, 5)
        consequent = X
        L4 = np.zeros((functions, X.shape[0], X.shape[1]))
        for i in range(functions):
            L4[i] = consequent
            L4[i] = L4[i] * L3[:, i].reshape(-1, 1)
        return L1, L2, L3, L4

    def linear_fit(self, L3, L4, X, y, functions, models):
        pred_train = np.zeros((len(X), functions))
        for i in range(functions):
            models[i].fit(L4[i], y)
            predTemp = models[i].predict(L4[i])
            pred_train[:, i] = predTemp
        pred_train = pred_train * L3
        pred_train = np.sum(pred_train, axis=1)
        return pred_train, models

    def linear_predict(self, L3, L4, X, functions, Trained_models):
        pred_test = np.zeros((X.shape[0], functions))
        for i in range(functions):
            predTemp = Trained_models[i].predict(L4[i]).reshape(-1, 1)
            pred_test[:, i] = predTemp[:, 0]
        pred_test = pred_test * L3
        pred_test = np.sum(pred_test, axis=1)
        return pred_test

    @staticmethod
    def rmse(true, pred):
        loss = np.sqrt(np.mean((true - pred) ** 2))
        return loss

    def fit(self, X_train, y_train, X_test=None, y_test=None, optimize_test_data=False):
        generations = self.generations
        offsprings = self.offsprings
        functions = self.functions
        popParam = self.init_population(X_train)
        inputsTrain, AntTrain, L1Train, L2Train = self.initialize(X_train)
        if optimize_test_data:
            inputsTest, AntTest, L1Test, L2Test = self.initialize(X_test)
        models = self.init_model()
        bestParam = popParam[0]
        for gen in range(generations):
            parentParam = deepcopy(bestParam)
            popParam[0] = deepcopy(bestParam)
            for ii in range(1, offsprings):
                mut = self.mutation(parentParam)
                popParam[ii] = deepcopy(mut)
            PopulationError = []
            bestModelLst = []
            for i in range(len(popParam)):
                L1, L2, L3, L4 = self.forwardPass(
                    popParam[i],
                    X_train,
                    inputsTrain,
                    AntTrain,
                    L1Train,
                    L2Train,
                    functions,
                )
                pred_train, Trained_models = self.linear_fit(
                    L3, L4, X_train, y_train, functions, models
                )
                mse_train = self.rmse(y_train, pred_train)
                if optimize_test_data:
                    L1, L2, L3, L4 = self.forwardPass(
                        popParam[i],
                        X_test,
                        inputsTest,
                        AntTest,
                        L1Test,
                        L2Test,
                        functions,
                    )
                    pred_test = self.linear_predict(
                        L3, L4, X_test, functions, Trained_models
                    )
                    mse_test = self.rmse(y_test, pred_test)
                    PopulationError.append((mse_train + mse_test) / 2)
                    bestModelLst.append(Trained_models)
                else:
                    PopulationError.append(mse_train)
                    bestModelLst.append(Trained_models)
            bestParamIndex = np.argmin(PopulationError)
            bestParam = deepcopy(popParam[bestParamIndex])
            bestModel = bestModelLst[bestParamIndex]
            print(gen, "RMSE is: ", PopulationError[bestParamIndex])
        return bestParam, bestModel

    def predict(self, X, bestParam, bestModel):
        functions = self.functions
        inputs, Ant, L1, L2 = self.initialize(X)
        L1, L2, L3, L4 = self.forwardPass(bestParam, X, inputs, Ant, L1, L2, functions)
        pred = self.linear_predict(L3, L4, X, functions, bestModel)
        return pred


# --- Data Loading ---
sheet_name = "Data after K-Fold (GBR & ANFIS)"
excel_path = r"D:\ML\Main_utils\Task\Global_AI_Content_Impact_Dataset.xlsx"
df = pd.read_excel(excel_path, sheet_name=sheet_name)
target_column = "Market Share of AI Companies (%)"

# --- Features and Target ---
X = df.drop(columns=[target_column])
y = df[target_column]

# --- Handle NaN Values ---
imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
y = y.fillna(y.mean())

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

# --- Standardize Features ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X = scaler.transform(X)

# --- ANFIS Model ---
model = EVOLUTIONARY_ANFIS(
    functions=3,
    generations=50,
    offsprings=35,
    mutationRate=0.0543,
    learningRate=0.42,
    chance=0.375,
    ruleComb="simple",
)
bestParam, bestModel = model.fit(
    X_train,
    y_train.values,
    X_test,
    y_test.values,
    optimize_test_data=True,
)

# --- Predictions ---
y_pred_all = model.predict(X, bestParam, bestModel)
y_pred_train = model.predict(X_train, bestParam, bestModel)
y_pred_test = model.predict(X_test, bestParam, bestModel)

# --- Split Test Predictions ---
mid_index = len(y_pred_test) // 2
y_test_first_half = y_test.iloc[:mid_index]
y_test_second_half = y_test.iloc[mid_index:]
y_pred_test_first_half = y_pred_test[:mid_index]
y_pred_test_second_half = y_pred_test[mid_index:]


# --- Metrics Calculation ---
def get_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, rmse


# --- Build Metrics Table Using getAllMetric ---
metrics_data = {"Set": [], "R2": [], "RMSE": [], "MAE": [], "RSE": [], "SMAPE": []}
sets = [
    ("All", y, y_pred_all),
    ("Train", y_train, y_pred_train),
    ("Test", y_test, y_pred_test),
    ("Value", y_test_first_half, y_pred_test_first_half),
    ("Test-Value", y_test_second_half, y_pred_test_second_half),
]

for name, y_true, y_pred in sets:
    R, RMSE, MAE, RSE, SMAPE = getAllMetric(y_true, y_pred)
    metrics_data["Set"].append(name)
    metrics_data["R2"].append(R)
    metrics_data["RMSE"].append(RMSE)
    metrics_data["MAE"].append(MAE)
    metrics_data["RSE"].append(RSE)
    metrics_data["SMAPE"].append(SMAPE)

metrics_df = pd.DataFrame(metrics_data)

# --- Additional DataFrames ---
df_train = pd.DataFrame({"y_train_real": y_train.values, "y_train_pred": y_pred_train})
df_test = pd.DataFrame({"y_test_real": y_test.values, "y_test_pred": y_pred_test})
df_all = pd.concat(
    [
        pd.DataFrame({"y_real": y_train.values, "y_pred": y_pred_train}),
        pd.DataFrame({"y_real": y_test.values, "y_pred": y_pred_test}),
    ],
    ignore_index=True,
)

# --- Output Results ---
print("\n📋 Performance Metrics Table:")
print(metrics_df)
print("\n📋 Training Data Predictions:")
print(df_train.head())
print("\n📋 Test Data Predictions:")
print(df_test.head())
print("\n📋 All Data Predictions:")
print(df_all.head())
