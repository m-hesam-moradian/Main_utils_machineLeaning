import random
import numpy as np
from copy import deepcopy
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


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
        self.chance = chance  # 50 percent chance of changing std.
        self.ruleComb = ruleComb
        self._noParam = 2  # mean and std

    def gaussian(self, x, mu, sig):
        return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))

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
        temp = np.asarray(arr)
        mean = temp[0]
        std = temp[1]
        meanShape = mean.shape
        stdShape = std.shape

        mean = mean.flatten()
        std = std.flatten()
        num = int(self.mutationRate * mean.size)

        if random.uniform(0, 1) > self.chance:
            inds = np.random.choice(mean.size, size=num)
            mean[inds] -= np.random.uniform(0, 1, size=num) * self.learningRate
        else:
            inds = np.random.choice(std.size, size=num)
            std[inds] -= np.random.uniform(0, 1, size=num) * self.learningRate
            std = np.where(std == 0, 0.0001, std)  # std cannot be zero

        temp[0] = mean.reshape(meanShape)
        temp[1] = std.reshape(stdShape)
        return temp

    def init_population(self, X):
        noParam = self._noParam
        functions = self.functions
        offsprings = self.offsprings
        bestParam = np.random.rand(noParam, X.shape[1], functions)
        parentParam = deepcopy(bestParam)
        popParam = [self.mutation(parentParam) for _ in range(offsprings)]
        return popParam

    def init_model(self, model=LinearRegression()):
        return [model for _ in range(self.functions)]

    def forwardPass(self, param, X, inputs, Ant, L1, L2, functions):
        noParam = self._noParam

        # Input layer
        for i in range(X.shape[1]):
            inputs[i] = np.repeat(X[:, i].reshape(-1, 1), functions, axis=1)

        # Antecedent parameters
        for ii in range(noParam):
            for i in range(X.shape[1]):
                Ant[ii] = np.repeat(param[ii][i, :].reshape(1, -1), X.shape[0], axis=0)

        # Membership values using Gaussian
        for i in range(X.shape[1]):
            L1[i, :, :] = self.gaussian(x=inputs[i], mu=Ant[0][i], sig=Ant[1][i])

        # Rule layer
        for j in range(functions):
            for i in range(1, X.shape[1]):
                L2[:, j] = L1[i - 1, :, j] * L1[i, :, j]

        summ = np.sum(L2, axis=1).reshape(-1, 1)

        epsilon = 1e-10
        L3 = L2 / (np.repeat(summ, functions, axis=1) + epsilon)

        L3 = np.round(L3, 5)

        # Consequent layer
        L4 = np.zeros((functions, X.shape[0], X.shape[1]))
        for i in range(functions):
            L4[i] = X * L3[:, i].reshape(-1, 1)

        return L1, L2, L3, L4

    def linear_fit(self, L3, L4, X, y, functions, models):
        pred_train = np.zeros((len(X), functions))
        for i in range(functions):
            models[i].fit(L4[i], y)
            pred_train[:, i] = models[i].predict(L4[i])
        pred_train = np.sum(pred_train * L3, axis=1)
        return pred_train, models

    def linear_predict(self, L3, L4, X, functions, Trained_models):
        pred_test = np.zeros((X.shape[0], functions))
        for i in range(functions):
            pred_test[:, i] = Trained_models[i].predict(L4[i])
        pred_test = np.sum(pred_test * L3, axis=1)
        return pred_test

    @staticmethod
    def rmse(true, pred):
        return np.sqrt(np.mean((true - pred) ** 2))

    def fit(self, X_train, y_train, X_test=None, y_test=None, optimize_test_data=False):
        popParam = self.init_population(X_train)
        inputsTrain, AntTrain, L1Train, L2Train = self.initialize(X_train)
        if optimize_test_data:
            inputsTest, AntTest, L1Test, L2Test = self.initialize(X_test)
        models = self.init_model()
        bestParam = popParam[0]

        for gen in range(self.generations):
            popParam[0] = deepcopy(bestParam)
            for ii in range(1, self.offsprings):
                popParam[ii] = deepcopy(self.mutation(bestParam))

            PopulationError = []
            bestModelLst = []

            for param in popParam:
                L1, L2, L3, L4 = self.forwardPass(
                    param,
                    X_train,
                    inputsTrain,
                    AntTrain,
                    L1Train,
                    L2Train,
                    self.functions,
                )
                pred_train, Trained_models = self.linear_fit(
                    L3, L4, X_train, y_train, self.functions, models
                )
                mse_train = self.rmse(y_train, pred_train)

                if optimize_test_data:
                    L1, L2, L3, L4 = self.forwardPass(
                        param,
                        X_test,
                        inputsTest,
                        AntTest,
                        L1Test,
                        L2Test,
                        self.functions,
                    )
                    pred_test = self.linear_predict(
                        L3, L4, X_test, self.functions, Trained_models
                    )
                    mse_test = self.rmse(y_test, pred_test)
                    PopulationError.append((mse_train + mse_test) / 2)
                else:
                    PopulationError.append(mse_train)

                bestModelLst.append(Trained_models)

            bestParamIndex = np.argmin(PopulationError)
            bestParam = deepcopy(popParam[bestParamIndex])
            bestModel = bestModelLst[bestParamIndex]
            print(gen, "RMSE is:", PopulationError[bestParamIndex])

        return bestParam, bestModel

    def predict(self, X, bestParam, bestModel):
        inputs, Ant, L1, L2 = self.initialize(X)
        L1, L2, L3, L4 = self.forwardPass(
            bestParam, X, inputs, Ant, L1, L2, self.functions
        )
        pred = self.linear_predict(L3, L4, X, self.functions, bestModel)
        return pred
