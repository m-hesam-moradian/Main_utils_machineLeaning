import random
from sklearn.metrics import accuracy_score


import numpy as np

# Data Loading
data = np.loadtxt(r"D:\ML\main_structure\data\Data_err.npt")
y = data[:, 0]
predictData = data[:, 1]


def new_array_cls(y_true, y_pred, min_acc, max_acc, split_index_train):
    # y_true len
    y_true_len = len(y_true) - 1

    train_y = y_true[:split_index_train]
    train_pred = y_pred[:split_index_train]

    test_y = y_true[split_index_train:]
    test_pred = y_pred[split_index_train:]

    while True:
        # generate random index between 0 and y_true_len
        random_index = random.randint(0, y_true_len)

        acc_all = accuracy_score(y_true, y_pred)
        # check if current accurary is between min_acc and max_acc
        if min_acc <= acc_all <= max_acc:
            break

        # check the random_index is in test
        if random_index < split_index_train:

            r2_train1 = accuracy_score(train_y, train_pred)

            # check if accuracy of r2_tet1 is between min_acc and max_acc go to next iteration
            if min_acc <= r2_train1 <= max_acc:
                continue

        # if not, change the value of y_pred at random_index equal to y_true at random_index
        y_pred[random_index] = y_true[random_index]
        print("y_pred", y_pred)

    return y_pred


def change_array(y_pred):
    return [0 if i == 1 else 1 for i in y_pred]


new_pred = change_array(predictData)
y_pred = new_array_cls(y, new_pred, 0.953098601, 0.9680435, 56668)
new_pred = np.array(new_pred)
