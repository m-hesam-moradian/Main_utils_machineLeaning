import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import precision_score, recall_score


# Objective function: multi-objective (maximize precision and recall)
def fobj(params, X_train, y_train, X_test, y_test):
    n_estimators = int(params[0])
    max_depth = int(params[1]) if params[1] > 0 else None
    min_samples_split = int(params[2])
    min_samples_leaf = int(params[3])

    model = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        bootstrap=True,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return np.array([-precision, -recall])  # Minimize negative scores


def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


def fast_non_dominated_sort(pop):
    fronts = [[]]
    S = [[] for _ in range(len(pop))]
    n = [0] * len(pop)
    rank = [0] * len(pop)

    for p in range(len(pop)):
        for q in range(len(pop)):
            if dominates(pop[p]["obj"], pop[q]["obj"]):
                S[p].append(q)
            elif dominates(pop[q]["obj"], pop[p]["obj"]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return fronts[:-1], rank


def bonobo_optimizer(X_train, y_train, X_test, y_test, dim=4, N=50, max_it=50):
    Var_min = np.array([100, 5, 2, 1])
    Var_max = np.array([500, 30, 20, 10])
    pop = [{"pos": np.random.uniform(Var_min, Var_max), "obj": None} for _ in range(N)]

    for i in range(N):
        pop[i]["obj"] = fobj(pop[i]["pos"], X_train, y_train, X_test, y_test)

    for it in range(max_it):
        fronts, rank = fast_non_dominated_sort(pop)
        new_pop = []

        for i in range(N):
            alpha = pop[np.random.choice(fronts[0])]["pos"]
            mate = pop[np.random.randint(N)]["pos"]
            r1 = np.random.rand(dim)
            child = (
                pop[i]["pos"]
                + 1.4 * r1 * (alpha - pop[i]["pos"])
                + 1.55 * (1 - r1) * (pop[i]["pos"] - mate)
            )
            child = np.clip(child, Var_min, Var_max)
            new_pop.append(
                {"pos": child, "obj": fobj(child, X_train, y_train, X_test, y_test)}
            )

        pop += new_pop
        fronts, rank = fast_non_dominated_sort(pop)
        sorted_pop = sorted(pop, key=lambda x: rank[pop.index(x)])
        pop = sorted_pop[:N]

        print(
            f"🦍 Iter {it+1}/{max_it} - Best Precision: {-pop[0]['obj'][0]:.4f}, Recall: {-pop[0]['obj'][1]:.4f}"
        )

    return pop[0]["pos"], -pop[0]["obj"]  # Return best params and their scores
