import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import mean_squared_error


def levy(dim, beta=1.5):
    sigma = (
        np.math.gamma(1 + beta)
        * np.sin(np.pi * beta / 2)
        / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    step = u / np.abs(v) ** (1 / beta)
    return step


def polr(A, R0, N, t, Tmax, r):
    th = (1 + t / Tmax) * A * np.pi * np.random.rand(N)
    R = (r - t / Tmax) * R0 * np.random.rand(N)
    xR = R * np.sin(th)
    yR = R * np.cos(th)
    xR /= np.max(np.abs(xR))
    yR /= np.max(np.abs(yR))
    return xR, yR


def rth_optimizer(fobj, N, Tmax, low, high, dim):
    Xpos = np.random.uniform(low, high, (N, dim))
    Xcost = np.array([fobj(x) for x in Xpos])
    Xbestcost = np.min(Xcost)
    Xbestpos = Xpos[np.argmin(Xcost)].copy()
    convergence_curve = []

    A, R0, r = 15, 0.5, 1.5

    for t in range(1, Tmax + 1):
        Xmean = np.mean(Xpos, axis=0)
        TF = 1 + np.sin(2.5 - t / Tmax)

        # High Soaring
        for i in range(N):
            Xnewpos = Xbestpos + (Xmean - Xpos[i]) * levy(dim) * TF
            Xnewpos = np.clip(Xnewpos, low, high)
            newcost = fobj(Xnewpos)
            if newcost < Xcost[i]:
                Xpos[i] = Xnewpos
                Xcost[i] = newcost
                if newcost < Xbestcost:
                    Xbestpos = Xnewpos.copy()
                    Xbestcost = newcost

        # Low Soaring
        Xmean = np.mean(Xpos, axis=0)
        aa = np.random.permutation(N)
        Xpos = Xpos[aa]
        Xcost = Xcost[aa]
        x, y = polr(A, R0, N, t, Tmax, r)
        for i in range(N - 1):
            step = Xpos[i] - Xmean
            Xnewpos = Xbestpos + (y[i] + x[i]) * step
            Xnewpos = np.clip(Xnewpos, low, high)
            newcost = fobj(Xnewpos)
            if newcost < Xcost[i]:
                Xpos[i] = Xnewpos
                Xcost[i] = newcost
                if newcost < Xbestcost:
                    Xbestpos = Xnewpos.copy()
                    Xbestcost = newcost

        # Stopping & Swooping
        Xmean = np.mean(Xpos, axis=0)
        TF = 1 + 0.5 * np.sin(2.5 - t / Tmax)
        b = np.random.permutation(N)
        Xpos = Xpos[b]
        Xcost = Xcost[b]
        x, y = polr(A, R0, N, t, Tmax, r)
        alpha = np.sin(2.5 - t / Tmax) ** 2
        G = 2 * (1 - t / Tmax)
        for i in range(N):
            step1 = Xpos[i] - TF * Xmean
            step2 = G * Xpos[i] - TF * Xbestpos
            Xnewpos = alpha * Xbestpos + x[i] * step1 + y[i] * step2
            Xnewpos = np.clip(Xnewpos, low, high)
            newcost = fobj(Xnewpos)
            if newcost < Xcost[i]:
                Xpos[i] = Xnewpos
                Xcost[i] = newcost
                if newcost < Xbestcost:
                    Xbestpos = Xnewpos.copy()
                    Xbestcost = newcost

        convergence_curve.append(Xbestcost)
        print(f"🦅 Iter {t}/{Tmax} - Best RMSE: {Xbestcost:.5f}")

    return Xbestcost, Xbestpos, convergence_curve


def objective(params, X_train, y_train, X_test, y_test):
    n_estimators = int(params[0])
    max_depth = int(params[1]) if params[1] > 0 else None
    min_samples_split = int(params[2])
    min_samples_leaf = int(params[3])

    model = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred, squared=False)
