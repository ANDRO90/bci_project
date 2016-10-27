import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from collections import OrderedDict


def main():
    TR = h5py.File('train.mat')

    pos_x_tr = np.array(TR['pos_x_tr']).T
    pos_y_tr = np.array(TR['pos_y_tr']).T

    feats_tr = np.load("features_tr.npy")

    X = feats_tr
    y = np.c_[pos_x_tr, pos_y_tr]

    RANDOM_STATE = 0

    ensemble_clfs = [
        ("RandomForestRegressor, max_features='sqrt'",
         RandomForestRegressor(warm_start=True, oob_score=True,
                                max_features="sqrt",
                                random_state=RANDOM_STATE)),
        ("RandomForestRegressor, max_features='log2'",
         RandomForestRegressor(warm_start=True, max_features='log2',
                                oob_score=True,
                                random_state=RANDOM_STATE)),
        ("RandomForestRegressor, max_features=None",
         RandomForestRegressor(warm_start=True, max_features=None,
                                oob_score=True,
                                random_state=RANDOM_STATE))
    ]

    # map a classifier name to a list of (<n_estimators>, <error rate>) pairs
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    min_estimators = 15
    max_estimators = 175

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            print i
            clf.set_params(n_estimators=i)
            clf.fit(X, y)

            # record the OOB error for the `n_estimators=i` setting
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # generate the "OOB error rate" vs. "n_estimators" plot
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.show()

if __name__ == "__main__":
    main()
