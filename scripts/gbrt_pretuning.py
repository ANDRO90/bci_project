import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn import ensemble


def main():
    TR = h5py.File('train.mat')

    pos_x_tr = np.array(TR['pos_x_tr']).T

    feats_tr = np.load("features_tr.npy")

    X = feats_tr
    y = np.ravel(pos_x_tr)  # pos_y_tr

    original_params = {'n_estimators': 1000, 'max_leaf_nodes': 5, 'max_depth': None, 'random_state': 2,
                       'min_samples_split': 5}

    plt.figure()

    settings = [('No shrinkage', 'orange',
                   {'learning_rate': 1.0, 'subsample': 1.0}),
                  ('learning_rate=0.1', 'turquoise',
                   {'learning_rate': 0.1, 'subsample': 1.0}),
                  ('subsample=0.5', 'blue',
                   {'learning_rate': 1.0, 'subsample': 0.5}),
                  ('learning_rate=0.1, subsample=0.5', 'gray',
                   {'learning_rate': 0.1, 'subsample': 0.5}),
                  ('learning_rate=0.1, max_features=2', 'magenta',
                   {'learning_rate': 0.1, 'max_features': 2})]

    for label, color, setting in settings:
        params = dict(original_params)
        params.update(setting)

        clf = ensemble.GradientBoostingRegressor(**params)
        clf.fit(X, y)

        # compute test set deviance
        test_deviance = np.zeros((params['n_estimators'],), dtype=np.float64)

        for i, y_pred in enumerate(clf.staged_decision_function(X)):
            test_deviance[i] = clf.loss_(y, y_pred)

        plt.plot((np.arange(test_deviance.shape[0]) + 1)[::5], test_deviance[::5],
                 '-', color=color, label=label)

    plt.legend(loc='upper left')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Test Set Deviance')

    plt.show()

if __name__ == "__main__":
    main()
