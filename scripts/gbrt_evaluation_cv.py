import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn import ensemble


def main():
    TR = h5py.File('train.mat')

    pos_x_tr = np.array(TR['pos_x_tr']).T
    pos_y_tr = np.array(TR['pos_y_tr']).T

    feats_tr = np.load("features_tr.npy")

    X = feats_tr
    y = np.ravel(pos_y_tr)  # pos_x_tr

    # fit regressor with out-of-bag estimates
    params = {'n_estimators': 1200, 'max_depth': 4,
              'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}

    n_estimators = params['n_estimators']
    x = np.arange(n_estimators) + 1

    def heldout_score(clf, X_test, y_test):
        score = np.zeros((n_estimators,), dtype=np.float64)
        for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
            score[i] = clf.loss_(y_test, y_pred)
        return score

    def cv_estimate(n_folds=3):
        cv = cross_validation.KFold(n=X.shape[0], n_folds=n_folds)
        cv_clf = ensemble.GradientBoostingRegressor(**params)
        val_scores = np.zeros((n_estimators,), dtype=np.float64)
        k = 0
        for train, test in cv:
            cv_clf.fit(X[train], y[train])
            val_scores += heldout_score(cv_clf, X[test], y[test])
            print k
            k += 1
        val_scores /= n_folds
        return val_scores

    # find best n_estimator using cross-validation
    cv_score = cv_estimate(3)

    # min loss according to cv
    cv_score -= cv_score[0]
    cv_best_iter = x[np.argmin(cv_score)]

    cv_color = list(map(lambda x: x / 256.0, (253, 192, 134)))

    # plot curves and vertical lines for best iterations
    plt.plot(x, cv_score, label='CV loss', color=cv_color)
    plt.axvline(x=cv_best_iter, color=cv_color)

    # add the vertical line to xticks
    xticks = plt.xticks()
    xticks_pos = np.array(xticks[0].tolist() +
                          [cv_best_iter])
    xticks_label = np.array(list(map(lambda t: int(t), xticks[0])) +
                            ['CV'])
    ind = np.argsort(xticks_pos)
    xticks_pos = xticks_pos[ind]
    xticks_label = xticks_label[ind]
    plt.xticks(xticks_pos, xticks_label)

    plt.legend(loc='upper right')
    plt.ylabel('normalized loss')
    plt.xlabel('number of iterations')

    plt.show()


if __name__ == "__main__":
    main()
