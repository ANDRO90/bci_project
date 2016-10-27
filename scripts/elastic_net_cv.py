import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn import cross_validation, linear_model


def main():
    TR = h5py.File('train.mat')

    pos_x_tr = np.array(TR['pos_x_tr']).T
    pos_y_tr = np.array(TR['pos_y_tr']).T

    feats_tr = np.load("features_tr.npy")
    feats_te = np.load("features_te.npy")

    elastic = linear_model.ElasticNet()

    alphas = np.logspace(-8, -1, 30)

    scores = list()
    scores_std = list()

    for alpha in alphas:
        elastic.alpha = alpha
        this_scores = cross_validation.cross_val_score(elastic, feats_tr, pos_y_tr)
        print alpha, this_scores
        scores.append(np.mean(this_scores))
        scores_std.append(np.std(this_scores))

    plt.figure(figsize=(4, 3))
    plt.semilogx(alphas, scores)
    plt.semilogx(alphas, np.array(scores) + np.array(scores_std) / np.sqrt(len(feats_tr)), 'b--')
    plt.semilogx(alphas, np.array(scores) - np.array(scores_std) / np.sqrt(len(feats_tr)), 'b--')
    plt.ylabel('CV score')
    plt.xlabel('alpha')
    plt.axhline(np.max(scores), linestyle='--', color='.5')

    elastic_cv = linear_model.ElasticNetCV(alphas=alphas)

    k_fold = cross_validation.KFold(len(feats_tr), 3)

    for k, (train, test) in enumerate(k_fold):
        elastic_cv.fit(feats_tr[train], pos_y_tr[train])

        print("[fold {0}] alpha: {1:.5f}, score: {2:.5f}".format(k, elastic_cv.alpha_, elastic_cv.score(feats_tr[test], pos_y_tr[test])))

    plt.show()
    

if __name__ == "__main__":
    main()
