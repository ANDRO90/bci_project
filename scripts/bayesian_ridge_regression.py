import numpy as np
import scipy.signal as sig
import h5py
from sklearn import linear_model


def main():
    TR = h5py.File('train.mat')

    pos_x_tr = np.array(TR['pos_x_tr']).T
    pos_y_tr = np.array(TR['pos_y_tr']).T

    feats_tr = np.load("features_tr.npy")
    feats_te = np.load("features_te.npy")

    bayesian_x = linear_model.BayesianRidge(compute_score=True)
    bayesian_x.fit(feats_tr, pos_x_tr)
    p_x = bayesian_x.predict(feats_tr)
    p_x = p_x.reshape(len(p_x), 1)
    e_x = p_x - pos_x_tr

    bayesian_y = linear_model.BayesianRidge(compute_score=True)
    bayesian_y.fit(feats_tr, pos_y_tr)
    p_y = bayesian_y.predict(feats_tr)
    p_y = p_y.reshape(len(p_y), 1)
    e_y = p_y - pos_y_tr

    e = np.vstack((e_x, e_y))
    total_error = e ** 2

    rmse_train = np.sqrt(np.sum(total_error) / len(total_error))
    print "train error:", rmse_train

    predict_x_te = bayesian_x.predict(feats_te)
    predict_y_te = bayesian_y.predict(feats_te)
    predict_x_te = predict_x_te.reshape(len(predict_x_te), 1)
    predict_y_te = predict_y_te.reshape(len(predict_y_te), 1)
    
    IDs = range(len(predict_x_te)+len(predict_y_te))

    all_pred_te = np.vstack((predict_x_te, predict_y_te))

    fw = open('bayesian_ridge_regression.csv', 'w')
    fw.write('Id,Prediction\n')
    for i in IDs:
        fw.write('%d,%.5f\n' % (IDs[i]+1, all_pred_te[i][0]))
    fw.close()

if __name__ == "__main__":
    main()