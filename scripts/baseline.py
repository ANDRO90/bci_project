import numpy as np
import h5py


def main():
    TR = h5py.File('train.mat')

    pos_x_tr = np.array(TR['pos_x_tr']).T
    pos_y_tr = np.array(TR['pos_y_tr']).T

    feats_tr = np.load("features_tr.npy")
    feats_te = np.load("features_te.npy")

    # perform linear regression on x and y position
    A = feats_tr.T.dot(feats_tr)
    b = feats_tr.T.dot(pos_x_tr)

    x_weights = np.linalg.pinv(A).dot(b)
    b = feats_tr.T.dot(pos_y_tr)
    y_weights = np.linalg.pinv(A).dot(b)

    predict_x = feats_tr.dot(x_weights)
    predict_y = feats_tr.dot(y_weights)

    print 'train rmse'
    print np.sqrt((1.0 / (2.0 * len(predict_x))) * np.sum(
        (np.vstack((predict_x, predict_y)) - np.vstack((pos_x_tr, pos_y_tr))) ** 2))

    # test predictions
    predict_x_te = feats_te.dot(x_weights)
    predict_y_te = feats_te.dot(y_weights)

    IDs = range(len(predict_x_te) + len(predict_y_te))

    all_pred_te = np.vstack((predict_x_te, predict_y_te))

    fw = open('baseline.csv', 'w')
    fw.write('Id,Prediction\n')
    for i in IDs:
        fw.write('%d,%.5f\n' % (IDs[i] + 1, all_pred_te[i][0]))


if __name__ == "__main__":
    main()