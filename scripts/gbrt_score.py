import numpy as np
import scipy.signal as sig
import h5py
from sklearn import ensemble

def main():
    TR = h5py.File('train.mat')

    pos_x_tr = np.array(TR['pos_x_tr']).T
    pos_y_tr = np.array(TR['pos_y_tr']).T

    feats_tr = np.load("features_tr.npy")
    feats_te = np.load("features_te.npy")

    X = feats_tr

    # retry with subsampling (bagging)
    params = {'n_estimators': 812, 'max_depth': 5, 'subsample': 0.3,
              'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}
    clf_x = ensemble.GradientBoostingRegressor(**params)
    clf_x.fit(X, pos_x_tr)
    p_x = clf_x.predict(feats_tr)
    p_x = p_x.reshape(len(p_x), 1)
    e_x = p_x - pos_x_tr

    # retry with subsampling (bagging)
    params = {'n_estimators': 1120, 'max_depth': 5, 'subsample': 0.3,
              'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}
    clf_y = ensemble.GradientBoostingRegressor(**params)
    clf_y.fit(X, pos_y_tr)
    p_y = clf_y.predict(feats_tr)
    p_y = p_y.reshape(len(p_y), 1)
    e_y = p_y - pos_y_tr

    e = np.vstack((e_x, e_y))
    total_error = e ** 2

    rmse_train = np.sqrt(np.sum(total_error) / len(total_error))
    print "train error:", rmse_train

    predict_x_te = clf_x.predict(feats_te)
    predict_y_te = clf_y.predict(feats_te)
    predict_x_te = predict_x_te.reshape(len(predict_x_te), 1)
    predict_y_te = predict_y_te.reshape(len(predict_y_te), 1)
    
    IDs = range(len(predict_x_te)+len(predict_y_te))

    all_pred_te = np.vstack((predict_x_te, predict_y_te))

    fw = open('gbr.csv', 'w')
    fw.write('Id,Prediction\n')
    for i in IDs:
        fw.write('%d,%.5f\n' % (IDs[i]+1, all_pred_te[i][0]))
    fw.close()
    


def Spectral_Estimation(signal, ts, t_d):

    Fs = 2034.5; # Hz
    
    # Define Band-pass Filter Frequencies
    Wn1 =   [8*2/Fs,  15*2/Fs]
    Wn2 =  [15*2/Fs,  30*2/Fs]
    Wn3 =  [30*2/Fs,  55*2/Fs]
    Wn4 =  [70*2/Fs, 115*2/Fs]
    Wn5 = [130*2/Fs, 175*2/Fs]
    
    # Define Band-pass Filter Coefficients
    O = 2; # filter order
    [b1_b,b1_a] = sig.butter(O, Wn1,btype='bandpass')
    [b2_b,b2_a] = sig.butter(O, Wn2,btype='bandpass')
    [b3_b,b3_a] = sig.butter(O, Wn3,btype='bandpass')
    [b4_b,b4_a] = sig.butter(O, Wn4,btype='bandpass')
    [b5_b,b5_a] = sig.butter(O, Wn5,btype='bandpass')
    
    # Define Low-pass Filter Frequency
    lp_Wn = 2* 2.0/Fs
    
    # Define Low-pass Filter Coefficients
    [lp_b, lp_a] = sig.butter(O, lp_Wn)
    
    # Band-pass Filter Signal
    b1 = sig.filtfilt(b1_b, b1_a, signal)
    b2 = sig.filtfilt(b2_b, b2_a, signal)
    b3 = sig.filtfilt(b3_b, b3_a, signal)
    b4 = sig.filtfilt(b4_b, b4_a, signal)
    b5 = sig.filtfilt(b5_b, b5_a, signal)
    
    # Low-pass and rectify
    b1 = sig.filtfilt(lp_b, lp_a, abs(b1))
    b2 = sig.filtfilt(lp_b, lp_a, abs(b2))
    b3 = sig.filtfilt(lp_b, lp_a, abs(b3))
    b4 = sig.filtfilt(lp_b, lp_a, abs(b4))
    b5 = sig.filtfilt(lp_b, lp_a, abs(b5))
    
    # Downsample using t_d
    ts = np.reshape(ts, len(ts))
    t_d = np.reshape(t_d, len(t_d))
    b1 = np.interp(t_d, ts, b1)  # error here!
    b2 = np.interp(t_d, ts, b2)
    b3 = np.interp(t_d, ts, b3)
    b4 = np.interp(t_d, ts, b4)
    b5 = np.interp(t_d, ts, b5)
    
    # Z-score
    (nothing,as_well,b1) = Running_Stats(b1)
    (nothing,as_well,b2) = Running_Stats(b1)
    (nothing,as_well,b3) = Running_Stats(b3)
    (nothing,as_well,b4) = Running_Stats(b4)
    (nothing,as_well,b5) = Running_Stats(b5)
    
    Estimates = np.vstack((b1, b2, b3, b4, b5)).T;


    return Estimates





def Running_Stats(x):

    M = np.zeros(np.shape(x));
    S = np.zeros(np.shape(x));
    
    M[0]= x[0];
    S[0] = 0.0;
    
    for ii in range(1,len(x)):
        M[ii] = M[ii-1] + (x[ii] - M[ii-1])/(ii+1);
        S[ii] = S[ii-1] + (x[ii] - M[ii-1]) * (x[ii] - M[ii]);
    
    mean_running = M;
    bottom = np.array(range(len(x)))#0:(len(x)-1);
    std_running = np.sqrt(np.divide(S,bottom))
    std_running[0] = 1.0;
    z_running = np.divide((x - M),std_running);
    return (mean_running, std_running, z_running)

if __name__ == "__main__":
    main()