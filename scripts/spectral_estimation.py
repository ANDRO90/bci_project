import numpy as np
import scipy.signal as sig
import h5py


def main():
    TR = h5py.File('train.mat')
    TE = h5py.File('test.mat')
    signal_tr = np.array(TR['signal_tr'],dtype='float64').T
    timestamps_tr = np.array(TR['timestamps_tr']).T
    time_sample_tr = np.array(TR['time_sample_tr']).T

    signal_te = np.array(TE['signal_te'],dtype='float64').T
    time_sample_te = np.array(TE['time_sample_te']).T
    timestamps_te = np.array(TE['timestamps_te']).T

    num_channels = 32
    
    # extract spectral features
    for ch in range(num_channels):
        print str(ch)
    
        ftr = Spectral_Estimation(signal_tr[:, ch], timestamps_tr, time_sample_tr)
        if ch == 0:
            feats_tr = ftr
        else:
            feats_tr = np.c_[feats_tr, ftr]
    
        fte = Spectral_Estimation(signal_te[:, ch], timestamps_te, time_sample_te)
        if ch == 0:
            feats_te = fte
        else:
            feats_te = np.c_[feats_te, fte]

    np.save("features_tr.npy", feats_tr)
    np.save("features_te.npy", feats_te)


def Spectral_Estimation(signal, ts, t_d):

    Fs = 2034.5  # Hz
    
    # Define Band-pass Filter Frequencies
    Wn1 = [8*2/Fs, 15*2/Fs]
    Wn2 = [15*2/Fs, 30*2/Fs]
    Wn3 = [30*2/Fs, 55*2/Fs]
    Wn4 = [70*2/Fs, 115*2/Fs]
    Wn5 = [130*2/Fs, 175*2/Fs]
    
    # Define Band-pass Filter Coefficients
    O = 2  # filter order
    [b1_b, b1_a] = sig.butter(O, Wn1, btype='bandpass')
    [b2_b, b2_a] = sig.butter(O, Wn2, btype='bandpass')
    [b3_b, b3_a] = sig.butter(O, Wn3, btype='bandpass')
    [b4_b, b4_a] = sig.butter(O, Wn4, btype='bandpass')
    [b5_b, b5_a] = sig.butter(O, Wn5, btype='bandpass')
    
    # Define Low-pass Filter Frequency
    lp_Wn = 2 * 2.0/Fs
    
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
    (nothing, as_well, b1) = Running_Stats(b1)
    (nothing, as_well, b2) = Running_Stats(b1)
    (nothing, as_well, b3) = Running_Stats(b3)
    (nothing, as_well, b4) = Running_Stats(b4)
    (nothing, as_well, b5) = Running_Stats(b5)
    
    estimates = np.vstack((b1, b2, b3, b4, b5)).T

    return estimates


def Running_Stats(x):

    M = np.zeros(np.shape(x))
    S = np.zeros(np.shape(x))
    
    M[0] = x[0]
    S[0] = 0.0
    
    for ii in range(1, len(x)):
        M[ii] = M[ii-1] + (x[ii] - M[ii-1])/(ii+1)
        S[ii] = S[ii-1] + (x[ii] - M[ii-1]) * (x[ii] - M[ii])
    
    mean_running = M
    bottom = np.array(range(len(x)))  # 0:(len(x)-1)
    std_running = np.sqrt(np.divide(S, bottom))
    std_running[0] = 1.0
    z_running = np.divide((x - M), std_running)
    return mean_running, std_running, z_running

if __name__ == "__main__":
    main()
