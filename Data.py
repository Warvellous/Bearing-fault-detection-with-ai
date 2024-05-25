#importing necessary libraries
import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import scipy
from scipy.stats import entropy
from scipy.optimize import curve_fit
import frequencyanalaysis as frq_anlysis
import time_analysis as t_analysis
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

#Data paths
dataset_path_1st = 'archive/1st_test/1st_test'
dataset_path_2nd = 'archive/2nd_test/2nd_test'
dataset_path_3rd = 'archive/3rd_test/4th_test/txt'

#dataset = pd.read_csv('archive/1st_test/1st_test/2003.10.22.12.06.24', sep='\t')

# extract clearence factor


def freq_features(dataset_path, id_set=None):
    time_features = ['mean','std','skew','kurtosis','freq_center','rmsf','root_variance','entropy']
    cols1 = ['B1_x','B1_y','B2_x','B2_y','B3_x','B3_y','B4_x','B4_y']
    cols2 = ['B1','B2','B3','B4']
    
    # initialize
    if id_set == 1:
        columns = [c+'_'+tf for c in cols1 for tf in time_features]
        data = pd.DataFrame(columns=columns)
    else:
        columns = [c+'_'+tf for c in cols2 for tf in time_features]
        data = pd.DataFrame(columns=columns)

        
        
    for filename in os.listdir(dataset_path):
        # read dataset
        raw_data = pd.read_csv(os.path.join(dataset_path, filename), sep='\t')
        signal = raw_data
        SAMPLING_RATE = 20000



        fft_values = pd.DataFrame({col: fft(signal[col].values) for col in signal.columns })
        fft_magnitudes = pd.DataFrame({col: np.abs(fft_values[col].values) for col in fft_values.columns})
        freqs = pd.DataFrame({col: fftfreq(((signal[col]).shape[0]), d=1/SAMPLING_RATE) for col in signal.columns})


        # freqfeatures
        mean = np.array(frq_anlysis.calculate_mean_frequency(fft_magnitudes, freqs, SAMPLING_RATE))
        std = np.array(frq_anlysis.calculate_std_frequency(fft_magnitudes, freqs, SAMPLING_RATE, mean))
        skew = np.array(frq_anlysis.calculate_skewness_frequency(fft_magnitudes, freqs, SAMPLING_RATE, mean, std))
        kurtosis = np.array(frq_anlysis.calculate_kurtosis_frequency(fft_magnitudes, freqs, SAMPLING_RATE, mean, std))
        freq_center=np.array(frq_anlysis.calculate_frequency_center(fft_magnitudes, freqs, SAMPLING_RATE))
        rmsf=np.array(frq_anlysis.calculate_rms_frequency(fft_magnitudes, freqs, SAMPLING_RATE))
        root_variance=np.array(frq_anlysis.calculate_root_variance_frequency(fft_magnitudes, freqs, SAMPLING_RATE, freq_center))
        entropy=np.array(frq_anlysis.calculate_shannon_entropy(fft_magnitudes, freqs, SAMPLING_RATE))



        
        if id_set == 1:

            mean = pd.DataFrame(mean.reshape(1,8), columns=[c+'_mean' for c in cols1])
            std = pd.DataFrame(std.reshape(1,8), columns=[c+'_std' for c in cols1])
            skew = pd.DataFrame(skew.reshape(1,8), columns=[c+'_skew' for c in cols1])
            kurtosis= pd.DataFrame(kurtosis.reshape(1,8), columns=[c+'_kurtosis' for c in cols1])
            freq_center= pd.DataFrame(freq_center.reshape(1,8), columns=[c+'_freq_center' for c in cols1])
            rmsf = pd.DataFrame(rmsf.reshape(1,8), columns=[c+'_rmsf' for c in cols1])
            root_variance = pd.DataFrame(root_variance.reshape(1,8), columns=[c+'_root_variance' for c in cols1])
            entropy = pd.DataFrame(entropy.reshape(1,8), columns=[c+'_entropy' for c in cols1])


            
        else:
            mean = pd.DataFrame(mean.reshape(1,4), columns=[c+'_mean' for c in cols2])
            std = pd.DataFrame(std.reshape(1,4), columns=[c+'_std' for c in cols2])
            skew = pd.DataFrame(skew.reshape(1,4), columns=[c+'_skew' for c in cols2])
            kurtosis= pd.DataFrame(kurtosis.reshape(1,4), columns=[c+'_kurtosis' for c in cols2])
            freq_center= pd.DataFrame(freq_center.reshape(1,4), columns=[c+'_freq_center' for c in cols2])
            rmsf = pd.DataFrame(rmsf.reshape(1,4), columns=[c+'_rmsf' for c in cols2])
            root_variance = pd.DataFrame(root_variance.reshape(1,4), columns=[c+'_root_variance' for c in cols2])
            entropy = pd.DataFrame(entropy.reshape(1,4), columns=[c+'_entropy' for c in cols2])


        mean.index = [filename]
        std.index = [filename]
        skew.index = [filename]
        kurtosis.index = [filename]
        freq_center.index = [filename]
        rmsf.index = [filename]
        root_variance.index = [filename]
        entropy.index = [filename]

        
        # concat
        merge = pd.concat([mean, std, skew, kurtosis, freq_center, rmsf, root_variance, entropy], axis=1)
        data = data._append(merge)
        
    if id_set == 1:
        cols = [c+'_'+tf for c in cols1 for tf in time_features]
        data = data[cols]
    else:
        cols = [c+'_'+tf for c in cols2 for tf in time_features]
        data = data[cols]
        
    data.index = pd.to_datetime(data.index, format='%Y.%m.%d.%H.%M.%S')
    data = data.sort_index()

    data.to_csv("freq_features_dataset_{}.csv".format(id_set))
    return data 


def time_features(dataset_path, id_set=None):
    time_features = ['mean','std','skew','kurtosis','entropy','rms','max','p2p']
    cols1 = ['B1_a','B1_b','B2_a','B2_b','B3_a','B3_b','B4_a','B4_b']
    cols2 = ['B1','B2','B3','B4']
    
    # initialize
    if id_set == 1:
        columns = [c+'_'+tf for c in cols1 for tf in time_features]
        data = pd.DataFrame(columns=columns)
    else:
        columns = [c+'_'+tf for c in cols2 for tf in time_features]
        data = pd.DataFrame(columns=columns)
    
    for filename in os.listdir(dataset_path):
        # read dataset
        raw_data = pd.read_csv(os.path.join(dataset_path, filename), sep='\t')
        
        # time features
        mean_abs = np.array(raw_data.abs().mean())
        std = np.array(raw_data.std())
        skew = np.array(raw_data.skew())
        kurtosis = np.array(raw_data.kurtosis())
        entropy = t_analysis.calculate_entropy(raw_data)
        rms = np.array(t_analysis.calculate_rms(raw_data))
        max_abs = np.array(raw_data.abs().max())
        p2p = t_analysis.calculate_p2p(raw_data)
        
        if id_set == 1:
            mean_abs = pd.DataFrame(mean_abs.reshape(1,8), columns=[c+'_mean' for c in cols1])
            std = pd.DataFrame(std.reshape(1,8), columns=[c+'_std' for c in cols1])
            skew = pd.DataFrame(skew.reshape(1,8), columns=[c+'_skew' for c in cols1])
            kurtosis = pd.DataFrame(kurtosis.reshape(1,8), columns=[c+'_kurtosis' for c in cols1])
            entropy = pd.DataFrame(entropy.reshape(1,8), columns=[c+'_entropy' for c in cols1])
            rms = pd.DataFrame(rms.reshape(1,8), columns=[c+'_rms' for c in cols1])
            max_abs = pd.DataFrame(max_abs.reshape(1,8), columns=[c+'_max' for c in cols1])
            p2p = pd.DataFrame(p2p.reshape(1,8), columns=[c+'_p2p' for c in cols1])
        else:
            mean_abs = pd.DataFrame(mean_abs.reshape(1,4), columns=[c+'_mean' for c in cols2])
            std = pd.DataFrame(std.reshape(1,4), columns=[c+'_std' for c in cols2])
            skew = pd.DataFrame(skew.reshape(1,4), columns=[c+'_skew' for c in cols2])
            kurtosis = pd.DataFrame(kurtosis.reshape(1,4), columns=[c+'_kurtosis' for c in cols2])
            entropy = pd.DataFrame(entropy.reshape(1,4), columns=[c+'_entropy' for c in cols2])
            rms = pd.DataFrame(rms.reshape(1,4), columns=[c+'_rms' for c in cols2])
            max_abs = pd.DataFrame(max_abs.reshape(1,4), columns=[c+'_max' for c in cols2])
            p2p = pd.DataFrame(p2p.reshape(1,4), columns=[c+'_p2p' for c in cols2])
            
        mean_abs.index = [filename]
        std.index = [filename]
        skew.index = [filename]
        kurtosis.index = [filename]
        entropy.index = [filename]
        rms.index = [filename]
        max_abs.index = [filename]
        p2p.index = [filename]
        
        # concat
        merge = pd.concat([mean_abs, std, skew, kurtosis, entropy, rms, max_abs, p2p], axis=1)
        data = data._append(merge)
        
    if id_set == 1:
        cols = [c+'_'+tf for c in cols1 for tf in time_features]
        data = data[cols]
    else:
        cols = [c+'_'+tf for c in cols2 for tf in time_features]
        data = data[cols]
        
    data.index = pd.to_datetime(data.index, format='%Y.%m.%d.%H.%M.%S')
    data = data.sort_index()
    data.to_csv("time_features_dataset_{}.csv".format(id_set))
    return data


#freq_features(dataset_path_2nd, id_set=2)
time_features(dataset_path_2nd, id_set=2)

