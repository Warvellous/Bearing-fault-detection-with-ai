import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

"""
file:///C:/Users/MERVE/OneDrive/Masa%C3%BCst%C3%BC/graduationlasttermins/Newclusteringalgorithm-basedfaultdiagnosisusingcompensationdistanceevaluationtechnique.pdf
New clustering algorithm-based fault diagnosis using
compensation distance evaluation technique
"""
# page 7/18, s(k)= positive_magnitudes, f(k)=positive_freqs



def calculate_mean_frequency(fft_magnitudes, freqs, sampling_rate):

    positive_freqs = freqs[freqs >= 0]
    positive_magnitudes = fft_magnitudes[freqs >= 0]
    mean_frequency = np.sum(positive_magnitudes) / len(positive_magnitudes)
    return mean_frequency


def calculate_mean_frequency(fft_magnitudes_df, freqs_df, sampling_rate):
    results = []
    
    for column in fft_magnitudes_df.columns:
        positive_freqs = freqs_df[column][freqs_df[column] >= 0]
        positive_magnitudes = fft_magnitudes_df[column][freqs_df[column] >= 0]
        mean_frequency = np.sum(positive_magnitudes) / len(positive_magnitudes)
        results.append(mean_frequency)
    
    return np.array(results)

def calculate_std_frequency(fft_magnitudes_df, freqs_df, sampling_rate, mean_frequencies):
    results = []

    for i, column in enumerate(fft_magnitudes_df.columns):
        positive_freqs = freqs_df[column][freqs_df[column] >= 0]
        positive_magnitudes = fft_magnitudes_df[column][freqs_df[column] >= 0]
        mean_frequency = mean_frequencies[i]
        std_frequency = np.sqrt(np.sum((positive_freqs - mean_frequency)**2 * positive_magnitudes) / np.sum(positive_magnitudes))
        results.append(std_frequency)
    
    return np.array(results)


def calculate_skewness_frequency(fft_magnitudes_df, freqs_df, sampling_rate, mean_frequencies, std_frequencies):
    results = []

    for i, column in enumerate(fft_magnitudes_df.columns):
        positive_freqs = freqs_df[column][freqs_df[column] >= 0]
        positive_magnitudes = fft_magnitudes_df[column][freqs_df[column] >= 0]
        mean_frequency = mean_frequencies[i]
        std_frequency = std_frequencies[i]
        skewness_frequency = np.sum((positive_freqs - mean_frequency)**3 * positive_magnitudes) / (np.sum(positive_magnitudes) * std_frequency**3)
        results.append(skewness_frequency)
    
    return np.array(results)

def calculate_kurtosis_frequency(fft_magnitudes_df, freqs_df, sampling_rate, mean_frequencies, std_frequencies):
    results = []

    for i, column in enumerate(fft_magnitudes_df.columns):
        positive_freqs = freqs_df[column][freqs_df[column] >= 0]
        positive_magnitudes = fft_magnitudes_df[column][freqs_df[column] >= 0]
        mean_frequency = mean_frequencies[i]
        std_frequency = std_frequencies[i]
        kurtosis_frequency = np.sum((positive_freqs - mean_frequency)**4 * positive_magnitudes) / (np.sum(positive_magnitudes) * std_frequency**4)
        results.append(kurtosis_frequency)
    
    return np.array(results)

def calculate_frequency_center(fft_magnitudes_df, freqs_df, sampling_rate):
    results = []

    for column in fft_magnitudes_df.columns:
        positive_freqs = freqs_df[column][freqs_df[column] >= 0]
        positive_magnitudes = fft_magnitudes_df[column][freqs_df[column] >= 0]
        frequency_center = np.sum(positive_freqs * positive_magnitudes) / np.sum(positive_magnitudes)
        results.append(frequency_center)
    
    return np.array(results)


def calculate_rms_frequency(fft_magnitudes_df, freqs_df, sampling_rate):
    results = []

    for column in fft_magnitudes_df.columns:
        positive_freqs = freqs_df[column][freqs_df[column] >= 0]
        positive_magnitudes = fft_magnitudes_df[column][freqs_df[column] >= 0]
        rms_frequency = np.sqrt(np.sum((positive_freqs**2) * positive_magnitudes) / np.sum(positive_magnitudes))
        results.append(rms_frequency)
    
    return np.array(results)

def calculate_root_variance_frequency(fft_magnitudes_df, freqs_df, sampling_rate, frequency_centers):
    results = []

    for i, column in enumerate(fft_magnitudes_df.columns):
        positive_freqs = freqs_df[column][freqs_df[column] >= 0]
        positive_magnitudes = fft_magnitudes_df[column][freqs_df[column] >= 0]
        frequency_center = frequency_centers[i]
        root_variance_frequency = np.sqrt(np.sum(((positive_freqs - frequency_center)**2) * positive_magnitudes) / np.sum(positive_magnitudes))
        results.append(root_variance_frequency)
    
    return np.array(results)

def calculate_shannon_entropy(fft_magnitudes_df, freqs_df, sampling_rate):
    results = []

    for column in fft_magnitudes_df.columns:
        positive_freqs = freqs_df[column][freqs_df[column] >= 0]
        positive_magnitudes = fft_magnitudes_df[column][freqs_df[column] >= 0]
        
        # Normalize the magnitudes to get probabilities
        magnitudes_sum = np.sum(positive_magnitudes)
        probabilities = positive_magnitudes / magnitudes_sum
        
        # Calculate Shannon entropy
        shannon_entropy = -np.sum(probabilities * np.log2(probabilities))
        results.append(shannon_entropy)
    
    return np.array(results)



# Shannon Entropisi
#shannon_entropy = calculate_shannon_entropy(fft_magnitudes, freqs, sampling_rate)
#print(f"Shannon Entropisi: {shannon_entropy}")

# Örnek sinyal ve örnekleme frekansı
# Örnek sinyal ve örnekleme frekansı
sampling_rate = 10000
time = np.arange(0, 1.0, 1.0 / sampling_rate)
frequency = 100
fft_magnitudes, freqs = np.sin(2 * np.pi * frequency * time) - (np.cos(8 * np.pi * frequency * time)/2)

# Fourier dönüşümünü al ve genlik spektrumunu hesapla
fft_values = fft(fft_magnitudes, freqs)
fft_magnitudes = np.abs(fft_values)
freqs = fftfreq(len(fft_magnitudes, freqs), d=1/sampling_rate)
positive_freqs = freqs[freqs >= 0]
positive_magnitudes = fft_magnitudes[freqs >= 0]

# Frekans spektrumunu çiz
plt.figure(figsize=(10, 6))
plt.plot(positive_freqs, positive_magnitudes)
plt.title('Frekans Spektrumu')
plt.xlabel('Frekans (Hz)')
plt.ylabel('Genlik')
plt.grid(True)
plt.show()

# Ortalama Frekans
mean_freq = calculate_mean_frequency(fft_magnitudes, freqs, sampling_rate)
print(f"Ortalama Frekans: {mean_freq} Hz")

# Standart Sapma Frekansı
std_freq = calculate_std_frequency(fft_magnitudes, freqs, sampling_rate, mean_freq)
print(f"Standart Sapma Frekansı: {std_freq} Hz")

# Çarpıklık Frekansı
skewness_freq = calculate_skewness_frequency(fft_magnitudes, freqs, sampling_rate, mean_freq, std_freq)
print(f"Çarpıklık Frekansı: {skewness_freq}")

# Basıklık Frekansı
kurtosis_freq = calculate_kurtosis_frequency(fft_magnitudes, freqs, sampling_rate, mean_freq, std_freq)
print(f"Basıklık Frekansı: {kurtosis_freq}")

# Frekans Merkezi
freq_center = calculate_frequency_center(fft_magnitudes, freqs, sampling_rate)
print(f"Frekans Merkezi: {freq_center} Hz")

# Karekök Ortalama Kare Frekans
rms_freq = calculate_rms_frequency(fft_magnitudes, freqs, sampling_rate)
print(f"Karekök Ortalama Kare Frekans: {rms_freq} Hz")

# Karekök Varyans Frekansı
root_variance_freq = calculate_root_variance_frequency(fft_magnitudes, freqs, sampling_rate, freq_center)
print(f"Karekök Varyans Frekansı: {root_variance_freq} Hz")

# Shannon Entropisi
shannon_entropy = calculate_shannon_entropy(fft_magnitudes, freqs, sampling_rate)
print(f"Shannon Entropisi: {shannon_entropy}")