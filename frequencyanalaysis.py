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

def calculate_std_frequency(fft_magnitudes, freqs, sampling_rate, mean_frequency):

    positive_freqs = freqs[freqs >= 0]
    positive_magnitudes = fft_magnitudes[freqs >= 0]
    std_frequency = np.sqrt(np.sum((positive_freqs - mean_frequency)**2 * positive_magnitudes) / np.sum(positive_magnitudes))
    return std_frequency

def calculate_skewness_frequency(fft_magnitudes, freqs, sampling_rate, mean_frequency, std_frequency):

    positive_freqs = freqs[freqs >= 0]
    positive_magnitudes = fft_magnitudes[freqs >= 0]
    skewness_frequency = np.sum((positive_freqs - mean_frequency)**3 * positive_magnitudes) / (np.sum(positive_magnitudes) * std_frequency**3)
    return skewness_frequency

def calculate_kurtosis_frequency(fft_magnitudes, freqs, sampling_rate, mean_frequency, std_frequency):

    positive_freqs = freqs[freqs >= 0]
    positive_magnitudes = fft_magnitudes[freqs >= 0]
    kurtosis_frequency = np.sum((positive_freqs - mean_frequency)**4 * positive_magnitudes) / (np.sum(positive_magnitudes) * std_frequency**4)
    return kurtosis_frequency

def calculate_frequency_center(fft_magnitudes, freqs, sampling_rate):

    positive_freqs = freqs[freqs >= 0]
    positive_magnitudes = fft_magnitudes[freqs >= 0]
    frequency_center = np.sum(positive_freqs * positive_magnitudes) / np.sum(positive_magnitudes)
    return frequency_center

def calculate_rms_frequency(fft_magnitudes, freqs, sampling_rate):

    positive_freqs = freqs[freqs >= 0]
    positive_magnitudes = fft_magnitudes[freqs >= 0]
    rms_frequency = np.sqrt(np.sum((positive_freqs**2) * positive_magnitudes) / np.sum(positive_magnitudes))
    return rms_frequency

def calculate_root_variance_frequency(fft_magnitudes, freqs, sampling_rate, frequency_center):

    positive_freqs = freqs[freqs >= 0]
    positive_magnitudes = fft_magnitudes[freqs >= 0]
    root_variance_frequency = np.sqrt(np.sum(((positive_freqs - frequency_center)**2) * positive_magnitudes) / np.sum(positive_magnitudes))
    return root_variance_frequency


def calculate_shannon_entropy(fft_magnitudes, freqs, sampling_rate):
    fft_values = fft(fft_magnitudes, freqs)
    fft_magnitudes = np.abs(fft_values)
    positive_magnitudes = fft_magnitudes[fftfreq(len(fft_magnitudes, freqs), d=1/sampling_rate) >= 0]
    normalized_magnitudes = positive_magnitudes / np.sum(positive_magnitudes)
    shannon_entropy = -np.sum(normalized_magnitudes * np.log(normalized_magnitudes + 1e-12))  # +1e-12 to avoid log(0)
    return shannon_entropy

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