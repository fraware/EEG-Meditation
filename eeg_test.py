import mne
import pywt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from scipy.signal import butter, filtfilt, welch, iirnotch

"""
================================
Spectral analysis of the Alpha Band
================================

This example shows how to extract the epochs from the dataset of a given
subject and then do a spectral analysis of the signals. The expected behavior
is that there should be a peak around 10 Hz for the 'closed' epochs, due to the
Alpha rhythm that appears when a person closes here eyes.

"""

# Define constants and paths
FS = 250  # Sampling frequency
FILE_PATH = r"C:\Users\mateo\OneDrive\Documents\GAP YEAR 2022-2023\Tech Internship Summer 2024\NextSense\EEG Project\earEEG_recording - no header.TXT"
CHANNEL = '6'  # The channel to be analyzed

# Define frequency bands
BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# Load the data from the EEG file
df = pd.read_csv(FILE_PATH, delim_whitespace=True)


# Extracting the relevant channel (Channel 6) and time data
channel_6_data = df['6']
time_data = df['time']

# Convert the time data to seconds for easier interpretation
time_data_seconds = (time_data - time_data[0]) / 1000

# Plotting the raw data from channel 6
plt.figure(figsize=(15, 5))
plt.plot(time_data_seconds, channel_6_data)
plt.title("Raw EEG Data (Channel 6)")
plt.xlabel("Time (seconds)")
plt.ylabel("EEG Signal Amplitude")
plt.grid(True)
plt.show()


# Sampling frequency
fs = 250

# Extracting the relevant channel (Channel 6) data
channel_6_data = df['6'].values

# Define the frequency bands
delta_band = (1, 4)
theta_band = (4, 8)
alpha_band = (8, 13)
beta_band = (13, 30)
gamma_band = (30, 45)

# Function to calculate power spectral density for a given frequency band
def band_power(data, fs, freq_band):
    freqs, psd = welch(data, fs, nperseg=1024)
    freq_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
    return freqs[freq_mask], psd[freq_mask]

# Calculate PSD for each band
delta_freqs, delta_psd = band_power(channel_6_data, fs, delta_band)
theta_freqs, theta_psd = band_power(channel_6_data, fs, theta_band)
alpha_freqs, alpha_psd = band_power(channel_6_data, fs, alpha_band)
beta_freqs, beta_psd = band_power(channel_6_data, fs, beta_band)
gamma_freqs, gamma_psd = band_power(channel_6_data, fs, gamma_band)

# Plotting the PSD for each frequency band
plt.figure(figsize=(15, 10))
plt.semilogy(delta_freqs, delta_psd, label='Delta Band')
plt.semilogy(theta_freqs, theta_psd, label='Theta Band')
plt.semilogy(alpha_freqs, alpha_psd, label='Alpha Band')
plt.semilogy(beta_freqs, beta_psd, label='Beta Band')
plt.semilogy(gamma_freqs, gamma_psd, label='Gamma Band')

plt.title("Power Spectral Density across Different Frequency Bands")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density (mV^2/Hz)")
plt.legend()
plt.show()

# The graph you've provided shows the Power Spectral Density (PSD) across different frequency bands for EEG data, plotted on a logarithmic scale. This type of graph is valuable for identifying the distribution of power across different frequency ranges, which can indicate various states of brain activity or the presence of artifacts.

# Calculate the PSD for the baseline (first 3 minutes) and meditation period (remaining 15 minutes)
baseline_data = channel_6_data[:3 * 60 * fs]  # First 3 minutes for baseline
meditation_data = channel_6_data[3 * 60 * fs:]  # Rest of the data for meditation

# Function to calculate PSD using Welch's method
def calculate_psd(data, fs):
    freqs, psd = welch(data, fs, nperseg=1024)
    return freqs, psd

# Calculate PSD for baseline and meditation data
baseline_freqs, baseline_psd = calculate_psd(baseline_data, fs)
meditation_freqs, meditation_psd = calculate_psd(meditation_data, fs)

# Plot the PSD for baseline and meditation periods
plt.figure(figsize=(15, 5))

# Plot baseline
plt.semilogy(baseline_freqs, baseline_psd, label='Baseline')

# Plot meditation
plt.semilogy(meditation_freqs, meditation_psd, label='Meditation')

# Adding labels and title
plt.title('Power Spectral Density during Baseline and Meditation')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (mV^2 / Hz)')
plt.legend()
plt.tight_layout()
plt.show()


# High-pass filter using filtfilt for zero-phase filtering
highpass_cutoff = 1
b_highpass, a_highpass = butter(N=6, Wn=highpass_cutoff/(0.5*fs), btype='high')
highpass_filtered_data = filtfilt(b_highpass, a_highpass, channel_6_data)

# Notch filter using filtfilt for zero-phase filtering
notch_freq = 60  # Powerline frequency in Hz (US Based)
quality_factor = 30  # Quality factor for the notch filter
b_notch, a_notch = iirnotch(notch_freq/(0.5*fs), quality_factor)
notch_filtered_data = filtfilt(b_notch, a_notch, highpass_filtered_data)

# Low-pass filter using filtfilt for zero-phase filtering
lowpass_cutoff = 13  # Cutoff frequency just above the alpha range
b_lowpass, a_lowpass = butter(N=5, Wn=lowpass_cutoff/(0.5*fs), btype='low')
lowpass_filtered_data = filtfilt(b_lowpass, a_lowpass, notch_filtered_data)

# Baseline Correction after artifact removal
baseline_samples = 3 * 60 * fs  # 3 minutes in samples
baseline_corrected_data = lowpass_filtered_data - np.mean(lowpass_filtered_data[:baseline_samples])

# Function to calculate maximum level of wavelet decomposition
def max_wavelet_level(data_length, wavelet='db4'):
    return pywt.dwt_max_level(data_length, pywt.Wavelet(wavelet).dec_len)

# Artifact rejection using wavelet with adaptive thresholding
def wavelet_artifact_rejection(data, wavelet='db4'):
    max_level = max_wavelet_level(len(data), wavelet)
    coeffs = pywt.wavedec(data, wavelet, level=max_level)
    
    # Adaptive threshold based on the median absolute deviation (MAD)
    threshold = np.median(np.abs(coeffs[-1] - np.median(coeffs[-1]))) / 0.6745
    threshold *= np.sqrt(2 * np.log(len(data)))
    
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
    
    reconstructed_signal = pywt.waverec(coeffs, wavelet)
    return reconstructed_signal

# Apply improved wavelet artifact rejection
artifact_rejected_data = wavelet_artifact_rejection(baseline_corrected_data)

# Plotting the filtered and baseline-corrected data
plt.figure(figsize=(15, 5))
plt.plot(time_data_seconds, artifact_rejected_data)
plt.title("Filtered and Baseline-Corrected EEG Data (Channel 6)")
plt.xlabel("Time (seconds)")
plt.ylabel("EEG Signal Amplitude")
plt.grid(True)
plt.show()


# Calculate PSD for baseline and meditation data
baseline_data = artifact_rejected_data[:3 * 60 * fs]  # First 3 minutes for baseline
meditation_data = artifact_rejected_data[3 * 60 * fs:]  # Rest of the data for meditation

baseline_freqs, baseline_psd = calculate_psd(baseline_data, fs)
meditation_freqs, meditation_psd = calculate_psd(meditation_data, fs)

# Plot the PSD for baseline and meditation periods
plt.figure(figsize=(15, 5))

# Plot baseline
plt.semilogy(baseline_freqs, baseline_psd, label='Baseline')

# Plot meditation
plt.semilogy(meditation_freqs, meditation_psd, label='Meditation')

# Adding labels and title
plt.title('Power Spectral Density during Baseline and Meditation')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (mV^2 / Hz)')
plt.legend()
plt.tight_layout()
plt.show()





# Perform Spectral Decomposition

# Feature Extraction: Calculating PSD in the alpha band (8-13 Hz)
alpha_low = 8
alpha_high = 13
alpha_frequencies, alpha_power_spectrum = welch(artifact_rejected_data, fs=fs, nperseg=1024) # 1024 samples per segment
alpha_band_mask = (alpha_frequencies >= alpha_low) & (alpha_frequencies <= alpha_high)
alpha_frequencies = alpha_frequencies[alpha_band_mask]
alpha_power_spectrum = alpha_power_spectrum[alpha_band_mask]

# Plotting the PSD within the alpha band
plt.figure(figsize=(15, 5))
plt.semilogy(alpha_frequencies, alpha_power_spectrum)
plt.title("Power Spectral Density in the Alpha Band (8-13 Hz)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectrum (dB/Hz)")
plt.grid(True)
plt.show()

# Segmentation and Alpha Wave Analysis
baseline_duration_seconds = 3 * 60
num_samples_baseline = baseline_duration_seconds * fs
remaining_data = baseline_corrected_data[num_samples_baseline:]
segment_length_seconds = baseline_duration_seconds
total_remaining_seconds = len(remaining_data) / fs
num_segments = int(total_remaining_seconds // segment_length_seconds)
segmented_alpha_powers = []
for i in range(num_segments):
    start_sample = i * segment_length_seconds * fs
    end_sample = (i + 1) * segment_length_seconds * fs
    segment_data = remaining_data[start_sample:end_sample]
    segment_frequencies, segment_power_spectrum = welch(segment_data, fs=fs, nperseg=1024)
    alpha_band_mask = (segment_frequencies >= alpha_low) & (segment_frequencies <= alpha_high)
    segment_alpha_power = segment_power_spectrum[alpha_band_mask].mean()
    segmented_alpha_powers.append(segment_alpha_power)
baseline_alpha_power = alpha_power_spectrum.mean()
segmented_alpha_powers.insert(0, baseline_alpha_power)

# Plotting the alpha band power for each segment including baseline
plt.figure(figsize=(15, 5))
plt.bar(range(len(segmented_alpha_powers)), segmented_alpha_powers)
plt.title("Alpha Band Power Across Different Segments")
plt.xlabel("Segment (0: Baseline, 1-N: Subsequent Segments)")
plt.ylabel("Average Alpha Power")
plt.show()


# Define the baseline and meditation data again for clarity
baseline_data = artifact_rejected_data[:3 * 60 * fs]
meditation_data = artifact_rejected_data[3 * 60 * fs:]

# Calculate PSD for the baseline
baseline_freqs, baseline_psd = welch(baseline_data, fs=fs, nperseg=min(1024, len(baseline_data)))
baseline_alpha_mask = (baseline_freqs >= alpha_low) & (baseline_freqs <= alpha_high)
baseline_alpha_psd = baseline_psd[baseline_alpha_mask]
baseline_alpha_freqs = baseline_freqs[baseline_alpha_mask]

# Calculate PSD for each meditation segment and overlay it with the baseline
plt.figure(figsize=(15, 5))

# Plot the baseline PSD
plt.plot(baseline_alpha_freqs, baseline_alpha_psd, label='Baseline', color='black', linewidth=2)

# Calculate the number of 3-minute segments in the meditation data
num_meditation_segments = len(meditation_data) // (3 * 60 * fs)

# Define colors for each meditation segment
colors = plt.cm.jet(np.linspace(0, 1, num_meditation_segments))

# Plot PSD for each meditation segment
for i in range(num_meditation_segments):
    start = i * 3 * 60 * fs
    end = (i + 1) * 3 * 60 * fs
    segment_data = meditation_data[start:end]
    segment_freqs, segment_psd = welch(segment_data, fs=fs, nperseg=min(1024, len(segment_data)))
    segment_alpha_mask = (segment_freqs >= alpha_low) & (segment_freqs <= alpha_high)
    segment_alpha_psd = segment_psd[segment_alpha_mask]
    segment_alpha_freqs = segment_freqs[segment_alpha_mask]

    # Plot the segment PSD
    plt.plot(segment_alpha_freqs, segment_alpha_psd, label=f'Meditation Segment {i+1}', color=colors[i])

# Adding labels and title
plt.title('PSD Comparison between Baseline and Meditation Segments')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (mV^2 / Hz)')

# Adding a legend
plt.legend()

# Show the plot
plt.show()


# Calculate the PSD for the baseline (first 3 minutes)
baseline_data = artifact_rejected_data[:3 * 60 * fs]
baseline_freqs, baseline_psd = welch(baseline_data, fs=fs, nperseg=min(1024, len(baseline_data)))
baseline_alpha_mask = (baseline_freqs >= alpha_low) & (baseline_freqs <= alpha_high)
baseline_alpha_psd = baseline_psd[baseline_alpha_mask]
baseline_alpha_freqs = baseline_freqs[baseline_alpha_mask]

# Initialize lists to store PSD values and errors for meditation segments
segmented_alpha_powers = []
segmented_alpha_se = []

# Calculate the number of full 3-minute meditation segments after the baseline
meditation_data = artifact_rejected_data[3 * 60 * fs:]
num_full_segments = len(meditation_data) // (3 * 60 * fs)

# Analyze each 3-minute meditation segment and compare to baseline
for i in range(num_full_segments):
    start = i * 3 * 60 * fs
    end = (i + 1) * 3 * 60 * fs
    segment_data = meditation_data[start:end]
    segment_freqs, segment_psd = welch(segment_data, fs=fs, nperseg=min(1024, len(segment_data)))
    segment_alpha_mask = (segment_freqs >= alpha_low) & (segment_freqs <= alpha_high)
    segment_alpha_psd = segment_psd[segment_alpha_mask]
    
    mean_psd = np.mean(segment_alpha_psd)
    se_psd = np.std(segment_alpha_psd) / np.sqrt(len(segment_alpha_psd))
    
    segmented_alpha_powers.append(segment_alpha_psd)
    segmented_alpha_se.append(se_psd)

# Create subplots for baseline and each meditation segment
num_plots = num_full_segments + 1  # Additional plot for the baseline
fig, axs = plt.subplots(1, num_plots, figsize=(15, 5), sharey=True)
if num_plots == 1:
    axs = [axs]  # If only one plot, wrap it in a list

# Plot the baseline PSD
axs[0].plot(baseline_alpha_freqs, baseline_alpha_psd, label='Baseline', color='black', linewidth=2)
axs[0].set_title('Baseline')
axs[0].set_xlabel('Frequency (Hz)')
axs[0].set_ylabel('Power Spectral Density (mV^2 / Hz)')

# Plot the PSD for each meditation segment in subsequent subplots
for i in range(num_full_segments):
    axs[i + 1].plot(segment_freqs[segment_alpha_mask], segmented_alpha_powers[i], label=f'Meditation Segment {i+1}', color='tab:blue', linewidth=2)
    axs[i + 1].set_title(f'Meditation Segment {i+1}')
    axs[i + 1].set_xlabel('Frequency (Hz)')

# Adjust the layout and add a common legend
fig.legend(*axs[0].get_legend_handles_labels(), loc='upper right')
plt.tight_layout()
plt.show()


# Statistical Analysis

# Define the baseline period and the meditation period
baseline_period = artifact_rejected_data[:3 * 60 * fs]
meditation_period = artifact_rejected_data[3 * 60 * fs:]

# Calculate PSD for the baseline period
baseline_freqs, baseline_psd = welch(baseline_period, fs=fs, nperseg=min(1024, len(baseline_period)))
baseline_alpha_mask = (baseline_freqs >= alpha_low) & (baseline_freqs <= alpha_high)
baseline_alpha_psd = baseline_psd[baseline_alpha_mask]

# Calculate PSD for the meditation period
meditation_freqs, meditation_psd = welch(meditation_period, fs=fs, nperseg=min(1024, len(meditation_period)))
meditation_alpha_mask = (meditation_freqs >= alpha_low) & (meditation_freqs <= alpha_high)
meditation_alpha_psd = meditation_psd[meditation_alpha_mask]

# ANOVA between baseline and entire meditation period
anova_result_entire = f_oneway(baseline_alpha_psd, meditation_alpha_psd)
anova_p_value_entire = anova_result_entire.pvalue

# Calculate the number of full 3-minute segments in the meditation data
num_segments = len(meditation_period) // (3 * 60 * fs)

# Function to calculate alpha power for each segment
def calculate_segment_powers(data, segment_length, fs, alpha_low, alpha_high):
    num_segments = int(len(data) / segment_length)
    segment_powers = []
    for i in range(num_segments):
        segment = data[i * segment_length:(i + 1) * segment_length]
        freqs, psd = welch(segment, fs=fs, nperseg=1024)
        alpha_mask = (freqs >= alpha_low) & (freqs <= alpha_high)
        alpha_power = psd[alpha_mask].mean()
        segment_powers.append(alpha_power)
    return segment_powers

# Calculate alpha power for each segment
segment_length_samples = 3 * 60 * fs
meditation_segment_powers = calculate_segment_powers(meditation_period, segment_length_samples, fs, alpha_low, alpha_high)

# Perform ANOVA for each meditation segment compared to baseline
anova_p_values_segments = []
for segment_power in meditation_segment_powers:
    anova_result_segment = f_oneway(baseline_alpha_psd, [segment_power] * len(baseline_alpha_psd))
    anova_p_values_segments.append(anova_result_segment.pvalue)

# Print ANOVA results
print(f"ANOVA p-value between baseline and entire meditation period: {anova_p_value_entire:.16f}")
for i, p_value in enumerate(anova_p_values_segments, 1):
    print(f"ANOVA p-value between baseline and meditation segment {i}: {p_value:.16f}")

