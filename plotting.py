import numpy as np
import matplotlib.pyplot as plt

def set_plot_properties(title: str, xlabel: str, ylabel: str, legend: bool = True):
    """Set common plot properties to avoid code duplication."""
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend()
    plt.grid(True)

def plot_raw_eeg(time_data: np.ndarray, eeg_data: np.ndarray, title: str) -> None:
    plt.figure(figsize=(15, 5))
    plt.plot(time_data, eeg_data, color='blue', linestyle='-', linewidth=1)
    set_plot_properties(title, "Time (seconds)", "EEG Signal Amplitude")
    plt.show()

def plot_psd(freqs: np.ndarray, psd: np.ndarray, label: str, title: str) -> None:
    plt.figure(figsize=(15, 5))
    plt.semilogy(freqs, psd, label=label)
    set_plot_properties(title, 'Frequency (Hz)', 'Power Spectral Density (mV^2 / Hz)')
    plt.show()

def plot_psd_bands(freqs_bands: list, psd_bands: list, bands_labels: list, title: str) -> None:
    plt.figure(figsize=(15, 10))
    for freqs, psd, label in zip(freqs_bands, psd_bands, bands_labels):
        plt.semilogy(freqs, psd, label=label)
    set_plot_properties(title, "Frequency (Hz)", "Power Spectral Density (mV^2/Hz)")
    plt.show()

def plot_psd_comparison(baseline_freqs: np.ndarray, baseline_psd: np.ndarray, 
                        meditation_segments_freqs: list, meditation_segments_psd: list, 
                        segment_labels: list, title: str) -> None:
    plt.figure(figsize=(15, 5))
    plt.plot(baseline_freqs, baseline_psd, label='Baseline', color='black', linewidth=2)
    colors = plt.cm.jet(np.linspace(0, 1, len(meditation_segments_freqs)))
    for freqs, psd, label, color in zip(meditation_segments_freqs, meditation_segments_psd, segment_labels, colors):
        plt.plot(freqs, psd, label=label, color=color)
    set_plot_properties(title, 'Frequency (Hz)', 'Power Spectral Density (mV^2 / Hz)')
    plt.show()

def plot_alpha_band_power(segmented_alpha_powers):
    plt.figure(figsize=(15, 5))
    plt.bar(range(len(segmented_alpha_powers)), segmented_alpha_powers)
    plt.title("Alpha Band Power Across Different Segments")
    plt.xlabel("Segment (0: Baseline, 1-N: Subsequent Segments)")
    plt.ylabel("Average Alpha Power")
    plt.show()

def plot_psd_baseline_meditation(baseline_freqs, baseline_psd, meditation_freqs, meditation_psd):
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

def plot_filtered_baseline_corrected_data(time_data, eeg_data):
    plt.figure(figsize=(15, 5))
    plt.plot(time_data, eeg_data)
    plt.title("Filtered and Baseline-Corrected EEG Data (Channel 6)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("EEG Signal Amplitude")
    plt.grid(True)
    plt.show()

def plot_psd_alpha_band(fs, eeg_data, alpha_low=8, alpha_high=13):
    from scipy.signal import welch

    alpha_frequencies, alpha_power_spectrum = welch(eeg_data, fs=fs, nperseg=1024)
    alpha_band_mask = (alpha_frequencies >= alpha_low) & (alpha_frequencies <= alpha_high)
    alpha_frequencies = alpha_frequencies[alpha_band_mask]
    alpha_power_spectrum = alpha_power_spectrum[alpha_band_mask]

    plt.figure(figsize=(15, 5))
    plt.semilogy(alpha_frequencies, alpha_power_spectrum)
    plt.title("Power Spectral Density in the Alpha Band (8-13 Hz)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectrum (dB/Hz)")
    plt.grid(True)
    plt.show()

def plot_psd_subplots(baseline_freqs, baseline_psd, segmented_powers):
    num_plots = len(segmented_powers) + 1  # Additional plot for the baseline
    fig, axs = plt.subplots(1, num_plots, figsize=(15, 5), sharey=True)
    if num_plots == 1:
        axs = [axs]  # If only one plot, wrap it in a list

    # Plot the baseline PSD
    axs[0].plot(baseline_freqs, baseline_psd, label='Baseline', color='black', linewidth=2)
    axs[0].set_title('Baseline')
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].set_ylabel('Power Spectral Density (mV^2 / Hz)')

    # Plot the PSD for each meditation segment in subsequent subplots
    for i, (freqs, psd) in enumerate(segmented_powers):
        axs[i + 1].plot(freqs, psd, label=f'Meditation Segment {i+1}', color='tab:blue', linewidth=2)
        axs[i + 1].set_title(f'Meditation Segment {i+1}')
        axs[i + 1].set_xlabel('Frequency (Hz)')

    # Adjust the layout and add a common legend
    fig.legend(*axs[0].get_legend_handles_labels(), loc='upper right')
    plt.tight_layout()
    plt.show()