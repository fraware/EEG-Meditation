import numpy as np
from scipy.signal import welch

def band_power(data: np.ndarray, fs: int, freq_band: tuple) -> tuple:
    """
    Calculate power spectral density (PSD) for a given frequency band using Welch's method.

    Parameters:
    data (np.ndarray): The signal data.
    fs (int): Sampling frequency.
    freq_band (tuple): Frequency band specified as (low_freq, high_freq).

    Returns:
    tuple: A tuple of arrays (frequencies, PSD values) within the specified frequency band.
    """
    freqs, psd = welch(data, fs, nperseg=1024)
    freq_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
    return freqs[freq_mask], psd[freq_mask]

def calculate_psd(data: np.ndarray, fs: int) -> tuple:
    """
    Calculate the Power Spectral Density (PSD) of a signal using Welch's method.

    Parameters:
    data (np.ndarray): The signal data.
    fs (int): Sampling frequency.

    Returns:
    tuple: A tuple of arrays (frequencies, PSD values) for the entire frequency spectrum.
    """
    return welch(data, fs, nperseg=1024)

def mean_band_power(data: np.ndarray, fs: int, freq_band: tuple) -> float:
    """
    Calculate the mean spectral power within a specific frequency band.

    Parameters:
    data (np.ndarray): The signal data.
    fs (int): Sampling frequency.
    freq_band (tuple): Frequency band specified as (low_freq, high_freq).

    Returns:
    float: The mean power spectral density within the specified frequency band.
    """
    freqs, psd = calculate_psd(data, fs)
    band_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
    return np.mean(psd[band_mask])

def calculate_band_powers(data: np.ndarray, fs: int, bands: dict) -> dict:
    """
    Calculate the power spectral density for each specified frequency band.

    Parameters:
    data (np.ndarray): The signal data.
    fs (int): Sampling frequency.
    bands (dict): A dictionary with band names as keys and frequency bands as values.

    Returns:
    dict: A dictionary with band names as keys and tuples (frequencies, PSD values) as values.
    """
    band_powers = {}
    for band_name, freq_band in bands.items():
        band_powers[band_name] = band_power(data, fs, freq_band)
    return band_powers

def segment_analysis(data: np.ndarray, fs: int, freq_band: tuple, segment_length_sec: int) -> list:
    """
    Segment data and analyze mean power in a specific frequency band for each segment.

    Parameters:
    data (np.ndarray): The signal data.
    fs (int): Sampling frequency.
    freq_band (tuple): Frequency band specified as (low_freq, high_freq).
    segment_length_sec (int): Length of each segment in seconds.

    Returns:
    list: A list of mean power values for each segment within the specified frequency band.
    """
    num_segments = int(len(data) / (segment_length_sec * fs))
    segment_powers = []
    for i in range(num_segments):
        segment_start = i * segment_length_sec * fs
        segment_end = segment_start + segment_length_sec * fs
        segment_power = mean_band_power(data[segment_start:segment_end], fs, freq_band)
        segment_powers.append(segment_power)
    return segment_powers

def segment_and_analyze_alpha_power(data: np.ndarray, fs: int, baseline_duration_seconds: int, alpha_low: float, alpha_high: float) -> list:
    """
    Segment data, analyze, and calculate the mean alpha power for each segment including baseline.

    Parameters:
    data (np.ndarray): The signal data.
    fs (int): Sampling frequency.
    baseline_duration_seconds (int): Duration of the baseline segment in seconds.
    alpha_low (float): Lower bound of the alpha frequency band.
    alpha_high (float): Upper bound of the alpha frequency band.

    Returns:
    list: A list of mean alpha power values for each segment including the baseline.
    """
    num_samples_baseline = baseline_duration_seconds * fs
    remaining_data = data[num_samples_baseline:]
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

    baseline_data = data[:num_samples_baseline]
    baseline_freqs, baseline_psd = welch(baseline_data, fs=fs, nperseg=1024)
    baseline_alpha_mask = (baseline_freqs >= alpha_low) & (baseline_freqs <= alpha_high)
    baseline_alpha_power = baseline_psd[baseline_alpha_mask].mean()
    segmented_alpha_powers.insert(0, baseline_alpha_power)

    return segmented_alpha_powers

def calculate_segmented_psd(data: np.ndarray, fs: int, segment_length: int, alpha_low: float, alpha_high: float) -> tuple:
    """
    Segment data and calculate the Power Spectral Density (PSD) and standard error in the alpha band for each segment.

    Parameters:
    data (np.ndarray): The signal data.
    fs (int): Sampling frequency.
    segment_length (int): Length of each segment in seconds.
    alpha_low (float): Lower bound of the alpha frequency band.
    alpha_high (float): Upper bound of the alpha frequency band.

    Returns:
    tuple: A tuple containing two lists, one for the PSD and one for the standard errors for each segment in the alpha band.
    """
    num_segments = len(data) // (segment_length * fs)
    segmented_powers = []
    segmented_se = []

    for i in range(num_segments):
        start = i * segment_length * fs
        end = (i + 1) * segment_length * fs
        segment_data = data[start:end]
        segment_freqs, segment_psd = welch(segment_data, fs=fs, nperseg=min(1024, len(segment_data)))
        alpha_mask = (segment_freqs >= alpha_low) & (segment_freqs <= alpha_high)
        alpha_psd = segment_psd[alpha_mask]
        
        mean_psd = np.mean(alpha_psd)
        se_psd = np.std(alpha_psd) / np.sqrt(len(alpha_psd))
        
        segmented_powers.append((segment_freqs[alpha_mask], alpha_psd))
        segmented_se.append(se_psd)

    return segmented_powers, segmented_se
