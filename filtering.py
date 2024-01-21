from scipy.signal import butter, filtfilt, iirnotch
import numpy as np
import pywt

def create_butter_filter(cutoff: float, fs: float, order: int = 6, btype: str = 'low') -> tuple:
    """
    Create a Butterworth filter.

    Parameters:
    cutoff (float): The cutoff frequency of the filter.
    fs (float): The sampling rate.
    order (int): The order of the filter.
    btype (str): The type of filter ('low' or 'high').

    Returns:
    tuple: Filter coefficients (b, a).
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return b, a

def apply_filter(data: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Apply the given filter to the data.

    Parameters:
    data (np.ndarray): The input data.
    b (np.ndarray): Numerator (b) coefficients of the filter.
    a (np.ndarray): Denominator (a) coefficients of the filter.

    Returns:
    np.ndarray: Filtered data.
    """
    return filtfilt(b, a, data)

def highpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 6) -> np.ndarray:
    """
    Apply a high-pass Butterworth filter.

    Parameters:
    data (np.ndarray): The input data.
    cutoff (float): The cutoff frequency.
    fs (float): The sampling rate.
    order (int): The order of the filter.

    Returns:
    np.ndarray: High-pass filtered data.
    """
    b, a = create_butter_filter(cutoff, fs, order, 'high')
    return apply_filter(data, b, a)

def lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 6) -> np.ndarray:
    """
    Apply a low-pass Butterworth filter.

    Parameters:
    data (np.ndarray): The input data.
    cutoff (float): The cutoff frequency.
    fs (float): The sampling rate.
    order (int): The order of the filter.

    Returns:
    np.ndarray: Low-pass filtered data.
    """
    b, a = create_butter_filter(cutoff, fs, order, 'low')
    return apply_filter(data, b, a)

def notch_filter(data: np.ndarray, freq: float, fs: float, Q: float = 30) -> np.ndarray:
    """
    Apply a notch filter.

    Parameters:
    data (np.ndarray): The input data.
    freq (float): The frequency to notch out.
    fs (float): The sampling rate.
    Q (float): Quality factor of the notch filter.

    Returns:
    np.ndarray: Notch filtered data.
    """
    nyq = 0.5 * fs
    normalized_freq = freq / nyq
    b, a = iirnotch(normalized_freq, Q)
    return apply_filter(data, b, a)

def baseline_correction(data: np.ndarray, baseline_samples: int) -> np.ndarray:
    """
    Perform baseline correction on the data.

    Parameters:
    data (np.ndarray): The input data.
    baseline_samples (int): Number of samples to use for the baseline.

    Returns:
    np.ndarray: Baseline corrected data.
    """
    baseline_value = np.mean(data[:baseline_samples])
    return data - baseline_value

def wavelet_artifact_rejection(data: np.ndarray, wavelet: str = 'db4') -> np.ndarray:
    """
    Apply wavelet-based artifact rejection.

    Parameters:
    data (np.ndarray): The input data.
    wavelet (str): The type of wavelet to use.

    Returns:
    np.ndarray: Data after wavelet artifact rejection.
    """
    max_level = pywt.dwt_max_level(len(data), pywt.Wavelet(wavelet).dec_len)
    coeffs = pywt.wavedec(data, wavelet, level=max_level)
    threshold = np.median(np.abs(coeffs[-1] - np.median(coeffs[-1]))) / 0.6745
    threshold *= np.sqrt(2 * np.log(len(data)))
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
    return pywt.waverec(coeffs, wavelet)
