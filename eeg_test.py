"""
====================================================
Spectral & Statistical Analysis of the Alpha Band
====================================================
"""

from constants import FS, FILE_PATH, CHANNEL, BANDS
from data_loading import load_data, get_channel_data, get_time_data
from filtering import highpass_filter, lowpass_filter, notch_filter, baseline_correction, wavelet_artifact_rejection
from plotting import plot_raw_eeg, plot_psd_bands, plot_psd_comparison, plot_psd_baseline_meditation,plot_filtered_baseline_corrected_data, plot_psd_alpha_band, plot_alpha_band_power, plot_psd_subplots
from spectral_analysis import calculate_psd, band_power, mean_band_power, calculate_band_powers, segment_analysis, segment_and_analyze_alpha_power, calculate_segmented_psd
from statistical_analysis import print_anova_results

def main():
    # Load data
    df = load_data(FILE_PATH)
    channel_data = get_channel_data(df, CHANNEL)
    time_data_seconds = get_time_data(df, 'time')

    # Plot raw EEG data
    plot_raw_eeg(time_data_seconds, channel_data, "Raw EEG Data (Channel 6)")

    # Calculate and plot PSD for each frequency band
    band_powers = calculate_band_powers(channel_data, FS, BANDS)
    plot_psd_bands([freqs for freqs, _ in band_powers.values()],
                   [psd for _, psd in band_powers.values()],
                   BANDS.keys(), "Power Spectral Density across Different Frequency Bands")

    # Calculate and plot PSD for baseline and meditation data
    baseline_data = channel_data[:3 * 60 * FS]
    meditation_data = channel_data[3 * 60 * FS:]
    baseline_freqs, baseline_psd = calculate_psd(baseline_data, FS)
    meditation_freqs, meditation_psd = calculate_psd(meditation_data, FS)
    plot_psd_baseline_meditation(baseline_freqs, baseline_psd, meditation_freqs, meditation_psd)

    # Data Filtering and Preprocessing
    filtered_data = highpass_filter(channel_data, 1, FS)
    filtered_data = notch_filter(filtered_data, 60, FS)
    filtered_data = lowpass_filter(filtered_data, 13, FS)
    baseline_corrected_data = baseline_correction(filtered_data, 3 * 60 * FS)
    artifact_rejected_data = wavelet_artifact_rejection(baseline_corrected_data)

    # Plot filtered and baseline-corrected EEG data
    plot_filtered_baseline_corrected_data(time_data_seconds, artifact_rejected_data)

    # Feature Extraction: Calculating PSD in the alpha band (8-13 Hz)
    plot_psd_alpha_band(FS, artifact_rejected_data)


    # Calculate PSD for filtered baseline and meditation data
    baseline_filtered_data = artifact_rejected_data[:3 * 60 * FS]
    meditation_filtered_data = artifact_rejected_data[3 * 60 * FS:]
    baseline_freqs, baseline_psd = calculate_psd(baseline_filtered_data, FS)
    meditation_freqs, meditation_psd = calculate_psd(meditation_filtered_data, FS)

    # Extract baseline alpha band frequencies and PSD
    alpha_low, alpha_high = BANDS['alpha']
    baseline_alpha_mask = (baseline_freqs >= alpha_low) & (baseline_freqs <= alpha_high)
    baseline_alpha_psd = baseline_psd[baseline_alpha_mask]
    baseline_alpha_freqs = baseline_freqs[baseline_alpha_mask]

    meditation_alpha_mask = (meditation_freqs >= alpha_low) & (meditation_freqs <= alpha_high)
    meditation_alpha_psd = meditation_psd[meditation_alpha_mask]
    meditation_alpha_freqs = meditation_freqs[meditation_alpha_mask]

    # Plot PSD for baseline and meditation
    plot_psd_baseline_meditation(baseline_freqs, baseline_psd, meditation_freqs, meditation_psd)

    # Segmentation and Alpha Wave Analysis
    segmented_alpha_powers = segment_and_analyze_alpha_power(baseline_corrected_data, FS, 3 * 60, alpha_low, alpha_high)

    # Plotting the alpha band power for each segment including baseline
    plot_alpha_band_power(segmented_alpha_powers)

    # Plot PSD for baseline and meditation in the alpha band
    plot_psd_comparison(baseline_alpha_freqs, baseline_alpha_psd, 
                        [meditation_alpha_freqs], [meditation_alpha_psd], 
                        ["Meditation"], "PSD Comparison between Baseline and Meditation Segments")

    # Calculate segmented PSD and errors
    segmented_powers, segmented_se = calculate_segmented_psd(meditation_filtered_data, FS, 3 * 60, alpha_low, alpha_high)

    # Create subplots for baseline and each meditation segment
    plot_psd_subplots(baseline_alpha_freqs, baseline_alpha_psd, segmented_powers)

    # Segment and analyze alpha band power in meditation segments
    segmented_alpha_power = segment_analysis(meditation_filtered_data, FS, BANDS['alpha'], 3 * 60)

    # Perform and print ANOVA results
    print_anova_results(baseline_filtered_data, meditation_filtered_data, segmented_alpha_power)

if __name__ == "__main__":
    main()
