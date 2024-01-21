# Define constants and configurations
FS = 250 # Sampling frequency
FILE_PATH = r"earEEG_recording.TXT"
CHANNEL = '6' # The channel to be analyzed

# Define frequency bands
BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}
