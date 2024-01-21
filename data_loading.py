import pandas as pd
from typing import Union, List

def remove_header(file_path: str) -> str:
    """
    Remove header from the file and save to a temporary file. 

    Parameters:
    file_path (str): The path to the EEG data file.

    """
    temp_file_path = file_path + '.tmp'  # Temporary file
    with open(file_path, 'r') as infile, open(temp_file_path, 'w') as outfile:
        for line in infile:
            if line.startswith('#column_names:'):
                # Remove '#column_names:' but keep the rest of the line
                outfile.write(line.replace('#column_names:', '').strip() + '\n')
            elif not line.startswith('#'):
                outfile.write(line)
    return temp_file_path

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load EEG data from a text file.

    Parameters:
    file_path (str): The path to the EEG data file.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the EEG data.

    Raises:
    FileNotFoundError: If the file at file_path does not exist.
    """
    try:
        temp_file_path = remove_header(file_path)  # Remove header
        df = pd.read_csv(temp_file_path, delim_whitespace=True)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    except Exception as e:
        raise Exception(f"An error occurred while loading the data: {e}")

def get_channel_data(df: pd.DataFrame, channel: Union[str, int]) -> pd.Series:
    """
    Extract channel data from the DataFrame.

    Parameters:
    df (pd.DataFrame): The EEG data DataFrame.
    channel (str or int): The channel column name or index.

    Returns:
    pd.Series: The extracted channel data.
    """
    if channel not in df.columns:
        raise ValueError(f"Channel '{channel}' not found in DataFrame.")
    return df[channel]

def get_time_data(df: pd.DataFrame, time_column: str) -> pd.Series:
    """
    Extract time data from the DataFrame and convert to seconds.

    Parameters:
    df (pd.DataFrame): The EEG data DataFrame.
    time_column (str): The time column name.

    Returns:
    pd.Series: The extracted time data in seconds.
    """
    if time_column not in df.columns:
        raise ValueError(f"Time column '{time_column}' not found in DataFrame.")
    time_data = df[time_column]
    if not pd.api.types.is_numeric_dtype(time_data):
        raise ValueError("Time data must be numeric.")
    return (time_data - time_data.iloc[0]) / 1000
