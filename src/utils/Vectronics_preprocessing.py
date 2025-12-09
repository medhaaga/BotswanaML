import numpy as np
import pandas as pd
import json

import config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from datetime import timedelta

from src.utils.data_prep import (combined_annotations,
                                adjust_behavior_and_durations,)

from src.utils.io import (get_video_labels_path,
                          get_audio_labels_path,
                          get_matched_data_path,
                          get_matched_metadata_path,
                          )

def load_annotations():
    # load matched acceleration and label pairs data, metadata, and summary

    all_annotations = combined_annotations(video_path=get_video_labels_path(), 
                                            audio_path=get_audio_labels_path(),
                                            id_mapping=config.id_mapping) # load annotations 

    all_annotations.Timestamp_start = pd.to_datetime(all_annotations.Timestamp_start)
    all_annotations.Timestamp_end = pd.to_datetime(all_annotations.Timestamp_end)
    all_annotations['duration'] = (all_annotations.Timestamp_end - all_annotations.Timestamp_start).dt.total_seconds()

    #extract behaviors of interest
    all_annotations['Behavior'] = all_annotations['Behavior'].replace(config.SUMMARY_COLLAPSE_BEHAVIORS_MAPPING) # collapse behaviors
    all_annotations = all_annotations[all_annotations['Behavior'].isin(config.SUMMARY_BEHAVIORS)]

    return all_annotations

def load_Vectronics_data_metadata():

    acc_data = pd.read_csv(get_matched_data_path())
    acc_metadata = pd.read_csv(get_matched_metadata_path())

    # convert acceleration strings to arrays
    acc_data['acc_x'] = acc_data['acc_x'].apply(json.loads)
    acc_data['acc_y'] = acc_data['acc_y'].apply(json.loads)
    acc_data['acc_z'] = acc_data['acc_z'].apply(json.loads)

    # convert timestamps to datetime objects
    acc_data['behavior_start'] = pd.to_datetime(acc_data['behavior_start'])
    acc_data['behavior_end'] = pd.to_datetime(acc_data['behavior_end'])
    acc_data['UTC date [yyyy-mm-dd]'] = pd.to_datetime(acc_data['UTC date [yyyy-mm-dd]'], format='%Y-%m-%d')

    # extract behaviors of interest
    acc_data = adjust_behavior_and_durations(acc_data, config.SUMMARY_COLLAPSE_BEHAVIORS_MAPPING, config.SUMMARY_BEHAVIORS)
    acc_metadata = acc_metadata.loc[acc_data.index]
    acc_data.reset_index()
    acc_metadata.reset_index()

    return acc_data, acc_metadata


def windowed_ptp_stats(arr, sub_window):
    """
    arr: numpy array of shape (W,)
    sub_window: integer (e.g., 2 * sampling_rate)

    Returns (max_ptp, mean_ptp) as float32
    """
    arr = np.asarray(arr)

    n = arr.shape[0]
    n_full = n // sub_window
    if n_full == 0:
        return np.float32(np.nan), np.float32(np.nan)

    # reshape to (num_windows, window_size)
    trimmed = arr[:n_full * sub_window].reshape(n_full, sub_window)

    ptp_vals = trimmed.ptp(axis=1)
    return np.float32(ptp_vals.max()), np.float32(ptp_vals.mean())

def process_column_vectorized(arr_matrix, sampling_rate=16):
    """
    arr_matrix: shape (num_windows, window_len)
    Returns dict with ptp_max and ptp_mean for all windows
    """

    sub_window = 2 * sampling_rate
    num_windows = arr_matrix.shape[0]

    ptp_max = np.zeros(num_windows, dtype=np.float32)
    ptp_mean = np.zeros(num_windows, dtype=np.float32)

    for i in range(num_windows):
        mx, mn = windowed_ptp_stats(arr_matrix[i], sub_window)
        ptp_max[i] = mx
        ptp_mean[i] = mn

    return ptp_max, ptp_mean

def split_row(row, chunk_size=480, sampling_rate=16):
    window_duration = chunk_size/sampling_rate
    length = len(row['acc_x'])
    n_chunks = length // chunk_size
    remainder = length % chunk_size

    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size

        chunks.append({
            'animal_id': row['individual ID'],
            'UTC date [yyyy-mm-dd]': row['UTC date [yyyy-mm-dd]'],
            'UTC time [yyyy-mm-dd HH:MM:SS]': pd.to_datetime(row['behavior_start']) + timedelta(seconds=i * window_duration),
            'behavior': row['behavior'],
            'acc_x': row['acc_x'][start:end],
            'acc_y': row['acc_y'][start:end],
            'acc_z': row['acc_z'][start:end],
            'duration': chunk_size / sampling_rate,
            'Source': row['Source'],
            'Confidence (H-M-L)': row['Confidence (H-M-L)'],
            'Eating intensity': row['Eating intensity']
        })

    if remainder > 0:
        chunks.append({
            'animal_id': row['individual ID'],
            'UTC date [yyyy-mm-dd]': row['UTC date [yyyy-mm-dd]'],
            'UTC time [yyyy-mm-dd HH:MM:SS]': pd.to_datetime(row['behavior_start']) + timedelta(seconds=n_chunks * window_duration),
            'behavior': row['behavior'],
            'acc_x': row['acc_x'][-remainder:],
            'acc_y': row['acc_y'][-remainder:],
            'acc_z': row['acc_z'][-remainder:],
            'duration': remainder / sampling_rate,
            'Source': row['Source'],
            'Confidence (H-M-L)': row['Confidence (H-M-L)'],
            'Eating intensity': row['Eating intensity']
        })

    return chunks

def create_max_windows(acc_data, window_duration=30.0, sampling_rate=16):
    # Apply the splitting to all rows and flatten the result
    split_chunks = []
    for _, row in acc_data.iterrows():
        split_chunks.extend(split_row(row, chunk_size=int(window_duration*sampling_rate), sampling_rate=sampling_rate))

    # Create the new DataFrame
    acc_data_split = pd.DataFrame(split_chunks)
    return acc_data_split


def add_ptp_features(df, sampling_rate=16):
    for col in ["acc_x", "acc_y", "acc_z"]:

        arr_matrix = np.stack(df[col].values)  # shape (num_windows, window_len)

        ptp_max, ptp_mean = process_column_vectorized(arr_matrix, sampling_rate)

        df[f"{col}_ptp_max"] = ptp_max
        df[f"{col}_ptp_mean"] = ptp_mean

    return df

def create_summary_data(acc_data_split, sampling_rate=16):
    acc_data_split['acc_x_mean'] = acc_data_split['acc_x'].apply(np.mean)
    acc_data_split['acc_y_mean'] = acc_data_split['acc_y'].apply(np.mean)
    acc_data_split['acc_z_mean'] = acc_data_split['acc_z'].apply(np.mean)

    acc_data_split = add_ptp_features(acc_data_split, sampling_rate=sampling_rate)

    acc_data_split = acc_data_split.drop(columns=['acc_x', 'acc_y', 'acc_z'])

    return acc_data_split

def create_data_splits(acc_data, feature_cols, test_size=0.2, val_size=0.25):
    acc_data = acc_data.dropna()

    X = acc_data[feature_cols].values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(acc_data['behavior'])

    # First: train+val and test split
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_val_idx, test_idx = next(sss1.split(X, y))

    X_train_val, y_train_val = X[train_val_idx], y[train_val_idx]
    X_test, y_test = X[test_idx], y[test_idx]  

    # Second: train and val split from train_val
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
    train_idx, val_idx = next(sss2.split(X_train_val, y_train_val))

    X_train, y_train = X_train_val[train_idx], y_train_val[train_idx]
    X_val, y_val = X_train_val[val_idx], y_train_val[val_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test

# creatng windows from continuous unlabeled data

def create_windowed_features(df, sampling_frequency=16, window_duration=None, window_length=None):

    if window_length is None and window_duration is None:
        raise ValueError('A window length/duration for the classification model is required.')

    if window_length is None:
        window_length = int(window_duration * sampling_frequency)

    if window_length is not None and window_duration is not None:
        assert window_length == int(window_duration * sampling_frequency), \
            "window length and window duration are not compatible according to provided sampling frequency."

    N = len(df)

    # non-overlapping windows
    num_windows = N // window_length
    if num_windows == 0:
        return pd.DataFrame()

    # slice arrays into contiguous windows (vectorized)
    def window_stack(col):
        arr = df[col].values.astype(np.float32)
        return arr[:num_windows * window_length].reshape(num_windows, window_length)

    X = window_stack('Acc X [g]')
    Y = window_stack('Acc Y [g]')
    Z = window_stack('Acc Z [g]')

    # timestamps
    ts = df['Timestamp'].values
    ts_start = ts[0:num_windows * window_length:window_length]
    ts_end   = ts[window_length - 1 : num_windows * window_length : window_length]

    # means
    x_mean = X.mean(axis=1).astype(np.float32)
    y_mean = Y.mean(axis=1).astype(np.float32)
    z_mean = Z.mean(axis=1).astype(np.float32)

    # vectorized PTP stats
    x_ptp_max, x_ptp_mean = process_column_vectorized(X, sampling_rate=16)
    y_ptp_max, y_ptp_mean = process_column_vectorized(Y, sampling_rate=16)
    z_ptp_max, z_ptp_mean = process_column_vectorized(Z, sampling_rate=16)

    # Build DataFrame (scalar columns only)
    out = pd.DataFrame({
        'Timestamp start [yyyy-mm-dd HH:MM:SS]': ts_start,
        'Timestamp end [yyyy-mm-dd HH:MM:SS]': ts_end,
        'acc_x_mean': x_mean,
        'acc_y_mean': y_mean,
        'acc_z_mean': z_mean,
        'acc_x_ptp_max': x_ptp_max,
        'acc_x_ptp_mean': x_ptp_mean,
        'acc_y_ptp_max': y_ptp_max,
        'acc_y_ptp_mean': y_ptp_mean,
        'acc_z_ptp_max': z_ptp_max,
        'acc_z_ptp_mean': z_ptp_mean,
    })

    # -------- NEW: remove windows with timestamp gaps --------
    actual_duration = (out['Timestamp end [yyyy-mm-dd HH:MM:SS]'] -
                       out['Timestamp start [yyyy-mm-dd HH:MM:SS]']).dt.total_seconds()

    out = out[actual_duration <= window_duration].reset_index(drop=True)

    return out

def modify_vectronics_labels(
    df,
    keep_confidence_levels=None,        # e.g., ['H', 'H/M']
    eating_to_other=None,               # e.g., ['M', 'L']
    eating_to_exclude=None               # e.g., ['M']
):
    """
    Modify behavior labels in a dataframe based on confidence and eating intensity.

    Args:
        df (pd.DataFrame): Input dataframe with columns 'Confidence (H-M-L)', 'behavior', 'Eating intensity'.
        keep_confidence_levels (list[str], optional): Confidence levels to keep. Rows not in this list are dropped.
        eating_to_other (list[str], optional): Eating intensities that should be converted to 'Other' behavior.
        eating_to_exclude (list[str], optional): Eating intensities to exclude for 'Eating' behavior after modification.

    Returns:
        pd.DataFrame: Modified dataframe.
    """

    df = df.copy()

    # Filter by confidence if specified
    if keep_confidence_levels is not None:
        df = df[df['Confidence (H-M-L)'].isin(keep_confidence_levels)].reset_index(drop=True)

    # Convert certain Eating intensities to 'Other'
    if eating_to_other is not None:
        df['behavior'] = df.apply(
            lambda row: 'Other' if (row['behavior'] == 'Eating' and
                                    pd.notna(row['Eating intensity']) and
                                    row['Eating intensity'] in eating_to_other)
                        else row['behavior'],
            axis=1
        )

    # Exclude certain Eating intensities
    if eating_to_exclude is not None:
        df = df.loc[~((df['behavior'] == 'Eating') &
                      df['Eating intensity'].isin(eating_to_exclude))].reset_index(drop=True)

    return df
