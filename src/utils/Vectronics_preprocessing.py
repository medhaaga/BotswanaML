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
    acc_data['UTC Date [yyyy-mm-dd]'] = pd.to_datetime(acc_data['UTC Date [yyyy-mm-dd]'], format='%Y-%m-%d')
    acc_data.rename(columns={'UTC Date [yyyy-mm-dd]': 'UTC date [yyyy-mm-dd]'}, inplace=True)

    # extract behaviors of interest
    acc_data = adjust_behavior_and_durations(acc_data, config.SUMMARY_COLLAPSE_BEHAVIORS_MAPPING, config.SUMMARY_BEHAVIORS)
    acc_metadata = acc_metadata.loc[acc_data.index]
    acc_data.reset_index()
    acc_metadata.reset_index()

    return acc_data, acc_metadata

def windowed_ptp_stats(arr, window=32):
    n_full_windows = len(arr) // window
    if n_full_windows == 0:
        return np.nan, np.nan  # Not enough data

    ptp_values = [np.ptp(arr[i*window:(i+1)*window]) for i in range(n_full_windows)]
    return np.max(ptp_values), np.mean(ptp_values)

# Apply to each column and create new stats columns
def process_column(df, col, sampling_rate=16):
    df[[f'{col}_ptp_max', f'{col}_ptp_mean']] = df[col].apply(
        lambda arr: pd.Series(windowed_ptp_stats(arr, window=int(2*sampling_rate)))
    )
    return df

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
            'Source': row['Source']
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
            'Source': row['Source']
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

def create_summary_data(acc_data_split, sampling_rate=16):
    acc_data_split['acc_x_mean'] = acc_data_split['acc_x'].apply(np.mean)
    acc_data_split['acc_y_mean'] = acc_data_split['acc_y'].apply(np.mean)
    acc_data_split['acc_z_mean'] = acc_data_split['acc_z'].apply(np.mean)

    for col in ['acc_x', 'acc_y', 'acc_z']:
        acc_data_split = process_column(acc_data_split, col, sampling_rate=sampling_rate)

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