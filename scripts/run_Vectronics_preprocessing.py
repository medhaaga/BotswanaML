import yaml
import pandas as pd
import sys
import numpy as np
import warnings
from tqdm import tqdm
sys.path.append('.')
sys.path.append('../')
sys.path.append('../../')

import src.utils.io as io
from src.utils.Vectronics_preprocessing import (create_max_windows,
                                                create_summary_data,
                                                load_annotations,
                                                create_windowed_features)
import config as config
from src.utils.data_prep import create_matched_data


def preprocess_labeled_Vectronics_data(save_preprocessed_data=True, window_duration=30.0, min_window_for_padding=30.0):

    all_annotations = load_annotations()
    metadata = pd.read_csv(io.get_metadata_path())


    _, acc_data, _, _ = create_matched_data(filtered_metadata=metadata, 
                                                            annotations=all_annotations, 
                                                            verbose=True, 
                                                            min_window_for_padding=min_window_for_padding,
                                                            min_matched_duration=window_duration)

    

    print(f"Creating windows of durations {window_duration}...")
    acc_data_split = create_max_windows(acc_data=acc_data, window_duration=window_duration, sampling_rate=config.SAMPLING_RATE)
    acc_data_split = acc_data_split[acc_data_split.duration >= window_duration]

    print(f"Creating summary statistics...")
    df_preprocessed = create_summary_data(acc_data_split, sampling_rate=config.SAMPLING_RATE)
    df_preprocessed = df_preprocessed.drop(columns=['duration'])

    print(f"Saving preprocessed data to {io.get_Vectronics_preprocessed_path(window_duration)}")
    if save_preprocessed_data:
        df_preprocessed.to_csv(io.get_Vectronics_preprocessed_path(window_duration), index=False)

    return df_preprocessed


def create_all_vectronics_summary_data(metadata: pd.DataFrame, 
                                       sampling_frequency: int = 16, 
                                       window_length: int = None, 
                                       window_duration: float = 30.0,
                                       save_data: bool = True):

    if window_length is None and window_duration is None:
        raise ValueError('A window length/duration for the classification model is required.')

    if window_length is None:
        window_length = int(window_duration * sampling_frequency)

    if window_length is not None and window_duration is not None:
        assert window_length == int(window_duration * sampling_frequency), \
            "window length and window duration are not compatible according to provided sampling frequency."

    grouped = metadata.groupby(["individual ID", "UTC Date [yyyy-mm-dd]"])
    group_keys = list(grouped.groups.keys())   # list of (individual, date) tuples
    np.random.shuffle(group_keys)    
    results = []
    i = 0

    for individual, date in tqdm(group_keys, total=len(group_keys)):
        print(f"Processing individual={individual}, date={date}")

        group = grouped.get_group((individual, date))

        # load all half-day files for this animal/day
        dfs = []
        for _, row in group.iterrows():
            df_half = pd.read_csv(row['file path'])
            df_half['Timestamp'] = pd.to_datetime(df_half['Timestamp'], utc=True, format='%Y-%m-%d %H:%M:%S.%f')
            dfs.append(df_half)

        full_day_data = pd.concat(dfs, ignore_index=True).sort_values("Timestamp")

        if len(full_day_data) < window_length:
            warnings.warn(f'{individual}-{date} has fewer samples than the window length. Skipped.')
            continue

        features = create_windowed_features(full_day_data, sampling_frequency=sampling_frequency, 
                                            window_duration=window_duration, window_length=window_length)
        features['animal_id'] = individual
        features['UTC date [yyyy-mm-dd]'] = date
        results.append(features) 

        i+=1
        if i == 20:
            break

    df = pd.concat(results, ignore_index=True)
    if save_data:
        print(f"Saving preprocessed data to {io.get_Vectronics_full_summary_path()}")
        df.to_csv(io.get_Vectronics_full_summary_path(), index=False)


    return df

if __name__ == '__main__':

    # Load config
    with open(config.VECTRONICS_PREPROCESSING_YAML) as f:
        Vectronics_preprocessing_config = yaml.safe_load(f)

    window_duration = Vectronics_preprocessing_config['window_duration']
    window_duration = 30.0

    _ = preprocess_labeled_Vectronics_data(save_preprocessed_data=True, 
                                           window_duration=window_duration, 
                                           min_window_for_padding=None)

    # metadata = pd.read_csv(io.get_metadata_path())
    # _ = create_all_vectronics_summary_data(metadata=metadata, 
    #                                    sampling_frequency=config.SAMPLING_RATE, 
    #                                    window_duration = 30.0,
    #                                    save_data=True)
