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


def smooth_behavior_segments(
    df,
    dur_threshold=5,     # short segment threshold
    gap_threshold=5,     # allowed gap for smoothing
    gap_auto_merge=5     # auto-merge same behavior if gap < 2 sec
):
    # Sort
    df = df.sort_values(["id", "Timestamp_start"]).reset_index(drop=True)

    results = []

    for pid, group in df.groupby("id"):
        segs = group.reset_index(drop=True)

        i = 0
        while i < len(segs):

            row = segs.loc[i]

            # -------------------------------
            # RULE 1: AUTO MERGE SAME-BEHAVIOR NEIGHBORS
            # -------------------------------
            if i > 0:
                prev = results[-1]
                gap_prev = (row.Timestamp_start - prev.Timestamp_end).total_seconds()

                if (
                    prev.Behavior == row.Behavior
                    and gap_prev <= gap_auto_merge
                ):
                    # merge row into prev
                    prev.Timestamp_end = row.Timestamp_end
                    prev.duration = (
                        prev.Timestamp_end - prev.Timestamp_start
                    ).total_seconds()
                    results[-1] = prev
                    i += 1
                    continue

            # -------------------------------
            # RULE 2: SHORT-SEGMENT SMOOTHING
            # -------------------------------
            if row.duration < dur_threshold:
                prev_ok = len(results) > 0
                next_ok = i < len(segs) - 1

                if prev_ok:
                    prev = results[-1]
                    gap1 = (row.Timestamp_start - prev.Timestamp_end).total_seconds()

                if next_ok:
                    nxt = segs.loc[i + 1]
                    gap2 = (nxt.Timestamp_start - row.Timestamp_end).total_seconds()

                merged = False

                # Case A: short surrounded by same behavior + small gaps
                if prev_ok and next_ok:
                    if (
                        prev.Behavior == nxt.Behavior
                        and row.duration < dur_threshold
                        and gap1 <= gap_threshold
                        and gap2 <= gap_threshold
                    ):
                        # merge prev, row, next
                        prev.Timestamp_end = nxt.Timestamp_end
                        prev.duration = (
                            prev.Timestamp_end - prev.Timestamp_start
                        ).total_seconds()
                        results[-1] = prev
                        # skip next segment
                        i += 2
                        continue

            # If none of the special rules apply → keep row
            results.append(row)
            i += 1

    # Return cleaned results
    out = pd.DataFrame(results)
    return out

def preprocess_labeled_Vectronics_data(save_preprocessed_data=True, window_duration=30.0, min_window_for_padding=30.0):

    # acc_data, _ = load_Vectronics_data_metadata()

    all_annotations = load_annotations()
    # all_annotations = smooth_behavior_segments(all_annotations)
    metadata = pd.read_csv(io.get_metadata_path())


    _, acc_data, _, _ = create_matched_data(filtered_metadata=metadata, 
                                                            annotations=all_annotations, 
                                                            verbose=True, 
                                                            min_window_for_padding=min_window_for_padding,
                                                            min_matched_duration=window_duration)

    

    print(f"Creating windows of durations {window_duration}...")
    acc_data_split = create_max_windows(acc_data=acc_data, window_duration=window_duration, sampling_rate=config.SAMPLING_RATE)

    # extract instances >= window duration
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
