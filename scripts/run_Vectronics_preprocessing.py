import yaml
import pandas as pd
import os
import sys
sys.path.append('.')
sys.path.append('../')
sys.path.append('../../')
import src.utils.io as io


from src.utils.Vectronics_preprocessing import (
                                                load_Vectronics_data_metadata,
                                                create_max_windows,
                                                create_summary_data,
                                                load_annotations)
import config as config
from src.utils.data_prep import create_matched_data


def preprocess_Vectronics_data(save_preprocessed_data=True, min_window_for_padding=30.0):

    # acc_data, _ = load_Vectronics_data_metadata()

    all_annotations = load_annotations()
    metadata = pd.read_csv(io.get_metadata_path())

    _, acc_data, _, _ = create_matched_data(filtered_metadata=metadata, 
                                                            annotations=all_annotations, 
                                                            verbose=True, 
                                                            min_window_for_padding=min_window_for_padding,
                                                            min_matched_duration=30.0)

    # Load config
    with open(config.VECTRONICS_PREPROCESSING_YAML) as f:
        Vectronics_preprocessing_config = yaml.safe_load(f)

    window_duration = Vectronics_preprocessing_config['window_duration']

    print(f"Creating windows of durations {window_duration}...")
    acc_data_split = create_max_windows(acc_data=acc_data, window_duration=window_duration, sampling_rate=config.SAMPLING_RATE)

    # extract instances >= window duration
    acc_data_split = acc_data_split[acc_data_split.duration >= window_duration]

    print(f"Creating summary statistics...")
    df_preprocessed = create_summary_data(acc_data_split, sampling_rate=config.SAMPLING_RATE)

    print(f"Saving preprocessed data to {io.get_Vectronics_preprocessed_path(min_window_for_padding)}")
    if save_preprocessed_data:
        df_preprocessed.to_csv(io.get_Vectronics_preprocessed_path(min_window_for_padding), index=False)

    return df_preprocessed

if __name__ == '__main__':

    _ = preprocess_Vectronics_data(save_preprocessed_data=True, min_window_for_padding=10.0)
