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
                                                create_summary_data)
import config as config


def preprocess_Vectronics_data(save_preprocessed_data=True):

    acc_data, _ = load_Vectronics_data_metadata()

    # Load config
    with open(config.VECTRONICS_PREPROCESSING_YAML) as f:
        Vectronics_preprocessing_config = yaml.safe_load(f)

    window_duration = Vectronics_preprocessing_config['window_duration']

    acc_data_split = create_max_windows(acc_data=acc_data, window_duration=window_duration, sampling_rate=config.SAMPLING_RATE)

    # extract instances >= window duration
    acc_data_split = acc_data_split[acc_data_split.duration >= window_duration]
    df_preprocessed = create_summary_data(acc_data_split, sampling_rate=config.SAMPLING_RATE)

    if save_preprocessed_data:
        df_preprocessed.to_csv(io.get_Vectronics_preprocessed_path(), index=False)

    return df_preprocessed

if __name__ == '__main__':

    _ = preprocess_Vectronics_data(save_preprocessed_data=True)
