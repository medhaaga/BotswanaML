import yaml
import pandas as pd
import os
import sys
import numpy as np
sys.path.append('.')
sys.path.append('../')

from src.utils.RVC_preprocessing import preprocess_data
import config as config
import src.utils.io as io

def load_RVC_data(data_path):

    if os.path.isfile(data_path) and data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
        df['animal_id'] = df['id'].str.capitalize()
        df["animal_id"] = df["animal_id"].apply(lambda x: x.upper() if x == 'Mj' else x)

    else:
        raise ValueError("The provided path is neither a folder nor a CSV file.")


    df = df.rename(columns={'UTC.time..yyyy.mm.dd.HH.MM.SS.': 'UTC time [yyyy-mm-dd HH:MM:SS]',
                            'GPS.time..s.': 'GPS time', 
                            'Max.accel.peak.X': 'acc_x_ptp_max',
                            'Max.accel.peak.Y': 'acc_y_ptp_max',
                            'Max.accel.peak.Z': 'acc_z_ptp_max',
                            'Mean.accel.peak.X': 'acc_x_ptp_mean',
                            'Mean.accel.peak.Y': 'acc_y_ptp_mean',
                            'Mean.accel.peak.Z': 'acc_z_ptp_mean',
                            'Mean.accel.X': 'acc_x_mean',
                            'Mean.accel.Y': 'acc_y_mean',
                            'Mean.accel.Z': 'acc_z_mean'
                            })
    df['UTC time [yyyy-mm-dd HH:MM:SS]'] = pd.to_datetime(df['UTC time [yyyy-mm-dd HH:MM:SS]'])
    df = df.sort_values(by=['animal_id', 'UTC time [yyyy-mm-dd HH:MM:SS]'])
    df['UTC date [yyyy-mm-dd]'] = df['UTC time [yyyy-mm-dd HH:MM:SS]'].dt.date
    return df

def load_RVC_metadata():
    metadata_df = pd.read_excel(io.get_RVC_merged_metadata_path())
    metadata_df.start_date_dd_mm_yyyy = pd.to_datetime(metadata_df.start_date_dd_mm_yyyy, format='%d/%m/%Y')
    metadata_df.end_date_dd_mm_yyyy = pd.to_datetime(metadata_df.end_date_dd_mm_yyyy, format='%d/%m/%Y')
    return metadata_df


def preprocess_RVC_data(data_path, save_preprocessed_data=True):
    
    # Load config
    with open(config.RVC_PREPROCESSING_YAML) as f:
        RVC_preprocessing_config = yaml.safe_load(f)

    # Load df
    print("Loading the RVC data...")
    df = load_RVC_data(data_path=data_path)

    # Load metadata_df
    metadata_df = load_RVC_metadata()

    # Preprocess df 
    df_preprocessed = preprocess_data(
        df=df,
        metadata_df=metadata_df,
        summary_dir=io.get_results_dir()
    )

    # Create behavior column
    df_preprocessed['behavior'] = np.select(
        [
            df_preprocessed['feeding_binary'] == 1,
            df_preprocessed['moving_binary'] == 1,
            df_preprocessed['resting_binary'] == 1
        ],
        ['Feeding', 'Moving', 'Stationary'],
        default=None
    )

    # Select columns
    df_preprocessed = df_preprocessed[RVC_preprocessing_config['feature_cols'] + RVC_preprocessing_config['helper_cols']]

    # save the preprocessed data
    if save_preprocessed_data:
        print("Saving the preprocessed RVC data...")
        df_preprocessed.to_csv(io.get_RVC_preprocessed_path(), index=False)

    return df_preprocessed

if __name__ == '__main__':

    _ = preprocess_RVC_data(data_path=config.RVC_ACC_ANNOTATED, save_preprocessed_data=True)
