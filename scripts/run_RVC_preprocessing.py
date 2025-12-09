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

    # Case 1: data_path is a directory containing several csv files
    if os.path.isdir(data_path):
        files = [file for file in os.listdir(data_path) if file.endswith('.csv')]
        dfs = []
        for file in files:
            df_temp = pd.read_csv(os.path.join(data_path, file))
            df_temp['animal_id'] = file.split('_')[0]
            df_temp = df_temp.sort_values(by='UTC.time..yyyy.mm.dd.HH.MM.SS.')
            dfs.append(df_temp)

        df = pd.concat(dfs)

    # Case 2: data_path is a single CSV file
    elif os.path.isfile(data_path) and data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
        file = os.path.basename(data_path)
        df['animal_id'] = df['id']

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
    metadata_df = pd.read_excel(config.RVC_MERGED_METADATA_PATH)
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

    # add weak labels
    df_labeled = pd.read_csv(config.HISTORIC_ACC_ANNOTATED_COMBINED)
    df_labeled = df_labeled.rename(columns={
                                            'UTC.time..yyyy.mm.dd.HH.MM.SS.': 'UTC time [yyyy-mm-dd HH:MM:SS]',
                                            'id': 'animal_id'
                                        })
    df_labeled['UTC time [yyyy-mm-dd HH:MM:SS]'] = pd.to_datetime(df_labeled['UTC time [yyyy-mm-dd HH:MM:SS]'])
    
    # the animal IDs need to be capitalized to match
    df_preprocessed['animal_id'] = df_preprocessed['animal_id'].str.capitalize()
    df_labeled['animal_id'] = df_labeled['animal_id'].str.capitalize()

    df_preprocessed = df_preprocessed.drop(columns=['moving_binary', 'resting_binary'], errors='ignore')
    df_preprocessed = df_preprocessed.merge(
                                            df_labeled[['UTC time [yyyy-mm-dd HH:MM:SS]', 'animal_id', 'feeding_binary', 'moving_binary', 'resting_binary']],
                                            on=['UTC time [yyyy-mm-dd HH:MM:SS]', 'animal_id'],
                                            how='left' 
                                            )

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

    _ = preprocess_RVC_data(data_path=config.HISTORIC_ACC_ANNOTATED, save_preprocessed_data=True)
