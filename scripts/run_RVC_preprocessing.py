import yaml
import pandas as pd
import os
import sys
sys.path.append('.')
sys.path.append('../')

from src.utils.RVC_preprocessing import preprocess_data
import config as config
import src.utils.io as io

def load_RVC_data():
    files = [file for file in os.listdir(config.HISTORIC_ACC) if file.endswith('.txt')]

    dfs = []
    for file in files:
        df_temp = pd.read_table(os.path.join(config.HISTORIC_ACC, file))
        df_temp['animal_id'] = file.split('_')[0]
        df_temp = df_temp.sort_values(by='UTC time (yyyy-mm-dd HH:MM:SS)')
        dfs.append(df_temp)

    df = pd.concat(dfs)
    df['UTC time (yyyy-mm-dd HH:MM:SS)'] = pd.to_datetime(df['UTC time (yyyy-mm-dd HH:MM:SS)'])
    df = df.rename(columns={'UTC time (yyyy-mm-dd HH:MM:SS)': 'UTC time [yyyy-mm-dd HH:MM:SS]',
                            'GPS time (s)': 'GPS time', 
                            'Max accel peak X': 'acc_x_ptp_max',
                            'Max accel peak Y': 'acc_y_ptp_max',
                            'Max accel peak Z': 'acc_z_ptp_max',
                            'Mean accel peak X': 'acc_x_ptp_mean',
                            'Mean accel peak Y': 'acc_y_ptp_mean',
                            'Mean accel peak Z': 'acc_z_ptp_mean',
                            'Mean accel X': 'acc_x_mean',
                            'Mean accel Y': 'acc_y_mean',
                            'Mean accel Z': 'acc_z_mean'
                            })
    df['UTC date [yyyy-mm-dd]'] = df['UTC time [yyyy-mm-dd HH:MM:SS]'].dt.date
    return df

def load_RVC_metadata():
    metadata_df = pd.read_excel(config.RVC_MERGED_METADATA_PATH)
    metadata_df.start_date_dd_mm_yyyy = pd.to_datetime(metadata_df.start_date_dd_mm_yyyy, format='%d/%m/%Y')
    metadata_df.end_date_dd_mm_yyyy = pd.to_datetime(metadata_df.end_date_dd_mm_yyyy, format='%d/%m/%Y')
    return metadata_df


def preprocess_RVC_data(save_preprocessed_data=True):
    
    # Load config
    with open(config.RVC_PREPROCESSING_YAML) as f:
        RVC_preprocessing_config = yaml.safe_load(f)

    # Load df
    print("Loading the RVC data...")
    df = load_RVC_data()

    # Load metadata_df
    metadata_df = load_RVC_metadata()

    # Preprocess df 
    df_preprocessed = preprocess_data(
        df=df,
        metadata_df=metadata_df,
        feature_cols=RVC_preprocessing_config['feature_cols'],
        helper_cols=RVC_preprocessing_config['helper_cols'],
        summary_dir=io.get_results_dir()
    )

    if save_preprocessed_data:
        print("Saving the preprocessed RVC data...")
        df_preprocessed.to_csv(io.get_RVC_preprocessed_path(), index=False)

    return df_preprocessed

if __name__ == '__main__':

    _ = preprocess_RVC_data(save_preprocessed_data=True)
