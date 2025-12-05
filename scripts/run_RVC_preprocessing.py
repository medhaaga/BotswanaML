import yaml
import pandas as pd
import os
import sys
from datetime import timedelta
sys.path.append('.')
sys.path.append('../')

from src.utils.RVC_preprocessing import preprocess_data
import config as config
import src.utils.io as io

def load_RVC_data(data_dir):
    files = [file for file in os.listdir(data_dir) if file.endswith('.csv')]

    dfs = []
    for file in files:
        df_temp = pd.read_csv(os.path.join(data_dir, file))
        df_temp['animal_id'] = file.split('_')[0]
        df_temp = df_temp.sort_values(by='UTC.time..yyyy.mm.dd.HH.MM.SS.')
        dfs.append(df_temp)

    df = pd.concat(dfs)
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
    df['UTC date [yyyy-mm-dd]'] = df['UTC time [yyyy-mm-dd HH:MM:SS]'].dt.date
    return df

def load_RVC_metadata():
    metadata_df = pd.read_excel(config.RVC_MERGED_METADATA_PATH)
    metadata_df.start_date_dd_mm_yyyy = pd.to_datetime(metadata_df.start_date_dd_mm_yyyy, format='%d/%m/%Y')
    metadata_df.end_date_dd_mm_yyyy = pd.to_datetime(metadata_df.end_date_dd_mm_yyyy, format='%d/%m/%Y')
    return metadata_df

def add_feeding_binary_from_sightings(df, sightings):
    """
    For each sighting event, mark a 60-min window around the event (-30 min to +30 min).
    In df, within that window: 
       feeding_binary = 0 if moving_binary == 1 OR resting_binary == 1,
       feeding_binary = 1 otherwise.
    Outside any window, feeding_binary = 0.
    """
    
    # Ensure datetimes exist
    df = df.copy()
    sightings = sightings.copy()

    df['UTC date [yyyy-mm-dd]'] = pd.to_datetime(df['UTC date [yyyy-mm-dd]']).dt.date
    df['UTC time [yyyy-mm-dd HH:MM:SS]'] = pd.to_datetime(df['UTC time [yyyy-mm-dd HH:MM:SS]'], utc=True)
    sightings['UTC date [yyyy-mm-dd]'] = pd.to_datetime(sightings['UTC date [yyyy-mm-dd]']).dt.date
    sightings['Sighting time [yyyy-mm-dd HH:MM:SS]'] = pd.to_datetime(sightings['Sighting time [yyyy-mm-dd HH:MM:SS]'], utc=True)

    sightings_unique = sightings.drop_duplicates(
        subset=['animal_id', 'UTC date [yyyy-mm-dd]', 'Sighting time [yyyy-mm-dd HH:MM:SS]']).copy()
    
    # Expand each sighting into a window
    sightings_unique['window_start'] = sightings_unique['Sighting time [yyyy-mm-dd HH:MM:SS]'] - timedelta(minutes=30)
    sightings_unique['window_end']   = sightings_unique['Sighting time [yyyy-mm-dd HH:MM:SS]'] + timedelta(minutes=30)

    # Prepare df for labeling
    df['feeding_binary'] = pd.NA  # default

    df['_orig_index'] = df.index

    # Merge df with sightings on (animal_id, utc_date)
    merged = df.merge(
        sightings_unique[['animal_id', 'UTC date [yyyy-mm-dd]', 'window_start', 'window_end']],
        on=['animal_id', 'UTC date [yyyy-mm-dd]'],
        how='left'
    )

    merged['window_start'] = pd.to_datetime(merged['window_start'], utc=True)
    merged['window_end']   = pd.to_datetime(merged['window_end'], utc=True)

    # Boolean mask: df timestamp lies inside the associated window
    in_window = (merged['window_start'].notna()) &\
                (merged['UTC time [yyyy-mm-dd HH:MM:SS]'] >= merged['window_start']) & \
                (merged['UTC time [yyyy-mm-dd HH:MM:SS]'] <= merged['window_end'])
    
    # Combine multiple possible windows per row with an "any" reduction
    idx_in_window = merged.loc[in_window, '_orig_index'].unique()


    # Compute "active" movement (treat missing as 0)
    moving_filled  = df['moving_binary'].fillna(0)
    resting_filled = df['resting_binary'].fillna(0)
    active = (moving_filled == 1) | (resting_filled == 1)

    # Assign values ONLY for rows inside windows
    df.loc[idx_in_window, 'feeding_binary'] = (~active.loc[idx_in_window]).astype(int)
    df = df.drop(columns=['_orig_index'])

    print(f"Total feeding labels from sightings: {df['feeding_binary'].sum()}")

    return df

def preprocess_RVC_data(data_dir, add_sightings=False, save_preprocessed_data=True):
    
    # Load config
    with open(config.RVC_PREPROCESSING_YAML) as f:
        RVC_preprocessing_config = yaml.safe_load(f)

    # Load df
    print("Loading the RVC data...")
    df = load_RVC_data(data_dir=data_dir)

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

    if add_sightings:
        sightings = pd.read_csv(io.get_sightings_path())
        df_preprocessed = add_feeding_binary_from_sightings(df_preprocessed, sightings)

    if save_preprocessed_data:
        print("Saving the preprocessed RVC data...")
        df_preprocessed.to_csv(io.get_RVC_preprocessed_path(), index=False)

    return df_preprocessed

if __name__ == '__main__':

    _ = preprocess_RVC_data(data_dir=config.HISTORIC_ACC_ANNOTATED, add_sightings=True, save_preprocessed_data=True)
