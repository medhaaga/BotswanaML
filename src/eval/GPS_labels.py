import sys
import os
sys.path.append('.')
sys.path.append('../')
sys.path.append('../..')
import numpy as np
import pandas as pd
import src.utils.io as io
from itertools import combinations
from tqdm import tqdm

def load_historic_gps(path):
    df = pd.read_csv(path, index_col=0)
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], format='%Y-%m-%d %H:%M:%S')
    df = df.sort_values(by=['id', 'timestamp_utc'])
    df = df[['id', 'timestamp_utc', 'lat', 'lon']]
    df['id'] = df['id'].replace('635 -Fiji ', '635 - Fiji ')
    return df

# Haversine distance (in meters)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in m
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def compute_time_dist_diff(df):

    df['time_diff [s]'] = df.groupby('id')['timestamp_utc'].diff().dt.total_seconds()
    df['distance [m]'] = haversine(df['lat'], df['lon'], df.groupby('id')['lat'].shift(), df.groupby('id')['lon'].shift())

    # Split the 'id' column into two new columns
    df[['collar_number', 'animal_id']] = df['id'].str.split(' - ', expand=True)
    df['animal_id'] = df['animal_id'].str.strip()
    df['collar_number'] = df['collar_number'].str.strip()

    # Drop the original 'id' column
    df = df.drop(columns=['id'])

    df.rename(columns={
        'timestamp_utc': 'UTC time [yyyy-mm-dd HH:MM:SS]',
        'lat': 'latitude',
        'lon': 'longitude',
    }, inplace=True)

    return df


def match_gps_to_RVC(gps_df, RVC_df):

    filtered_gps_df = gps_df.merge(RVC_df[['animal_id', 'UTC date [yyyy-mm-dd]']].drop_duplicates(),
                                    on=['animal_id', 'UTC date [yyyy-mm-dd]'],
                                    how='inner')
    return filtered_gps_df


def extract_moving_gps(df, time_diff_threshold=450, distance_threshold=20):
    
    # Ensure your DataFrame is sorted by animal_id and timestamp
    df = df.sort_values(by=['animal_id', 'collar_number', 'UTC time [yyyy-mm-dd HH:MM:SS]'])

    # Create shifted timestamps (previous row per animal/collar)
    df['timestamp_prev'] = df.groupby(['animal_id', 'collar_number'])['UTC time [yyyy-mm-dd HH:MM:SS]'].shift()
    df['timestamp_prev'] = pd.to_datetime(df['timestamp_prev'], format='%Y-%m-%d %H:%M:%S')

    # Apply your filters
    filtered = df[
        (df['time_diff [s]'] <= time_diff_threshold) &
        (df['distance [m]'] >= distance_threshold)
    ].copy()

    # Add start and end timestamps
    filtered['timestamp_start [yyyy-mm-dd HH:MM:SS]'] = filtered['timestamp_prev']
    filtered['timestamp_end [yyyy-mm-dd HH:MM:SS]'] = filtered['UTC time [yyyy-mm-dd HH:MM:SS]']
    filtered['UTC date [yyyy-mm-dd]'] = filtered['timestamp_end [yyyy-mm-dd HH:MM:SS]'].dt.date.astype(str)

    # Drop helper column
    filtered = filtered.drop(columns=['timestamp_prev', 'UTC time [yyyy-mm-dd HH:MM:SS]'])

    # Optional: reorder for clarity
    filtered = filtered[['animal_id', 'collar_number',
                        'timestamp_start [yyyy-mm-dd HH:MM:SS]', 
                        'timestamp_end [yyyy-mm-dd HH:MM:SS]',
                        'UTC date [yyyy-mm-dd]',
                        'time_diff [s]', 'distance [m]']]
        
    return filtered

def within_circle(latitudes, longitudes, diameter=100):
    # Compute max pairwise distance among 3 points
    max_dist = 0
    for i, j in combinations(range(3), 2):
        d = haversine(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
        max_dist = max(max_dist, d)
    return max_dist <= diameter

def find_consecutive_windows(subdf, time_diff_threshold=450, distance_threshold=100):

    # Keep original index for output
    subdf = subdf.reset_index()  # keep original index in a column called 'index'
    
    valid_idx = []
    for i in tqdm(range(len(subdf) - 2)):
        window = subdf.iloc[i:i+3]
        # Check time condition
        if window['time_diff [s]'].iloc[1] < time_diff_threshold and window['time_diff [s]'].iloc[2] < distance_threshold:
            # Check spatial condition
            if within_circle(window['latitude'].values, window['longitude'].values, diameter=distance_threshold):
                valid_idx.extend(window['index'].tolist())  # use original indices
    return valid_idx

def extract_feeding_gps(df, time_diff_threshold=450, distance_threshold=100):

    df = df.sort_values(by=['animal_id', 'collar_number', 'UTC time [yyyy-mm-dd HH:MM:SS]'])

    # Ensure your DataFrame is sorted by animal_id and timestamp
    df = df.sort_values(by=['animal_id', 'collar_number', 'UTC time [yyyy-mm-dd HH:MM:SS]'])

    # Create shifted timestamps (previous row per animal/collar)
    df['timestamp_prev'] = df.groupby(['animal_id', 'collar_number'])['UTC time [yyyy-mm-dd HH:MM:SS]'].shift()
    df['timestamp_prev'] = pd.to_datetime(df['timestamp_prev'], format='%Y-%m-%d %H:%M:%S')

    # Add start and end timestamps
    df['timestamp_start [yyyy-mm-dd HH:MM:SS]'] = df['timestamp_prev']
    df['timestamp_end [yyyy-mm-dd HH:MM:SS]'] = df['UTC time [yyyy-mm-dd HH:MM:SS]']
    df['UTC date [yyyy-mm-dd]'] = df['timestamp_end [yyyy-mm-dd HH:MM:SS]'].dt.date.astype(str)
    df = df.drop(columns=['timestamp_prev', 'UTC time [yyyy-mm-dd HH:MM:SS]'])

    valid_indices = []
    for _, subdf in df.groupby(['animal_id', 'collar_number']):
        valid_indices.extend(find_consecutive_windows(subdf, time_diff_threshold=time_diff_threshold, distance_threshold=distance_threshold))

    # Extract those rows
    df_valid = df.loc[sorted(set(valid_indices))].copy()
    df_valid = df_valid[['animal_id', 'collar_number',
                         'UTC date [yyyy-mm-dd]',
                         'timestamp_start [yyyy-mm-dd HH:MM:SS]',
                         'timestamp_end [yyyy-mm-dd HH:MM:SS]',
                         'time_diff [s]', 'distance [m]']]
    return df_valid


if __name__ == "__main__":

    data_dir = io.get_data_path()
    path = os.path.join(data_dir, 'dog-all-gps-cleaned.csv')
    moving_save_path = io.get_gps_moving_path()
    feeding_save_path = io.get_gps_feeding_path()


    print("Loading RVC data...")
    RVC_df = pd.read_csv(io.get_RVC_preprocessed_path())
    RVC_df['UTC date [yyyy-mm-dd]'] = pd.to_datetime(RVC_df['UTC date [yyyy-mm-dd]'], format='%Y-%m-%d').dt.date.astype(str)
    print("Loading GPS data...")
    df = load_historic_gps(path)
    df = compute_time_dist_diff(df)

    print("Extracting moving instances from GPS data...")
    moving_df = extract_moving_gps(df, time_diff_threshold=450, distance_threshold=20)
    moving_df = match_gps_to_RVC(moving_df, RVC_df)

    print(f"Saved {len(moving_df)} moving GPS points and saved to {moving_save_path}")
    moving_df.to_csv(moving_save_path, index=False)

    print("Extracting feeding instances from GPS data...")
    feeding_df = extract_feeding_gps(df, time_diff_threshold=450, distance_threshold=100)
    feeding_df = match_gps_to_RVC(feeding_df, RVC_df)

    print(f"Saved {len(feeding_df)} feeding GPS points and saved to {feeding_save_path}")
    feeding_df.to_csv(feeding_save_path, index=False)





