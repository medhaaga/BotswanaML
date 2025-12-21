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

    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
    df.rename(columns={
        'timestamp_utc': 'UTC time [yyyy-mm-dd HH:MM:SS]',
        'lat': 'latitude',
        'lon': 'longitude',
    }, inplace=True)

    # Add previous and current timestamps for clarity
    df['timestamp_prev [yyyy-mm-dd HH:MM:SS]'] = df.groupby('id')['UTC time [yyyy-mm-dd HH:MM:SS]'].shift()

    # Compute time difference in seconds
    df['time_diff [s]'] = (df['UTC time [yyyy-mm-dd HH:MM:SS]'] - df['timestamp_prev [yyyy-mm-dd HH:MM:SS]']).dt.total_seconds()

    # Compute distance in meters between consecutive GPS points
    df['distance [m]'] = haversine(
        df['latitude'], df['longitude'],
        df.groupby('id')['latitude'].shift(),
        df.groupby('id')['longitude'].shift()
    )

    # Split the 'id' column into collar number and animal ID
    df[['collar_number', 'animal_id']] = df['id'].str.split(' - ', expand=True)
    df['animal_id'] = df['animal_id'].str.strip()
    df['collar_number'] = df['collar_number'].str.strip()

    df = df.drop(columns=['id'])
    df['UTC date [yyyy-mm-dd]'] = df['UTC time [yyyy-mm-dd HH:MM:SS]'].dt.date.astype(str)

    return df

def match_gps_to_RVC(gps_df, RVC_df):

    filtered_gps_df = gps_df.merge(RVC_df[['animal_id', 'UTC date [yyyy-mm-dd]']].drop_duplicates(),
                                    on=['animal_id', 'UTC date [yyyy-mm-dd]'],
                                    how='inner')
    return filtered_gps_df


def extract_moving_gps(df, time_diff_threshold=450, distance_threshold=200):
    
    # Ensure your DataFrame is sorted by animal_id and timestamp
    df = df.sort_values(by=['animal_id', 'collar_number', 'UTC time [yyyy-mm-dd HH:MM:SS]'])

    # Create shifted timestamps (previous row per animal/collar)
    df['timestamp_prev [yyyy-mm-dd HH:MM:SS]'] = pd.to_datetime(df['timestamp_prev [yyyy-mm-dd HH:MM:SS]'], format='%Y-%m-%d %H:%M:%S')

    # Apply your filters
    filtered = df[
        (df['time_diff [s]'] <= time_diff_threshold) &
        (df['distance [m]'] >= distance_threshold)
    ].copy()

    # Add start and end timestamps
    filtered['timestamp_start [yyyy-mm-dd HH:MM:SS]'] = filtered['timestamp_prev [yyyy-mm-dd HH:MM:SS]']
    filtered['timestamp_end [yyyy-mm-dd HH:MM:SS]'] = filtered['UTC time [yyyy-mm-dd HH:MM:SS]']

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

def load_and_merge_behavior_data(RVC_df, moving_df, label_name="Moving"):

    RVC_df = RVC_df.copy()
    moving_df = moving_df.copy()

    # Datetime conversion
    RVC_df["UTC time [yyyy-mm-dd HH:MM:SS]"] = pd.to_datetime(RVC_df["UTC time [yyyy-mm-dd HH:MM:SS]"])
    moving_df["timestamp_start [yyyy-mm-dd HH:MM:SS]"] = pd.to_datetime(moving_df["timestamp_start [yyyy-mm-dd HH:MM:SS]"])
    moving_df["timestamp_end [yyyy-mm-dd HH:MM:SS]"] = pd.to_datetime(moving_df["timestamp_end [yyyy-mm-dd HH:MM:SS]"])

    labeled_mask = np.zeros(len(RVC_df), dtype=bool)

    # Process per animal_id and date to reduce memory usage
    for (aid, date), group in tqdm(moving_df.groupby(["animal_id", "UTC date [yyyy-mm-dd]"])):

        idx = (RVC_df["animal_id"] == aid) & (RVC_df["UTC date [yyyy-mm-dd]"] == date)
        times = RVC_df.loc[idx, "UTC time [yyyy-mm-dd HH:MM:SS]"]

        # Create intervals
        intervals = pd.IntervalIndex.from_arrays(group["timestamp_start [yyyy-mm-dd HH:MM:SS]"], 
                                                 group["timestamp_end [yyyy-mm-dd HH:MM:SS]"], 
                                                 closed='both')

        # Boolean mask: True if time in any interval
        mask = times.apply(lambda t: intervals.contains(t).any())
        labeled_mask[idx] = mask.values

    labeled_df = RVC_df[labeled_mask].copy()
    labeled_df["behavior"] = label_name

    unlabeled_df = RVC_df[~labeled_mask].copy()
    unlabeled_df["behavior"] = None

    return labeled_df, unlabeled_df


def find_consecutive_windows(subdf, time_diff_threshold=450, distance_threshold=100):
    
    subdf = subdf.sort_values('UTC time [yyyy-mm-dd HH:MM:SS]')
    valid_idx = []

    for i in range(len(subdf) - 2):
        window = subdf.iloc[i:i+3]

        # Time condition: all two consecutive gaps < threshold
        if (
            window['time_diff [s]'].iloc[1] < time_diff_threshold and
            window['time_diff [s]'].iloc[2] < time_diff_threshold
        ):
            # Spatial condition
            if within_circle(window['latitude'].values, window['longitude'].values, diameter=distance_threshold):
                valid_idx.extend(window.index.tolist())
    return valid_idx


def extract_feeding_gps(df, time_diff_threshold=450, distance_threshold=100):

    df = df.sort_values(by=['animal_id', 'collar_number', 'UTC time [yyyy-mm-dd HH:MM:SS]'])

    valid_indices = []
    for key, subdf in df.groupby(['animal_id', 'collar_number']):
        print(key)
        valid_indices.extend(find_consecutive_windows(
            subdf,
            time_diff_threshold=time_diff_threshold,
            distance_threshold=distance_threshold
        ))

    # select valid rows
    feeding_df = df.loc[sorted(set(valid_indices))].copy()

    # Ensure the dataframe is sorted correctly by animal and time
    feeding_df = feeding_df.sort_values(by=['animal_id', 'collar_number', 'UTC time [yyyy-mm-dd HH:MM:SS]'])

    # Calculate time difference between consecutive rows for each animal
    time_gaps = feeding_df.groupby(['animal_id', 'collar_number'])['UTC time [yyyy-mm-dd HH:MM:SS]'].diff().dt.total_seconds()

    # A new bout starts where the gap is larger than the threshold or where it's the first point (NaN)
    is_new_bout = (time_gaps > time_diff_threshold) | (time_gaps.isnull())

    # Assign a unique ID to each bout by taking the cumulative sum of the 'is_new_bout' boolean series
    feeding_df['bout_id'] = is_new_bout.cumsum()
    bout_df = feeding_df.groupby(['animal_id', 'collar_number', 'bout_id']).agg(
                                        bout_start=('UTC time [yyyy-mm-dd HH:MM:SS]', 'min'),
                                        bout_end=('UTC time [yyyy-mm-dd HH:MM:SS]', 'max')).reset_index()
    
    bout_df['UTC date [yyyy-mm-dd]'] = bout_df['bout_start'].dt.date.astype(str)

    return bout_df


if __name__ == "__main__":

    data_dir = io.get_data_path()
    raw_gps_path = os.path.join(data_dir, 'dog-all-gps-cleaned.csv')
    matched_gps_path = os.path.join(data_dir, 'matched_gps.csv')
    moving_save_path = io.get_gps_moving_path()
    feeding_save_path = io.get_gps_feeding_path()
    matched_moving_path = io.get_matched_gps_moving_path()


    print("Loading RVC data...")
    RVC_df = pd.read_csv(io.get_RVC_preprocessed_path())
    RVC_df['UTC date [yyyy-mm-dd]'] = pd.to_datetime(RVC_df['UTC date [yyyy-mm-dd]'], format='%Y-%m-%d').dt.date.astype(str)
    
    print("Loading GPS data...")
    df = load_historic_gps(raw_gps_path)
    df = compute_time_dist_diff(df)
    df = match_gps_to_RVC(df, RVC_df)
    df.to_csv(matched_gps_path)

    df = pd.read_csv(matched_gps_path)

    print("Extracting moving instances from GPS data...")
    moving_df = extract_moving_gps(df, time_diff_threshold=450, distance_threshold=200)

    print(f"Saved {len(moving_df)} moving GPS points and saved to {moving_save_path}")
    moving_df.to_csv(moving_save_path, index=False)

    print("Matching moving bouts with acceleration data...")
    labeled_df, _ = load_and_merge_behavior_data(RVC_df, moving_df, label_name="Moving")
    labeled_df.to_csv(matched_moving_path, index=False)

    print("Extracting feeding instances from GPS data...")
    feeding_df = extract_feeding_gps(df, time_diff_threshold=450, distance_threshold=100)

    # print(f"Saved {len(feeding_df)} feeding GPS points and saved to {feeding_save_path}")
    # feeding_df.to_csv(feeding_save_path, index=False)





