import sys
import os
from tqdm import tqdm
sys.path.append('.')
sys.path.append('../')
sys.path.append('../..')
import pandas as pd
import src.utils.io as io
import config as config
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from matplotlib.patches import Circle

R = 6371008.8  # Earth radius in meters
radius_m = 100
radius_deg_lat = radius_m / 111320

def load_sightings(file_path: str) -> pd.DataFrame:
    """
    Load wildlife sighting data from a CSV file, process timestamps, and expand multiple IDs.

    This function performs the following steps:
    1. Reads the CSV file into a pandas DataFrame.
    2. Combines 'Date' and 'Btime' columns into a single datetime column.
    3. Parses the combined string into a timezone-aware datetime object (Botswana time zone).
    4. Keeps only the relevant columns: 'ID', 'Btime', and 'timestamp'.
    5. Expands comma-separated IDs into multiple rows, ensuring each row has a single ID.
    6. Strips any leading/trailing whitespace from the IDs.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing the sightings data.

    Returns
    -------
    pd.DataFrame
        A cleaned DataFrame with one row per ID, containing:
        - ID : str → individual identifier
        - Btime : str → original time string from the CSV
        - timestamp : datetime64[ns, Africa/Gaborone] → parsed and localized timestamp
    """
    
    # Step 1: Read CSV into a DataFrame
    df = pd.read_csv(file_path)

    # Step 2: Combine 'Date' and 'Btime' columns into a single datetime string
    df["datetime_str"] = df["Date"].astype(str) + " " + df["Btime"].astype(str)

    # Step 3: Parse combined string into datetime with known format (MM/DD/YYYY hh:mm:ss AM/PM)
    df["timestamp"] = pd.to_datetime(
        df["datetime_str"],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce"
    )

    # Step 4: Localize the timestamp to Botswana time zone (Africa/Gaborone, UTC+2) and convert to UTC
    df["timestamp"] = df["timestamp"].dt.tz_localize("Africa/Gaborone")
    df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

    # Step 5: Keep only the relevant columns
    df = df[['ID', 'timestamp']]

    # Step 6: Split comma-separated IDs into lists
    df["ID"] = df["ID"].str.split(",")

    # Step 7: Expand list of IDs into separate rows
    df = df.explode("ID")

    # Step 8: Remove leading/trailing whitespace from IDs
    df["ID"] = df["ID"].str.strip()
    df = df.rename(columns={"ID": "animal_id", "timestamp": "UTC time [yyyy-mm-dd HH:MM:SS]"})

    return df

def match_sightings(sightings_df: pd.DataFrame, RVC_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row in sightings_df, find rows in RVC_df with matching animal_id
    and UTC time within ±30 seconds.
    
    Returns a concatenated DataFrame of all matching RVC_df rows, with a df1_index
    column indicating which sightings_df row it matched.
    """
    
    # Convert timestamps to datetime with UTC
    sightings_df['UTC time [yyyy-mm-dd HH:MM:SS]'] = pd.to_datetime(
        sightings_df['UTC time [yyyy-mm-dd HH:MM:SS]'], utc=True
    )
    RVC_df['UTC time [yyyy-mm-dd HH:MM:SS]'] = pd.to_datetime(
        RVC_df['UTC time [yyyy-mm-dd HH:MM:SS]'], utc=True
    )

    # Define time window
    time_window = pd.Timedelta(minutes=30)

    # Dictionary to store results
    results = []

    # Wrap iterrows with tqdm to show a progress bar
    for idx, row in tqdm(sightings_df.iterrows(), total=len(sightings_df), desc="Matching sightings"):
        animal = row['animal_id']
        t1 = row['UTC time [yyyy-mm-dd HH:MM:SS]']
        
        # Filter RVC_df: same animal_id and timestamp within ±30 seconds
        matching_rows = RVC_df[
            (RVC_df['animal_id'] == animal) &
            (RVC_df['UTC time [yyyy-mm-dd HH:MM:SS]'] >= t1 - time_window) &
            (RVC_df['UTC time [yyyy-mm-dd HH:MM:SS]'] <= t1)
        ]
        
        # Store df1 row info alongside matched rows
        if not matching_rows.empty:
            matching_rows = matching_rows.copy()
            matching_rows['sightings_index'] = idx
            matching_rows['Sighting time [yyyy-mm-dd HH:MM:SS]'] = t1
            results.append(matching_rows)

    # Concatenate all results into a single DataFrame
    df_matches = pd.concat(results, ignore_index=True)

    return df_matches


def run_dbscan(gps_day):
    coords = gps_day[['latitude', 'longitude']].values
    epsilon = 0.1 / 6371.0088  # 100 m in radians
    db = DBSCAN(eps=epsilon, min_samples=5, metric='haversine')
    gps_day['cluster'] = db.fit_predict(np.radians(coords))
    return gps_day

def circle_radius_deg(lon_lat, radius_m):
    lat = lon_lat[0]
    return radius_m / (111320 * np.cos(np.radians(lat)))

def plot_cluster_visuals(cluster_points):
    """Plot centroid + 100m circle."""
    centroid_lat = cluster_points['latitude'].mean()
    centroid_lon = cluster_points['longitude'].mean()
    r = circle_radius_deg((centroid_lat, centroid_lon))

    plt.scatter(centroid_lon, centroid_lat, color='black',
                s=50, marker='x', linewidths=1, zorder=5)
    circle = Circle((centroid_lon, centroid_lat), r,
                    edgecolor='black', facecolor='none',
                    linestyle='--', linewidth=0.7, alpha=0.8)
    plt.gca().add_patch(circle)

def plot_cluster_points(cluster_points, color):
    """Grey shapes: circles <450s, squares >=450s."""
    round_pts = cluster_points[cluster_points['time_diff [s]'] < 450]
    square_pts = cluster_points[cluster_points['time_diff [s]'] >= 450]

    if not round_pts.empty:
        plt.scatter(round_pts['longitude'], round_pts['latitude'],
                    color=color, marker='o', s=40, alpha=0.9)
    if not square_pts.empty:
        plt.scatter(square_pts['longitude'], square_pts['latitude'],
                    color=color, marker='s', s=40, alpha=0.9)

def create_sightings_clusters(sightings: pd.DataFrame, 
                              matched_gps: pd.DataFrame):

    all_cluster_points = []
    grouped_sightings = sightings.groupby(['animal_id', 'UTC date [yyyy-mm-dd]'])

    plot_dir = os.path.join(io.get_sightings_dir(), 'gps_clusters')
    os.makedirs(plot_dir, exist_ok=True)

    for (animal_id, date), _ in tqdm(grouped_sightings, desc="Processing animal-days"):

        gps_day = matched_gps[
            (matched_gps['animal_id'] == animal_id) &
            (matched_gps['UTC date [yyyy-mm-dd]'] == date)
        ].copy()

        if gps_day.empty:
            continue

        gps_day = gps_day.sort_values(by='UTC time [yyyy-mm-dd HH:MM:SS]')
        gps_day = run_dbscan(gps_day)

        valid_cluster_points = gps_day[gps_day['cluster'] != -1]
        all_cluster_points.append(valid_cluster_points)

        unique_labels = np.sort(np.unique(gps_day['cluster']))

        valid_cluster_ids = np.sort(valid_cluster_points['cluster'].unique())
        color_list = [config.COLOR_LIST[i % len(config.COLOR_LIST)] for i in range(len(valid_cluster_ids))]
        cluster_colors = {cid: color_list[i] for i, cid in enumerate(valid_cluster_ids)}
        print(len(valid_cluster_ids))


        plt.figure(figsize=(10, 6))
        plt.plot(gps_day['longitude'], gps_day['latitude'],
                linestyle='--', linewidth=1, alpha=0.6, color='black')

        cluster_handles = []
        cluster_labels = []

        for idx, label in enumerate(unique_labels):

            cluster_points = gps_day[gps_day['cluster'] == label]
            color = cluster_colors[label] if label != -1 else 'gray'

            # ----- Summary output table -----
            if label != -1:
                valid_points = gps_day[gps_day['cluster'] != -1]


            # ----- Plot centroid + 100m circle -----
            if label != -1:
                plot_cluster_visuals(cluster_points, color)

            # ----- Legend entry -----
            handle = plt.scatter([], [], color=color, s=60)
            cluster_handles.append(handle)
            cluster_labels.append(
                f"Cluster {label+1} ({len(cluster_points)})" if label != -1 else "Outliers"
            )

            # ----- Plot actual GPS points -----
            plot_cluster_points(cluster_points, color)

        # Marker legend
        marker_handles = [
            plt.scatter([], [], color='gray', marker='o', s=60),
            plt.scatter([], [], color='gray', marker='s', s=60)
        ]
        marker_labels = [f"$<$ 450 s", f"$>$ 450 s"]

        legend1 = plt.legend(cluster_handles, cluster_labels,
                            loc='upper left', bbox_to_anchor=(1.0, 1.0))
        legend2 = plt.legend(marker_handles, marker_labels,
                            loc='upper left', bbox_to_anchor=(1.0, 0.25))
        plt.gca().add_artist(legend1)

        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'{animal_id} | {date}')
        plt.grid(True)
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig(os.path.join(plot_dir, f"{animal_id}_{date}.png"))
        plt.close()


    # -------- Final Summary DataFrame -------- #

    cluster_points_df = pd.concat(all_cluster_points, ignore_index=True)
    cluster_points_df = cluster_points_df[
        ['UTC time [yyyy-mm-dd HH:MM:SS]',
        'latitude',
        'longitude',
        'timestamp_prev [yyyy-mm-dd HH:MM:SS]',
        'time_diff [s]',
        'distance [m]',
        'collar_number',
        'animal_id',
        'UTC date [yyyy-mm-dd]',
        'cluster']
    ]
    cluster_points_df.to_csv(os.path.join(io.get_data_path(), 'GPS_clusters.csv'))
    return cluster_points_df


if __name__ == "__main__":

    file_path = os.path.join(io.get_data_path(), 'sightings.csv')
    print("Loading sightings...")
    sightings_df = load_sightings(file_path)

    print("Loading RVC data...")
    RVC_df = pd.read_csv(io.get_RVC_preprocessed_path())

    print("Matching sightings with RVC data..")
    matched_sightings = match_sightings(sightings_df, RVC_df)
    matched_sightings.to_csv(io.get_sightings_path(), index=False)
    print(f"Number of matched sightings: {matched_sightings.sightings_index.nunique()}/{len(sightings_df)}")

    print("Creating clusters for sightings data..")
    matched_sightings = pd.read_csv(io.get_sightings_path())
    matched_gps = pd.read_csv(io.get_matched_gps_path())
    create_sightings_clusters(matched_sightings, matched_gps)

