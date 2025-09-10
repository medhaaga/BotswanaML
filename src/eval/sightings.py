import sys
import os
from tqdm import tqdm
sys.path.append('.')
sys.path.append('../')
sys.path.append('../..')

import numpy as np
import pandas as pd
import src.utils.io as io


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
    time_window = pd.Timedelta(seconds=30)

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
            results.append(matching_rows)

    # Concatenate all results into a single DataFrame
    df_matches = pd.concat(results, ignore_index=True)

    return df_matches

if __name__ == "__main__":

    file_path = os.path.join(io.get_data_path(), 'sightings.csv')
    print("Loading sightings...")
    sightings_df = load_sightings(file_path)

    print("Loading RVC data...")
    RVC_df = pd.read_csv(io.get_RVC_preprocessed_path())

    print("Matching sightings with RVC data..")
    matched_sightings = match_sightings(sightings_df, RVC_df)
    matched_sightings.to_csv(os.path.join(io.get_data_path(), 'matched_sightings.csv'), index=False)
    print(f"Number of matched sightings: {matched_sightings.sightings_index.nunique()}/{len(sightings_df)}")