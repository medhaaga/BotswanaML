import sys
sys.path.append('.')
sys.path.append('../')
sys.path.append('../../')
import os
import json
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
import config as config
import src.utils.io as io


def extract_six_point_cal(data_streams):
    cal_dict = {'X': None, 'Y': None, 'Z': None}
    for ds in data_streams:
        for channel in ds.findall('Channel'):
            desc = channel.findtext('Description', default='')
            if desc.startswith('X Acceleration') and cal_dict['X'] is None:
                text = channel.findtext('SixPointCal')
                cal_dict['X'] = [int(val.strip()) for val in text.split(',')]
            elif desc.startswith('Y Acceleration') and cal_dict['Y'] is None:
                text = channel.findtext('SixPointCal')
                cal_dict['Y'] = [int(val.strip()) for val in text.split(',')]
            elif desc.startswith('Z Acceleration') and cal_dict['Z'] is None:
                text = channel.findtext('SixPointCal')
                cal_dict['Z'] = [int(val.strip()) for val in text.split(',')]
    return cal_dict

def extract_x_acc_range(data_streams):
    x_range = None
    for ds in data_streams:
        for channel in ds.findall('Channel'):
            desc = channel.findtext('Description', default='')
            if desc.startswith('X Acceleration') and x_range is None:
                range_text = channel.findtext('Range').strip()
                if range_text:
                    try:
                        if range_text.endswith('g'):
                            x_range = float(range_text[:-1]) 
                        else:
                            x_range = float(range_text)
                    except ValueError:
                        x_range = range_text
                break  # Found it, no need to check more channels
        if x_range is not None:
            break  # Found it, no need to check more data streams
    return x_range

def parse_xml_file(filepath):
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

    filename = os.path.basename(filepath)
    parts = filename.replace('.XML', '').split('_')
    if len(parts) < 3:
        print(f"Filename format incorrect: {filename}")
        return None
    serial_no, name = int(parts[0]), parts[1]

    # Extract hardware and firmware information
    hardware = root.findtext('Hardware')
    hw_serial = root.findtext('HardwareSerial')
    firmware = root.findtext('Firmware')
    fw_version = root.findtext('FirmwareMajorVersion')

    # Extract six-point calibration
    six_point_cals = {'X': None, 'Y': None, 'Z': None}
    data_streams = root.findall('DataStream')
    six_point_cals = extract_six_point_cal(data_streams)

    # extract accelerometer range
    acc_range = extract_x_acc_range(data_streams)

    sensitivity_X = (max(six_point_cals['X']) - min(six_point_cals['X']))/2
    sensitivity_Y = (max(six_point_cals['Y']) - min(six_point_cals['Y']))/2
    sensitivity_Z = (max(six_point_cals['Z']) - min(six_point_cals['Z']))/2
    offset_X = (sum(six_point_cals['X']) - min(six_point_cals['X']) - max(six_point_cals['X'])) / (len(six_point_cals['X']) - 2)
    offset_Y = (sum(six_point_cals['Y']) - min(six_point_cals['Y']) - max(six_point_cals['Y'])) / (len(six_point_cals['Y']) - 2)
    offset_Z = (sum(six_point_cals['Z']) - min(six_point_cals['Z']) - max(six_point_cals['Z'])) / (len(six_point_cals['Z']) - 2)

    return {
        'collar_number': serial_no,
        'animal_id': name,
        'hardware': hardware,
        'hardware_serial': hw_serial,
        'firmware': firmware,
        'firmware_major_version': fw_version,
        'range': acc_range,
        'six_point_X': six_point_cals['X'],
        'six_point_Y': six_point_cals['Y'],
        'six_point_Z': six_point_cals['Z'],
        'sensitivity_X': sensitivity_X,
        'sensitivity_Y': sensitivity_Y,
        'sensitivity_Z': sensitivity_Z,
        'offset_X': offset_X,
        'offset_Y': offset_Y,
        'offset_Z': offset_Z
    }

def process_folder(folder_path):
    records = []
    for fname in os.listdir(folder_path):
        if fname.endswith('.XML'):
            file_path = os.path.join(folder_path, fname)
            record = parse_xml_file(file_path)
            if record:
                records.append(record)
    return pd.DataFrame(records)


def match_metadata(row, metadata_df):
    aid = row['animal_id']
    ts = row['UTC time']
    
    # All matching rows for this animal
    candidates = metadata_df[metadata_df['animal_id'] == aid]

    for _, meta_row in candidates.iterrows():
        start = meta_row['start_date_dd_mm_yyyy']
        end = meta_row['end_date_dd_mm_yyyy']

        if pd.isna(start) and pd.isna(end):
            pass  # match if both are NA
        elif pd.notna(start) and pd.notna(end):
            if not (start <= ts <= end):
                continue  # skip if not in range
        else:
            continue  # incomplete date info

        # Compute desired values
        six_point_X = meta_row['six_point_X']
        six_point_Y = meta_row['six_point_Y']
        six_point_Z = meta_row['six_point_Z']

        try:
            sensitivity_X = (max(six_point_X) - min(six_point_X))/2
            sensitivity_Y = (max(six_point_Y) - min(six_point_Y))/2
            sensitivity_Z = (max(six_point_Z) - min(six_point_Z))/2
            offset_X = (sum(six_point_X) - min(six_point_X) - max(six_point_X)) / (len(six_point_X) - 2)
            offset_Y = (sum(six_point_Y) - min(six_point_Y) - max(six_point_Y)) / (len(six_point_Y) - 2)
            offset_Z = (sum(six_point_Z) - min(six_point_Z) - max(six_point_Z)) / (len(six_point_Z) - 2)
    
        except Exception:
            sensitivity_X = np.nan
            sensitivity_Y = np.nan
            sensitivity_Z = np.nan
            offset_X = np.nan
            offset_Y = np.nan
            offset_Z = np.nan

        return {
            'collar_number': meta_row['collar_number'],
            'sensitivity_X': sensitivity_X,
            'sensitivity_Y': sensitivity_Y,
            'sensitivity_Z': sensitivity_Z,
            'offset_X': offset_X,
            'offset_Y': offset_Y,
            'offset_Z': offset_Z
        }

    # If no match
    return {'collar_number': np.nan, 'sensitivity_X': np.nan, 'sensitivity_Y': np.nan, 'sensitivity_Z': np.nan, 'offset_X': np.nan, 'offset_Y': np.nan, 'offset_Z': np.nan}


def calibrate_RVC_data(df, metadata_df):

    df.loc[:,'collar_number'] = 0.0
    df.loc[:,'range'] = 0.0
    df.loc[:,'firmware_major_version'] = 0.0

    columns_to_modify = [
        'acc_x_ptp_max', 'acc_y_ptp_max', 'acc_z_ptp_max',
        'acc_x_ptp_mean', 'acc_y_ptp_mean', 'acc_z_ptp_mean',
        'acc_x_mean', 'acc_y_mean', 'acc_z_mean',
        'collar_number', 'range', 'firmware_major_version'
    ]
    df.loc[:, columns_to_modify] = df[columns_to_modify].astype(float)

    # Step 0: preprocess metadata
    metadata_df['start_date_dd_mm_yyyy'] = pd.to_datetime(metadata_df['start_date_dd_mm_yyyy'], format='%d/%m/%Y', errors='coerce')
    metadata_df['end_date_dd_mm_yyyy'] = pd.to_datetime(metadata_df['end_date_dd_mm_yyyy'], format='%d/%m/%Y', errors='coerce')

    # Step 1: Initialize matched mask
    matched_mask = pd.Series(False, index=df.index)

    # Step 2: Loop through metadata and apply transformations
    for _, meta_row in metadata_df.iterrows():
        aid = meta_row['animal_id']
        start = meta_row['start_date_dd_mm_yyyy']
        end = meta_row['end_date_dd_mm_yyyy']
        collar = meta_row['collar_number']
        range_value = meta_row['range']
        firmware_major_version = meta_row['firmware_major_version']

        sensitivity_X = meta_row['sensitivity_X']
        sensitivity_Y = meta_row['sensitivity_Y']
        sensitivity_Z = meta_row['sensitivity_Z']
        offset_X = meta_row['offset_X']
        offset_Y = meta_row['offset_Y']
        offset_Z = meta_row['offset_Z']

        # Build mask
        if pd.isna(start) and pd.isna(end):
            mask = df['animal_id'] == aid
        elif pd.notna(start) and pd.notna(end):
            mask = (df['animal_id'] == aid) & (df['UTC time [yyyy-mm-dd HH:MM:SS]'] >= start) & (df['UTC time [yyyy-mm-dd HH:MM:SS]'] <= end)
        else:
            continue  # skip if only one of start/end is missing

        # Accumulate matched rows
        matched_mask |= mask

        # Check if any sensitivity or offset is NaN
        if any(pd.isna([sensitivity_X, sensitivity_Y, sensitivity_Z, offset_X, offset_Y, offset_Z])):
            # Set all target columns to NaN
            df.loc[mask, columns_to_modify] = np.nan
            continue  # skip this iteration

        if mask.any():
            # Assign calibrated and collar values
            df.loc[mask, 'collar_number'] = collar
            df.loc[mask, 'range'] = range_value
            df.loc[mask, 'firmware_major_version'] = firmware_major_version

            df.loc[mask, 'acc_x_ptp_max'] = df.loc[mask, 'acc_x_ptp_max'] / sensitivity_X
            df.loc[mask, 'acc_y_ptp_max'] = df.loc[mask, 'acc_y_ptp_max'] / sensitivity_Y
            df.loc[mask, 'acc_z_ptp_max'] = df.loc[mask, 'acc_z_ptp_max'] / sensitivity_Z

            df.loc[mask, 'acc_x_ptp_mean'] = df.loc[mask, 'acc_x_ptp_mean'] / sensitivity_X
            df.loc[mask, 'acc_y_ptp_mean'] = df.loc[mask, 'acc_y_ptp_mean'] / sensitivity_Y
            df.loc[mask, 'acc_z_ptp_mean'] = df.loc[mask, 'acc_z_ptp_mean'] / sensitivity_Z

            df.loc[mask, 'acc_x_mean'] = df.loc[mask, 'acc_x_mean'].apply(lambda x: (x - offset_X) / sensitivity_X)
            df.loc[mask, 'acc_y_mean'] = df.loc[mask, 'acc_y_mean'].apply(lambda x: (x - offset_Y) / sensitivity_Y)
            df.loc[mask, 'acc_z_mean'] = df.loc[mask, 'acc_z_mean'].apply(lambda x: (x - offset_Z) / sensitivity_Z)
        else:
            print(f"No data found for animal_id {meta_row['animal_id']} in the date range {start} to {end}")


    df.loc[~matched_mask, columns_to_modify] = np.nan
    print(f"Number of rows without calibration metadata: {pd.isna(df['collar_number']).sum()}/{len(df)}")

    # Step 3: Drop rows that were not calibrated
    df = df.dropna(subset=columns_to_modify).reset_index(drop=True)

    return df


def assign_collar_number_sensitivity_offset(df, metadata):

    df['UTC time'] = pd.to_datetime(df['UTC time'])
    metadata['start_date_dd_mm_yyyy'] = pd.to_datetime(metadata['start_date_dd_mm_yyyy'], format='%d/%m/%Y', errors='coerce')
    metadata['end_date_dd_mm_yyyy'] = pd.to_datetime(metadata['end_date_dd_mm_yyyy'], format='%d/%m/%Y', errors='coerce')

    # Prepare empty columns
    df['collar_number'] = np.nan
    df['sensitivity_X'] = np.nan
    df['sensitivity_Y'] = np.nan
    df['sensitivity_Z'] = np.nan
    df['offset_X'] = np.nan
    df['offset_Y'] = np.nan
    df['offset_Z'] = np.nan

    # Iterate over metadata rows 
    for _, meta_row in metadata.iterrows():
        aid = meta_row['animal_id']
        start = meta_row['start_date_dd_mm_yyyy']
        end = meta_row['end_date_dd_mm_yyyy']
        collar = meta_row['collar_number']

        six_point_X = meta_row['six_point_X']
        six_point_Y = meta_row['six_point_Y']
        six_point_Z = meta_row['six_point_Z']

        try:
            sensitivity_X = (max(six_point_X) - min(six_point_X))/2
            sensitivity_Y = (max(six_point_Y) - min(six_point_Y))/2
            sensitivity_Z = (max(six_point_Z) - min(six_point_Z))/2
            offset_X = (sum(six_point_X) - min(six_point_X) - max(six_point_X)) / (len(six_point_X) - 2)
            offset_Y = (sum(six_point_Y) - min(six_point_Y) - max(six_point_Y)) / (len(six_point_Y) - 2)
            offset_Z = (sum(six_point_Z) - min(six_point_Z) - max(six_point_Z)) / (len(six_point_Z) - 2)
        except Exception:
            sensitivity_X = sensitivity_Y = sensitivity_Z = np.nan
            offset_X = offset_Y = offset_Z = np.nan

        # Build mask for df rows to update
        if pd.isna(start) and pd.isna(end):
            mask = df['animal_id'] == aid
        elif pd.notna(start) and pd.notna(end):
            mask = (df['animal_id'] == aid) & (df['UTC time'] >= start) & (df['UTC time'] <= end)
        else:
            continue  # skip if one of start/end is missing

        # Assign values to matched rows
        df.loc[mask, 'collar_number'] = collar
        df.loc[mask, 'sensitivity_X'] = sensitivity_X
        df.loc[mask, 'sensitivity_Y'] = sensitivity_Y
        df.loc[mask, 'sensitivity_Z'] = sensitivity_Z
        df.loc[mask, 'offset_X'] = offset_X
        df.loc[mask, 'offset_Y'] = offset_Y
        df.loc[mask, 'offset_Z'] = offset_Z

    return df

# Convert strings to lists if necessary
def safe_convert_to_list(x):
    if isinstance(x, list):
        return x
    try:
        return json.loads(x)
    except (TypeError, json.JSONDecodeError):
        return x  # Leave it unchanged if it can't be parsed

def threshold_RVC(df):
    # Columns and corresponding bounds
    ptp_cols_max = ["acc_x_ptp_max", "acc_y_ptp_max", "acc_z_ptp_max"]
    ptp_cols_mean = ["acc_x_ptp_mean", "acc_y_ptp_mean", "acc_z_ptp_mean"]
    mean_cols = ["acc_x_mean", "acc_y_mean", "acc_z_mean"]

    n = len(df)

    # Apply thresholds for ptp max and ptp mean (±2 * range)
    for col in ptp_cols_max + ptp_cols_mean:
        df = df[(df[col] >= 0) & (df[col] <= 2 * df["range"])]

    # Apply thresholds for mean (±range)  
    for col in mean_cols:
        df = df[(df[col] >= -df["range"]) & (df[col] <= df["range"])]

    print(f"Number of outliers removed: {n - len(df)}/{n}.")

    return df


if __name__ == "__main__":

    print("=" * 40)
    if os.path.exists(io.get_RVC_metadata_path()):
        print(f"Loading existing metadata from {io.get_RVC_metadata_path()}.")
        metadata_df = pd.read_excel(io.get_RVC_metadata_path())
    else:
        print(f"No existing metadata found at {io.get_RVC_metadata_path()}.")
        metadata_df = None

    xml_df = process_folder(io.get_RVC_header_files_dir())
    df_merged = pd.merge(
    metadata_df,
    xml_df,
    on=['collar_number', 'animal_id'],
    how='left'  # use 'left' to preserve df_main structure
    )
    df_merged.to_excel(io.get_RVC_merged_metadata_path(), index=False)
    print(f"Saved merged metadata to {io.get_RVC_merged_metadata_path()}.")

    print("=" * 40)

