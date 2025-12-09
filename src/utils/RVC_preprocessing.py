import pandas as pd
from .RVC_calibration import (calibrate_RVC_data,
                              threshold_RVC
                              )


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from dataframe."""
    before_shape = df.shape
    df = df.drop_duplicates()
    after_shape = df.shape
    print(f"Removed {before_shape[0] - after_shape[0]} duplicates.")
    return df

def preprocess_data(df, metadata_df, summary_dir=None):
    # Remove duplicates
    df = remove_duplicates(df)

    # Calibration
    print("Calibrating the RVC data...")
    df = calibrate_RVC_data(df, metadata_df)
    if summary_dir is not None:
        grouped_df = return_grouped_summary(df, metadata_df)
        grouped_df.to_csv(f"{summary_dir}/RVC_data_summary.csv", index=False)

    # Thresholding
    print("Thresholding the RVC data...")
    df = threshold_RVC(df)
    if summary_dir is not None:
        grouped_df = return_grouped_summary(df, metadata_df)
        grouped_df.to_csv(f"{summary_dir}/truncated_RVC_data_summary.csv", index=False)
    df['UTC date [yyyy-mm-dd]'] = pd.to_datetime(df['UTC date [yyyy-mm-dd]'])
    df = df[df['UTC date [yyyy-mm-dd]'].dt.year >= 2000]

    return df

def return_grouped_summary(df, metadata_df):
    grouped_df = (
        df.groupby(['animal_id', 'collar_number', 'UTC date [yyyy-mm-dd]'])
        .agg(
            count=('animal_id', 'count'),  
            Max_peak_X_min=('acc_x_ptp_max', 'min'),
            Max_peak_X_max=('acc_x_ptp_max', 'max'),
            Max_peak_Y_min=('acc_y_ptp_max', 'min'),
            Max_peak_Y_max=('acc_y_ptp_max', 'max'),
            Max_peak_Z_min=('acc_z_ptp_max', 'min'),
            Max_peak_Z_max=('acc_z_ptp_max', 'max'),
            Mean_peak_X_min=('acc_x_ptp_mean', 'min'),
            Mean_peak_X_max=('acc_x_ptp_mean', 'max'),
            Mean_peak_Y_min=('acc_y_ptp_mean', 'min'),
            Mean_peak_Y_max=('acc_y_ptp_mean', 'max'),
            Mean_peak_Z_min=('acc_z_ptp_mean', 'min'),
            Mean_peak_Z_max=('acc_z_ptp_mean', 'max'),
            Mean_X_min=('acc_x_mean', 'min'),
            Mean_X_max=('acc_x_mean', 'max'),
            Mean_Y_min=('acc_y_mean', 'min'),
            Mean_Y_max=('acc_y_mean', 'max'),
            Mean_Z_min=('acc_z_mean', 'min'),
            Mean_Z_max=('acc_z_mean', 'max'),
        )
        .reset_index()
    )
    grouped_df = grouped_df.merge(
        metadata_df[['animal_id', 'collar_number', 'hardware', 'hardware_serial', 'firmware', 'firmware_major_version', 'range']],
        on=['animal_id', 'collar_number'],
        how='left'
    )
    return grouped_df
