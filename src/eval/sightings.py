import sys
import os
from tqdm import tqdm
sys.path.append('.')
sys.path.append('../')
sys.path.append('../..')
import yaml
import numpy as np
import pandas as pd
import src.utils.io as io
import config as config
import torch
from src.eval.eval_utils import evaluate_label_distribution
import src.methods.dann as dann
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import matplotlib.dates as mdates
from src.utils import preprocess
import src.utils.datasets as datasets
from collections import Counter
# Graphing Parameters
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from datetime import timedelta

mpl.rcParams['lines.markersize'] = 12
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams["axes.labelsize"] = 20
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['text.usetex'] = False

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

def plot_signal_and_scores(timestamps: pd.Series,
                           signal: np.ndarray,
                           scores: np.ndarray,
                           label_encoder: LabelEncoder,
                           probs: np.ndarray = None,  # add probs array
                           plot_path: str = None,
                           matched_sightings: pd.DataFrame = None,
                           title: str = None):
    """
    Plot tri-axial signal and softmax scores over time with optional suptitle
    and table of label distribution (probs).
    
    Returns:
        fig: matplotlib Figure
        axes: tuple(ax_signal, ax_online)
    """
   
    sns.set_style("whitegrid")

    timestamps = pd.to_datetime(timestamps)
    n_classes = scores.shape[1]
    y = np.arange(n_classes)

    # Flatten for scatter plotting
    X, Y = np.meshgrid(timestamps, y)
    X_flat, Y_flat = X.flatten(), Y.flatten()
    color_flat = scores.T.flatten()
    y_labels = label_encoder.inverse_transform(y)

    # Figure and GridSpec
    fig = plt.figure(figsize=(20, 8))  # a little wider to fit table
    if title:
        fig.suptitle(title, fontsize=25, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 3, width_ratios=[30, 1, 10], height_ratios=[1, 1], wspace=0.2, hspace=0.6)
    
    ax_signal = fig.add_subplot(gs[0, 0])
    ax_online = fig.add_subplot(gs[1, 0])
    cbar_ax = fig.add_subplot(gs[1, 1])
    table_ax = fig.add_subplot(gs[:, 2])  # span both rows

    table_ax.axis('off')  # hide axes for table

    # Plot tri-axial signals
    colors = ['black', 'blue', 'maroon']
    labels = ['X Signal', 'Y Signal', 'Z Signal']
    for i in range(3):
        ax_signal.plot(timestamps, signal[:, i], label=labels[i], color=colors[i], linewidth=1., alpha=0.6)
    ax_signal.set_xlabel("Time (h)")
    ax_signal.set_ylabel("Amplitude (g)")
    ax_signal.set_title("Summary Signal")
    ax_signal.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax_signal.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    # Highlight matched sightings if provided
    if matched_sightings is not None and not matched_sightings.empty:
        matched_sightings['Sighting time [yyyy-mm-dd HH:MM:SS]'] = pd.to_datetime(matched_sightings['Sighting time [yyyy-mm-dd HH:MM:SS]'])
        for j, (_, row) in enumerate(matched_sightings.iterrows()):
            start = row['Sighting time [yyyy-mm-dd HH:MM:SS]'] - timedelta(minutes=30)
            end   = row['Sighting time [yyyy-mm-dd HH:MM:SS]'] + timedelta(minutes=30)
            ax_signal.axvspan(start, end, color='pink', alpha=0.4, label="Sighting" if j == 0 else None)

    ax_signal.legend(loc='upper right', bbox_to_anchor=(1.26, 1.0), fontsize=20)

    # Plot softmax scores
    scatter = ax_online.scatter(X_flat, Y_flat, c=color_flat, cmap='Blues', s=140, marker='s', alpha=0.7)
    ax_online.set_xlabel("Time (h)")
    ax_online.set_yticks(y)
    ax_online.set_yticklabels(y_labels)
    ax_online.set_ylim(-1, n_classes)
    ax_online.set_title("Predicted Scores")
    ax_online.sharex(ax_signal)

    # Colorbar
    cbar = plt.colorbar(scatter, cax=cbar_ax)
    cbar.set_label("Softmax Score", fontsize=20)
    pos = cbar_ax.get_position()
    cbar_ax.set_position([pos.x0 - 0.02, pos.y0 + 0.01, pos.width, pos.height - 0.01])

    # Add table of label distribution if probs provided
    if probs is not None:
        cell_text = []
        for i, p in enumerate(probs):
            label = label_encoder.inverse_transform([i])[0]
            cell_text.append([label, f"{p:.2f}%"])
        table = table_ax.table(cellText=cell_text,
                               colLabels=["Behavior", "Probability"],
                               cellLoc='center',
                               colLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(15)
        table.scale(1, 2)  # scale width/height

    # Save figure
    if plot_path:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    plt.close(fig)
    return fig, (ax_signal, ax_online)


def plot_signal_and_feeding(
    timestamps: pd.Series,
    signal: np.ndarray,
    labels: np.ndarray,
    plot_path: str = None,
    matched_sightings: pd.DataFrame = None,
    title: str = None,
    label_encoder=None
):
    """
    Plot tri-axial signal with feeding events, optional matched sightings,
    and a table of label distribution.

    Parameters
    ----------
    timestamps : pd.Series
        Time indices of the signal.
    signal : np.ndarray
        Shape (N, 3), tri-axial accelerometer signal.
    labels : np.ndarray
        Labels per timestamp (same length as timestamps).
    plot_path : str, optional
        If provided, save the plot at this path.
    matched_sighting : pd.DataFrame, optional
        DataFrame with 'UTC time [yyyy-mm-dd HH:MM:SS]' for event highlighting.
    title : str, optional
        Title for the plot.
    label_encoder : LabelEncoder, optional
        If provided, will be used to inverse transform labels.
        Otherwise raw labels are shown.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    sns.set_style("whitegrid")

    # Input validation
    assert len(timestamps) == signal.shape[0] == labels.shape[0], \
        "timestamps, signal, and labels must be the same length"

    timestamps = pd.to_datetime(timestamps)

    # Initialize figure and GridSpec
    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.2)

    ax_signal = fig.add_subplot(gs[0, 0])
    table_ax = fig.add_subplot(gs[0, 1])
    table_ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=20, fontweight="bold", y=0.98)

    # --- Plot tri-axial signals ---
    colors = ["black", "blue", "maroon"]
    signal_names = ["X Signal", "Y Signal", "Z Signal"]

    for i in range(3):
        ax_signal.plot(
            timestamps, signal[:, i],
            label=signal_names[i],
            color=colors[i],
            linewidth=1.0,
            alpha=0.7,
        )

    ax_signal.set_xlabel("Time (h)")
    ax_signal.set_ylabel("Amplitude (g)")

    # Format x-axis as hours
    ax_signal.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax_signal.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    # Highlight matched sightings if provided
    if matched_sightings is not None and not matched_sightings.empty:
        matched_sightings['Sighting time [yyyy-mm-dd HH:MM:SS]'] = pd.to_datetime(matched_sightings['Sighting time [yyyy-mm-dd HH:MM:SS]'])
        for j, (_, row) in enumerate(matched_sightings.iterrows()):
            start = row['Sighting time [yyyy-mm-dd HH:MM:SS]'] - timedelta(minutes=30)
            end   = row['Sighting time [yyyy-mm-dd HH:MM:SS]'] + timedelta(minutes=30)
            ax_signal.axvspan(start, end, color='pink', alpha=0.4, label="Sighting" if j == 0 else None)
            
    # --- Feeding events ---
    feeding_mask = labels == "Feeding"
    if feeding_mask.any():
        ax_signal.scatter(
            timestamps[feeding_mask],
            np.full(np.sum(feeding_mask), ax_signal.get_ylim()[1] * 0.9),
            color="orange",
            marker="|",
            s=200,
            label="Feeding",
        )

    ax_signal.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=5)

    # --- Compute label distribution ---
    label_counts = Counter(labels)
    total = len(labels)
    cell_text = []
    for lbl, count in label_counts.items():
        # Decode if label_encoder is provided
        if label_encoder is not None and isinstance(lbl, (int, np.integer)):
            lbl_name = label_encoder.inverse_transform([lbl])[0]
        else:
            lbl_name = str(lbl)
        proportion = count / total * 100
        cell_text.append([lbl_name, f"{proportion:.2f}%"])

    # --- Add table ---
    table = table_ax.table(
        cellText=cell_text,
        colLabels=["Behavior", "Proportion"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.scale(1, 2)

    # Save if needed
    if plot_path:
        directory = os.path.dirname(plot_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig

def make_sightings_plots_from_dann(matched_sightings: pd.DataFrame, device: str = 'cpu') -> dict:
    """
    Evaluate DANN model on each unique animal-day and generate plots.

    Returns:
        results: dict mapping (animal_id, date) -> {softmax_scores, label_distribution}
    """
    results = {}
    grouped = matched_sightings.groupby(['animal_id', 'UTC date [yyyy-mm-dd]'])

    with open(config.VECTRONICS_PREPROCESSING_YAML) as f:
        vectronics_config = yaml.safe_load(f)
    feature_cols = vectronics_config['feature_cols']

    print("Loading source data (Vectronics)...")
    vectronics_df = pd.read_csv(io.get_Vectronics_preprocessed_path(10.0))
    X_src = vectronics_df[feature_cols].values

    print("Loading target data (RVC)...")
    RVC_df = pd.read_csv(io.get_RVC_preprocessed_path())

    # --------------------------
    # create transform 
    # --------------------------

    pos_idx = [0, 1, 2, 3, 4, 5]
    center_idx = [6, 7, 8]

    # compute global lows/highs once
    lows, highs = preprocess.compute_combined_quantiles(
        datasets=[X_src],
        pos_idx=pos_idx,
        center_idx=center_idx,
        low_q=0.00,
        high_q=1.00,
    )
    # define transform
    transform = preprocess.TransformAndScale(
        pos_idx=pos_idx,
        center_idx=center_idx,
        lows=lows,
        highs=highs
    )
    
    # --------------------------
    # create paths 
    # --------------------------

    model_dir = os.path.join(io.get_domain_adaptation_results_dir(), "dann")
    plot_dir = os.path.join(io.get_sightings_dir(), "dann", "eval_plots")
    os.makedirs(plot_dir, exist_ok=True)

    # --------------------------
    # load models
    # --------------------------
    feature_extractor = dann.FeatureExtractor(in_dim=len(feature_cols))
    feature_extractor.load_state_dict(torch.load(os.path.join(model_dir, "feature_extractor.pth"), map_location=device))
    feature_extractor.to(device).eval()

    label_classifier = dann.LabelClassifier(in_dim=64, n_classes=len(config.SUMMARY_BEHAVIORS))
    label_classifier.load_state_dict(torch.load(os.path.join(model_dir, "label_classifier.pth"), map_location=device))
    label_classifier.to(device).eval()

    label_encoder = LabelEncoder()
    label_encoder.fit(config.SUMMARY_BEHAVIORS)

    for (animal_id, date), group_df in tqdm(grouped, desc="Processing animal-days"):

        day_data = RVC_df[(RVC_df['animal_id'] == animal_id) &
                          (RVC_df['UTC date [yyyy-mm-dd]'] == date)]

        if day_data.empty:
            continue

        timestamps = pd.to_datetime(day_data['UTC time [yyyy-mm-dd HH:MM:SS]'])
        day_acc_data = day_data[feature_cols].values
        dataset = DataLoader(datasets.NumpyDataset(X=day_acc_data, y=None, transform=transform), batch_size=1024, shuffle=True, num_workers=4, pin_memory=True)
        firmware_major_version = int(np.unique(day_data['firmware_major_version'])[0])

        # Evaluate model
        softmax_scores, label_distribution = evaluate_label_distribution(
            feat_model=feature_extractor,
            clf_model=label_classifier,
            data=dataset,
            n_classes=len(config.SUMMARY_BEHAVIORS),
            label_encoder=label_encoder,
            device=device,
        )

        # Generate plots
        plot_path = os.path.join(plot_dir, f"{animal_id}_{date}.png")
        plot_signal_and_scores(
            timestamps=timestamps,
            signal=day_acc_data,
            scores=softmax_scores,
            label_encoder=label_encoder,
            probs=label_distribution,
            plot_path=plot_path,
            matched_sightings=group_df,
            title=f"{animal_id} | {date} | Version-{firmware_major_version}"
        )
        

        results[(animal_id, date)] = {
            "softmax_scores": softmax_scores,
            "label_distribution": label_distribution
        }

    return results

def make_sightings_plots_from_weak_labels(
        matched_sightings: pd.DataFrame,
        RVC_df: pd.DataFrame,
        verbose: bool = False
    ) -> int:
    """
    Generate and save signal plots with weak labels for each animal-day.

    Parameters
    ----------
    matched_sightings : pd.DataFrame
        DataFrame with matched sightings (must include ['animal_id', 'UTC date [yyyy-mm-dd]']).
    RVC_df : pd.DataFrame
        Full dataframe of accelerometer and label data (must include 'behavior').
    config : object
        Config module or object with SUMMARY_BEHAVIORS and VECTRONICS_PREPROCESSING_YAML.
    verbose : bool, default False
        If True, print skipped animal-days.

    Returns
    -------
    dict
        Mapping from (animal_id, date) → saved plot path.
    """
    assert "behavior" in RVC_df.columns, "RVC_df must have a 'behavior' column"

    # Group sightings once
    grouped = matched_sightings.groupby(["animal_id", "UTC date [yyyy-mm-dd]"])

    # Load config once
    with open(config.VECTRONICS_PREPROCESSING_YAML) as f:
        vectronics_config = yaml.safe_load(f)
    feature_cols = vectronics_config["feature_cols"]

    # Create plot directory
    plot_dir = os.path.join(io.get_sightings_dir(), "weak labels", "eval_plots")
    os.makedirs(plot_dir, exist_ok=True)

    for (animal_id, date), sighting_df in tqdm(grouped, desc="Processing animal-days"):
        # Subset to day-specific data
        day_data = RVC_df[
            (RVC_df["animal_id"] == animal_id)
            & (RVC_df["UTC date [yyyy-mm-dd]"] == date)
        ]

        if day_data.empty:
            if verbose:
                print(f"Skipping {animal_id} on {date}: no RVC data found.")
            continue

        timestamps = pd.to_datetime(day_data["UTC time [yyyy-mm-dd HH:MM:SS]"])
        signals = day_data[feature_cols].values
        labels = day_data["behavior"].values
        firmware_major_version = int(day_data["firmware_major_version"].iloc[0])

        # Define plot path
        plot_path = os.path.join(plot_dir, f"{animal_id}_{date}.png")

        # Generate and save plot
        plot_signal_and_feeding(
            timestamps=timestamps,
            signal=signals,
            labels=labels,
            plot_path=plot_path,
            matched_sightings=sighting_df,
            title=f"{animal_id} | {date} | Version-{firmware_major_version}",
        )

    return 0



if __name__ == "__main__":

    # file_path = os.path.join(io.get_data_path(), 'sightings.csv')
    # print("Loading sightings...")
    # sightings_df = load_sightings(file_path)

    # print("Loading RVC data...")
    # RVC_df = pd.read_csv(io.get_RVC_preprocessed_path())

    # print("Matching sightings with RVC data..")
    # matched_sightings = match_sightings(sightings_df, RVC_df)
    # matched_sightings.to_csv(os.path.join(io.get_data_path(), 'matched_sightings.csv'), index=False)
    # print(f"Number of matched sightings: {matched_sightings.sightings_index.nunique()}/{len(sightings_df)}")

    matched_sightings = pd.read_csv(os.path.join(io.get_data_path(), 'matched_sightings.csv'))
    make_sightings_plots_from_dann(matched_sightings, device='cuda')
