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


def plot_signal_and_scores(timestamps: pd.Series,
                           signal: np.ndarray,
                           scores: np.ndarray,
                           label_encoder: LabelEncoder,
                           predictions: np.array = None,
                           probs: np.ndarray = None,
                           plot_path: str = None,
                           matched_sightings: pd.DataFrame = None,
                           matched_gps_moving: pd.DataFrame = None,
                           matched_gps_feeding: pd.DataFrame = None,
                           title: str = None):
    """
    Plot tri-axial signal, softmax scores, and moving distance over time
    with optional suptitle and table of label distribution (probs).
    
    Returns:
        fig: matplotlib Figure
        axes: tuple(ax_signal, ax_online, ax_moving)
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
    fig = plt.figure(figsize=(25, 15))
    if title:
        fig.suptitle(title, fontsize=25, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(
        3, 3,
        width_ratios=[30, 1, 15],
        height_ratios=[1, 1, 1],
        wspace=0.25, hspace=0.7
    )

    ax_signal = fig.add_subplot(gs[0, 0])
    ax_moving = fig.add_subplot(gs[1, 0], sharex=ax_signal)
    ax_online = fig.add_subplot(gs[2, 0], sharex=ax_signal)
    cbar_ax = fig.add_subplot(gs[2, 1])
    table_ax = fig.add_subplot(gs[:, 2])

    table_ax.axis('off')

    # --- Plot tri-axial signals ---
    colors = ['black', 'blue', 'maroon']
    labels = ['X Signal', 'Y Signal', 'Z Signal']
    for i in range(3):
        ax_signal.plot(timestamps, signal[:, i], label=labels[i], color=colors[i], linewidth=1., alpha=0.6)
    ax_signal.set_ylabel("Amplitude (g)")
    ax_signal.set_title("Summary Signal")
    ax_signal.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax_signal.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    # --- Highlight matched sightings ---
    if matched_sightings is not None and not matched_sightings.empty:
        matched_sightings = matched_sightings.copy()
        matched_sightings['Sighting time [yyyy-mm-dd HH:MM:SS]'] = pd.to_datetime(
            matched_sightings['Sighting time [yyyy-mm-dd HH:MM:SS]']
        )
        for j, (_, row) in enumerate(matched_sightings.iterrows()):
            start = row['Sighting time [yyyy-mm-dd HH:MM:SS]'] - timedelta(minutes=30)
            end = row['Sighting time [yyyy-mm-dd HH:MM:SS]'] + timedelta(minutes=30)
            ax_signal.axvspan(start, end, color='pink', alpha=0.4, label="Sighting" if j == 0 else None)

    # --- Highlight matched GPS moving instances ---
    if matched_gps_moving is not None and not matched_gps_moving.empty:
        matched_gps_moving = matched_gps_moving.copy()
        matched_gps_moving['timestamp_start [yyyy-mm-dd HH:MM:SS]'] = pd.to_datetime(
            matched_gps_moving['timestamp_start [yyyy-mm-dd HH:MM:SS]']
        )
        matched_gps_moving['timestamp_end [yyyy-mm-dd HH:MM:SS]'] = pd.to_datetime(
            matched_gps_moving['timestamp_end [yyyy-mm-dd HH:MM:SS]']
        )
        for j, (_, row) in enumerate(matched_gps_moving.iterrows()):
            start = row['timestamp_start [yyyy-mm-dd HH:MM:SS]']
            end = row['timestamp_end [yyyy-mm-dd HH:MM:SS]']
            ax_signal.axvspan(start, end, color='lightblue', alpha=0.4, label="GPS Moving" if j == 0 else None)

    # --- Highlight matched GPS feeding instances ---
    if matched_gps_feeding is not None and not matched_gps_feeding.empty:
        matched_gps_feeding = matched_gps_feeding.copy()
        matched_gps_feeding['timestamp_start [yyyy-mm-dd HH:MM:SS]'] = pd.to_datetime(
            matched_gps_feeding['timestamp_start [yyyy-mm-dd HH:MM:SS]']
        )
        matched_gps_feeding['timestamp_end [yyyy-mm-dd HH:MM:SS]'] = pd.to_datetime(
            matched_gps_feeding['timestamp_end [yyyy-mm-dd HH:MM:SS]']
        )
        for j, (_, row) in enumerate(matched_gps_feeding.iterrows()):
            start = row['timestamp_start [yyyy-mm-dd HH:MM:SS]']
            end = row['timestamp_end [yyyy-mm-dd HH:MM:SS]']
            ax_signal.axvspan(start, end, color='olive', alpha=0.4, label="GPS Feeding" if j == 0 else None)

    # --- Add orange dots for predicted Feeding ---
    if predictions is not None:
        if np.issubdtype(predictions.dtype, np.number):
            pred_labels = label_encoder.inverse_transform(predictions)
        else:
            pred_labels = predictions
        feeding_mask = (pred_labels == "Feeding")
        feeding_t = timestamps[feeding_mask]
        feeding_y = np.full(len(feeding_t), np.max(signal))
        ax_signal.scatter(
            feeding_t, feeding_y,
            color='red', s=40, label="Predicted Feeding", zorder=5
        )

    ax_signal.legend(
        loc='upper left', bbox_to_anchor=(1.02, 1.0),
        fontsize=15, frameon=False
    )


    # --- Plot GPS moving horizontal bars with y = distance [m] ---
    if matched_gps_moving is not None and not matched_gps_moving.empty:
        matched_gps_moving = matched_gps_moving.copy()
        matched_gps_moving['timestamp_start [yyyy-mm-dd HH:MM:SS]'] = pd.to_datetime(
            matched_gps_moving['timestamp_start [yyyy-mm-dd HH:MM:SS]'])
        matched_gps_moving['timestamp_end [yyyy-mm-dd HH:MM:SS]'] = pd.to_datetime(
            matched_gps_moving['timestamp_end [yyyy-mm-dd HH:MM:SS]'])

        for _, row in matched_gps_moving.iterrows():
            ax_moving.plot(
                [row['timestamp_start [yyyy-mm-dd HH:MM:SS]'], row['timestamp_end [yyyy-mm-dd HH:MM:SS]']],
                [row['distance [m]'], row['distance [m]']],
                color='blue', linewidth=3
            )


    ax_moving.set_ylabel("Distance [m]")
    ax_moving.set_title("GPS Moving Segments")
    ax_moving.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

    # --- Plot softmax scores ---
    scatter = ax_online.scatter(X_flat, Y_flat, c=color_flat, cmap='Blues', s=140, marker='s', alpha=0.7)
    ax_online.set_yticks(y)
    ax_online.set_yticklabels(y_labels)
    ax_online.set_ylim(-1, n_classes)
    ax_online.set_title("Predicted Scores")
    ax_online.set_xlabel("Time (h)")
    ax_online.sharex(ax_signal)

    # --- Colorbar ---
    cbar = plt.colorbar(scatter, cax=cbar_ax)
    cbar.set_label("Softmax Score", fontsize=18)
    pos = cbar_ax.get_position()
    cbar_ax.set_position([pos.x0 - 0.02, pos.y0 + 0.01, pos.width * 0.5, pos.height - 0.02])

    # --- Probability Table ---
    if probs is not None:
        cell_text = []
        for i, p in enumerate(probs):
            label = label_encoder.inverse_transform([i])[0]
            cell_text.append([label, f"{p:.2f}%"])
        table = table_ax.table(
            cellText=cell_text,
            colLabels=["Behavior", "Probability"],
            cellLoc='center', colLoc='center', loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(18)
        table.scale(0.8, 3.0)

    # --- Save figure ---
    if plot_path:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    plt.close(fig)
    return fig, (ax_signal, ax_online, ax_moving)

# def plot_signal_and_scores(timestamps: pd.Series,
#                            signal: np.ndarray,
#                            scores: np.ndarray,
#                            label_encoder: LabelEncoder,
#                            predictions: np.array = None,
#                            probs: np.ndarray = None,  # add probs array
#                            plot_path: str = None,
#                            matched_sightings: pd.DataFrame = None,
#                            matched_gps_moving: pd.DataFrame = None,
#                            matched_gps_feeding: pd.DataFrame = None,
#                            title: str = None):
#     """
#     Plot tri-axial signal and softmax scores over time with optional suptitle
#     and table of label distribution (probs).
    
#     Returns:
#         fig: matplotlib Figure
#         axes: tuple(ax_signal, ax_online)
#     """
   
#     sns.set_style("whitegrid")

#     timestamps = pd.to_datetime(timestamps)
#     n_classes = scores.shape[1]
#     y = np.arange(n_classes)

#     # Flatten for scatter plotting
#     X, Y = np.meshgrid(timestamps, y)
#     X_flat, Y_flat = X.flatten(), Y.flatten()
#     color_flat = scores.T.flatten()
#     y_labels = label_encoder.inverse_transform(y)

#     # Figure and GridSpec
#     fig = plt.figure(figsize=(25, 8))  # a little wider to fit table
#     if title:
#         fig.suptitle(title, fontsize=25, fontweight='bold', y=0.98)

#     gs = gridspec.GridSpec(2, 3, width_ratios=[30, 1, 15], height_ratios=[1, 1], wspace=0.25, hspace=0.6)

#     ax_signal = fig.add_subplot(gs[0, 0])
#     ax_online = fig.add_subplot(gs[1, 0])
#     cbar_ax = fig.add_subplot(gs[1, 1])
#     table_ax = fig.add_subplot(gs[:, 2])  # span both rows

#     table_ax.axis('off')  # hide axes for table

#     # Plot tri-axial signals
#     colors = ['black', 'blue', 'maroon']
#     labels = ['X Signal', 'Y Signal', 'Z Signal']
#     for i in range(3):
#         ax_signal.plot(timestamps, signal[:, i], label=labels[i], color=colors[i], linewidth=1., alpha=0.6)
#     ax_signal.set_xlabel("Time (h)")
#     ax_signal.set_ylabel("Amplitude (g)")
#     ax_signal.set_title("Summary Signal")
#     ax_signal.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
#     ax_signal.xaxis.set_major_locator(mdates.HourLocator(interval=2))

#     # Highlight matched sightings if provided
#     if matched_sightings is not None and not matched_sightings.empty:
#         matched_sightings['Sighting time [yyyy-mm-dd HH:MM:SS]'] = pd.to_datetime(matched_sightings['Sighting time [yyyy-mm-dd HH:MM:SS]'])
#         for j, (_, row) in enumerate(matched_sightings.iterrows()):
#             start = row['Sighting time [yyyy-mm-dd HH:MM:SS]'] - timedelta(minutes=30)
#             end   = row['Sighting time [yyyy-mm-dd HH:MM:SS]'] + timedelta(minutes=30)
#             ax_signal.axvspan(start, end, color='pink', alpha=0.4, label="Sighting" if j == 0 else None)

#     # Highlight matched GPS moving instances if provided
#     if matched_gps_moving is not None and not matched_gps_moving.empty:
#         matched_gps_moving = matched_gps_moving.copy()
#         matched_gps_moving['timestamp_start [yyyy-mm-dd HH:MM:SS]'] = pd.to_datetime(matched_gps_moving['timestamp_start [yyyy-mm-dd HH:MM:SS]'])
#         matched_gps_moving['timestamp_end [yyyy-mm-dd HH:MM:SS]'] = pd.to_datetime(matched_gps_moving['timestamp_end [yyyy-mm-dd HH:MM:SS]'])

#         for j, (_, row) in enumerate(matched_gps_moving.iterrows()):
#             start = row['timestamp_start [yyyy-mm-dd HH:MM:SS]']
#             end   = row['timestamp_end [yyyy-mm-dd HH:MM:SS]'] 
#             ax_signal.axvspan(start, end, color='lightblue', alpha=0.4, label="GPS Moving" if j == 0 else None)

#     # Highlight matched GPS feeding instances if provided
#     if matched_gps_feeding is not None and not matched_gps_moving.empty:
#         matched_gps_feeding = matched_gps_feeding.copy()
#         matched_gps_feeding['timestamp_start [yyyy-mm-dd HH:MM:SS]'] = pd.to_datetime(matched_gps_moving['timestamp_start [yyyy-mm-dd HH:MM:SS]'])
#         matched_gps_feeding['timestamp_end [yyyy-mm-dd HH:MM:SS]'] = pd.to_datetime(matched_gps_moving['timestamp_end [yyyy-mm-dd HH:MM:SS]'])

#         for j, (_, row) in enumerate(matched_gps_feeding.iterrows()):
#             start = row['timestamp_start [yyyy-mm-dd HH:MM:SS]']
#             end   = row['timestamp_end [yyyy-mm-dd HH:MM:SS]'] 
#             ax_signal.axvspan(start, end, color='olive', alpha=0.4, label="GPS Feeding" if j == 0 else None)
    
#     # --- Add orange dots where predictions == 'Feeding' ---
#     if predictions is not None:
#         # Decode predictions if numeric
#         if np.issubdtype(predictions.dtype, np.number):
#             pred_labels = label_encoder.inverse_transform(predictions)
#         else:
#             pred_labels = predictions

#         # Boolean mask for "Feeding"
#         feeding_mask = (pred_labels == "Feeding")

#         # y-position for dots: mean of tri-axial signals at each feeding timestamp
#         feeding_t = timestamps[feeding_mask]
#         feeding_y = np.full(len(feeding_t), signal.max().item())

#         # Plot orange dots
#         ax_signal.scatter(
#             feeding_t,
#             feeding_y,
#             color='red',
#             s=40,
#             label="Predicted Feeding",
#             zorder=5
#         )

#     ax_signal.legend(
#     loc='upper left',
#     bbox_to_anchor=(1.02, 1.0),
#     fontsize=16,
#     frameon=False)

#     # Plot softmax scores
#     scatter = ax_online.scatter(X_flat, Y_flat, c=color_flat, cmap='Blues', s=140, marker='s', alpha=0.7)
#     ax_online.set_xlabel("Time (h)")
#     ax_online.set_yticks(y)
#     ax_online.set_yticklabels(y_labels)
#     ax_online.set_ylim(-1, n_classes)
#     ax_online.set_title("Predicted Scores")
#     ax_online.sharex(ax_signal)

#     # Colorbar
#     cbar = plt.colorbar(scatter, cax=cbar_ax)
#     cbar.set_label("Softmax Score", fontsize=20)
#     pos = cbar_ax.get_position()
#     cbar_ax.set_position([pos.x0 - 0.02, pos.y0 + 0.01, pos.width, pos.height - 0.01])

#     # Add table of label distribution if probs provided
#     if probs is not None:
#         cell_text = []
#         for i, p in enumerate(probs):
#             label = label_encoder.inverse_transform([i])[0]
#             cell_text.append([label, f"{p:.2f}%"])
#         table = table_ax.table(cellText=cell_text,
#                                colLabels=["Behavior", "Probability"],
#                                cellLoc='center',
#                                colLoc='center',
#                                loc='center')
#         table.auto_set_font_size(False)
#         table.set_fontsize(20)
#         table.scale(0.8, 3.0)  # scale width/height
    

#     # Save figure
#     if plot_path:
#         os.makedirs(os.path.dirname(plot_path), exist_ok=True)
#         plt.savefig(plot_path, dpi=300, bbox_inches='tight')

#     plt.close(fig)
#     return fig, (ax_signal, ax_online)


def plot_signal_and_behaviors(
    timestamps: pd.Series,
    signal: np.ndarray,
    labels: np.ndarray,
    behaviors: list = ['Feeding', 'Moving'],
    plot_path: str = None,
    matched_sightings: pd.DataFrame = None,
    matched_gps: pd.DataFrame = None,
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

    # Highlight matched GPS moving instances if provided
    if matched_gps is not None and not matched_gps.empty:
        matched_gps = matched_gps.copy()
        matched_gps['timestamp_start [yyyy-mm-dd HH:MM:SS]'] = pd.to_datetime(matched_gps['timestamp_start [yyyy-mm-dd HH:MM:SS]'])
        matched_gps['timestamp_end [yyyy-mm-dd HH:MM:SS]'] = pd.to_datetime(matched_gps['timestamp_end [yyyy-mm-dd HH:MM:SS]'])

        for j, (_, row) in enumerate(matched_gps.iterrows()):
            start = row['timestamp_start [yyyy-mm-dd HH:MM:SS]']
            end   = row['timestamp_end [yyyy-mm-dd HH:MM:SS]'] 
            ax_signal.axvspan(start, end, color='lightblue', alpha=0.4, label="GPS Moving" if j == 0 else None)

    # --- Behavior events ---
    behavior_colors = sns.color_palette("husl", len(behaviors))
    for b in behaviors:
        mask = labels == b
        print(len(mask))
        if mask.any():
            ax_signal.scatter(
                timestamps[mask],
                np.full(np.sum(mask), ax_signal.get_ylim()[1] * 0.9),
                color=behavior_colors[behaviors.index(b)],
                marker="|",
                s=200,
                label=b,
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

def make_sightings_plots_from_model(
        model: torch.nn.Module,
        data: torch.Tensor,
        metadata: pd.DataFrame,
        matched_sightings: pd.DataFrame, 
        matched_gps_moving: pd.DataFrame = None,
        matched_gps_feeding: pd.DataFrame = None,
        device: str = 'cpu',
        model_name: str = 'coral'
    ) -> dict:
    """
    Evaluate a DANN model on each unique (animal_id, date) pair and generate plots.

    Args:
        model (torch.nn.Module): Trained DANN model.
        data (np.ndarray): Input feature array aligned with `metadata` (rows correspond to metadata rows).
        metadata (pd.DataFrame): Metadata containing at least ['animal_id', 'UTC date [yyyy-mm-dd]', 'UTC time [yyyy-mm-dd HH:MM:SS]', 'firmware_major_version'].
        matched_sightings (pd.DataFrame): DataFrame of sighting events with ['animal_id', 'UTC date [yyyy-mm-dd]'].
        matched_gps (pd.DataFrame, optional): GPS data matched to sightings (same grouping keys).
        device (str, optional): Device to run inference on ('cpu' or 'cuda').

    Returns:
        dict: Mapping from (animal_id, date) -> {'softmax_scores': np.ndarray, 'label_distribution': np.ndarray}.
    """

    results = {}
    label_encoder = LabelEncoder()
    label_encoder.fit(config.SUMMARY_BEHAVIORS)

    metadata['UTC time [yyyy-mm-dd HH:MM:SS]'] = pd.to_datetime(metadata['UTC time [yyyy-mm-dd HH:MM:SS]'], errors='coerce')

    # Group sightings by animal-day
    grouped_sightings = matched_sightings.groupby(['animal_id', 'UTC date [yyyy-mm-dd]'])

    # Ensure output directory exists
    plot_dir = os.path.join(io.get_sightings_dir(), model_name, 'eval_plots')
    os.makedirs(plot_dir, exist_ok=True)

    print(len(matched_gps_feeding))

    for (animal_id, date), sighting_group in tqdm(grouped_sightings, desc="Processing animal-days"):
        # Select metadata for this animal-day
        day_mask = (
            (metadata['animal_id'] == animal_id) &
            (metadata['UTC date [yyyy-mm-dd]'] == date)
        )
        day_indices = np.where(day_mask)[0]

        if len(day_indices) == 0:
            continue  # skip if no matching metadata rows

        # Prepare data and timestamps
        timestamps = metadata.loc[day_indices, 'UTC time [yyyy-mm-dd HH:MM:SS]']
        X_day = data[day_indices]
        firmware_version = metadata.loc[day_indices, 'firmware_major_version'].iloc[0]

        # Get matching GPS subset (optional)
        gps_moving_group, gps_feeding_group = None, None
        if matched_gps_moving is not None:
            gps_moving_group = matched_gps_moving[
                (matched_gps_moving['animal_id'] == animal_id) &
                (matched_gps_moving['UTC date [yyyy-mm-dd]'] == date)
            ]
        if matched_gps_feeding is not None:
            gps_feeding_group = matched_gps_feeding[
                (matched_gps_feeding['animal_id'] == animal_id) &
                (matched_gps_feeding['UTC date [yyyy-mm-dd]'] == date)
            ]

        print(len(gps_moving_group), len(gps_feeding_group))

        # Evaluate model on this animal-day
        predictions, softmax_scores, label_distribution = evaluate_label_distribution(
            model=model,
            data=X_day,
            n_classes=len(config.SUMMARY_BEHAVIORS),
            label_encoder=label_encoder,
            device=device,
            verbose=False
        )

        # Generate and save plot
        plot_path = os.path.join(plot_dir, f"{animal_id}_{date}.png")
        plot_signal_and_scores(
            timestamps=timestamps,
            signal=X_day.numpy(),
            predictions=predictions,
            scores=softmax_scores,
            label_encoder=label_encoder,
            probs=label_distribution,
            plot_path=plot_path,
            matched_sightings=sighting_group,
            matched_gps_moving=gps_moving_group,
            matched_gps_feeding=gps_feeding_group,
            title=f"{animal_id} | {date} | Version-{int(firmware_version)}"
        )

        # Store results
        results[(animal_id, date)] = {
            "softmax_scores": softmax_scores,
            "label_distribution": label_distribution
        }

    return results


def make_sightings_plots_from_labels(
        matched_sightings: pd.DataFrame,
        matched_gps: pd.DataFrame,
        RVC_df: pd.DataFrame,
        verbose: bool = False,
        label_name: str = "weak labels"
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
    plot_dir = os.path.join(io.get_sightings_dir(), label_name, "eval_plots")
    os.makedirs(plot_dir, exist_ok=True)

    for (animal_id, date), group_df in tqdm(grouped, desc="Processing animal-days"):

        # Subset to day-specific data
        day_data = RVC_df[
            (RVC_df["animal_id"] == animal_id)
            & (RVC_df["UTC date [yyyy-mm-dd]"] == date)
        ]

        group_gps = matched_gps[(matched_gps['animal_id'] == animal_id) & (matched_gps['UTC date [yyyy-mm-dd]'] == date)]

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
        plot_signal_and_behaviors(
            timestamps=timestamps,
            signal=signals,
            labels=labels,
            behaviors=['Feeding', 'Moving'],
            plot_path=plot_path,
            matched_sightings=group_df,
            matched_gps=group_gps,
            title=f"{animal_id} | {date} | Version-{firmware_major_version}",
        )

    return 0

def make_sightings_plots_from_clusters(
        cluster_labels: np.ndarray,
        cluster_scores: np.ndarray,
        data: torch.Tensor,
        metadata: pd.DataFrame,
        matched_sightings: pd.DataFrame, 
        matched_gps: pd.DataFrame = None,
        label_name: str = "dann",
    ) -> dict:
    """
    Evaluate a DANN model on each unique (animal_id, date) pair and generate plots.

    Args:
        model (torch.nn.Module): Trained DANN model.
        data (np.ndarray): Input feature array aligned with `metadata` (rows correspond to metadata rows).
        metadata (pd.DataFrame): Metadata containing at least ['animal_id', 'UTC date [yyyy-mm-dd]', 'UTC time [yyyy-mm-dd HH:MM:SS]', 'firmware_major_version'].
        matched_sightings (pd.DataFrame): DataFrame of sighting events with ['animal_id', 'UTC date [yyyy-mm-dd]'].
        matched_gps (pd.DataFrame, optional): GPS data matched to sightings (same grouping keys).
        device (str, optional): Device to run inference on ('cpu' or 'cuda').

    Returns:
        dict: Mapping from (animal_id, date) -> {'softmax_scores': np.ndarray, 'label_distribution': np.ndarray}.
    """

    results = {}
    label_encoder = LabelEncoder()
    label_encoder.fit(config.SUMMARY_BEHAVIORS)

    metadata['UTC time [yyyy-mm-dd HH:MM:SS]'] = pd.to_datetime(metadata['UTC time [yyyy-mm-dd HH:MM:SS]'], errors='coerce')

    # Group sightings by animal-day
    grouped_sightings = matched_sightings.groupby(['animal_id', 'UTC date [yyyy-mm-dd]'])

    # Ensure output directory exists
    plot_dir = os.path.join(io.get_sightings_dir(), label_name, 'calibrated')
    os.makedirs(plot_dir, exist_ok=True)

    for (animal_id, date), sighting_group in tqdm(grouped_sightings, desc="Processing animal-days"):
        # Select metadata for this animal-day
        day_mask = (
            (metadata['animal_id'] == animal_id) &
            (metadata['UTC date [yyyy-mm-dd]'] == date)
        )
        day_indices = np.where(day_mask)[0]

        if len(day_indices) == 0:
            continue  # skip if no matching metadata rows

        # Prepare data and timestamps
        timestamps = metadata.loc[day_indices, 'UTC time [yyyy-mm-dd HH:MM:SS]']
        X_day = data[day_indices]
        cluster_labels_day = cluster_labels[day_indices]
        cluster_scores_day = cluster_scores[cluster_labels_day]
        firmware_version = metadata.loc[day_indices, 'firmware_major_version'].iloc[0]

        # Get matching GPS subset (optional)
        gps_group = None
        if matched_gps is not None:
            gps_group = matched_gps[
                (matched_gps['animal_id'] == animal_id) &
                (matched_gps['UTC date [yyyy-mm-dd]'] == date)
            ]

        label_distribution = np.mean(cluster_scores_day, axis=0)*100

        # Generate and save plot
        plot_path = os.path.join(plot_dir, f"{animal_id}_{date}.png")
        plot_signal_and_scores(
            timestamps=timestamps,
            signal=X_day,
            scores=cluster_scores_day,
            label_encoder=label_encoder,
            probs=label_distribution,
            plot_path=plot_path,
            matched_sightings=sighting_group,
            matched_gps=gps_group,
            title=f"{animal_id} | {date} | Version-{int(firmware_version)}"
        )

        # Store results
        results[(animal_id, date)] = {
            "softmax_scores": cluster_scores_day,
            "label_distribution": label_distribution
        }

    return results