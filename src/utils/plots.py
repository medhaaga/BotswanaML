import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
import config as config
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import matplotlib.dates as mdates
# Graphing Parameters
import matplotlib as mpl
mpl.rcParams['lines.markersize'] = 12
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['xtick.labelsize'] = 25
mpl.rcParams['ytick.labelsize'] = 25
mpl.rcParams["axes.labelsize"] = 25
mpl.rcParams['legend.fontsize'] = 25
mpl.rcParams['axes.titlesize'] = 25
mpl.rcParams['text.usetex'] = False

def multi_label_predictions(dir, label_encoder, split='test', plot_confusion=True, return_accuracy=False, return_precision=False, return_recall=False, return_f1=False, plot_path=None, average='macro'):
    
    if split == 'test':
        y = np.load(os.path.join(dir, 'test_true_classes.npy'))
        predictions = np.load(os.path.join(dir, 'test_predictions.npy'))

    elif split == 'val':
        y = np.load(os.path.join(dir, 'val_true_classes.npy'))
        predictions = np.load(os.path.join(dir, 'val_predictions.npy'))
    else:
        raise ValueError

    # plot confusion matrices

    if plot_confusion:
        cm = confusion_matrix(y, predictions, normalize='true')
        class_names = label_encoder.inverse_transform(np.arange(len(np.unique(y))))

        plt.clf()

        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar=False, square=True, linewidths=0,
                    annot_kws={"size": 20},)

        ax.set_xlabel("Predicted Label", fontsize=30, labelpad=20)
        ax.set_ylabel("True Label", fontsize=30, labelpad=20)
        ax.set_xticklabels(class_names, fontsize=25, rotation=90)
        ax.set_yticklabels(class_names, fontsize=25, rotation=0)

        plt.tight_layout()
        if plot_path:
            plt.savefig(plot_path, format="png", bbox_inches="tight")
        plt.show()
    
    if return_accuracy:
        label_accuracies = accuracy_score(y, predictions) 
        return label_accuracies
    
    if return_precision:
        label_precisions = precision_score(y, predictions, average=average, zero_division=0) 
        return label_precisions
    
    if return_recall:
        label_recalls = recall_score(y, predictions, average=average, zero_division=0)
        return label_recalls
    
    if return_f1:
        label_f1s = f1_score(y, predictions, average=average, zero_division=0) 
        return label_f1s


def plot_signal_and_online_predictions(time, signal, online_avg, window_length, hop_length, window_duration, label_encoder, plot_dir=None, half_day_behaviors=None):
    """
    Plots the raw signal and the online predictions.

    Parameters:
    ---------------
    - time: time stamps of the signal (1D array).
    - signal: The raw signal data (2D array).
    - online_avg: The average online prediction probabilities (2D array).
    - window_length: Length of each window for smoothening.
    - hop_length: Overlap length between windows in data points.
    - window_duration: Duration of each window in seconds.
    - label_encoder: A label encoder for behavior labels.
    - sampling_rate: The sampling rate of the signal (Hz).
    - plot_dir: Directory where the plot will be saved. If None, the plot is not saved.
    """
    
    sns.set_style("whitegrid")  # Set seaborn style at the beginning

    time = pd.to_datetime(time)

    # Calculate x-axis in hours
    x = window_duration * (np.arange(online_avg.shape[-1]) * hop_length + window_length / 2) / 3600
    
    # y-axis labels for each row in online_avg
    y = np.arange(online_avg.shape[0])

    # Create a mesh grid for plotting
    X, Y = np.meshgrid(x, y)
    color_intensity = online_avg

    # Flatten the mesh grid and color intensity for scatter plot
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    color_flat = color_intensity.flatten()

    # Get inverse transformed labels for y-axis
    y_labels = label_encoder.inverse_transform(y)

    # Create figure and GridSpec
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[30, 1], height_ratios=[1, 1], wspace=0.2, hspace=0.6)

    # Create subplots
    ax_signal = fig.add_subplot(gs[0, 0])
    ax_online = fig.add_subplot(gs[1, 0])
    cbar_ax = fig.add_subplot(gs[1, 1])

    # Plot the signal
    ax_signal.plot(time, signal[0,:], label='X Signal', color='black', linewidth=.5, alpha=0.6)
    ax_signal.plot(time, signal[1,:], label='Y Signal', color='blue', linewidth=.5, alpha=0.5)
    ax_signal.plot(time, signal[2,:], label='Z Signal', color='maroon', linewidth=.5, alpha=0.4)
    ax_signal.set_xlabel('Time (h)')
    ax_signal.set_ylabel("Amplitude (g)")
    ax_signal.set_title("Raw Signal")
    ax_signal.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax_signal.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    
    # Plot behaviors

    if half_day_behaviors is not None:
        colors = dict(zip(config.RAW_BEHAVIORS, sns.color_palette("husl", len(config.RAW_BEHAVIORS))))
        behaviors = half_day_behaviors['behavior'].unique()

        for i in range(len(half_day_behaviors)):
            behavior = half_day_behaviors.iloc[i]['behavior']
            ax_signal.axvspan(half_day_behaviors.iloc[i]['behavior_start'], 
                              half_day_behaviors.iloc[i]['behavior_end'], 
                              color=colors[behavior], 
                              alpha=0.3)
        legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[behavior], markersize=10, alpha=0.3, label=behavior)
                    for behavior in label_encoder.classes_]
        ax_signal.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.26, 1.00), fontsize=25)
    else:
        ax_signal.legend(loc='upper right', bbox_to_anchor=(1.26, 1.00), fontsize=25)




    # Plot the online predictions
    scatter = ax_online.scatter(X_flat, Y_flat, c=color_flat, cmap='Blues', s=140, marker='s', alpha=0.7)
    ax_online.set_xlabel("Time (h)")
    ax_online.set_yticks(y)
    ax_online.set_yticklabels(y_labels)
    ax_online.set_ylim(-1, len(y))
    ax_online.set_title(f"Online Predictions, $s = {window_length}$")


    # Add colorbar to the scatter plot, placed in the separate axis
    cbar = plt.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Softmax Score', fontsize=25)

    # Shift colorbar up so it doesn't align with the x-axis label
    pos = cbar_ax.get_position()
    cbar_ax.set_position([pos.x0 - 0.02, pos.y0 + 0.01, pos.width, pos.height-0.01])

    # Adjust layout to fit everything
    plt.tight_layout()  # Reduce right space for colorbar

    # Save plot if plot_dir is specified
    if plot_dir is not None:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)  # Create directory if it doesn't exist
        plt.savefig(os.path.join(plot_dir, f'window_length_{window_length}.png'), dpi=300, bbox_inches='tight')
    
    return fig, (ax_signal, ax_online)