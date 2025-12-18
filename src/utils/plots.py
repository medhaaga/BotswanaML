import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
import config as config
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    roc_curve,
    auc
)

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
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
mpl.rcParams['text.usetex'] = True

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

def multi_label_predictions(dir, label_encoder, split='test', plot_confusion=True,
                            return_accuracy=False, return_precision=False, return_recall=False,
                            return_f1=False, plot_path=None, average='macro', threshold=0.5):
    
    # Load predictions and true labels
    if split == 'test':
        results = np.load(os.path.join(dir, "test_results.npz"))
        y_true = results["true_classes"]        # shape: (N, C)
        y_pred = results["predictions"]         # shape: (N, C)
    elif split == 'val':
        results = np.load(os.path.join(dir, "val_results.npz"))
        y_true = results["true_classes"]        # shape: (N, C)
        y_pred = results["predictions"]         # shape: (N, C)
    elif split == 'target_test':
        results = np.load(os.path.join(dir, "target_test_results.npz"))
        y_true = results["true_classes"]        # shape: (N, C)
        y_pred = results["predictions"]         # shape: (N, C)
    elif split == 'target_val':
        results = np.load(os.path.join(dir, "target_val_results.npz"))
        y_true = results["true_classes"]        # shape: (N, C)
        y_pred = results["predictions"]         # shape: (N, C)
    else:
        raise ValueError("split must be 'test' or 'val'")

    
    n_classes = y_true.shape[1]
    class_names = label_encoder.inverse_transform(np.arange(n_classes))

    # ===================
    # Plot per-class confusion matrices
    # ===================
    if plot_confusion:
        fig, axes = plt.subplots(nrows=1, ncols=n_classes, figsize=(5 * n_classes, 5))
        if n_classes == 1:
            axes = [axes]  # Make it iterable
        
        for i in range(n_classes):
            cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1])
            # Row-wise normalization with division safety
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1   # prevent division-by-zero
            cm_norm = cm / row_sums

                    
            sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                        xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
                        cbar=False, square=True, annot_kws={"size": 25}, ax=axes[i])
            axes[i].set_title(class_names[i])
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("True")
        
        plt.tight_layout()
        if plot_path:
            plt.savefig(plot_path, format="png", bbox_inches="tight")
        plt.show()

    # ===================
    # Compute metrics
    # ===================
    results = {}
    if return_accuracy:
        results['accuracy'] = accuracy_score(y_true, y_pred)
    if return_precision:
        results['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    if return_recall:
        results['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    if return_f1:
        results['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    if len(results) == 1:
        return list(results.values())[0]
    return results

def plot_multiclass_roc(labels, preds, label_encoder=None):


    labels = np.array(labels)
    preds = np.array(preds)

    n_classes = preds.shape[1]
    possible_classes = np.arange(n_classes)
    present_classes = np.unique(labels)

    # binarize only for present classes
    y_bin = label_binarize(labels, classes=possible_classes)

    plt.figure(figsize=(8, 6))

    for c in present_classes:
        fpr, tpr, _ = roc_curve(y_bin[:, c], preds[:, c])
        roc_auc = auc(fpr, tpr)

        if label_encoder is not None:
            name = label_encoder.inverse_transform([c])[0]
        else:
            name = f"Class: {c}"

        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.2f})")

    # chance line
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_multiclass_threshold_diagnostics(
    true_behaviors,
    prediction_scores,
    label_encoder,
    n_thresholds=101
):
    n_samples, n_classes = prediction_scores.shape
    thresholds = np.linspace(0.0, 1.0, n_thresholds)

    # Identify which classes are actually present
    present_classes = np.unique(true_behaviors)
    present_classes = present_classes[present_classes < n_classes]

    if len(present_classes) == 0:
        raise ValueError("No valid classes present in true_behaviors.")

    # One-vs-rest binarization (full width, but we'll index later)
    y_true_bin = label_binarize(
        true_behaviors,
        classes=np.arange(n_classes)
    )

    # Containers (only for present classes)
    precision_curves = {}
    recall_curves = {}
    pr_curves = {}
    roc_curves = {}
    aucs = {}

    # ---------- Precision / Recall vs threshold ----------
    for c in present_classes:
        precision_vals = []
        recall_vals = []

        for t in thresholds:
            y_pred_bin = (prediction_scores[:, c] >= t).astype(int)

            precision_vals.append(
                precision_score(
                    y_true_bin[:, c],
                    y_pred_bin,
                    zero_division=0
                )
            )
            recall_vals.append(
                recall_score(
                    y_true_bin[:, c],
                    y_pred_bin,
                    zero_division=0
                )
            )

        precision_curves[c] = np.array(precision_vals)
        recall_curves[c] = np.array(recall_vals)

    # Macro averages (only over present classes)
    precision_macro = np.mean(
        np.stack(list(precision_curves.values())), axis=0
    )
    recall_macro = np.mean(
        np.stack(list(recall_curves.values())), axis=0
    )

    # ---------- Precision–Recall curves ----------
    recall_grid = np.linspace(0, 1, 1000)
    interp_precisions = []

    for c in present_classes:
        p, r, _ = precision_recall_curve(
            y_true_bin[:, c],
            prediction_scores[:, c]
        )
        pr_curves[c] = (r, p)
        interp_precisions.append(
            np.interp(recall_grid, r[::-1], p[::-1])
        )

    pr_macro = np.mean(interp_precisions, axis=0)

    # ---------- ROC curves ----------
    fpr_grid = np.linspace(0, 1, 1000)
    interp_tprs = []

    for c in present_classes:
        fpr, tpr, _ = roc_curve(
            y_true_bin[:, c],
            prediction_scores[:, c]
        )
        roc_curves[c] = (fpr, tpr)
        aucs[c] = auc(fpr, tpr)
        interp_tprs.append(np.interp(fpr_grid, fpr, tpr))

    roc_macro = np.mean(interp_tprs, axis=0)
    auc_macro = auc(fpr_grid, roc_macro)

    # ---------- Plotting ----------
    fig, axes = plt.subplots(2, 2, figsize=(17, 12))

    # Precision vs threshold
    for c in present_classes:
        axes[0, 0].plot(
            thresholds,
            precision_curves[c], 
            linewidth=2,
            label=label_encoder.inverse_transform([c])[0]
        )
    axes[0, 0].plot(thresholds, precision_macro, linewidth=3, linestyle='--', label="Average", color='black')
    axes[0, 0].set_title("Precision vs Threshold")
    axes[0, 0].set_xlabel("Threshold")
    axes[0, 0].set_ylabel("Precision")
    axes[0, 0].grid(True)

    # Recall vs threshold
    for c in present_classes:
        axes[0, 1].plot(
            thresholds,
            recall_curves[c],
            linewidth=2,
            label=label_encoder.inverse_transform([c])[0]
        )
    axes[0, 1].plot(thresholds, recall_macro, linewidth=3, linestyle='--',color='black', label="Average")
    axes[0, 1].set_title("Recall vs Threshold")
    axes[0, 1].set_xlabel("Threshold")
    axes[0, 1].set_ylabel("Recall")
    axes[0, 1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1,)

    axes[0, 1].grid(True)

    # Precision–Recall curve
    for c in present_classes:
        r, p = pr_curves[c]
        axes[1, 0].plot(
            r, p,
            linewidth=2,
            label=label_encoder.inverse_transform([c])[0]
        )
    axes[1, 0].plot(recall_grid, pr_macro, linewidth=3, linestyle='--',color='black', label="Average")
    axes[1, 0].set_title("Precision–Recall Curve")
    axes[1, 0].set_xlabel("Recall")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].grid(True)

    # ROC curve
    for c in present_classes:
        fpr, tpr = roc_curves[c]
        axes[1, 1].plot(
            fpr, tpr,
            linewidth=2,
            label=f"{label_encoder.inverse_transform([c])[0]} (AUC={aucs[c]:.2f})"
        )
    axes[1, 1].plot(
        fpr_grid,
        roc_macro,
        linewidth=3,
        color='black', 
        linestyle='--',
        label=f"Average (AUC={auc_macro:.2f})"
    )
    axes[1, 1].plot([0, 1], [0, 1], linestyle="--")
    axes[1, 1].set_title("ROC Curve")
    axes[1, 1].set_xlabel("False Positive Rate")
    axes[1, 1].set_ylabel("True Positive Rate")
    axes[1, 1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1,)

    # axes[1, 1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=2)

    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_signal_and_online_predictions(time, signal, online_avg, online_avg_times, window_length, label_encoder, plot_path=None, half_day_behaviors=None):
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

    # Convert all timestamps to datetime objects for plotting
    time = pd.to_datetime(time)
    x_online = pd.to_datetime(online_avg_times) # Use the passed-in midpoint timestamps
        
    # y-axis labels for each row in online_avg
    y = np.arange(online_avg.shape[0])

    # Create a mesh grid for plotting using the new timestamps
    X, Y = np.meshgrid(x_online, y)
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
    ax_signal.plot(time, signal[0,:], label='X Signal', color='black', linewidth=1., alpha=0.6)
    ax_signal.plot(time, signal[1,:], label='Y Signal', color='blue', linewidth=1., alpha=0.6)
    ax_signal.plot(time, signal[2,:], label='Z Signal', color='maroon', linewidth=1., alpha=0.6)
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

    # Share the x-axis between the two plots to ensure perfect alignment
    ax_online.sharex(ax_signal)

    # Add colorbar to the scatter plot, placed in the separate axis
    cbar = plt.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Softmax Score', fontsize=25)

    # Shift colorbar up so it doesn't align with the x-axis label
    pos = cbar_ax.get_position()
    cbar_ax.set_position([pos.x0 - 0.02, pos.y0 + 0.01, pos.width, pos.height-0.01])

    # Adjust layout to fit everything
    # plt.tight_layout()  # Reduce right space for colorbar

    # Save plot if plot_dir is specified
    if plot_path is not None:
        if not os.path.exists(os.path.dirname(plot_path)):
            os.makedirs(os.path.dirname(plot_path))  # Create directory if it doesn't exist
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)
    return fig, (ax_signal, ax_online)

def plot_scores(time, scores, label_encoder=None, plot_path=None, plot_title=""):
    time = pd.to_datetime(time)
    y = np.arange(scores.shape[0])
    
    if label_encoder is not None:
        y_labels = label_encoder.inverse_transform(y)
    else:
        y_labels = y

    fig, ax = plt.subplots(figsize=(15, 4))

    for i in range(scores.shape[0]):
        ax.scatter(time, [i]*len(time), c=scores[i], cmap='Blues', s=140, marker='s', alpha=0.7)

    ax.set_xlabel("Time (h)")
    ax.set_yticks(y)
    ax.set_yticklabels(y_labels)
    ax.set_ylim(-1, len(y))
    ax.set_title(plot_title)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())


    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Softmax Score', fontsize=25)

    if plot_path:
        directory = os.path.dirname(plot_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    plt.close(fig)

def plot_feature_histograms(X_src, X_targets, bins=50, fname="feature_hists.png"):
    """
    Plot histograms of features for source and multiple target domains.

    Parameters
    ----------
    X_src : np.ndarray, shape (n_samples, n_features)
        Source dataset (train).
    X_targets : list of np.ndarray
        List of target domain arrays, each shape (n_samples, n_features).
    bins : int
        Number of histogram bins.
    fname : str
        Path to save the figure.
    """
    n_features = X_src.shape[1]
    n_rows, n_cols = 3, 3
    assert n_features <= n_rows * n_cols, "Increase subplot grid size for >9 features"

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))
    axes = axes.ravel()

    colors = ["C0", "C1", "C2", "C3", "C4"]  # extend if more domains
    labels = ["Train"] + [f"Target{i+1}" for i in range(len(X_targets))]

    for j in range(n_features):
        ax = axes[j]
        ax.hist(X_src[:, j], bins=bins, color=colors[0],
                histtype="step", density=True, label=labels[0])
        for i, Xt in enumerate(X_targets):
            ax.hist(Xt[:, j], histtype="step", bins=bins, color=colors[i+1], density=True, label=labels[i+1])
        ax.set_title(f"Feature {j}", fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        if j == 0:
            ax.legend()
        ax.set_yscale("log")
        ax.set_ylabel("Log Density")
    # Hide unused subplots
    for k in range(n_features, n_rows * n_cols):
        axes[k].axis("off")

    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"Saved histogram plot to {fname}")

def make_multilabel_confusion_matrices(y_true, y_pred, label_encoder):

    n_classes = y_true.shape[1]
    fig, axes = plt.subplots(nrows=1, ncols=n_classes, figsize=(5 * n_classes, 5))

    for i in range(n_classes):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1])
        # Row-wise normalization with division safety
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1   # prevent division-by-zero
        cm_norm = cm / row_sums

                
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
                    cbar=False, square=True, annot_kws={"size": 25}, ax=axes[i])
        axes[i].set_title(label_encoder.classes_[i])
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")

    plt.tight_layout()
    plt.show()