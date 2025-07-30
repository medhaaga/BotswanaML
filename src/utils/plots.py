import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

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
