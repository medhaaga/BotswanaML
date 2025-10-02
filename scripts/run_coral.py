import sys
import os
import yaml
import json
sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")
import argparse
import numpy as np
import torch
from src.utils import preprocess
from src.utils.train import train_coral, multi_label_eval_loop
import pandas as pd
import src.utils.io as io   
import src.utils.datasets as datasets
import config as config
from sklearn.preprocessing import LabelEncoder
from src.eval.eval_utils import evaluate_label_distribution
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    # ---------------- Feature setup ----------------
    parser.add_argument("--pos_idx", nargs="+", type=int, default=[0,1,2,3,4,5],
                        help="Indices of positive-only features")
    parser.add_argument("--center_idx", nargs="+", type=int, default=[6,7,8],
                        help="Indices of zero-centered features")

    # ---------------- Preprocessing ----------------
    parser.add_argument("--n_sample_per_target", type=int, default=200000,
                        help="Number of samples to draw from each target for computing mean/std")

    # ---------------- Model ----------------
    parser.add_argument("--feat_dim", type=int, default=128,
                        help="Dimension of feature extractor's hidden layer")

    # ---------------- Training ----------------
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lambda_coral", type=float, default=1.0, help="Weight for domain loss")
    parser.add_argument("--test_frac", type=float, default=0.2, help="Fraction of train set to reserve for validation")


    # ---------------- Output ----------------
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    return parser

def main():
    # --------------------------
    # Parse arguments
    # --------------------------
    parser = parse_arguments()
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    root_dir = os.path.join(io.get_domain_adaptation_results_dir(), "coral")
    os.makedirs(dir, exist_ok=True)
    np.random.seed(seed=args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # --------------------------
    # Load data
    # --------------------------

    print("Loading source data (Vectronics)...")
    with open(config.VECTRONICS_PREPROCESSING_YAML) as f:
        Vectronics_preprocessing_config = yaml.safe_load(f)
    Vectronics_feature_cols = Vectronics_preprocessing_config['feature_cols']

    min_duration_before_padding = 15.0
    vectronics_df = pd.read_csv(io.get_Vectronics_preprocessed_path(min_duration_before_padding))
    X_src = vectronics_df[Vectronics_feature_cols].values
    y_src = vectronics_df['behavior'].values

    args.input_dim = X_src.shape[-1]
    args.n_classes = len(np.unique(y_src))

    # encode the labels
    label_encoder = LabelEncoder()
    y_src = label_encoder.fit_transform(y_src)
    n_classes = len(np.unique(y_src))


    print("Loading target data (RVC)...")
    RVC_df = pd.read_csv(io.get_RVC_preprocessed_path())
    X_targets = [RVC_df.loc[RVC_df.firmware_major_version == 2.0, Vectronics_feature_cols].values,
                RVC_df.loc[RVC_df.firmware_major_version == 3.0, Vectronics_feature_cols].values]

    # --------------------------
    # create datasets and dataloaders 
    # --------------------------

    # compute global lows/highs once
    lows, highs = preprocess.compute_combined_quantiles(
        datasets=[X_src],
        pos_idx=args.pos_idx,
        center_idx=args.center_idx,
        low_q=0.00,
        high_q=1.00,
    )
    # define transform
    transform = preprocess.TransformAndScale(
        pos_idx=args.pos_idx,
        center_idx=args.center_idx,
        lows=lows,
        highs=highs
    )

    X_train, X_temp, y_train, y_temp = train_test_split(X_src, y_src, test_size=2*args.test_frac, random_state=42, stratify=y_src)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=args.test_frac, random_state=42, stratify=y_temp)
     
    print(f"Train data: {X_train.shape}")
    print(f"Val data: {X_val.shape}")
    for i, Xt in enumerate(X_targets):
        print(f"Target data {i+1}: {Xt.shape}")
    print(f"Number of classes: {n_classes}")

    # Build datasets
    train_dataset = datasets.NumpyDataset(X=X_train, y=y_train, transform=transform)
    val_dataset   = datasets.NumpyDataset(X=X_val, y=y_val, transform=transform)
    test_dataset   = datasets.NumpyDataset(X=X_test, y=y_test, transform=transform)

    # Build dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader   = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    target_loaders = [
        DataLoader(datasets.NumpyDataset(X=Xt, y=None, transform=transform), batch_size=1024, shuffle=True, num_workers=4, pin_memory=True)
        for Xt in X_targets
    ]

    for i, loader in enumerate(target_loaders):

        # ---------------- Train DANN ----------------
        results = train_coral(train_loader, val_loader, test_loader, loader, args, device)
        model = results['model']

        #############################################
        ###### Save objects
        ##############################################

        dir = os.path.join(root_dir, f'target{i+1}')
        os.makedirs(dir, exist_ok=True)

        torch.save(model, os.path.join(dir, 'model.pt'))
        
        json_training_stats_file = os.path.join(dir, 'training_stats.json')
        with open(json_training_stats_file, 'w') as f:
            json.dump(results['training_stats'], f, indent=4)

        # Save test results
        test_results_path = os.path.join(dir, 'test_results.npz')
        np.savez(
        test_results_path,
        true_classes=results['test_true_classes'],
        predictions=results['test_predictions'],
        scores=results['test_scores'])

        # Save val results
        val_results_path = os.path.join(dir, 'val_results.npz')
        np.savez(
            val_results_path,
            true_classes=results['val_true_classes'],
            predictions=results['val_predictions'],
            scores=results['val_scores'])

        # Evaluate on target domains
        _ = evaluate_label_distribution(model=model, 
                                            data=loader,
                                            n_classes=n_classes, 
                                            label_encoder=label_encoder,
                                            device=device, )


if __name__ == "__main__":
    main()
