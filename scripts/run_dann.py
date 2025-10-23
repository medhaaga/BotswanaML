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
from src.utils.train import train_dann
import pandas as pd
import src.utils.io as io   
import src.utils.datasets as datasets
import config as config
from sklearn.preprocessing import LabelEncoder
from src.eval.eval_utils import evaluate_label_distribution
from src.eval.plot_utils import make_sightings_plots_from_model
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    # ---------------- Feature setup ----------------
    parser.add_argument("--pos_idx", nargs="+", type=int, default=[0,1,2,3,4,5],
                        help="Indices of positive-only features")
    parser.add_argument("--center_idx", nargs="*", type=int, default=[6,7,8],
                        help="Indices of zero-centered features")

    # ---------------- Preprocessing ----------------
    parser.add_argument("--n_sample_per_target", type=int, default=200000,
                        help="Number of samples to draw from each target for computing mean/std")

    # ---------------- Training ----------------
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lambda_domain", type=float, default=1.0, help="Weight for domain loss")
    parser.add_argument("--test_frac", type=float, default=0.2, help="Fraction of train set to reserve for validation")

        # ---------------- Model ----------------
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Dimension of feature extractor's hidden layer")

    # ---------------- Output ----------------
    parser.add_argument("--plot_hists", action="store_true",
                        help="Whether to save histograms before and after preprocessing")
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
    root_dir = os.path.join(io.get_domain_adaptation_results_dir(), "dann")
    os.makedirs(root_dir, exist_ok=True)
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
    args.input_dim, args.n_classes = X_src.shape[-1], len(np.unique(y_src))

    # encode the labels
    label_encoder = LabelEncoder()
    y_src = label_encoder.fit_transform(y_src)
    n_classes = len(np.unique(y_src))


    print("Loading target data (RVC)...")
    RVC_df = pd.read_csv(io.get_RVC_preprocessed_path())
    X_targets = [RVC_df.loc[RVC_df.firmware_major_version == 2.0],
                RVC_df.loc[RVC_df.firmware_major_version == 3.0]]
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
        highs=highs,
        clip_to_quantile=False
    )

    print(len(RVC_df), transform(torch.tensor(RVC_df[Vectronics_feature_cols].values, dtype=torch.float32)).shape)

    X_train, X_temp, y_train, y_temp = train_test_split(X_src, y_src, test_size=2*args.test_frac, random_state=42, stratify=y_src)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=args.test_frac, random_state=42, stratify=y_temp)
     
    print(f"Train data: {X_train.shape}")
    print(f"Val data: {X_val.shape}")
    print(f"Number of classes: {n_classes}")

    # Build datasets
    train_dataset = datasets.NumpyDataset(X=X_train, y=y_train, transform=transform)
    val_dataset   = datasets.NumpyDataset(X=X_val, y=y_val, transform=transform)
    test_dataset   = datasets.NumpyDataset(X=X_test, y=y_test, transform=transform)

    # Build dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader   = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    target_train_loaders = []
    target_test_loaders = []

    for Xt in X_targets:
        X_target_full = Xt[Vectronics_feature_cols].values

        idx = np.random.permutation(len(X_target_full))
        target_train_idx = idx[:X_train.shape[0]]
        target_test_idx = idx[X_train.shape[0]:]  # remaining go to test

        X_target_train = X_target_full[target_train_idx]
        X_target_test  = X_target_full[target_test_idx]

        # Build target train/test datasets
        target_train_dataset = datasets.NumpyDataset(X=X_target_train, y=None, transform=transform)
        target_test_dataset  = datasets.NumpyDataset(X=X_target_test, y=None, transform=transform)

        # Create loaders
        target_train_loader = DataLoader(target_train_dataset, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True)
        target_test_loader  = DataLoader(target_test_dataset,  batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)

        target_train_loaders.append(target_train_loader)
        target_test_loaders.append(target_test_loader)

    for i, loader in enumerate(target_train_loaders):

        print(f"Training DANN for target domain {i+1}...")

        # ---------------- Train DANN ----------------

        results = train_dann(train_loader, val_loader, test_loader, [loader], args, device)
        model = results['model']

        #############################################
        ###### Save objects
        ##############################################

        dir = os.path.join(root_dir, f'target{i+1}')
        os.makedirs(dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(dir, 'model.pt'))
        
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

        #############################################
        ###### Evaluation
        ##############################################

        _, _, _ = evaluate_label_distribution(model=model, 
                                            data=target_test_loaders[i],
                                            n_classes=n_classes, 
                                            label_encoder=label_encoder,
                                            device=device, 
                                            verbose=True)

        # Plot sightings
        matched_sightings = pd.read_csv(os.path.join(io.get_data_path(), 'matched_sightings.csv'))
        matched_gps = pd.read_csv(io.get_gps_moving_path())

        print('dann')
        
        make_sightings_plots_from_model(model=model, 
                                   data=transform(torch.tensor(X_targets[i][Vectronics_feature_cols].values, dtype=torch.float32)),
                                   metadata=RVC_df[RVC_df.firmware_major_version == (2.0 if i==0 else 3.0)].reset_index(drop=True),
                                   matched_sightings=matched_sightings, 
                                   matched_gps=matched_gps, 
                                   device=device,
                                   model_name='dann')



if __name__ == "__main__":
    main()
