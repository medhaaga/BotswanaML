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
import pandas as pd
import src.utils.io as io   
import src.utils.datasets as datasets
from src.utils import preprocess
from src.utils.train import train_coral
from src.utils.data_prep import setup_multilabel_dataloaders
from src.utils.Vectronics_preprocessing import modify_vectronics_labels
import config as config
from sklearn.preprocessing import LabelEncoder
from src.eval.eval_utils import evaluate_label_distribution

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def parse_arguments():
    parser = argparse.ArgumentParser()

    # ---------------- Experimental setup ----------------
    parser.add_argument("--exp_name", type=str, default="DEFAULT", help="Base name of experiment")

    # ---------------- Feature setup ----------------
    parser.add_argument("--pos_idx", nargs="+", type=int, default=[0,1,2,3,4,5],
                        help="Indices of positive-only features")
    parser.add_argument("--center_idx", nargs="+", type=int, default=None,
                        help="Indices of zero-centered features")
    parser.add_argument("--source_padding_duration", type=float, default=None,
                        help="Padding duration used in creation of preprocessed source data")
    
    # ---------------- Preprocessing ----------------
    parser.add_argument("--keep_confidence_levels", nargs="*", type=str, default=None,
                        help="list of str of confidence levels from ['H', 'M', 'H/M']")
    parser.add_argument("--eating_to_other", nargs="*", type=str, default=None,
                        help="list of str of eating intensities to convert to Other behavior. Values in ['H', 'M', 'L']")
    parser.add_argument("--eating_to_exclude", nargs="*", type=str, default=None,
                        help="list of str of eating intensities to exclude form data. Values in ['H', 'M', 'L']")
    parser.add_argument("--n_sample_per_target", type=int, default=200000,
                        help="Number of samples to draw from each target for computing mean/std")

    # ---------------- Model ----------------
    parser.add_argument("--feat_dim", type=int, default=128,
                        help="Dimension of feature extractor's hidden layer")

    # ---------------- Training ----------------
    parser.add_argument("--theta", type=float, default=0.3)
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay (L2 regularization)")
    parser.add_argument("--lambda_coral", type=float, default=1.0, help="Weight for domain loss")
    parser.add_argument("--test_frac", type=float, default=0.2, help="Fraction of train set to reserve for validation")


    # ---------------- Output ----------------
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    return parser


def main():
    # -----------------------------------------------
    # Parse arguments, create dir, set seeds
    # -----------------------------------------------
    parser = parse_arguments()
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    root_dir = os.path.join(io.get_domain_adaptation_results_dir(), "coral")
    root_dir = io.get_exp_dir(output_root=root_dir, exp_name=args.exp_name)
    os.makedirs(root_dir, exist_ok=True)
    with open(os.path.join(root_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

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

    vectronics_df = pd.read_csv(io.get_Vectronics_preprocessed_path(args.source_padding_duration))
    vectronics_df = modify_vectronics_labels(vectronics_df, 
                                             keep_confidence_levels=args.keep_confidence_levels,
                                             eating_to_other=args.eating_to_other,
                                             eating_to_exclude=args.eating_to_exclude)
    
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
    X_targets = [RVC_df.loc[RVC_df.firmware_major_version == 2.0],
                RVC_df.loc[RVC_df.firmware_major_version == 3.0]]
    
    left_limit = np.quantile(X_src, 0.0, axis=0)
    right_limit = np.quantile(X_src, 1.0, axis=0)
    right_limit = np.where(right_limit == left_limit, left_limit + 1e-6, right_limit)

    for i, Xt in enumerate(X_targets):
        Xt = Xt[Vectronics_feature_cols].values
        mask = (Xt >= left_limit).all(axis=1) & (Xt <= right_limit).all(axis=1) & (Xt > 0.0).all(axis=1)
        X_targets[i] = X_targets[i][mask].reset_index(drop=True)

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

    X_train, X_temp, y_train, y_temp = train_test_split(X_src, y_src, test_size=2*args.test_frac, random_state=42, stratify=y_src)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    train_loader, val_loader, test_loader = setup_multilabel_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, args, n_outputs=n_classes, transform=transform)

    target_train_loaders = []
    target_test_loaders = []

    for i, Xt in enumerate(X_targets):
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
        target_train_loader = DataLoader(target_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        target_test_loader  = DataLoader(target_test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        target_train_loaders.append(target_train_loader)
        target_test_loaders.append(target_test_loader)


    print(f"Train data: {X_train.shape}")
    print(f"Val data: {X_val.shape}")
    for i, Xt in enumerate(X_targets):
        print(f"Target data {i+1}: {Xt.shape}")
    print(f"Number of classes: {n_classes}")

    for i, loader in enumerate(target_train_loaders):

        print(f"Training CORAL for target domain {i+1}...")

        # ---------------- Train CORAL ----------------
        results = train_coral(train_loader, val_loader, test_loader, loader, args, device)
        model = results['model']

        #############################################
        ###### Save objects
        ##############################################

        dir = os.path.join(root_dir, f'target{i+1}')
        os.makedirs(dir, exist_ok=True)
        print(dir)

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
        

if __name__ == "__main__":
    main()
