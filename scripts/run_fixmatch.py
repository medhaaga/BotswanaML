import sys
import os
import yaml
from tqdm import tqdm
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
from src.utils.train import train_fixmatch
from src.utils.Vectronics_preprocessing import modify_vectronics_labels
from src.utils.data_prep import setup_multilabel_dataloaders
import config as config
from sklearn.preprocessing import LabelEncoder
from src.eval.eval_utils import evaluate_multilabel_distribution

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
    parser.add_argument("--fixmatch_threshold", type=float, default=0.95,
                        help="Threshold for classifying as strong label")   
    parser.add_argument("--lambda_target", type=float, default=1.0,
                        help="Weight for labeled target loss")
    parser.add_argument("--lambda_unsup", type=float, default=0.0,
                        help="Weight for unlabaled target loss") 

    # ---------------- Training ----------------
    parser.add_argument("--theta", type=float, default=0.3)
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay (L2 regularization)")
    parser.add_argument("--source_test_frac", type=float, default=0.2, help="Fraction of train set to reserve for validation")
    parser.add_argument("--target_val_frac", type=float, default=0.25, help="Fraction of train set to reserve for validation")
    parser.add_argument("--target_test_frac", type=float, default=0.25, help="Fraction of train set to reserve for validation")

    # ---------------- Output ----------------
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    return parser

def weak_augment(x, noise_std=0.01):
            noise = torch.randn_like(x) * noise_std
            return x + noise

def strong_augment(x, noise_std=0.05, dropout_prob=0.1):
    noise = torch.randn_like(x) * noise_std
    x_aug = x + noise
    mask = np.random.rand(*x.shape) < dropout_prob
    x_aug[mask] = 0  # random feature dropout
    return x_aug


def main():
    # --------------------------
    # Parse arguments
    # --------------------------
    parser = parse_arguments()
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if args.lambda_target == 0.0:
        root_dir = os.path.join(io.get_domain_adaptation_results_dir(), "fixmatch_self_supervised")
    else:
        root_dir = os.path.join(io.get_domain_adaptation_results_dir(), "fixmatch_semi_supervised")
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

    print("Loading lableded source data (Vectronics)...")
    labeled_vectronics_df = pd.read_csv(io.get_Vectronics_preprocessed_path(args.source_padding_duration))
    labeled_vectronics_df = modify_vectronics_labels(labeled_vectronics_df, 
                                             keep_confidence_levels=args.keep_confidence_levels,
                                             eating_to_other=args.eating_to_other,
                                             eating_to_exclude=args.eating_to_exclude)

    print("Loading target data (RVC)...")
    RVC_df = pd.read_csv(io.get_RVC_preprocessed_path())
    RVC_df = RVC_df.drop_duplicates().reset_index(drop=True)

    # create labeled source, unlabeled source, and target data tensors
    X_src = labeled_vectronics_df[Vectronics_feature_cols].values
    y_src = labeled_vectronics_df['behavior'].values

    # encode the labels
    label_encoder = LabelEncoder()
    y_src = label_encoder.fit_transform(y_src)
    n_classes = len(np.unique(y_src))

    args.input_dim = X_src.shape[-1]
    args.n_classes = len(np.unique(y_src))

    left_limit = np.quantile(X_src, 0.0, axis=0)
    right_limit = np.quantile(X_src, 1.0, axis=0)
    right_limit = np.where(right_limit == left_limit, left_limit + 1e-6, right_limit)

    Xt = RVC_df[Vectronics_feature_cols].values
    mask = (Xt >= left_limit).all(axis=1) & (Xt <= right_limit).all(axis=1) & (Xt > 0.0).all(axis=1)
    RVC_df = RVC_df[mask].reset_index(drop=True)

    labeled_mask = RVC_df['behavior'].notna()

    RVC_labeled_df = RVC_df[labeled_mask].reset_index(drop=True)
    RVC_unlabeled_df = RVC_df[~labeled_mask].reset_index(drop=True)
    
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

    # creating dataloaders for labeled source data
    X_train, X_temp, y_train, y_temp = train_test_split(X_src, y_src, test_size=2*args.source_test_frac, random_state=42, stratify=y_src)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    train_loader, val_loader, test_loader = setup_multilabel_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, args, n_outputs=n_classes, transform=transform)

    print("SOURCE SHAPES:")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    classes, counts = np.unique(y_src, return_counts=True)
    print(f"Class distribution:")
    for cls, count in zip(classes, counts):
        print(f" - {label_encoder.inverse_transform([cls])[0]}: {count}, ({count / X_src.shape[0]:.2%})")
    print("")

    # creating dataloaders for labeled test data
    target_labeled_train_loaders = []
    target_labeled_test_loaders = []
    target_labeled_val_loaders = []
    target_unlabeled_loaders = []

    for i, sensor_version in enumerate(RVC_df.firmware_major_version.unique().astype(int)):

        df = RVC_labeled_df.loc[(RVC_labeled_df.behavior == 'Feeding') & (RVC_labeled_df.firmware_major_version == sensor_version)].reset_index(drop=True)
        group_ids = list(df.groupby(['animal_id', 'UTC date [yyyy-mm-dd]']).groups.keys())
        n_feeding_days = len(group_ids)

        assert args.target_val_frac + args.target_test_frac < 1, "The fraction of val and test split of target data should add to values < 1"
        target_train_frac = 1.0 - (args.target_val_frac + args.target_test_frac)

        train_feeding_days, val_feeding_days, test_feeding_days = group_ids[:int(target_train_frac*n_feeding_days)], \
                                                                    group_ids[int(target_train_frac*n_feeding_days): int((target_train_frac+args.target_val_frac)*n_feeding_days)], \
                                                                    group_ids[int((target_train_frac+args.target_val_frac)*n_feeding_days): ]
        
        # save the (animal_id, date) apiirs for each target split
        def tuples_to_df(tuples_list, split_name):
            return pd.DataFrame(tuples_list, columns=['animal_id', 'UTC date [yyyy-mm-dd]']).assign(split=split_name)

        train_df = tuples_to_df(train_feeding_days, "train")
        val_df   = tuples_to_df(val_feeding_days, "val")
        test_df  = tuples_to_df(test_feeding_days, "test")
        target_splits_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        target_splits_df.to_csv(os.path.join(root_dir, 'target_splits.csv'), index=False)

        train_mask = RVC_labeled_df[['animal_id', 'UTC date [yyyy-mm-dd]']].apply(tuple, axis=1).isin(train_feeding_days)
        val_mask = RVC_labeled_df[['animal_id', 'UTC date [yyyy-mm-dd]']].apply(tuple, axis=1).isin(val_feeding_days)
        test_mask = RVC_labeled_df[['animal_id', 'UTC date [yyyy-mm-dd]']].apply(tuple, axis=1).isin(test_feeding_days)

        X_t_train, y_t_train = RVC_labeled_df[train_mask][Vectronics_feature_cols].values, RVC_labeled_df[train_mask]['behavior'].values
        X_t_val, y_t_val = RVC_labeled_df[val_mask][Vectronics_feature_cols].values, RVC_labeled_df[val_mask]['behavior'].values
        X_t_test, y_t_test = RVC_labeled_df[test_mask][Vectronics_feature_cols].values, RVC_labeled_df[test_mask]['behavior'].values
        n_target = X_t_train.shape[0] + X_t_val.shape[0] + X_t_test.shape[0]

        y_t_train = label_encoder.transform(y_t_train)
        y_t_val = label_encoder.transform(y_t_val)
        y_t_test = label_encoder.transform(y_t_test)
        
        print(f"TARGET - {i+1} SHAPES:")
        print(f"Train: {X_t_train.shape}, Val: {X_t_val.shape}, Test: {X_t_test.shape}")
        classes, counts = np.unique(np.concatenate([y_t_train, y_t_val, y_t_test]), return_counts=True)
        print(f"Class distribution:")
        for cls, count in zip(classes, counts):
            print(f" - {label_encoder.inverse_transform([cls])[0]}: {count}, ({count / n_target:.2%})")
        print("")

        target_l_train_loader, target_l_val_loader, target_l_test_loader = setup_multilabel_dataloaders(X_t_train, y_t_train, 
                                                                                                            X_t_val, y_t_val, 
                                                                                                            X_t_test, y_t_test, 
                                                                                                            args, n_outputs=n_classes,
                                                                                                            transform=transform)
        target_labeled_train_loaders.append(target_l_train_loader)
        target_labeled_val_loaders.append(target_l_val_loader)
        target_labeled_test_loaders.append(target_l_test_loader)

        target_u_ds = datasets.NumpyDataset(X=RVC_unlabeled_df.loc[RVC_unlabeled_df.firmware_major_version == sensor_version][Vectronics_feature_cols].values, y=None, transform=transform)
        target_u_loader = DataLoader(target_u_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        target_unlabeled_loaders.append(target_u_loader)
    

    for i, sensor_version in enumerate(RVC_df.firmware_major_version.unique().astype(int)):

        print(f"Training FixMatch for target domain {i+1}...")

        # ---------------- Train FixMatch ----------------

        results = train_fixmatch(
                                train_loader=train_loader, 
                                val_loader=val_loader, 
                                test_loader=test_loader,
                                target_labeled_train_loader=target_labeled_train_loaders[i],
                                target_labeled_val_loader=target_labeled_val_loaders[i],
                                target_labeled_test_loader=target_labeled_test_loaders[i],  
                                target_unlabeled_loader=target_unlabeled_loaders[i],
                                args=args,
                                device=device, 
                                weak_augment=weak_augment, 
                                strong_augment=strong_augment, 
                                threshold=0.5
                            )
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

        # Save test results on source data
        test_results_path = os.path.join(dir, 'test_results.npz')
        np.savez(
        test_results_path,
        true_classes=results['test_true_classes'],
        predictions=results['test_predictions'],
        scores=results['test_scores'])

        # Save val results on source data
        val_results_path = os.path.join(dir, 'val_results.npz')
        np.savez(
            val_results_path,
            true_classes=results['val_true_classes'],
            predictions=results['val_predictions'],
            scores=results['val_scores'])
        
        # Save target val results on target data
        target_test_results_path = os.path.join(dir, 'target_val_results.npz')
        np.savez(
            target_test_results_path,
            true_classes=results['target_val_true_classes'],
            predictions=results['target_val_predictions'],
            scores=results['target_val_scores'])
        
        # Save target test results on target data
        target_test_results_path = os.path.join(dir, 'target_test_results.npz')
        np.savez(
            target_test_results_path,
            true_classes=results['target_test_true_classes'],
            predictions=results['target_test_predictions'],
            scores=results['target_test_scores'])


        #############################################
        ###### Evaluation
        ##############################################

        eval_df = RVC_df.loc[RVC_df.firmware_major_version == sensor_version].reset_index(drop=True)

        _, _, _ = evaluate_multilabel_distribution(model=model, 
                                            data=transform(torch.tensor(eval_df[Vectronics_feature_cols].values, dtype=torch.float32)),
                                            label_encoder=label_encoder,
                                            device=device, 
                                            threshold=0.5,
                                            verbose=True)
        
if __name__ == "__main__":
    main()
