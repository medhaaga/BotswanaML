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
from src.utils.data_prep import setup_multilabel_dataloaders
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
    parser.add_argument("--remove_outliers", type=int, default=1,
                        help="whether to remove target samples outside the source domain")

    # ---------------- Preprocessing ----------------
    parser.add_argument("--n_sample_per_target", type=int, default=200000,
                        help="Number of samples to draw from each target for computing mean/std")

    # ---------------- Model ----------------
    parser.add_argument("--feat_dim", type=int, default=128,
                        help="Dimension of feature extractor's hidden layer")
    parser.add_argument("--fixmatch_threshold", type=float, default=0.95,
                        help="Threshold for classifying as strong label")    
    parser.add_argument("--model_name", type=str, default='A', help="Model name for saving results")

    # ---------------- Training ----------------
    parser.add_argument("--theta", type=float, default=0.3)
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay (L2 regularization)")
    parser.add_argument("--source_test_frac", type=float, default=0.2, help="Fraction of train set to reserve for validation")
    parser.add_argument("--target_test_frac", type=float, default=0.005, help="Fraction of train set to reserve for validation")
    parser.add_argument("--lambda_target", type=float, default=100.0,
                        help="Weight for labeled target loss")
    parser.add_argument("--lambda_unsup", type=float, default=0.1,
                        help="Weight for unlabaled target loss")

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

def modify_vectronics_labels(df, model_name='A'):

    if model_name == 'A':
        return df
    elif model_name == 'B':
        df =  df[df['Confidence (H-M-L)'].isin(['H', 'H/M'])].reset_index(drop=True)
    elif model_name == 'C':
        # modify 'Feeding' labels based on eating intensity
        df['behavior'] = df.apply(
                lambda row: (
                    'Other' if (pd.notna(row['Eating intensity']) and row['Eating intensity'] in ['M', 'L'])
                    else row['behavior']
                ),
                axis=1
            )
    elif model_name == 'D':
        df =  df[df['Confidence (H-M-L)'].isin(['H', 'H/M'])].reset_index(drop=True)

        # modify 'Feeding' labels based on eating intensity
        df['behavior'] = df.apply(
            lambda row: (
                'Other' if (pd.notna(row['Eating intensity']) and row['Eating intensity'] in ['M', 'L'])
                else row['behavior']
            ),
            axis=1
        )
    elif model_name == 'E':
        df['behavior'] = df.apply(
            lambda row: (
                'Other' if (pd.notna(row['Eating intensity']) and row['Eating intensity'] in ['L'])
                else row['behavior']
            ),
            axis=1
        )
        df = df.loc[~((df["behavior"] == "Eating") & (df["Eating intensity"] == "M"))].reset_index(drop=True)
    
    elif model_name == 'F':
        df =  df[df['Confidence (H-M-L)'].isin(['H', 'H/M'])].reset_index(drop=True)
        df['behavior'] = df.apply(
            lambda row: (
                'Other' if (pd.notna(row['Eating intensity']) and row['Eating intensity'] in ['L'])
                else row['behavior']
            ),
            axis=1
        )
        df = df.loc[~((df["behavior"] == "Eating") & (df["Eating intensity"] == "M"))].reset_index(drop=True)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return df



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
        root_dir = os.path.join(io.get_domain_adaptation_results_dir(), "fixmatch_semi_supervised", f"lambda{args.lambda_target}")
        
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
    vectronics_df = modify_vectronics_labels(vectronics_df, model_name=args.model_name)
    
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
    RVC_df = RVC_df.drop_duplicates().reset_index(drop=True)

    if args.remove_outliers:
        left_limit = np.quantile(X_src, 0.0, axis=0)
        right_limit = np.quantile(X_src, 1.0, axis=0)
        right_limit = np.where(right_limit == left_limit, left_limit + 1e-6, right_limit)
    
        Xt = RVC_df[Vectronics_feature_cols].values
        mask = (Xt >= left_limit).all(axis=1) & (Xt <= right_limit).all(axis=1) & (Xt > 0.0).all(axis=1)
        RVC_df = RVC_df[mask].reset_index(drop=True)

    cols = ['feeding_binary', 'moving_binary', 'resting_binary']
    RVC_df[cols] = RVC_df[cols].fillna(0)
    RVC_df['behavior'] = np.select(
    [
        RVC_df['feeding_binary'] == 1,
        RVC_df['moving_binary'] == 1,
        RVC_df['resting_binary'] == 1
    ],
    ['Feeding', 'Moving', 'Stationary'], default=None) 

    labeled_mask = RVC_df['behavior'].notna()

    RVC_labeled_df = RVC_df[labeled_mask]
    RVC_unlabaled_df = RVC_df[~labeled_mask] 

    X_labeled_targets = [RVC_labeled_df.loc[RVC_labeled_df.firmware_major_version == 2.0][Vectronics_feature_cols].values,
                            RVC_labeled_df.loc[RVC_labeled_df.firmware_major_version == 3.0][Vectronics_feature_cols].values]
    y_labeled_targets = [RVC_labeled_df.loc[RVC_labeled_df.firmware_major_version == 2.0]['behavior'].values,
                            RVC_labeled_df.loc[RVC_labeled_df.firmware_major_version == 3.0]['behavior'].values]
    y_labeled_targets = [label_encoder.transform(y) for y in y_labeled_targets]


    X_unlabeled_targets = [RVC_unlabaled_df.loc[RVC_unlabaled_df.firmware_major_version == 2.0][Vectronics_feature_cols].values,
                            RVC_unlabaled_df.loc[RVC_unlabaled_df.firmware_major_version == 3.0][Vectronics_feature_cols].values]
    
    
    print(f"Source data: {X_src.shape}")
    for i in range(2):
        print(f"Labeled Target data {i+1}: {X_labeled_targets[i].shape}")
        print(f"Unlabeled Target data {i+1}: {X_unlabeled_targets[i].shape}")
    print(f"Number of classes: {n_classes}")

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

    print("Source shapes:")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}, Num of classes: {len(np.unique(y_src))}")

    # creating dataloaders for labaled test data
    target_labeled_train_loaders = []
    target_labeled_test_loaders = []
    target_labeled_val_loaders = []
    target_unlabeled_loaders = []

    for i in range(len(X_unlabeled_targets)):
        
        # split into train/test
        X_t_train, X_t_temp, y_t_train, y_t_temp = train_test_split(X_labeled_targets[i], y_labeled_targets[i], 
                                                                    test_size=2*args.target_test_frac, 
                                                                    random_state=42, stratify=y_labeled_targets[i])
        X_t_val, X_t_test, y_t_val, y_t_test = train_test_split(X_t_temp, y_t_temp, 
                                                                test_size=0.5, 
                                                                random_state=42, stratify=y_t_temp)
        if i==0:
            print("Target-1 shapes:")
            print(f"Train: {X_t_train.shape}, Val: {X_t_val.shape}, Test: {X_t_test.shape}, Num of classes: {len(np.unique(y_labeled_targets[i]))}")
        target_l_train_loader, target_l_val_loader, target_l_test_loader = setup_multilabel_dataloaders(X_t_train, y_t_train, 
                                                                                                        X_t_val, y_t_val, 
                                                                                                        X_t_test, y_t_test, 
                                                                                                        args, n_outputs=n_classes,
                                                                                                        transform=transform)
        
        target_u_ds = datasets.NumpyDataset(X=X_unlabeled_targets[i], y=None, transform=transform)
        target_u_loader = DataLoader(target_u_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        target_labeled_train_loaders.append(target_l_train_loader)
        target_labeled_val_loaders.append(target_l_val_loader)
        target_labeled_test_loaders.append(target_l_test_loader)
        target_unlabeled_loaders.append(target_u_loader)

    sightings = pd.read_csv(io.get_sightings_path())
    group_ids = list(sightings.groupby(['animal_id', 'UTC date [yyyy-mm-dd]']).groups.keys())
    n_sightings_days = len(group_ids)
    train_sightings, val_sightings, test_sightings = group_ids[:int(0.5*n_sightings_days)], group_ids[int(0.5*n_sightings_days): int(0.75*n_sightings_days)], group_ids[int(0.75*n_sightings_days): ]
    train_mask = RVC_labeled_df[['animal_id', 'UTC date [yyyy-mm-dd]']].apply(tuple, axis=1).isin(train_sightings)
    val_mask = RVC_labeled_df[['animal_id', 'UTC date [yyyy-mm-dd]']].apply(tuple, axis=1).isin(val_sightings)
    test_mask = RVC_labeled_df[['animal_id', 'UTC date [yyyy-mm-dd]']].apply(tuple, axis=1).isin(test_sightings)

    X_t_train, y_t_train = RVC_labeled_df[train_mask][Vectronics_feature_cols].values, RVC_labeled_df[train_mask]['behavior'].values
    X_t_val, y_t_val = RVC_labeled_df[val_mask][Vectronics_feature_cols].values, RVC_labeled_df[val_mask]['behavior'].values
    X_t_test, y_t_test = RVC_labeled_df[test_mask][Vectronics_feature_cols].values, RVC_labeled_df[test_mask]['behavior'].values
    y_t_train = label_encoder.transform(y_t_train)
    y_t_val = label_encoder.transform(y_t_val)
    y_t_test = label_encoder.transform(y_t_test)
    print("Target-2 shapes:")
    print(f"Train: {X_t_train.shape}, Val: {X_t_val.shape}, Test: {X_t_test.shape}, Num of classes: {len(np.unique(y_t_train))}")

    target_l_train_loader, target_l_val_loader, target_l_test_loader = setup_multilabel_dataloaders(X_t_train, y_t_train, 
                                                                                                        X_t_val, y_t_val, 
                                                                                                        X_t_test, y_t_test, 
                                                                                                        args, n_outputs=n_classes,
                                                                                                        transform=transform)
    target_labeled_train_loaders[-1] = target_l_train_loader
    target_labeled_val_loaders[-1] = target_l_val_loader
    target_labeled_test_loaders[-1] = target_l_test_loader
    

    for i in range(len(X_unlabeled_targets)):

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

        
        dir = os.path.join(root_dir, args.model_name, f'target{i+1}')
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

        eval_df = RVC_df.loc[RVC_df.firmware_major_version == (2.0+i)].reset_index(drop=True)

        _, _, _ = evaluate_label_distribution(model=model, 
                                            data=transform(torch.tensor(eval_df[Vectronics_feature_cols].values, dtype=torch.float32)),
                                            n_classes=n_classes, 
                                            label_encoder=label_encoder,
                                            device=device, 
                                            verbose=True)
        
        # plot sightings
        matched_sightings = pd.read_csv(io.get_sightings_path())
        matched_gps = pd.read_csv(io.get_matched_gps_path())
        matched_gps_moving = pd.read_csv(io.get_gps_moving_path())

        if args.lambda_target == 0.0:
            plot_dir = os.path.join(io.get_sightings_dir(), 'fixmatch_self_supervised', args.model_name, 'uncalibrated')
        else:
            plot_dir = os.path.join(io.get_sightings_dir(), 'fixmatch_semi_supervised', f"lambda{args.lambda_target}", args.model_name, 'uncalibrated')

        os.makedirs(plot_dir, exist_ok=True)
        make_sightings_plots_from_model(model=model, 
                                   data=transform(torch.tensor(eval_df[Vectronics_feature_cols].values, dtype=torch.float32)),
                                   metadata=eval_df,
                                   matched_sightings=matched_sightings, 
                                   matched_gps=matched_gps,
                                   matched_gps_moving=matched_gps_moving, 
                                   device=device,
                                   plot_dir=plot_dir)

if __name__ == "__main__":
    main()
