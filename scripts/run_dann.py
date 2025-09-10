import sys
import os
import yaml
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
import config as config
from src.utils.plots import plot_feature_histograms
from sklearn.preprocessing import LabelEncoder
from src.eval.eval_utils import evaluate_label_distribution
from sklearn.model_selection import train_test_split

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

    # ---------------- Training ----------------
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lambda_domain", type=float, default=0.1, help="Weight for domain loss")
    parser.add_argument("--val_frac", type=float, default=0.2, help="Fraction of train set to reserve for validation")


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

    min_duration_before_padding = 10.0
    vectronics_df = pd.read_csv(io.get_Vectronics_preprocessed_path(min_duration_before_padding))
    X_src = vectronics_df[Vectronics_feature_cols].values
    y_src = vectronics_df['behavior'].values
    label_encoder = LabelEncoder()
    y_src = label_encoder.fit_transform(y_src)

    print("Loading target data (RVC)...")
    RVC_df = pd.read_csv(io.get_RVC_preprocessed_path())
    X_targets = [RVC_df.loc[RVC_df.firmware_major_version == 2.0, Vectronics_feature_cols].values,
                RVC_df.loc[RVC_df.firmware_major_version == 3.0, Vectronics_feature_cols].values]

    # --------------------------
    # Preprocess (compute quantiles using all domains)
    # --------------------------
    lows, highs = preprocess.compute_combined_quantiles(
        datasets=[X_src], pos_idx=args.pos_idx, center_idx=args.center_idx,
        low_q=0.00, high_q=1.00,
    )
    X_src_prep = preprocess.transform_and_scale(X_src, args.pos_idx, args.center_idx, lows, highs)
    X_targets_prep = [preprocess.transform_and_scale(Xt, args.pos_idx, args.center_idx, lows, highs)
                    for Xt in X_targets]
    X_train, X_val, y_train, y_val = train_test_split(X_src_prep, y_src, test_size=args.val_frac, random_state=42, stratify=y_src)
    
    if args.plot_hists:
        plot_feature_histograms(X_src, X_targets, fname=os.path.join(io.get_figures_dir(), "raw_feature_hists.png"))
        plot_feature_histograms(X_src_prep, X_targets_prep, fname=os.path.join(io.get_figures_dir(), "preprocessed_feature_hists.png"))


    print(f"Source data shape: {X_src_prep.shape}")
    for i, Xt in enumerate(X_targets_prep):
        print(f"Target data {i+1} shape: {Xt.shape}")
    print(f"Number of classes: {len(np.unique(y_src))}")

    # ---------------- Train DANN ----------------
    n_classes = len(np.unique(y_src))
    best_models, history, best_val_labels, best_val_preds = train_dann(
        X_train, y_train, X_val, y_val, X_targets_prep,
        n_classes=n_classes,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_domain=args.lambda_domain,
        device=device,
    )

    print("Training complete.")

    # Save models
    dir = os.path.join(io.get_domain_adaptation_results_dir(), "dann")
    os.makedirs(dir, exist_ok=True)
    # Save models
    for name, model in best_models.items():
        torch.save(model.state_dict(), os.path.join(dir, f"{name}.pth"))
    history = {k: np.array(v) for k, v in history.items()}

    np.savez_compressed(os.path.join(dir, "training_history.npz"), **history)
    np.savez_compressed(os.path.join(dir, "val_preds.npz"), val_true=best_val_labels, val_preds=best_val_preds)
    print(f"Objects saved to {dir}")

    # Evaluate on target domains
    for i, Xt in enumerate(X_targets_prep):
        evaluate_label_distribution(best_models['feature_extractor'], best_models['label_classifier'], Xt, n_classes, device=device,
                                    domain_name=f"Target{i+1}", label_encoder=label_encoder)


if __name__ == "__main__":
    main()
