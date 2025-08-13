###########################################################
### This script requires access to raw accelerometery data
### Adjust paths accordingly for using your data
###########################################################


# System & OS
import os
import sys
import argparse
import warnings
import random as random
sys.path.append('.')
sys.path.append('../')

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import config as config
from sklearn.preprocessing import LabelEncoder
from src.utils.plots import plot_signal_and_online_predictions

# Script imports



from config.settings import (RAW_BEHAVIORS)

# for reproducible results, conduct online evaluations for dog jessie with seed 23. This is 


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_duration", type=float, default=12.937)
    parser.add_argument("--window_duration_percentile", type=float, default=50)
    parser.add_argument("--window_length", type=int, default=206)
    parser.add_argument("--score_hop_length", type=int, default=None)
    parser.add_argument("--smoothening_window_length", type=int, default=10)
    parser.add_argument("--smoothening_hop_length", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--experiment_name", type=str, default='no_split', choices=['no_split', 'interdog', 'interyear', 'interAMPM'])
    parser.add_argument("--kernel_size", type=int, default=5, help="size fo kernel for CNN")
    parser.add_argument("--n_channels", type=int, default=32, help="number of output channels for the first CNN layer")
    parser.add_argument("--n_CNNlayers", type=int, default=5, help="number of convolution layers")
    parser.add_argument("--theta", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dog", type=str, default='jessie', choices=['jessie', 'ash', 'palus', 'green', 'fossey'])


    return parser

def online_smoothening(scores, window_len, hop_len):

    scores = scores.reshape(-1,scores.shape[-1]) #(number of classes, number of windows)
    n_windows = 1+ (scores.shape[-1] - window_len)//hop_len

    online_avg = np.zeros((scores.shape[0], n_windows))

    for i in range(n_windows):
        start = i*hop_len
        online_avg[:,i] = np.mean(scores[:, start:start+window_len], axis=-1)
        
    return online_avg


def all_online_eval(model_dir, metadata, device, sampling_frequency=16, window_length=None, window_duration=None, dir=None, plot=False):

    if (window_length is None) & (window_duration is None):
        raise ValueError('A window length/duratioon for the classification model is required.')
    
    if (window_length is None) & (window_duration is not None):
        window_length = int(window_duration*sampling_frequency)

    if (window_length is not None) & (window_duration is not None):
        assert window_length == int(window_duration*sampling_frequency), "window length and window duration are not compatible according to provided sampling frequency."
    

    # check if model and window duration are compatible
    model = torch.load(os.path.join(model_dir, 'model.pt'),  weights_only=False, map_location=device)
    zero_signal = torch.zeros(1, 3, window_length).to(device)
    assert model[:-2](zero_signal).shape[-1] == model[-2].in_features, "Window duration and model not compatible"

    # fit the label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(RAW_BEHAVIORS)

    for _, row in tqdm(metadata.iterrows(), total = len(metadata)):

        individual, half_day = row['individual ID'], row['half day [yyyy-mm-dd_am/pm]']

        windows = []
        half_day_acc = []

        half_day_data = pd.read_csv(row['file path'])
        half_day_data['Timestamp'] = pd.to_datetime(half_day_data['Timestamp'], utc=True)

        if len(half_day_data) < window_length:
            warnings.warn(f'half day {individual}-{half_day} has lesser data than window length. Skipped.')
        
        start_index = 0

        while start_index + window_length < len(half_day_data):

            end_index = start_index  + window_length
            window = half_day_data.iloc[start_index:end_index]

            # Collect timestamps for start and end of the window
            window_start = window['Timestamp'].iloc[0]
            window_end = window['Timestamp'].iloc[-1]
            windows.append({'Timestamp start': window_start, 'Timestamp end': window_end})

            # Collect values for tensor
            window_values = window[['Acc X [g]', 'Acc Y [g]', 'Acc Z [g]']].values
            half_day_acc.append(window_values)

            start_index = end_index

        # Create DataFrame for windows
        half_day_online_evals = pd.DataFrame(windows)
        half_day_online_evals['individual ID'] = [individual]*len(half_day_online_evals)

        # Convert list of arrays to a PyTorch tensor
        half_day_acc = np.array(half_day_acc).reshape(len(half_day_acc), window_length, 3)
        half_day_acc = np.transpose(half_day_acc, (0,2,1)) # (number of windows, 3, window length)
        half_day_acc = torch.tensor(half_day_acc, dtype=torch.float32)

        with torch.no_grad():
            scores = model(half_day_acc.to(device))

        half_day_online_evals['Prediction scores'] = np.max(F.softmax(scores, dim=1).cpu().numpy(), axis=1)
        half_day_online_evals['Most probable behavior'] = label_encoder.inverse_transform(np.argmax(scores.cpu().numpy(), axis=1))


        if dir is not None:
            eval_dir = os.path.join(dir, 'evals')
            os.makedirs(eval_dir, exist_ok=True)
            eval_path = os.path.join(eval_dir, os.path.basename(row['file path']))
            half_day_online_evals.to_csv(eval_path, index=False)

            if plot is True:

                scores = np.array(scores.unsqueeze(0).detach().cpu().numpy()) # (1, number of windows, number of classes)
                scores = np.transpose(scores, (0,2,1))
                online_avg = online_smoothening(scores, 1, 1)

                plot_dir = os.path.join(dir, 'plots')
                os.makedirs(plot_dir, exist_ok=True)
                plot_path = os.path.join(plot_dir, os.path.splitext(os.path.basename(row['file path']))[0]+'.png')
                plot_signal_and_online_predictions(
                    half_day_data['Timestamp'], 
                    np.array([half_day_data['Acc X [g]'].values, half_day_data['Acc Y [g]'].values, half_day_data['Acc Z [g]'].values]),
                    online_avg, 
                    window_length=1, 
                    hop_length=1,  # Assuming hop_length is equal to window_length
                    window_duration=window_duration, 
                    label_encoder=label_encoder, 
                    plot_dir=plot_path, 
                    half_day_behaviors=None
                )


if __name__ == '__main__':

    parser = parse_arguments()
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model_config = {'experiment_name': args.experiment_name,
                    'n_CNNlayers': args.n_CNNlayers,
                    'n_channels': args.n_channels,
                    'kernel_size': args.kernel_size,
                    'theta': args.theta,
                    'window_duration_percentile': args.window_duration_percentile
                    }

    smoothening_config = {'smoothening_window_length': args.smoothening_window_length,
                          'smoothening_hop_length': args.smoothening_hop_length,
                          'score_hop_length': args.score_hop_length
                          }


   


    

