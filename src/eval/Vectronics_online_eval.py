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
sys.path.append('../../')

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import config as config
from sklearn.preprocessing import LabelEncoder
from src.utils.plots import plot_signal_and_online_predictions
import src.utils.io as io
# Script imports



from config.settings import (RAW_BEHAVIORS,
                             SUMMARY_BEHAVIORS)

# for reproducible results, conduct online evaluations for dog jessie with seed 23. This is 


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_duration", type=float, default=14.937)
    parser.add_argument("--window_duration_percentile", type=float, default=50)
    parser.add_argument("--window_length", type=int, default=206)
    parser.add_argument("--score_hop_length", type=int, default=None)
    parser.add_argument("--smoothening_window_length", type=int, default=1)
    parser.add_argument("--smoothening_hop_length", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--experiment_name", type=str, default='no_split', choices=['no_split', 'interdog', 'interyear', 'interAMPM'])
    parser.add_argument("--kernel_size", type=int, default=5, help="size fo kernel for CNN")
    parser.add_argument("--n_channels", type=int, default=64, help="number of output channels for the first CNN layer")
    parser.add_argument("--n_CNNlayers", type=int, default=5, help="number of convolution layers")
    parser.add_argument("--theta", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--individual", type=str, default='green', choices=['jessie', 'ash', 'palus', 'green', 'fossey'])


    return parser

def online_smoothening(scores, start_times, window_len, hop_len):

    scores = scores.reshape(-1, scores.shape[-1]) #(number of classes, number of windows)

    if scores.ndim == 1:
        scores = scores.reshape(1, -1)

    #  Validate that the number of timestamps matches the number of scores
    if len(start_times) != scores.shape[1]:
        raise ValueError("Length of start_times must match the number of scores.")

    n_windows = 1+ (scores.shape[-1] - window_len)//hop_len

    online_avg = np.zeros((scores.shape[0], n_windows))
    midpoint_times = np.zeros(n_windows, dtype='datetime64[ns]')

    for i in range(n_windows):
        start_idx = i * hop_len
        end_idx = start_idx + window_len

        online_avg[:,i] = np.mean(scores[:, start_idx:end_idx], axis=-1)

        # Get the start time of the first element and the last element in the window
        window_start_time = start_times[start_idx]
        window_end_time = start_times[end_idx - 1] 

        # Calculate the midpoint time of the window's time span
        midpoint_times[i] = midpoint_times[i] = window_start_time + (window_end_time - window_start_time) / 2
        
    return online_avg, midpoint_times


def all_online_raw_eval(model_dir, metadata, device, sampling_frequency=16, window_length=None, window_duration=None, smoothening_window_length=1, smoothening_hop_length=1, dir=None, plot=False):

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

    # group by individual ID and date
    grouped = metadata.groupby(["individual ID", "UTC Date [yyyy-mm-dd]"])

    for (individual, date), group in tqdm(grouped, total=len(grouped)):
        print(f"Processing individual={individual}, date={date}")

        # load one or two half-day files
        dfs = []
        for _, row in group.iterrows():
            df_half = pd.read_csv(row['file path'])
            df_half['Timestamp'] = pd.to_datetime(df_half['Timestamp'], utc=True, format='%Y-%m-%d %H:%M:%S.%f')
            dfs.append(df_half)

        # merge & sort
        full_day_data = pd.concat(dfs, ignore_index=True).sort_values("Timestamp")

        if len(full_day_data) < window_length:
            warnings.warn(f'{individual}-{date} has fewer samples than window length. Skipped.')
            continue

        start_index = 0
        # sliding windows
        windows, acc_segments = [], []

        while start_index + window_length < len(full_day_data):

            end_index = start_index  + window_length
            window = full_day_data.iloc[start_index:end_index]

            # Collect timestamps for start and end of the window
            window_start = window['Timestamp'].iloc[0]
            window_end = window['Timestamp'].iloc[-1]
            windows.append({'Timestamp start': window_start, 
                            'Timestamp end': window_end})

            # Collect values for tensor
            acc_segments.append(window[['Acc X [g]', 'Acc Y [g]', 'Acc Z [g]']].values)
            start_index = end_index

        # Create DataFrame for windows
        evals = pd.DataFrame(windows)
        evals['individual ID'] = individual
        evals['UTC Date [yyyy-mm-dd]'] = date

        # Convert list of arrays to a PyTorch tensor
        acc_segments = np.array(acc_segments).reshape(len(acc_segments), window_length, 3)
        acc_segments = np.transpose(acc_segments, (0,2,1)) # (number of windows, 3, window length)
        acc_segments = torch.tensor(acc_segments, dtype=torch.float32)

        with torch.no_grad():
            scores = model(acc_segments.to(device))

        # results dataframe
        evals['Prediction scores'] = np.max(F.softmax(scores, dim=1).cpu().numpy(), axis=1)
        evals['behavior'] = label_encoder.inverse_transform(np.argmax(scores.cpu().numpy(), axis=1))


        if dir is not None:
            eval_dir = os.path.join(dir, 'evals')
            os.makedirs(eval_dir, exist_ok=True)
            eval_path = os.path.join(eval_dir, '_'.join(os.path.basename(row['file path']).split('_')[:-1])+'.csv')
            evals.to_csv(eval_path, index=False)

            if plot is True:

                # 1. Prepare scores array with shape (num_classes, num_windows)
                scores_np = np.array(scores.unsqueeze(0).detach().cpu().numpy()) # (1, number of windows, number of classes)
                scores_np = np.transpose(scores_np, (0,2,1)) # (1, number of classes, number of windows)

                # 2. Get the start timestamp for each score window from the 'evals' DataFrame
                initial_start_times = evals['Timestamp start'].values

                # 3. Call the smoothing function with the scores AND their timestamps
                smoothed_scores, smoothed_start_times = online_smoothening(
                    scores=scores_np,
                    start_times=initial_start_times,
                    window_len=smoothening_window_length,
                    hop_len=smoothening_hop_length
                )

                plot_dir = os.path.join(dir, 'plots')
                os.makedirs(plot_dir, exist_ok=True)
                plot_path = os.path.join(plot_dir, '_'.join(os.path.basename(row['file path']).split('_')[:-1])+'.png')
                plot_signal_and_online_predictions(
                    time=full_day_data['Timestamp'], 
                    signal=np.array([full_day_data['Acc X [g]'].values, full_day_data['Acc Y [g]'].values, full_day_data['Acc Z [g]'].values]),
                    online_avg=smoothed_scores, 
                    online_avg_times=smoothed_start_times,
                    window_length=window_length, 
                    label_encoder=label_encoder, 
                    plot_path=plot_path, 
                    half_day_behaviors=None
                )

def extract_feeding_events(dir):
    
    feeding_events = []
    data_dir = os.path.join(dir, 'evals')

    for fname in tqdm(os.listdir(data_dir)):
        if fname.endswith(".csv"):  # check file extension
            file_path = os.path.join(data_dir, fname)
            df = pd.read_csv(file_path)

            # filter rows where Behavior == "Feeding"
            feeding = df[df["behavior"] == "Feeding"]
            feeding_events.append(feeding)

    # concatenate all feeding rows into one DataFrame
    feeding_df = pd.concat(feeding_events, ignore_index=True)
    # feeding_df = feeding_df.rename(columns={'Most probable behavior': 'behavior'})

    # save the result
    feeding_df.to_csv(os.path.join(dir, "feeding_events.csv"), index=False)

def windowed_ptp_stats(arr, window=32):
    n_full_windows = len(arr) // window
    if n_full_windows == 0:
        return np.nan, np.nan  # Not enough data

    ptp_values = [np.ptp(arr[i*window:(i+1)*window]) for i in range(n_full_windows)]
    return np.float32(np.max(ptp_values)), np.float32(np.mean(ptp_values))

# Apply to each column and create new stats columns
def process_column(df, col, sampling_rate=16):
    df[[f'{col}_ptp_max', f'{col}_ptp_mean']] = df[col].apply(
        lambda arr: pd.Series(windowed_ptp_stats(arr, window=int(2*sampling_rate)))
    )
    return df

def all_online_summary_eval(model_dir, metadata, device, sampling_frequency=16, window_length=None, window_duration=None, smoothening_window_length=1, smoothening_hop_length=1, plot=True, dir=None):

    if (window_length is None) & (window_duration is None):
        raise ValueError('A window length/duratioon for the classification model is required.')
    
    if (window_length is None) & (window_duration is not None):
        window_length = int(window_duration*sampling_frequency)

    if (window_length is not None) & (window_duration is not None):
        assert window_length == int(window_duration*sampling_frequency), "window length and window duration are not compatible according to provided sampling frequency."
    
    model = torch.load(os.path.join(model_dir, 'model.pt'),  weights_only=False, map_location=device)

    # fit the label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(SUMMARY_BEHAVIORS)

    # group by individual ID and date
    grouped = metadata.groupby(["individual ID", "UTC Date [yyyy-mm-dd]"])

    for (individual, date), group in tqdm(grouped, total=len(grouped)):
        print(f"Processing individual={individual}, date={date}")

        # load one or two half-day files
        dfs = []
        for _, row in group.iterrows():
            df_half = pd.read_csv(row['file path'])
            df_half['Timestamp'] = pd.to_datetime(df_half['Timestamp'], utc=True, format='%Y-%m-%d %H:%M:%S.%f')
            dfs.append(df_half)

        # merge & sort
        full_day_data = pd.concat(dfs, ignore_index=True).sort_values("Timestamp")

        if len(full_day_data) < window_length:
            warnings.warn(f'{individual}-{date} has fewer samples than window length. Skipped.')
            continue

        start_index = 0

        # sliding windows
        acc_data = []

        while start_index + window_length < len(full_day_data):

            end_index = start_index  + window_length
            window = full_day_data.iloc[start_index:end_index]
            # Collect timestamps for start and end of the window
            window_start = window['Timestamp'].iloc[0]
            window_end = window['Timestamp'].iloc[-1]
            acc_data.append({'Timestamp start': window_start, 
                            'Timestamp end': window_end,
                            'acc_x': window['Acc X [g]'].values,
                            'acc_y': window['Acc Y [g]'].values,
                            'acc_z': window['Acc Z [g]'].values
                            })
            start_index = end_index

        # Create DataFrame for windows
        acc_data = pd.DataFrame(acc_data)
        acc_data['individual ID'] = individual
        acc_data['UTC Date [yyyy-mm-dd]'] = date

        print(f"Total {len(acc_data)} in this day.")
        
        acc_data['acc_x_mean'] = acc_data['acc_x'].apply(np.mean).astype(np.float32)
        acc_data['acc_y_mean'] = acc_data['acc_y'].apply(np.mean).astype(np.float32)
        acc_data['acc_z_mean'] = acc_data['acc_z'].apply(np.mean).astype(np.float32)

        for col in ['acc_x', 'acc_y', 'acc_z']:
            acc_data = process_column(acc_data, col, sampling_rate=config.SAMPLING_RATE)

        feature_cols = ['acc_x_ptp_max',
                                'acc_y_ptp_max',
                                'acc_z_ptp_max',
                                'acc_x_ptp_mean',
                                'acc_y_ptp_mean',
                                'acc_z_ptp_mean',
                                'acc_x_mean',
                                'acc_y_mean',
                                'acc_z_mean']
    
        acc_data = acc_data.drop(columns=['acc_x', 'acc_y', 'acc_z'])
        X = np.array(acc_data[feature_cols].values).reshape(len(acc_data), 9)
        X = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            scores = model(X.to(device))

        # results dataframe
        acc_data['Prediction scores'] = np.max(F.softmax(scores, dim=1).cpu().numpy(), axis=1)
        acc_data['behavior'] = label_encoder.inverse_transform(np.argmax(scores.cpu().numpy(), axis=1))

        if dir is not None:
            eval_dir = os.path.join(dir, 'evals')
            os.makedirs(eval_dir, exist_ok=True)
            eval_path = os.path.join(eval_dir, '_'.join(os.path.basename(row['file path']).split('_')[:-1])+'.csv')
            acc_data.to_csv(eval_path, index=False)

            if plot is True:

                # 1. Prepare scores array with shape (num_classes, num_windows)
                scores_np = np.array(scores.unsqueeze(0).detach().cpu().numpy()) # (1, number of windows, number of classes)
                scores_np = np.transpose(scores_np, (0,2,1)) # (1, number of classes, number of windows)

                # 2. Get the start timestamp for each score window from the 'evals' DataFrame
                initial_start_times = acc_data['Timestamp start'].values

                # 3. Call the smoothing function with the scores AND their timestamps
                smoothed_scores, smoothed_start_times = online_smoothening(
                    scores=scores_np,
                    start_times=initial_start_times,
                    window_len=smoothening_window_length,
                    hop_len=smoothening_hop_length
                )
    
                plot_dir = os.path.join(dir, 'plots')
                os.makedirs(plot_dir, exist_ok=True)
                plot_path = os.path.join(plot_dir, '_'.join(os.path.basename(row['file path']).split('_')[:-1])+'.png')
                plot_signal_and_online_predictions(
                    time=full_day_data['Timestamp'], 
                    signal=np.array([full_day_data['Acc X [g]'].values, full_day_data['Acc Y [g]'].values, full_day_data['Acc Z [g]'].values]),
                    online_avg=smoothed_scores, 
                    online_avg_times=smoothed_start_times,
                    window_length=window_length, 
                    label_encoder=label_encoder, 
                    plot_path=plot_path, 
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
    
    window_duration = args.window_duration
    window_length = int(window_duration * config.SAMPLING_RATE)

    model_dir = io.get_results_path('no_split', args.n_CNNlayers, args.n_channels, args.kernel_size, args.theta)
    model = torch.load(os.path.join(model_dir, 'model.pt'), map_location=device)

    metadata = pd.read_csv(io.get_metadata_path())
    metadata = metadata[metadata['individual ID'] == args.individual]
    metadata['UTC Date [yyyy-mm-dd]'] = pd.to_datetime( metadata['UTC Date [yyyy-mm-dd]'], format='%Y-%m-%d')

    all_online_raw_eval(model_dir=model_dir, 
                    metadata=metadata, 
                    device=device, 
                    sampling_frequency=config.SAMPLING_RATE, 
                    window_length=window_length,
                    window_duration=window_duration, 
                    smoothening_window_length=args.smoothening_window_length, 
                    smoothening_hop_length=args.smoothening_hop_length,  
                    dir=config.VECTRONICS_BEHAVIOR_EVAL_PATH, 
                    plot=True)
    
    # extract_feeding_events(config.VECTRONICS_BEHAVIOR_EVAL_PATH)
    # window_duration = 30.0
    # training_results_dir = os.path.join(io.get_results_dir(), 'summary_training_results')
    # model_dir = os.path.join(training_results_dir, f"duration{window_duration}_theta{args.theta}_seed{args.seed}")

    # all_online_summary_eval(model_dir, 
    #                         metadata, 
    #                         device, 
    #                         sampling_frequency=config.SAMPLING_RATE, 
    #                         window_length=None, 
    #                         window_duration=window_duration, 
    #                         smoothening_window_length=1, 
    #                         smoothening_hop_length=1, 
    #                         plot=True, 
    #                         dir=config.VECTRONICS_SUMMARY_BEHAVIOR_EVAL_PATH)
    # extract_feeding_events(config.VECTRONICS_SUMMARY_BEHAVIOR_EVAL_PATH)
    







    


        

