import sys
import os
sys.path.append('.')
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from pytz import timezone
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
import torch
import warnings
import time
import ast
import json
from scipy.signal import butter, filtfilt
from config.settings import (SAMPLING_RATE,
                             id_mapping
)
from src.utils.io import (format_time,
                          get_metadata_path,
                          get_video_labels_path,
                          get_audio_labels_path,
                          get_matched_data_path, 
                          get_matched_metadata_path,
                          get_matched_summary_path,
)
            

def combined_annotations(video_path, audio_path, id_mapping):

    """Combine the annotations from gold and silver labels.
    
    Arguments 
    --------------------
    path: dictionary = a sictionary of paths to folders of different AWD
    id_mapping: dictionary =  dict for matching the id names in annotations to names used uniformly
    
    Returns 
    --------------------
    all_annotations: Pandas dataframe = dataframe of all annotations with time stamps

    """
    video_annotations = pd.read_csv(video_path) # load video annotations
    audio_annotations = pd.read_csv(audio_path) # load audio annotations

    video_annotations['id'] = video_annotations['id'].replace(id_mapping)
    audio_annotations['Individual'] = audio_annotations['Individual'].replace(id_mapping)

    audio_annotations = audio_annotations[audio_annotations['Confidence (H-M-L)'].isin(['H', 'H/M'])]
    
    audio_annotations = audio_annotations.assign(Source='Audio')
    video_annotations = video_annotations.assign(Source='Video')


    annotations_columns = ['id', 'Behavior', 'Timestamp_start', 'Timestamp_end', 'Source']
    rename_dict = {'Individual': 'id', 'Behaviour': 'Behavior', 'Timestamp_start_utc': 'Timestamp_start', 'Timestamp_end_utc': 'Timestamp_end'}
    audio_annotations = audio_annotations.rename(columns=rename_dict)

    botswana_timezone = 'Africa/Gaborone'
    video_annotations['Timestamp_start'] = pd.to_datetime(video_annotations['Timestamp_start'], format='%Y/%m/%d %H:%M:%S')
    video_annotations['Timestamp_end'] = pd.to_datetime(video_annotations['Timestamp_end'], format='%Y/%m/%d %H:%M:%S')

    # localize to botswana time and the change clock time to utc
    video_annotations['Timestamp_start'] = video_annotations['Timestamp_start'].apply(lambda x: x.tz_localize(botswana_timezone).tz_convert('UTC'))
    video_annotations['Timestamp_end'] = video_annotations['Timestamp_end'].apply(lambda x: x.tz_localize(botswana_timezone).tz_convert('UTC'))
    
    # now remove time zone information
    video_annotations['Timestamp_start'] = video_annotations['Timestamp_start'].dt.tz_localize(None)
    video_annotations['Timestamp_end'] = video_annotations['Timestamp_end'].dt.tz_localize(None)

    # make sure timestamp is in a given format
    video_annotations['Timestamp_start'] = video_annotations['Timestamp_start'].dt.strftime('%Y/%m/%d %H:%M:%S')
    video_annotations['Timestamp_end'] = video_annotations['Timestamp_end'].dt.strftime('%Y/%m/%d %H:%M:%S')


    all_annotations = pd.concat([video_annotations[annotations_columns], audio_annotations[annotations_columns]])

    return all_annotations


def create_matched_data(filtered_metadata, annotations, verbose=True):
    
    """Match the files in metadata with available annotations

    Arguments 
    ---------------
    filtered_metadata: pandas dataframe with columns ['file path', 'individual ID', 'year', 'UTC Date [yyyy-mm-dd]', 'am/pm', 'half day [yyyy-mm-dd_am/pm]', 'avg temperature [C]']
    annotations: pandas dataframe with columns ['id', 'Behavior', 'Timestamp_start', 'Timestamp_end', 'Source']

    Returns
    ----------------
    acc_summary: pandas dataframe = summary of the matched acceleration files with columns 

        id: string
            individual ID
        date_am_pm_id: string
            In format yyyy-mm-dd_{am/pm} 
        annotations: string 
            behavior class
        acc: float
            matched acceleration duration
        number of matched acc: int 
            number of matched annotations in the half day

    acc_data: pandas dataframe = final matched windows of acc data  with columns

        individual ID: string
            individual ID
        behavior: string
            behvior annotation
        behavior_start: string 
            behavior start time in format %Y/%m/%d %H:%M:%S
        behavior_end: string 
            behavior end time in format %Y/%m/%d %H:%M:%S
        duration: float
            duration of the matched behavior
        year: int 
            year of behavior observation
        UTC Date [yyyy-mm-dd]: string 
            date of behavior observation
        am/pm: string 
            AM or PM time of behavior observation
        half day [yyyy-mm-dd_am/pm]: string 
            half day of behavior observation
        avg temperature [C]: float 
            average temperature on the half day of behavior observation
        acc_x: list-like object
            acceleration data along X axis
        acc_y: list-like object
            acceleration data along Y axis
        acc_z: list-like object 
            acceleration data along Z axis
        Source: string 
            source of behavior annotation (video, audio, etc)

    acc_data_metadata: pandas dataframe = metadata of the acceleration segments matched with annotations

        file_path: string
            file path where the half-day segment of the acceleration snippet is stored
        individual ID: string
            individual ID
        year: int 
            year of behavior observation
        UTC Date [yyyy-mm-dd]: string 
            date of behavior observation
        am/pm: string 
            AM or PM time of behavior observation
        half day [yyyy-mm-dd_am/pm]: string 
            half day of behavior observation
        avg temperature [C]: float 
            average temperature on the half day of behavior observation

    """
    # create dataframes for saving matched acceleration and behavior data

    cols = ['individual ID', 'behavior', 'behavior_start', 'behavior_end', 'duration', 'year', 'UTC Date [yyyy-mm-dd]', 'am/pm',  'half day [yyyy-mm-dd_am/pm]', 'avg temperature [C]', 'acc_x', 'acc_y', 'acc_z', 'Source']
    acc_data = pd.DataFrame(columns=cols, index=[])
    acc_data_metadata = pd.DataFrame(columns=filtered_metadata.columns, index=[])
    acc_summary = pd.DataFrame(columns=['id', 'date_am_pm_id', 'annotations', 'acc', 'number of matched acc'], index=[])


    # loop over all unique individuals in filtered_metadata
    for (i, individual) in enumerate(filtered_metadata['individual ID'].unique()):

        ## find annotations for this individual
        annotations_orig = annotations[annotations['id'] == individual]
        individual_annotations = annotations_orig.copy()

        # Format and add helper columns to the annotations dataframe
        individual_annotations['Timestamp_start'] = pd.to_datetime(annotations_orig['Timestamp_start'], format='%Y/%m/%d %H:%M:%S')
        individual_annotations['Timestamp_end'] = pd.to_datetime(annotations_orig['Timestamp_end'], format='%Y/%m/%d %H:%M:%S')
        individual_annotations['date'] = individual_annotations['Timestamp_start'].dt.date
        individual_annotations['am/pm'] = pd.to_datetime(individual_annotations['Timestamp_start'], format="%Y/%m/%d %H:%M:%S").dt.strftime('%p').str.lower() 
        individual_annotations['half day [yyyy-mm-dd_am/pm]'] = individual_annotations['date'].astype(str) + '_' + individual_annotations['am/pm']
        
        # create submetadata file for this individual
        individual_metadata = filtered_metadata[filtered_metadata['individual ID'] == individual]

        if verbose:
            print('individual {} has {} halfdays in the filtered metadata.'.format(individual, len(individual_metadata)))

        for unique_period_loop in tqdm(individual_metadata['half day [yyyy-mm-dd_am/pm]'].unique(), desc=f'Processing unique half days for {individual}'):

            annotation_available = unique_period_loop in individual_annotations['half day [yyyy-mm-dd_am/pm]'].values

            if annotation_available:

                annotations_loop = individual_annotations[individual_annotations['half day [yyyy-mm-dd_am/pm]'] == unique_period_loop]
                
                # if the acceleration file is available for this individual and half day, read it
                
                acc_file_path = individual_metadata.loc[individual_metadata['half day [yyyy-mm-dd_am/pm]'] == unique_period_loop, 'file path'].values[0]
                acc_loop = pd.read_csv(acc_file_path)
                acc_loop['Timestamp'] = pd.to_datetime(acc_loop['Timestamp'], format='mixed', utc=True)

                for _, row in annotations_loop.iterrows():
                        
                    behaviour_start_time = row['Timestamp_start'].to_pydatetime().replace(tzinfo=timezone('UTC'))
                    behaviour_end_time = row['Timestamp_end'].to_pydatetime().replace(tzinfo=timezone('UTC'))

                    if not pd.isnull(behaviour_end_time):
                        acc_summary.loc[len(acc_summary)] = [individual, unique_period_loop, 0, 0.0, 0]
                        acc_summary.at[acc_summary.index[-1], 'annotations'] += (behaviour_end_time - behaviour_start_time).total_seconds() 
            
                    
                    if (not pd.isnull(behaviour_end_time)) & (behaviour_end_time > behaviour_start_time):          

                        # log the duration of audio avalilable for the behaviour

                        behaviour_acc = acc_loop[(acc_loop['Timestamp'] >= behaviour_start_time) & (acc_loop['Timestamp'] <= behaviour_end_time)].sort_values('Timestamp')
                        
                        # log the duration of acc avalilable for the behaviour
                        if len(behaviour_acc) > 0:

                            # duration of the acceleration data that is matched with the behavior
                            matched_duration = (behaviour_acc.iloc[len(behaviour_acc)-1]['Timestamp'] - behaviour_acc.iloc[0]['Timestamp']).total_seconds()

                            if matched_duration < min_duration - 1:
                                print(f"Behavior duration: {(behaviour_end_time - behaviour_start_time).total_seconds()}, Acc duration: {matched_duration}")

                            acc_summary.at[acc_summary.index[-1], 'acc'] += matched_duration
                            acc_summary.at[acc_summary.index[-1], 'number of matched acc'] += 1

                            acc_data.loc[len(acc_data)] = [individual, 
                                                            row['Behavior'], 
                                                            behaviour_acc.iloc[0]['Timestamp'], 
                                                            behaviour_acc.iloc[len(behaviour_acc)-1]['Timestamp'], 
                                                            matched_duration, 
                                                            individual_metadata.loc[individual_metadata['half day [yyyy-mm-dd_am/pm]'] == unique_period_loop, 'year'].values[0],
                                                            individual_metadata.loc[individual_metadata['half day [yyyy-mm-dd_am/pm]'] == unique_period_loop, 'UTC Date [yyyy-mm-dd]'].values[0],
                                                            individual_metadata.loc[individual_metadata['half day [yyyy-mm-dd_am/pm]'] == unique_period_loop, 'am/pm'].values[0],
                                                            unique_period_loop,
                                                            individual_metadata.loc[individual_metadata['half day [yyyy-mm-dd_am/pm]'] == unique_period_loop, 'avg temperature [C]'].values[0],
                                                            behaviour_acc['Acc X [g]'].to_list(),
                                                            behaviour_acc['Acc Y [g]'].to_list(),
                                                            behaviour_acc['Acc Z [g]'].to_list(),
                                                            row['Source']]

                            acc_data_metadata.loc[len(acc_data_metadata)] = individual_metadata.loc[individual_metadata['half day [yyyy-mm-dd_am/pm]'] == unique_period_loop].values[0].tolist()
            
            else:
                acc_summary.loc[len(acc_summary)] = [individual, unique_period_loop, 0, 0.0, 0]

    return acc_summary, acc_data, acc_data_metadata


def windowed_ptp_stats(arr, window=32):
    n_full_windows = len(arr) // window
    if n_full_windows == 0:
        return np.nan, np.nan  # Not enough data

    ptp_values = [np.ptp(arr[i*window:(i+1)*window]) for i in range(n_full_windows)]
    return np.max(ptp_values), np.mean(ptp_values)

# Apply to each column and create new stats columns
def process_column(df, col, sampling_rate=16):
    df[[f'{col}_ptp_max', f'{col}_ptp_mean']] = df[col].apply(
        lambda arr: pd.Series(windowed_ptp_stats(arr, window=int(2*sampling_rate)))
    )
    return df

def split_row(row, chunk_size=480, sampling_rate=16):
    length = len(row['acc_x'])
    n_chunks = length // chunk_size
    remainder = length % chunk_size

    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size

        chunks.append({
            'behavior': row['behavior'],
            'acc_x': row['acc_x'][start:end],
            'acc_y': row['acc_y'][start:end],
            'acc_z': row['acc_z'][start:end],
            'duration': chunk_size / sampling_rate
        })

    if remainder > 0:
        chunks.append({
            'behavior': row['behavior'],
            'acc_x': row['acc_x'][-remainder:],
            'acc_y': row['acc_y'][-remainder:],
            'acc_z': row['acc_z'][-remainder:],
            'duration': remainder / sampling_rate
        })

    return chunks

def create_max_windows(acc_data, window_duration=30.0, sampling_rate=16):
    # Apply the splitting to all rows and flatten the result
    split_chunks = []
    for _, row in acc_data.iterrows():
        split_chunks.extend(split_row(row, chunk_size=int(window_duration*sampling_rate), sampling_rate=sampling_rate))

    # Create the new DataFrame
    acc_data_split = pd.DataFrame(split_chunks)
    return acc_data_split

def create_summary_data(acc_data_split, sampling_rate=16):
    acc_data_split['acc_x_mean'] = acc_data_split['acc_x'].apply(np.mean)
    acc_data_split['acc_y_mean'] = acc_data_split['acc_y'].apply(np.mean)
    acc_data_split['acc_z_mean'] = acc_data_split['acc_z'].apply(np.mean)

    for col in ['acc_x', 'acc_y', 'acc_z']:
        acc_data_split = process_column(acc_data_split, col, sampling_rate=sampling_rate)

    return acc_data_split

def create_data_splits(acc_data, feature_cols, test_size=0.2, val_size=0.25):
    acc_data = acc_data.dropna()

    X = acc_data[feature_cols].values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(acc_data['behavior'])

    # First: train+val and test split
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_val_idx, test_idx = next(sss1.split(X, y))

    X_train_val, y_train_val = X[train_val_idx], y[train_val_idx]
    X_test, y_test = X[test_idx], y[test_idx]  

    # Second: train and val split from train_val
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
    train_idx, val_idx = next(sss2.split(X_train_val, y_train_val))

    X_train, y_train = X_train_val[train_idx], y_train_val[train_idx]
    X_val, y_val = X_train_val[val_idx], y_train_val[val_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test

def give_balanced_weights(theta, y):
    n_classes = len(np.unique(y))
    weights = theta*np.ones(n_classes)/n_classes + (1-theta)*np.unique(y, return_counts=True)[1]/len(y)
    return weights

def setup_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, args):

    n_outputs = len(np.unique(np.concatenate((y_train, y_val, y_test))))
    
    weights = give_balanced_weights(args.theta, y_train)
    # sample_weights = np.unique(y_train, return_counts=True)[1]
    y_weights = torch.tensor([weights[i] for i in y_train], dtype=torch.float32)
    sampler = WeightedRandomSampler(y_weights, len(y_weights))

    # converting to one-hot vectors
    y_train = np.eye(n_outputs)[y_train]
    y_val = np.eye(n_outputs)[y_val]
    y_test = np.eye(n_outputs)[y_test]

    # Convert data and labels to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # Create DataLoader for training and testing
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

def adjust_behavior_and_durations(df, collapse_behavior_mapping, behaviors, verbose=False):

    """
        1. Collapse behaviors to coarser classes.
        2. filter out behaviors of interest
        3. remove behaviors of shorter duration
        4. remove running, moving, eatng, and marking behaviors shorter than 8 sec
    """

    if verbose:
        duration_before_filter = df.duration.sum()
        print(f'Total behavior duration before filtering - {duration_before_filter/3600}')  

    # collapse classes
    df['behavior'] = df['behavior'].replace(collapse_behavior_mapping) # collapse behaviors
    df = df[df['behavior'].isin(behaviors)]

    if verbose:
        duration_sum = df.duration.sum()
        print(f'Total duration after filtering out chosen behaviors is {duration_sum/3600} hrs.')

    # filtration
    df = df[df['duration'] >= 1]
    idx = [(df['behavior'].isin(['Running', 'Feeding', 'Moving'])) & (df['duration'] < 8) & (df['Source'] == 'Video')][0] # running from video lables of duration lesser than 8 sec are unreliable
    df = df[~idx]

    if verbose:
        duration_sum = df.duration.sum()
        print(f'Total behavior duration after filtering is {duration_sum/3600} hrs.')

    return df


def get_exp_filter_profiles(exp_name):

    """Return the train and test filter profiles for our different experiments."""

    train_filter_profile = {'individual ID': None,
                   'year': None,
                   'UTC Date [yyyy-mm-dd]': None,
                   'am/pm': None,
                   'half day [yyyy-mm-dd_am/pm]': None,
                   'avg temperature [C]': None}

    test_filter_profile = {'individual ID': None,
                'year': None,
                'UTC Date [yyyy-mm-dd]': None,
                'am/pm': None,
                'half day [yyyy-mm-dd_am/pm]': None,
                'avg temperature [C]': None}

    if exp_name == 'no_split':
        pass
        
    elif exp_name == 'interdog':
        train_filter_profile['individual ID'] = ['jessie', 'palus', 'ash', 'fossey']
        test_filter_profile['individual ID'] = ['green',]
    
    elif exp_name == 'interyear':
        train_filter_profile['year'] = [2021,]
        test_filter_profile['year'] = [2022,]
    
    elif exp_name == 'interAMPM':
        train_filter_profile['am/pm'] = ['am',]
        test_filter_profile['am/pm'] = ['pm',]

    elif exp_name == 'test_interyear':
        train_filter_profile['year'] = [2022,]
        test_filter_profile['year'] = [2025,]

    else:
        raise ValueError("Unspecified experiment name")

    return train_filter_profile, test_filter_profile


def train_test_metadata_split(train_metadata, test_metadata, test_size=0.2, random_state=0):
    
    """Create a split of train and test metedata so that there is no overlap. 
    Takes the train and test metadat, finds overlapping rows and divides them between 
    train and test data according to proportion of test data.

    Arguments
    --------------------
    train_metadata: pandas DataFrame
    test_metadata: pandas DataFrame
    test_size: float = 0.2

    Returns 
    ---------------------
    train_metadata: pandas DataFrame
    test_metadata: pandas DataFrame

    """
    
    # Perform inner join to extract overlapping rows
    overlapping_df = pd.merge(train_metadata, test_metadata, on=None, how='inner')

    # Perform left join to remove overlapping rows from df1
    df1_no_overlap = pd.merge(train_metadata, overlapping_df, on=None, how='left', indicator=True)
    df1_no_overlap = df1_no_overlap[df1_no_overlap['_merge'] == 'left_only']
    df1_no_overlap.drop(columns='_merge', inplace=True)

    # Perform right join to remove overlapping rows from df2
    df2_no_overlap = pd.merge(overlapping_df, test_metadata, on=None, how='right', indicator=True)
    df2_no_overlap = df2_no_overlap[df2_no_overlap['_merge'] == 'right_only']
    df2_no_overlap.drop(columns='_merge', inplace=True)

    overlap_train, overlap_test = train_test_split(overlapping_df, test_size=test_size, shuffle=True, random_state=random_state)

    df1 = pd.concat([df1_no_overlap, overlap_train])
    df2 = pd.concat([df2_no_overlap, overlap_test])

    return df1, df2

def split_overlapping_indices(train_indices, test_indices, behaviors, split_ratio=0.5):

    # Convert indices to sets for easier manipulation
    train_set = set(train_indices)
    test_set = set(test_indices)
    
    # Find overlapping indices
    overlapping_indices = train_set & test_set
    
    # Find non-overlapping indices
    train_non_overlap = train_set - overlapping_indices
    test_non_overlap = test_set - overlapping_indices
    
    # Convert sets back to sorted lists
    train_non_overlap = sorted(list(train_non_overlap))
    test_non_overlap = sorted(list(test_non_overlap))
    
    # Convert overlapping_indices to a sorted list
    overlapping_indices = np.array(sorted(list(overlapping_indices)))
    print(f'Overlapping indices of shape = {overlapping_indices.shape}')

    # train validation split 
    stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=split_ratio, random_state=42)

    for train_index, test_index in stratified_splitter.split(overlapping_indices, behaviors[overlapping_indices]):
        train_overlap_split, test_overlap_split = list(overlapping_indices[train_index]), list(overlapping_indices[test_index])
    
    # Combine non-overlapping indices with the split overlapping indices
    final_train_indices = train_non_overlap + train_overlap_split
    final_test_indices = test_non_overlap + test_overlap_split
    
    return final_train_indices, final_test_indices

def filter_data(metadata, filter_profile):

        """Filters the index from metadata that satisfy metadata constraints.

        Arguments
        --------------
        metadata: pd.DataFrame
        filter_profile: dictionary-like object
        """

        filter_idx = np.arange(len(metadata))

        # filter desired individual ID
        if filter_profile['individual ID'] is not None:
                assert isinstance(filter_profile['individual ID'], list), "individual ID filter should be a list"
                filter_idx = [idx for idx in filter_idx if metadata.iloc[idx]['individual ID'] in filter_profile['individual ID']]

        # filter desired year
        if filter_profile['year'] is not None:
                assert isinstance(filter_profile['year'], list), "year filter should be a list"
                filter_idx = [idx for idx in filter_idx if metadata.iloc[idx]['year'] in filter_profile['year']]

        # filter desired dates
        if filter_profile['UTC Date [yyyy-mm-dd]'] is not None:
                assert isinstance(filter_profile['UTC Date [yyyy-mm-dd]'], list), "year filter should be a list"

                date_idx = []
                for date_range in filter_profile['UTC Date [yyyy-mm-dd]']:
                        assert date_range[0] is None or date_range[1] is None or date_range[0] < date_range[1], "Incorrect date range"
                        assert isinstance(date_range, tuple), "Each entry of the date filter should be a tuple of lower and upper range of desired dates. Provide -1 for entire tail."
                        range_idx = filter_idx
                        if date_range[0] is not None:
                                lower_limit = datetime.strptime(date_range[0], '%Y-%m-%d')
                                range_idx = [idx for idx in filter_idx if datetime.strptime(metadata.iloc[idx]['UTC Date [yyyy-mm-dd]'], "%Y-%m-%d") >= lower_limit]
                        
                        if date_range[1] is not None:
                                upper_limit = datetime.strptime(date_range[1], '%Y-%m-%d')
                                range_idx = [idx for idx in range_idx if datetime.strptime(metadata.iloc[idx]['UTC Date [yyyy-mm-dd]'], "%Y-%m-%d") <= upper_limit]
                        date_idx.extend(range_idx)
                filter_idx = date_idx

        # filter desired am/pd ID
        if filter_profile['am/pm'] is not None:
                assert isinstance(filter_profile['am/pm'], list), "am/pm filter should be a list"
                filter_idx = [idx for idx in filter_idx if metadata.iloc[idx]['am/pm'] in filter_profile['am/pm']]

        # filter desired half days
        if filter_profile['half day [yyyy-mm-dd_am/pm]'] is not None:
                assert isinstance(filter_profile['half day [yyyy-mm-dd_am/pm]'], list), "half day [yyyy-mm-dd_am/pm] filter should be a list"
                half_day_idx = []
                for date_range in filter_profile['half day [yyyy-mm-dd_am/pm]']:
                        assert date_range[0] is None or date_range[1] is None or date_range[0] < date_range[1], "Incorrect date range"

                        range_idx = filter_idx
                        if date_range[0] is not None:
                                range_idx = [idx for idx in filter_idx if metadata.iloc[idx]['half day [yyyy-mm-dd_am/pm]'] >= date_range[0]]
                        
                        if date_range[1] is not None:
                                range_idx = [idx for idx in range_idx if metadata.iloc[idx]['half day [yyyy-mm-dd_am/pm]'] <= date_range[1]]
                        half_day_idx.extend(range_idx)
                filter_idx = half_day_idx

        # filter desired average temperature ranges
        if filter_profile['avg temperature [C]'] is not None:
                assert isinstance(filter_profile['avg temperature [C]'], list), "avg temperature [C] filter should be a list"

                temp_idx = []
                for temp_range in filter_profile['avg temperature [C]']:

                        assert temp_range[0] is None or temp_range[1] is None or temp_range[0] < temp_range[1], "Incorrect temp range"
                        range_idx = filter_idx
                        if temp_range[0] is not None:
                                range_idx = [idx for idx in filter_idx if metadata.iloc[idx]['avg temperature [C]'] >= temp_range[0]]
                        
                        if temp_range[1] is not None:
                                range_idx = [idx for idx in range_idx if metadata.iloc[idx]['avg temperature [C]'] <= temp_range[1]]
                        temp_idx.extend(range_idx)
                filter_idx = temp_idx

        return filter_idx

def match_train_test_df(metadata, all_annotations, collapse_behavior_mapping, behaviors, args):

    """match train and test data based on filter profiles and create train-test df
    
    Arguments:
    ----------------
    metadata: pd DataFrame
    all_annotations: pd DataFrame = all annotations from videos and audios
    collapse_behavior_mapping: dictionary 
    behaviors: list = list of desired behaviors
    args: dictionary
    
    Returns
    -----------------
    df_train: pd DataFrame
    df_test: pd DataFrame
    """

    # get filter profiles for this experiment
    train_filter_profile, test_filter_profile = get_exp_filter_profiles(args.experiment_name) 
    train_filtered_metadata = metadata.iloc[filter_data(metadata, train_filter_profile)]
    test_filtered_metadata = metadata.iloc[filter_data(metadata, test_filter_profile)]

    if len(pd.merge(train_filtered_metadata, test_filtered_metadata, on=None, how='inner')):
        warnings.warn("train and test filters overlap", UserWarning)
        print(f'Before overlap, \nno. of train half days: {len(train_filtered_metadata)}, no. of test half days: {len(test_filtered_metadata)}')
        train_filtered_metadata, test_filtered_metadata = train_test_metadata_split(train_filtered_metadata, test_filtered_metadata, args.train_test_split)
        print(f'After removing overlaps, \nno. of train half days: {len(train_filtered_metadata)}, no. of test half days: {len(test_filtered_metadata)}')
    else:
        print(f'No overlaps. \nno. of train half days: {len(train_filtered_metadata)}, no. of test half days: {len(test_filtered_metadata)}')

    t1 = time.time()

    # match filtered data with annotations
    _, df_train, _ = create_matched_data(train_filtered_metadata, all_annotations)
    _, df_test, _ = create_matched_data(test_filtered_metadata, all_annotations)

    t2 = time.time()

    print("")
    print("==================================")
    print(f'Data frames matched in {format_time(t2-t1)}.')

    df_train = adjust_behavior_and_durations(df_train, collapse_behavior_mapping, behaviors)
    df_test = adjust_behavior_and_durations(df_test, collapse_behavior_mapping, behaviors)

    df_train.reset_index()
    df_test.reset_index()

    return df_train, df_test

def load_matched_train_test_df(collapse_behavior_mapping, behaviors, exp_name, acc_data_path, acc_metadata_path, train_test_split=0.2):


    """load pre-matched accereation-behavior data. Create train and test dataframes based on filter profiles
    
    Arguments:
    ----------------
    collapse_behavior_mapping: dictionary 
    behaviors: list = list of desired behaviors
    args.experiment_name: string = name of experiment provided to `get_exp_filter_profiles` function.
    acc_data_path: string = path where matched acceleration data is stored
    acc_metadata_path: string = path where matched acceleration metadata is stored
    train_test_split: float = proportion of test data in overlapping data points
    
    Returns
    -----------------
    df_train: pd DataFrame
    df_test: pd DataFrame
    """

    train_filter_profile, test_filter_profile = get_exp_filter_profiles(exp_name) 
    
    acc_data = pd.read_csv(acc_data_path)
    acc_data_metadata = pd.read_csv(acc_metadata_path)

    acc_data['acc_x'] = acc_data['acc_x'].apply(json.loads)
    acc_data['acc_y'] = acc_data['acc_y'].apply(json.loads)
    acc_data['acc_z'] = acc_data['acc_z'].apply(json.loads)
    

    acc_data = adjust_behavior_and_durations(acc_data, collapse_behavior_mapping, behaviors)
    acc_data_metadata = acc_data_metadata.loc[acc_data.index]

    acc_data.reset_index()
    acc_data_metadata.reset_index()

    print(f'Total number of matched annotations: {len(acc_data)}')

    train_filter_idx = filter_data(acc_data_metadata, train_filter_profile)
    test_filter_idx = filter_data(acc_data_metadata, test_filter_profile)


    if len(set(train_filter_idx) & set(test_filter_idx)):
        warnings.warn("train and test filters overlap", UserWarning)
        print(f'Before overlap, \nno. of train observations: {len(train_filter_idx)}, no. of test observations: {len(test_filter_idx)}')
        train_filter_idx, test_filter_idx = split_overlapping_indices(train_filter_idx, test_filter_idx, acc_data['behavior'].values, split_ratio=train_test_split)
        print(f'After removing overlaps, \nno. of train observations: {len(train_filter_idx)}, no. of test observations: {len(test_filter_idx)}')
    else:
        print(f'No overlaps. \nno. of train observations: {len(train_filter_idx)}, no. of test observations: {len(test_filter_idx)}')


    df_train = acc_data.iloc[train_filter_idx]
    df_test = acc_data.iloc[test_filter_idx]

    df_train.reset_index()
    df_test.reset_index()

    return df_train, df_test

def repeat_or_truncate_list(lst, fixed_length, reuse=False, min_length=16):

    """
    Pad the list with repitition of data if it's shorter than fixed_length.
    Truncate the list if it's longer than fixed_length

    Arguments
    ---------
    lst: list 
    fixed_length: int
    reuse: indicator on whether to extract multiple windows from the list
    min_length: minimum length to repeat, discard otherwise
    
    """
    
    if reuse:
        bundle = []
        while len(lst) >= fixed_length:
                bundle.append(lst[:fixed_length])
                lst = lst[fixed_length:]
        if len(lst) >= min_length:
                lst = lst*(fixed_length//len(lst)) 
                bundle.append(lst + lst[:(fixed_length - len(lst))])
                
        return bundle

    else:
        if len(lst) < fixed_length:
                # Repeat the list if it's shorter than fixed_length
                lst = lst*(fixed_length//len(lst)) 
                return [lst + lst[:(fixed_length - len(lst))]]

        else:
                # Truncate the list if it's longer than fixed_length
                return [lst[len(lst)-fixed_length:]]


def pad_or_truncate_list(lst, fixed_length):
    """
    Pad the list with 0 if it's shorter than fixed_length.
    Truncate the list if it's longer than fixed_length

    Arguments
    ---------
    lst: list 
    fixed_length: int
    
    """
    if len(lst) < fixed_length:
        # Pad the list with 0 if it's shorter than fixed_length
        return [0] * (fixed_length - len(lst)) + lst
    else:
        # Truncate the list if it's longer than fixed_length
        return lst[len(lst)-fixed_length:]
    
def create_padded_or_truncated_data(df, fixed_length, padding='repeat', reuse_behaviors=[], min_duration=1.0):
        
    """Load the dataset and make the acc sequence along x, y, z of fixed length.

    Arguments
    ---------
    df: Pandas DataFrame = dataframe with columns `acc_x`, `acc_y`, `acc_z`, and `behavior`.
    fixed_length: int = fix the length of accelerometer sequences
    padding: str = type of padding for accelerations smaller than fixed length
    reuse_behaviors: 

    Return 
    ------
    X: array-like objective (N, C, L)
        independent variables
        N is number of observations, C is number of input channels, L is the length of features 

    Y: array-like objective (N,)
        categorical labels
    """

    # Apply pad_or_truncate_list to all cells in columns x, y, z

    if len(df) == 0:
        raise ValueError('No data provided')


    df_new = pd.DataFrame(columns=['acc_x', 'acc_y', 'acc_z'])
    df_metadata = pd.DataFrame(columns=['individual ID', 'year', 'UTC Date [yyyy-mm-dd]', 'am/pm', 'half day [yyyy-mm-dd_am/pm]', 'avg temperature [C]', 'Source'])
    
    if padding == 'zeros':
        df_new['acc_x'] = df['acc_x'].apply(pad_or_truncate_list, args=(fixed_length,))
        df_new['acc_y'] = df['acc_y'].apply(pad_or_truncate_list, args=(fixed_length,))
        df_new['acc_z'] = df['acc_z'].apply(pad_or_truncate_list, args=(fixed_length,))
        
    elif padding == 'repeat':

        expanded_rows = []
        for _, row in df.iterrows():

            reuse = row['behavior'] in reuse_behaviors

            acc_x_windows = repeat_or_truncate_list(row['acc_x'], fixed_length, reuse=reuse, min_length=min_duration*SAMPLING_RATE)
            acc_y_windows = repeat_or_truncate_list(row['acc_y'], fixed_length, reuse=reuse, min_length=min_duration*SAMPLING_RATE)
            acc_z_windows = repeat_or_truncate_list(row['acc_z'], fixed_length, reuse=reuse, min_length=min_duration*SAMPLING_RATE)

            assert len(acc_x_windows) == len(acc_y_windows) == len(acc_z_windows) 

            for x, y, z in zip(acc_x_windows, acc_y_windows, acc_z_windows):
                expanded_rows.append({'acc_x': x, 'acc_y': y, 'acc_z': z, 'behavior': row['behavior']})
                df_metadata.loc[len(df_metadata)] = row[['individual ID', 'year', 'UTC Date [yyyy-mm-dd]', 'am/pm', 'half day [yyyy-mm-dd_am/pm]', 'avg temperature [C]', 'Source']].values

        df_new = pd.DataFrame(expanded_rows)
    else:
        raise ValueError


    arr_list = []
    for col in ['acc_x', 'acc_y', 'acc_z']:
        arr_list.append(np.array(df_new[col].to_list()))


    X = np.stack(arr_list, axis=2)
    X = np.transpose(X, (0, 2, 1))

    y = df_new['behavior'].values

    return X, y, df_metadata

def apply_band_pass_filter(data, cutoff_frequency=0.0, sampling_rate=16, btype='high', N=5, axis=2):

    """Apply a high, low, or band pass filter

    data: Numpy 3-D array 
    cutoff_frequency: Optional = scalar for high or low pass, array of size 2 for bandpass
    sampling_rate: int 
    btype: str = type of bandpass filter. Choices = ['high', 'low', 'bandpass']
    N: int = order of bandpass cutoff
    axis: int = axis along the temporal component of the time series
    
    """

    if cutoff_frequency == 0.0 and btype=='high':
        return data

    if cutoff_frequency == 0.5 * sampling_rate and btype=='low':
        return data

    nyquist = 0.5 * sampling_rate

    # High/Low-pass Butterworth filter
    if btype in ['high', 'low']:
        normal_cutoff = cutoff_frequency / nyquist
        b, a = butter(N=N, Wn=normal_cutoff, btype=btype, analog=False)

    elif btype == 'bandpass':
        assert len(cutoff_frequency) == 2
        low = cutoff_frequency[0]/nyquist
        high = cutoff_frequency[1]/nyquist
        b, a = butter(N=N, Wn=[low, high], btype='band', analog=False)
    
    # Apply the filter to each time series in the data
    filtered_data = filtfilt(b, a, data, axis=axis)
    
    return filtered_data

def setup_data_objects(metadata, all_annotations, collapse_behavior_mapping, 
                        behaviors, args, reuse_behaviors=[], acc_data_path=None, 
                        acc_metadata_path=None):

    """
    Arguments
    -----------------------
    metadata: Pandas Dataframe = metadata on all acceleration segments
    all_annotations: Pandas Dataframe = information on data frames
    collapse_behavior_mapping: dictionary 
    behaviors: list = list of behaviors of interest
    args: dictionary 
    match: bool = whether to match behaviors or use a pre-matched dataframe
    acc_data_path: string = path where matched acceleration data is stored, default=None
    acc_metadata_path: string = path where matched acceleration metadata is stored, default=None
    

    Returns 
    ----------------------
    X_train     : (n, d, T) np ndarray = train acceleration, n = no. of samples, d = no. of features, T = time axis            
    y_train     : (n, ) np ndarray    = train labels, n = no. of samples
    z_train     : pandas dataframe     = metadata associated with the train observations                                       
    X_val       : (n, d, T) np ndarray = val acceleration, n = no. of samples, d = no. of features, T = time axis           
    y_val       : (n, ) np ndarray    = val labels, n = no. of samples
    z_val       : pandas dataframe     = metadata associated with the validation observations                                  
    X_test      : (n, d, T) np ndarray = test acceleration, n = no. of samples, d = no. of features, T = time axis             
    y_test      : (n, ) np ndarray    = test labels, n = no. of samples
    z_test      : pandas dataframe     = metadata associated with the test observations                                        
    """

    t1 = time.time()
    if args.match or (acc_data_path is None) or (acc_metadata_path is None):
        print('Matching acceleration-behavior pairs...')
        df_train, df_test = match_train_test_df(metadata, all_annotations, collapse_behavior_mapping, behaviors, args)
    else:
        print('Using pre-matched acceleration-behavior pairs...')
        df_train, df_test = load_matched_train_test_df(collapse_behavior_mapping=collapse_behavior_mapping, 
                                                        behaviors=behaviors, 
                                                        exp_name=args.experiment_name, 
                                                        acc_data_path=acc_data_path,
                                                        acc_metadata_path=acc_metadata_path,
                                                        train_test_split=args.train_test_split)

    print("")
    print("==================================")
    print(f"Matching annotations to acceleration snippets takes {time.time() - t1:3f} seconds")

    t2 = time.time()
    # fix sequence max length and truncate/pad data to create X, y, and z.
    if args.window_duration_percentile is not None:
        max_acc_duration = np.percentile(np.concatenate((df_train['duration'].values, df_test['duration'].values), axis=0), args.window_duration_percentile)
    elif args.window_duration is not None:
        max_acc_duration = args.window_duration
    else:
        ValueError("Both window_duration_percentile and window_duration cannot be None in arguments.")
        
    max_steps = int(max_acc_duration*SAMPLING_RATE)
    X, y, z = create_padded_or_truncated_data(df_train, max_steps, padding=args.padding, reuse_behaviors=reuse_behaviors, min_duration=args.min_duration)
    X_test, y_test, z_test = create_padded_or_truncated_data(df_test, max_steps, padding=args.padding, reuse_behaviors=reuse_behaviors, min_duration=args.min_duration)
    print(f"Creating fixed-duration windows takes {time.time() - t2:3f} seconds.")

    print("")
    print("==================================")
    print(f"Time series duration window = {max_acc_duration}")

    # Band filter - no filter by default
    X = apply_band_pass_filter(X, args.cutoff_frequency, SAMPLING_RATE, btype=args.filter_type, N=args.cutoff_order, axis=2)
    X_test = apply_band_pass_filter(X_test, args.cutoff_frequency, SAMPLING_RATE, btype=args.filter_type, N=args.cutoff_order, axis=2)

    # standardize data
    if args.normalization:
        X = (X - np.mean(X, axis=0, keepdims=True))/np.std(X, axis=0, keepdims=True)
        X_test = (X_test - np.mean(X_test, axis=0, keepdims=True))/np.std(X_test, axis=0, keepdims=True)

    # label encoding
    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate((y, y_test)))
    y = label_encoder.transform(y)
    y_test = label_encoder.transform(y_test)
    
    # train validation split 
    stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.train_val_split, random_state=42)

    for train_index, val_index in stratified_splitter.split(X, y):
        X_train, X_val, y_train, y_val, z_train, z_val = X[train_index], X[val_index], y[train_index], y[val_index], z.iloc[train_index], z.iloc[val_index]

    return X_train, y_train, z_train, X_val, y_val, z_val, X_test, y_test, z_test, label_encoder


if __name__ == '__main__':
     
    metadata = pd.read_csv(get_metadata_path()) # load metadata
     
    all_annotations = combined_annotations(video_path=get_video_labels_path(), 
                                            audio_path=get_audio_labels_path(),
                                            id_mapping=id_mapping) # load annotations 
    
    print(f"Total number of annotations: {len(all_annotations)}")
    acc_summary, acc_data, acc_data_metadata = create_matched_data(filtered_metadata=metadata, annotations=all_annotations, verbose=True)
    acc_summary.to_csv(get_matched_summary_path(), index=False)
    acc_data.to_csv(get_matched_data_path(), index=False)
    acc_data_metadata.to_csv(get_matched_metadata_path(), index=False)