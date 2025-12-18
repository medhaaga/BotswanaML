import sys
import os
sys.path.append('.')
sys.path.append('../')
sys.path.append('../../')

from datetime import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm
from pytz import timezone
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import warnings
import time
import json
from scipy.signal import butter, filtfilt
import config as config
from src.utils.io import (format_time,
                          get_vectronics_metadata_path,
                          get_video_labels_path,
                          get_audio_labels_path,
                          get_vectronics_data_path, 
                          get_vectronics_acc_metadata_path,
                          get_vectronics_summary_path,
                          get_vectronics_annotations_summary_path
)
import src.utils.datasets as datasets
            

def process_chunk_vectronics(chunk, individual, file_dir, verbose=False):

    '''Save each chunk of acceleration data in respective half day segments

    Arguments
    --------------
    chunk: pd Dataframe
    individual: individual ID
    file_dir: path-like object = directory to save segments 

    '''

    # Expected columns - [UTC Date[mm/dd], UTC DateTime, Milliseconds, Acc X [g], Acc Y [g], Acc Z [g], Temperature [Celsius]]

    ts = pd.to_datetime(
    chunk['UTC Date[mm/dd/yyyy]'] + ' ' + chunk['UTC DateTime'],
    format='%m/%d/%Y %H:%M:%S',
    cache=True)

    chunk['Timestamp'] = ts + pd.to_timedelta(chunk['Milliseconds'], unit='ms')
    chunk['date'] = ts.dt.date.astype(str)
    chunk['am_pm'] = ts.dt.hour.lt(12).map({True: 'am', False: 'pm'})
    chunk['date_am_pm_id'] = chunk['date'] + '_' + chunk['am_pm']

    unique_half_days = chunk['date_am_pm_id'].unique()
    print(f"{'Half days in chunk:':<30} {(unique_half_days)}, chunk duration: {np.round(len(chunk)/(config.SAMPLING_RATE * 3600), 2)} hrs.")

    for half_day, df in chunk.groupby('date_am_pm_id', sort=False):
        
        file_name = os.path.join(file_dir, '{}_{}.csv'.format(individual, half_day))

        df.to_csv(
            file_name,
            mode='a',
            header=not os.path.exists(file_name),
            index=False
        )

        if verbose:
            print(f'Wrote {len(df)} rows → {file_name}')


def combine_acc_vectronics(individual, acc_filepaths, max_chunks=None, verbose=False):

    '''break the yearly csv files for each individual into chunks

    Arguments
    --------------
    individual: individual ID
    acc_filepaths: list of path-like object for the CSV files for the individual. 
                   The basename of files is the year data was collected in (example - [2022.csv, 2023.csv, 2024.csv])
    max_chunks: stop reading a csv after these many chunks
    
    '''

    # loop over csv files for each year for each individual

    for path in acc_filepaths:

        print(f"{'Handling the csv:':<30} {path}")

        file_dir = os.path.join(os.path.dirname(path), 'combined_acc')
        os.makedirs(file_dir, exist_ok=True)
        # for filename in os.listdir(file_dir):
        #     file_path = os.path.join(file_dir, filename)
        #     if os.path.isfile(file_path):
        #         os.remove(file_path)

        chunk_size = 10**6  # Adjust the chunk size based on your available memory

        # ---- compute total number of chunks for tqdm ----
        with open(path, "r") as f:
            n_rows = sum(1 for _ in f) - 1  # subtract header

        total_chunks = (n_rows + chunk_size - 1) // chunk_size
        if max_chunks is not None:
            total_chunks = min(total_chunks, max_chunks)

        num_chunks = 0
        year = os.path.basename(path).split('.')[0]

        reader = pd.read_csv(
            path,
            chunksize=chunk_size,
            skiprows=1  # only if file has no header
        )

        for chunk in tqdm(
            reader,
            total=total_chunks,
            desc=f"Reading {year}",
            unit="chunk"
        ):

            num_chunks += 1
            year = os.path.basename(path).split('.')[0]
            chunk['UTC Date[mm/dd/yyyy]'] = chunk['UTC Date[mm/dd]'] + '/' + year
            process_chunk_vectronics(chunk, individual, file_dir, verbose=verbose)
            del chunk

            if max_chunks is not None and num_chunks == max_chunks:
                break

        time.sleep(10)


def create_vectronics_halfday_segments(path_mappings, max_chunks=None, verbose=False):
    
    """
    create segments by reading accelerometer data in chunks. Saves the segments in a 
    directory titled "combined_acc" inside the data directory of each individual.
    
    Parameters:
    - path_mappings (dict): A dictionary where keys are individual names (str) and values are file paths (str) to their data directories.
    - max_chunks (int, optional): The maximum number of chunks to process per individual. Default is 0 (no limit).
    
    """
    
    individual_outputs = pd.DataFrame({'location': list(path_mappings.values()),
                            'id': list(path_mappings.keys())})

    individuals = individual_outputs['id'].values

    individual_acc_filepaths = [[os.path.join(individual_outputs.iloc[i]['location'], file) for file in os.listdir(individual_outputs.iloc[i]['location']) if file.endswith('csv')] for i in range(len(individual_outputs))]
        
    # Sample data for the loop
    data = zip(individuals, individual_acc_filepaths)
    
    for individual, acc_filepaths in data:
        print(f"{'Processing individual:':<30} {individual}")
        print(f"{'Files for this individual :':<30}", [os.path.basename(file) for file in acc_filepaths])
        combine_acc_vectronics(individual, acc_filepaths, max_chunks=max_chunks, verbose=verbose)
        print("")

def create_metadata(path_mappings, metadata_path):

    """
    Generates metadata from accelerometer data files for multiple individuals.

    Parameters:
    - path_mappings (dict): A dictionary where keys are individual names (str) and values are file paths (str) to their data directories.
    - metadata_path (str): The file path where the generated metadata CSV file will be saved.

    Metadata columns 
    -----------

    file path: string
        path-like object of where the half day segment is stored
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

    ## Read in your combined annotations

    data_locations = pd.DataFrame({'id': list(path_mappings.keys()),
                                'location': list(path_mappings.values()),
                                'combined_acc_location': [os.path.join(file, 'combined_acc') for file in list(path_mappings.values())],
                                'Outputs_location': [os.path.join(file, 'Outputs') for file in list(path_mappings.values())]}
                                )

    metadata = pd.DataFrame(columns = ['file path', 'individual ID', 'year', 'UTC Date [yyyy-mm-dd]', 'am/pm', 'half day [yyyy-mm-dd_am/pm]'])
    data_locations_existing = data_locations[data_locations['combined_acc_location'].apply(os.path.isdir)].reset_index(drop=True)

    # Step 3: Extract individuals and their corresponding filepaths
    individuals = data_locations_existing['id'].values
    individuals_acc_filepaths = [
        [
            os.path.join(dir_path, file)
            for file in os.listdir(dir_path)
            if file.endswith('csv')
        ]
        for dir_path in data_locations_existing['combined_acc_location']
    ]
    
    # Sample data for the loop
    data = zip(individuals, individuals_acc_filepaths)

    for individual, acc_filepaths in data:

        print('individual {} has {} halfdays.'.format(individual, len(acc_filepaths)))

        for file_path in tqdm(acc_filepaths):

            basename = os.path.basename(file_path).split('.')[0]
            date = basename.split('_')[1]
            year = date.split('-')[0]
            am_pm = basename.split('_')[2]
            half_day = date + '_' + am_pm
            # avg_temp = pd.read_csv(file_path, usecols=['Temperature [Celsius]'])['Temperature [Celsius]'].mean()

            metadata.loc[len(metadata)] = [file_path, individual, year, date, am_pm, half_day]

    metadata.to_csv(metadata_path, index=False)

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
    audio_annotations = pd.read_csv(audio_path, encoding='Windows-1252') # load audio annotations

    video_annotations['id'] = video_annotations['id'].replace(id_mapping)
    audio_annotations['Individual'] = audio_annotations['Individual'].replace(id_mapping)

    
    audio_annotations = audio_annotations.assign(Source='Audio')
    video_annotations = video_annotations.assign(Source='Video')

    annotations_columns = ['id', 'Behavior', 'Timestamp_start', 'Timestamp_end', 'Source', 'Confidence (H-M-L)', 'Eating intensity']
    
    rename_dict = {'Individual': 'id', 'Behaviour': 'Behavior', 'Timestamp_start_utc': 'Timestamp_start', 'Timestamp_end_utc': 'Timestamp_end'}
    audio_annotations = audio_annotations.rename(columns=rename_dict)

    # audio_annotations = audio_annotations[audio_annotations['Confidence (H-M-L)'].isin(['H', 'H/M'])]
    audio_annotations = audio_annotations[(audio_annotations['Behavior'] == 'Other') | (audio_annotations['Confidence (H-M-L)'].isin(['H', 'H/M']))]
    video_annotations['Confidence (H-M-L)'] = 'H'  # all video annotations are high confidence


    # --- VIDEO timestamps: dd/mm/yyyy
    botswana_timezone = 'Africa/Gaborone'
    video_annotations['Timestamp_start'] = pd.to_datetime(video_annotations['Timestamp_start'], format='%d/%m/%Y %H:%M:%S', errors='coerce') 
    video_annotations['Timestamp_end'] = pd.to_datetime(video_annotations['Timestamp_end'], format='%d/%m/%Y %H:%M:%S', errors='coerce')

    # localize to botswana time and the change clock time to utc
    video_annotations['Timestamp_start'] = (
        video_annotations['Timestamp_start']
        .dt.tz_localize(botswana_timezone)
        .dt.tz_convert('UTC')
        .dt.tz_localize(None)
    )

    video_annotations['Timestamp_end'] = (
        video_annotations['Timestamp_end']
        .dt.tz_localize(botswana_timezone)
        .dt.tz_convert('UTC')
        .dt.tz_localize(None)
    )

    # --- AUDIO timestamps: yyyy/mm/dd
    audio_annotations['Timestamp_start'] = pd.to_datetime(audio_annotations['Timestamp_start'], format='%Y/%m/%d %H:%M:%S', errors='coerce')
    audio_annotations['Timestamp_end'] = pd.to_datetime(audio_annotations['Timestamp_end'], format='%Y/%m/%d %H:%M:%S', errors='coerce')

    # Audio should already be UTC, but drop tz if present
    audio_annotations['Timestamp_start'] = audio_annotations['Timestamp_start'].dt.tz_localize(None)
    audio_annotations['Timestamp_end'] = audio_annotations['Timestamp_end'].dt.tz_localize(None)

    # Convert all timestamps to common output format
    video_annotations['Timestamp_start'] = video_annotations['Timestamp_start'].dt.strftime('%Y/%m/%d %H:%M:%S')
    video_annotations['Timestamp_end'] = video_annotations['Timestamp_end'].dt.strftime('%Y/%m/%d %H:%M:%S')

    audio_annotations['Timestamp_start'] = audio_annotations['Timestamp_start'].dt.strftime('%Y/%m/%d %H:%M:%S')
    audio_annotations['Timestamp_end'] = audio_annotations['Timestamp_end'].dt.strftime('%Y/%m/%d %H:%M:%S')


    all_annotations = pd.concat([video_annotations[annotations_columns], audio_annotations[annotations_columns]]).reset_index(drop=True)

    return all_annotations


def create_matched_data(filtered_metadata, annotations, verbose=True, min_window_for_padding=None, min_matched_duration=None):
    
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

    cols = ['individual ID', 'behavior', 'behavior_start', 'behavior_end', 'duration', 
            'year', 'UTC date [yyyy-mm-dd]', 'am/pm',  'half day [yyyy-mm-dd_am/pm]', 'avg temperature [C]', 
            'acc_x', 'acc_y', 'acc_z', 'Source', 'Confidence (H-M-L)', 'Eating intensity']
    
    acc_data = pd.DataFrame(columns=cols, index=[])
    acc_data_metadata = pd.DataFrame(columns=filtered_metadata.columns, index=[])
    acc_summary = pd.DataFrame(columns=['id', 'date_am_pm_id', 'annotations', 'acc', 'number of matched acc'], index=[])

    # Format and add helper columns to the annotations dataframe
    annotations['Timestamp_start'] = pd.to_datetime(annotations['Timestamp_start'], format='%Y/%m/%d %H:%M:%S')
    annotations['Timestamp_end'] = pd.to_datetime(annotations['Timestamp_end'], format='%Y/%m/%d %H:%M:%S')
    annotations['duration'] = (annotations['Timestamp_end'] - annotations['Timestamp_start']).dt.total_seconds()
    annotations['date'] = annotations['Timestamp_start'].dt.date
    annotations['am/pm'] = pd.to_datetime(annotations['Timestamp_start'], format="%Y/%m/%d %H:%M:%S").dt.strftime('%p').str.lower() 
    annotations['half day [yyyy-mm-dd_am/pm]'] = annotations['date'].astype(str) + '_' + annotations['am/pm']

    annotations_summary = annotations.copy()
    annotations_summary['match'] = 0
    annotations_summary['matched_duration'] = 0.0


    # loop over all unique individuals in filtered_metadata
    for (i, individual) in enumerate(filtered_metadata['individual ID'].unique()):

        ## find annotations for this individual
        annotations_orig = annotations[annotations['id'] == individual]
        individual_annotations = annotations_orig.copy()

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

                for row_idx, row in annotations_loop.iterrows():
                        
                    behaviour_start_time = row['Timestamp_start'].to_pydatetime().replace(tzinfo=timezone('UTC'))
                    behaviour_end_time = row['Timestamp_end'].to_pydatetime().replace(tzinfo=timezone('UTC'))

                    if not pd.isnull(behaviour_end_time):
                        acc_summary.loc[len(acc_summary)] = [individual, unique_period_loop, 0, 0.0, 0]
                        acc_summary.at[acc_summary.index[-1], 'annotations'] += (behaviour_end_time - behaviour_start_time).total_seconds() 
            
                    if (not pd.isnull(behaviour_end_time)) & (behaviour_end_time > behaviour_start_time):   

                        behaviour_acc = acc_loop[(acc_loop['Timestamp'] >= behaviour_start_time) &
                             (acc_loop['Timestamp'] <= behaviour_end_time)].sort_values('Timestamp')
                               
                        if len(behaviour_acc) > 0:
                            
                            # duration of the acceleration data that is matched with the behavior
                            matched_duration = (behaviour_acc.iloc[-1]['Timestamp'] - behaviour_acc.iloc[0]['Timestamp']).total_seconds()

                            if (min_matched_duration is not None) and (min_window_for_padding is not None) and (matched_duration < min_matched_duration) and (matched_duration >= min_window_for_padding):
                                padding = (min_matched_duration - matched_duration) / 2.0
                                padding = timedelta(seconds=padding)

                                # Expand start and end times symmetrically
                                start_time = behaviour_start_time - padding
                                end_time = behaviour_end_time + padding
                                acc_start, acc_end = acc_loop['Timestamp'].min(), acc_loop['Timestamp'].max()
                                start_time = max(start_time, acc_start)
                                end_time = min(end_time, acc_end)

                                behaviour_acc = acc_loop[(acc_loop['Timestamp'] >= start_time) &
                                     (acc_loop['Timestamp'] <= end_time)].sort_values('Timestamp')
                                
                                matched_duration = (behaviour_acc.iloc[-1]['Timestamp'] - behaviour_acc.iloc[0]['Timestamp']).total_seconds()

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
                                                            row['Source'],
                                                            row['Confidence (H-M-L)'],
                                                            row['Eating intensity']
                            ]

                            acc_data_metadata.loc[len(acc_data_metadata)] = individual_metadata.loc[individual_metadata['half day [yyyy-mm-dd_am/pm]'] == unique_period_loop].values[0].tolist()
                            annotations_summary.at[row_idx, 'match'] = 1
                            annotations_summary.at[row_idx, 'matched_duration'] = matched_duration
                            
            
            else:
                acc_summary.loc[len(acc_summary)] = [individual, unique_period_loop, 0, 0.0, 0]

    return acc_summary, acc_data, acc_data_metadata, annotations_summary

def give_balanced_weights(theta, y, n_classes_total):
    """
    Compute class weights for rebalancing, even if some classes are missing in y.
    theta: float between 0 and 1 for balancing uniform vs empirical distribution
    y: array of integer class labels (may have gaps)
    n_classes_total: total number of classes (including missing ones)
    """
    classes_present, class_counts = np.unique(y, return_counts=True)
    n_present = len(classes_present)

    empirical_weights = class_counts / len(y)
    uniform_weights = np.ones(n_present) / n_present
    combined_weights = theta * uniform_weights + (1 - theta) * empirical_weights
    class_to_weight = dict(zip(classes_present, combined_weights))

    # For missing classes, assign weight 0
    for cls in range(n_classes_total):
        if cls not in class_to_weight:
            class_to_weight[cls] = 0.0

    return class_to_weight, classes_present, class_counts

def setup_multilabel_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, args, n_outputs, transform=None, verbose=False):
    
    # --- Compute class weights for arbitrary labels ---
    class_to_weight, classes, class_counts = give_balanced_weights(args.theta, y_train, n_outputs)

    if verbose:
        print("Class weights:")
        for label in range(n_outputs):
            if label in classes:
                print(f"{label} -> {class_to_weight[label] / class_counts[np.where(classes == label)[0][0]]}")

    # Map y_train labels to weights
    y_weights = torch.tensor([class_to_weight[label] / class_counts[np.where(classes == label)[0][0]] for label in y_train],
                             dtype=torch.float32)
    sampler = WeightedRandomSampler(y_weights, len(y_weights))
    
    # --- One-hot encoding ---
    y_train_oh = np.eye(n_outputs)[y_train]
    y_val_oh   = np.eye(n_outputs)[y_val]
    y_test_oh  = np.eye(n_outputs)[y_test]
    
    # --- Create datasets ---
    train_dataset = datasets.NumpyDataset(X=X_train, y=y_train_oh, transform=transform)
    val_dataset   = datasets.NumpyDataset(X=X_val, y=y_val_oh, transform=transform)
    test_dataset  = datasets.NumpyDataset(X=X_test, y=y_test_oh, transform=transform)
    
    # --- DataLoaders ---
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    val_dataloader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
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

            acc_x_windows = repeat_or_truncate_list(row['acc_x'], fixed_length, reuse=reuse, min_length=min_duration*config.SAMPLING_RATE)
            acc_y_windows = repeat_or_truncate_list(row['acc_y'], fixed_length, reuse=reuse, min_length=min_duration*config.SAMPLING_RATE)
            acc_z_windows = repeat_or_truncate_list(row['acc_z'], fixed_length, reuse=reuse, min_length=min_duration*config.SAMPLING_RATE)

            assert len(acc_x_windows) == len(acc_y_windows) == len(acc_z_windows) 

            for x, y, z in zip(acc_x_windows, acc_y_windows, acc_z_windows):
                expanded_rows.append({'acc_x': x, 'acc_y': y, 'acc_z': z, 'behavior': row['behavior']})
                df_metadata.loc[len(df_metadata)] = row[['individual ID', 'year', 'UTC date [yyyy-mm-dd]', 'am/pm', 'half day [yyyy-mm-dd_am/pm]', 'avg temperature [C]', 'Source']].values

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

    max_steps = int(max_acc_duration*config.SAMPLING_RATE)
    X, y, z = create_padded_or_truncated_data(df_train, max_steps, padding=args.padding, reuse_behaviors=reuse_behaviors, min_duration=args.min_duration)
    X_test, y_test, z_test = create_padded_or_truncated_data(df_test, max_steps, padding=args.padding, reuse_behaviors=reuse_behaviors, min_duration=args.min_duration)
    print(f"Creating fixed-duration windows takes {time.time() - t2:3f} seconds.")

    print("")
    print("==================================")
    print(f"Time series duration window = {max_acc_duration}")

    # Band filter - no filter by default
    X = apply_band_pass_filter(X, args.cutoff_frequency, config.SAMPLING_RATE, btype=args.filter_type, N=args.cutoff_order, axis=2)
    X_test = apply_band_pass_filter(X_test, args.cutoff_frequency, config.SAMPLING_RATE, btype=args.filter_type, N=args.cutoff_order, axis=2)

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

    # create half-day segments of vectronics data by reading it in chunks
    create_vectronics_halfday_segments(config.VECTRONICS_PATHS, max_chunks=None)

    # create a metadata of the vectronics data
    create_metadata(config.VECTRONICS_PATHS, get_vectronics_metadata_path())
     
    # read the vectronics metadata and load annotations
    metadata = pd.read_csv(get_vectronics_metadata_path()) # load metadata
    all_annotations = combined_annotations(video_path=get_video_labels_path(), 
                                            audio_path=get_audio_labels_path(),
                                            id_mapping=config.id_mapping) # load annotations 

    # match annotations and vectronics data
    acc_summary, acc_data, acc_data_metadata, annotations_summary = create_matched_data(filtered_metadata=metadata, 
                                                                                        annotations=all_annotations, 
                                                                                        verbose=True, 
                                                                                        min_window_for_padding=None,
                                                                                        min_matched_duration=None)
         
    # save matched data, metadata, summaries                                                                              
    acc_summary.to_csv(get_vectronics_summary_path(), index=False)
    acc_data.to_csv(get_vectronics_data_path(), index=False)
    acc_data_metadata.to_csv(get_vectronics_acc_metadata_path(), index=False)
    annotations_summary.to_csv(get_vectronics_annotations_summary_path(), index=False)    