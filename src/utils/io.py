import os
import yaml
import os
import torch
import json
import shutil
import datetime
import numpy as np

def get_project_root() -> str:
    """Returns the root directory of the project."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_path(levels, main_dir):
    path = main_dir
    for item in levels:
        path = os.path.join(path, item + "/")
        if not os.path.exists(path):
            os.mkdir(path)
    return path

def get_data_path():
    data_path = os.path.join(get_project_root(), 'data')
    os.makedirs(data_path, exist_ok=True)
    return data_path

def get_matched_data_path():
    path = os.path.join(get_data_path(), 'matched_acc_data.csv')
    return path

def get_matched_metadata_path():
    path = os.path.join(get_data_path(), 'matched_acc_metadata.csv')
    return path

def get_matched_annotations_summary_path():
    path = os.path.join(get_data_path(), 'matched_annotations_summary.csv')
    return path     

def get_matched_summary_path():
    path = os.path.join(get_data_path(), 'matched_acc_summary.csv')
    return path

def get_metadata_path():
    path = os.path.join(get_data_path(), 'metadata.csv')
    return path

def get_RVC_metadata_path():
    path = os.path.join(get_data_path(), 'RVC_merged_metadata.xlsx')
    return path


def get_RVC_preprocessed_path():
    path = os.path.join(get_data_path(), 'RVC_preprocessed.csv')
    return path

def get_Vectronics_preprocessed_path(duration):
    path = os.path.join(get_data_path(), f'Vectronics_preprocessed_duration{duration}.csv')
    return path

def get_video_labels_path():
    data_path = os.path.join(get_project_root(), 'data')
    os.makedirs(data_path, exist_ok=True)
    path = os.path.join(data_path, 'video_labels.csv')
    return path

def get_audio_labels_path():
    path = os.path.join(get_data_path(), '2025_08_15_awd_audio_annotations.csv')
    return path

def get_RVC_historic_data_path():
    path = os.path.join(get_data_path(), 'RVC_historic_data.csv')
    return path

def get_results_dir():
    current_path = get_project_root()
    path = os.path.join(current_path, 'results')
    os.makedirs(path, exist_ok=True)
    return path

def get_results_path(exp_name, n_CNNlayers, n_channels, kernel_size, theta):
    results_dir = get_results_dir()
    os.makedirs(results_dir, exist_ok=True)
    levels = ['predictions', exp_name, 'conv_layers_'+str(n_CNNlayers), \
                 'n_channels_'+str(n_channels), 'kernel_size_'+str(kernel_size), \
                 'theta_'+str(theta)]
    
    return get_path(levels, results_dir)

def get_online_pred_path(halfday):
    results_dir = get_results_dir()
    levels = ['online_predictions', halfday]
    return get_path(levels, results_dir)

def get_figures_dir():
    current_path = get_project_root()
    path = os.path.join(current_path, 'figures')
    os.makedirs(path, exist_ok=True)
    return path

# utility functions
def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


## Test paths
def get_test_matched_data_path():
    data_path = get_path(['data', 'test'], get_project_root())
    os.makedirs(data_path, exist_ok=True)
    path = os.path.join(data_path, 'matched_acc_data.csv')
    return path

def get_test_matched_metadata_path():
    data_path = get_path(['data', 'test'], get_project_root())
    os.makedirs(data_path, exist_ok=True)
    path = os.path.join(data_path, 'matched_acc_metadata.csv')
    return path

def get_test_matched_summary_path():
    data_path = get_path(['data', 'test'], get_project_root())
    os.makedirs(data_path, exist_ok=True)
    path = os.path.join(data_path, 'matched_acc_summary.csv')
    return path

def get_test_metadata_path():
    data_path = os.path.join(get_project_root(), 'data')
    os.makedirs(data_path, exist_ok=True)
    path = os.path.join(data_path, 'metadata.csv')
    return path

def get_results_dir():
    current_path = get_project_root()
    path = os.path.join(current_path, 'results')
    os.makedirs(path, exist_ok=True)
    return path

def get_sightings_dir():
    results_dir = get_results_dir()
    sightings_dir = os.path.join(results_dir, 'sightings')
    os.makedirs(sightings_dir, exist_ok=True)
    return sightings_dir

def get_domain_adaptation_results_dir():
    path = os.path.join(get_results_dir(), 'domain_adaptation_training_results')
    os.makedirs(path, exist_ok=True)
    return path

def get_test_results_dir():
    path = get_path(['results', 'test'], get_project_root())
    os.makedirs(path, exist_ok=True)
    return path

def get_test_results_path(exp_name, n_CNNlayers, n_channels, kernel_size, theta, window_duration_percentile):
    results_dir = get_test_results_dir()
    os.makedirs(results_dir, exist_ok=True)
    levels = ['predictions', exp_name, 'conv_layers_'+str(n_CNNlayers), \
             'n_channels_'+str(n_channels), 'kernel_size_'+str(kernel_size), \
             'theta_'+str(theta), 'duration_'+str(window_duration_percentile)]
    return get_path(levels, results_dir)

def load_config(config_name, config_type):
    """Loads a YAML config file."""
    path = os.path.join('configs', config_type, f"{config_name}.yml")
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_results(results, model, configs, run_name=None):
    """Saves training results, model, and configs."""
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    results_dir = os.path.join('results', run_name)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Saving results to {results_dir}")

    # Save training statistics
    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results['training_stats'], f, indent=4)
        
    # Save predictions and true values
    for split in ['val', 'test']:
        np.save(os.path.join(results_dir, f'{split}_true.npy'), results[f'{split}_true'])
        np.save(os.path.join(results_dir, f'{split}_preds.npy'), results[f'{split}_preds'])
        np.save(os.path.join(results_dir, f'{split}_scores.npy'), results[f'{split}_scores'])

    # Save the best model
    torch.save(model.state_dict(), os.path.join(results_dir, 'best_model.pt'))
    
    # Copy the config files for perfect reproducibility
    for config_type, config_name in configs.items():
        shutil.copy(
            os.path.join('configs', config_type, f"{config_name}.yml"),
            os.path.join(results_dir, f"{config_type}_config.yml")
        )


if __name__ == '__main__':
    print(get_results_path('no_split', 5, 32, 5, 0.0, 50))