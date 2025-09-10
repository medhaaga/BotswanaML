import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # config/

RVC_PREPROCESSING_YAML = os.path.join(BASE_DIR, 'RVC_preprocessing.yaml')
VECTRONICS_PREPROCESSING_YAML = os.path.join(BASE_DIR, 'Vectronics_preprocessing.yaml')

HISTORIC_ACC = '/mnt/ssd/medhaaga/wildlife/historic'

# dictionary with individual ID as key and path to the directory which stores that individual's acceleration CSV files as value.
AWD_VECTRONICS_PATHS = {
                        'jessie': "/mnt/ssd/medhaaga/wildlife/Vectronics/2022_44934_Samurai_Jessie",
                        'green': "/mnt/ssd/medhaaga/wildlife/Vectronics/2021_44915_Samurai_Green",
                        'palus': "/mnt/ssd/medhaaga/wildlife/Vectronics/2021_44910_Aqua_Palus",
                        'ash': "/mnt/ssd/medhaaga/wildlife/Vectronics/2021_44904_Ninja_Ash",
                        'fossey': '/mnt/ssd/medhaaga/wildlife/Vectronics/2022_44907_Aqua_Fossey'
                        }

# path to metadata and behavior annotations file. 
# If two different sources of annotations exist, like audio and video, provide these paths here.
VECTRONICS_METADATA_PATH = "/mnt/ssd/medhaaga/wildlife/Vectronics/metadata.csv"
VECTRONICS_VIDEO_ANNOTATIONS_PATH = "/mnt/ssd/medhaaga/wildlife/Vectronics/annotations_combined.csv"
VECTRONICS_AUDIO_ANNOTATIONS_PATH = "/mnt/ssd/medhaaga/wildlife/Vectronics/silver_labels_annotations.csv"
VECTRONICS_ANNOTATIONS_PATH = "/mnt/ssd/medhaaga/wildlife/Vectronics/all_annotations.csv"

# RVC HEADER PATHS
RVC_HEADER_FILES_PATH = '/home/medhaaga/BotswanaML/data/RVC_header_files/'
RVC_METADATA_PATH = '/home/medhaaga/BotswanaML/data/RVC_metadata.xlsx'
RVC_MERGED_METADATA_PATH = '/home/medhaaga/BotswanaML/data/RVC_merged_metadata.xlsx'

# dictionary with individual ID as key and path to the directory which stores that individual's acceleration CSV files as value.
AWD_VECTRONICS_PATHS = {'jessie': "/mnt/ssd/medhaaga/wildlife/Vectronics/2022_44934_Samurai_Jessie",
 'green': "/mnt/ssd/medhaaga/wildlife/Vectronics/2021_44915_Samurai_Green",
 'palus': "/mnt/ssd/medhaaga/wildlife/Vectronics/2021_44910_Aqua_Palus",
 'ash': "/mnt/ssd/medhaaga/wildlife/Vectronics/2021_44904_Ninja_Ash",
 'fossey': '/mnt/ssd/medhaaga/wildlife/Vectronics/2022_44907_Aqua_Fossey'}

VECTRONICS_BEHAVIOR_EVAL_PATH = "/mnt/ssd/medhaaga/wildlife/Vectronics/behavior_evaluations"
RVC_BEHAVIOR_EVAL_PATH = "/mnt/ssd/medhaaga/wildlife/historic/behavior_evaluations"
VECTRONICS_SUMMARY_BEHAVIOR_EVAL_PATH = "/mnt/ssd/medhaaga/wildlife/Vectronics/behavior_evaluations_summary"

