import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # config/

RVC_PREPROCESSING_YAML = os.path.join(BASE_DIR, 'RVC_preprocessing.yaml')
VECTRONICS_PREPROCESSING_YAML = os.path.join(BASE_DIR, 'Vectronics_preprocessing.yaml')

HISTORIC_ACC_UNANNOTATED = '/mnt/ssd/medhaaga/wildlife/historic/unannotated'
HISTORIC_ACC_ANNOTATED = '/mnt/ssd/medhaaga/wildlife/historic/annotated'
HISTORIC_ACC_ANNOTATED_COMBINED = '/mnt/ssd/medhaaga/wildlife/historic/2025_12_05_acceleration_annotated.csv'


# dictionary with individual ID as key and path to the directory which stores that individual's acceleration CSV files as value.
AWD_VECTRONICS_PATHS = {
                        'jessie': "/mnt/ssd/medhaaga/wildlife/Vectronics/2022_44934_Samurai_Jessie",
                        'green': "/mnt/ssd/medhaaga/wildlife/Vectronics/2021_44915_Samurai_Green",
                        'palus': "/mnt/ssd/medhaaga/wildlife/Vectronics/2021_44910_Aqua_Palus",
                        'ash': "/mnt/ssd/medhaaga/wildlife/Vectronics/2021_44904_Ninja_Ash",
                        'fossey': '/mnt/ssd/medhaaga/wildlife/Vectronics/2022_44907_Aqua_Fossey'
                        }

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

