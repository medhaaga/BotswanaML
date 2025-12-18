import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # config/

RVC_PREPROCESSING_YAML = os.path.join(BASE_DIR, 'RVC_preprocessing.yaml')
VECTRONICS_PREPROCESSING_YAML = os.path.join(BASE_DIR, 'Vectronics_preprocessing.yaml')

# RVC annotated data
RVC_ACC_ANNOTATED = '/mnt/ssd/medhaaga/wildlife/historic/2025_12_05_acceleration_annotated.csv'

# dictionary with individual ID as key and path to the directory which stores that individual's acceleration CSV files as value.
VECTRONICS_PATHS = {
                        'jessie': "/mnt/ssd/medhaaga/wildlife/Vectronics/2022_44934_Samurai_Jessie",
                        'green': "/mnt/ssd/medhaaga/wildlife/Vectronics/2021_44915_Samurai_Green",
                        'palus': "/mnt/ssd/medhaaga/wildlife/Vectronics/2021_44910_Aqua_Palus",
                        'ash': "/mnt/ssd/medhaaga/wildlife/Vectronics/2021_44904_Ninja_Ash",
                        'fossey': '/mnt/ssd/medhaaga/wildlife/Vectronics/2022_44907_Aqua_Fossey'
                        }

