
import pytz
import pandas as pd

# don't change these
DATE_FORMAT = "%Y%m%d_%H%M%S"
TIMEZONE = pytz.utc
SAMPLING_RATE = 16

# no need to change these. We use it to map a separate encoding of individual ID to the globally used individual IDs
id_mapping = {'2021_ninja_ash': 'ash', '2021_aqua_palus': 'palus', '2021_samurai_green': 'green', 
            '2022_aqua_fossey': 'fossey', '2022_ninja_birch': 'birch', '2022_roman_bishop': 'bishop',
            '2022_samurai_jessie': 'jessie', '2022_royal_rossignol': 'rossignol', 'Jessie': 'jessie',
            'Fossey': 'fossey', 'Palus': 'palus', 'Ash': 'ash', 'Green': 'green',
            'Birch': 'birch', 'Bishop': 'bishop', 'Rossignol': 'rossignol'}

# map fine behavior classifications in annottaions file to coarser behavior classes
RAW_COLLAPSE_BEHAVIORS_MAPPING = {'Lying (head up)': 'Vigilant', 
                                    'Lying (head down)': 'Resting',
                                    'Walking': 'Moving',
                                    'Trotting': 'Moving',
                                    'Running': 'Running',
                                    'Standing': 'Vigilant',
                                    'Sitting':  'Vigilant',
                                    'Marking (scent)': 'Marking',
                                    'Interaction': 'Other',
                                    'Rolling': 'Marking',
                                    'Scratching': 'Other',
                                    'Drinking': 'Other',
                                    'Dig': 'Other',
                                    'Capture?': 'Other',
                                    'Eating': 'Feeding',
                                    }


# no need to change these. We use it to map a separate encoding of individual ID to the globally used individual IDs
id_mapping = {'2021_ninja_ash': 'ash', '2021_aqua_palus': 'palus', '2021_samurai_green': 'green', 
            '2022_aqua_fossey': 'fossey', '2022_ninja_birch': 'birch', '2022_roman_bishop': 'bishop',
            '2022_samurai_jessie': 'jessie', '2022_royal_rossignol': 'rossignol', 'Jessie': 'jessie'}

# map fine behavior classifications in annottaions file to coarser behavior classes
SUMMARY_COLLAPSE_BEHAVIORS_MAPPING = {'Lying (head up)': 'Stationary', 
                                    'Lying (head down)': 'Stationary',
                                    'Walking': 'Moving',
                                    'Trotting': 'Moving',
                                    'Running': 'Running',
                                    'Standing': 'Stationary',
                                    'Sitting':  'Stationary',
                                    'Marking (scent)': 'Marking',
                                    'Interaction': 'Other',
                                    'Rolling': 'Marking',
                                    'Scratching': 'Other',
                                    'Drinking': 'Other',
                                    'Dig': 'Other',
                                    'Capture?': 'Other',
                                    'Eating': 'Feeding',
                                    }


# behaviors of interest for classification 
RAW_BEHAVIORS = ['Feeding', 'Moving', 'Resting', 'Running', 'Vigilant']
SUMMARY_BEHAVIORS = ['Feeding', 'Moving', 'Running', 'Stationary']
