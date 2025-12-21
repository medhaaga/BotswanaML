
import pytz

# don't change these
DATE_FORMAT = "%Y%m%d_%H%M%S"
TIMEZONE = pytz.utc
SAMPLING_RATE = 16
COLOR_LIST = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf"
]

# no need to change these. We use it to map a separate encoding of individual ID to the globally used individual IDs
id_mapping = {'2021_ninja_ash': 'ash', '2021_aqua_palus': 'palus', '2021_samurai_green': 'green', 
            '2022_aqua_fossey': 'fossey', '2022_ninja_birch': 'birch', '2022_roman_bishop': 'bishop',
            '2022_samurai_jessie': 'jessie', '2022_royal_rossignol': 'rossignol', 
            'Ash': 'ash', 'Palus': 'palus', 'Green': 'green', 
            'Fossey': 'fossey', 'Birch': 'birch', 'Bishop': 'bishop',
            'Jessie': 'jessie', 'Rossignol': 'rossignol'
            }

# map fine behavior classifications in annottaions file to coarser behavior classes
SUMMARY_COLLAPSE_BEHAVIORS_MAPPING = {'Lying (head up)': 'Stationary', 
                                    'Lying (head down)': 'Stationary',
                                    'Walking': 'Moving',
                                    'Trotting': 'Moving',
                                    'Running': 'Running',
                                    'Standing': 'Stationary',
                                    'Sitting':  'Stationary',
                                    'Marking (scent)': 'Other',
                                    'Interaction': 'Other',
                                    'Rolling': 'Other',
                                    'Scratching': 'Other',
                                    'Drinking': 'Other',
                                    'Dig': 'Other',
                                    'Capture?': 'Other',
                                    'Eating': 'Feeding',
                                    }


# behaviors of interest for classification 
SUMMARY_BEHAVIORS = ['Feeding', 'Moving', 'Other', 'Running', 'Stationary']
