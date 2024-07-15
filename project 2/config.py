import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the data directory
DATA_DIR = os.path.join(BASE_DIR, 'Data')

# List of CSV files
CSV_FILES = [
    os.path.join(DATA_DIR, 'ASSAM-BARPETA.csv'),
    os.path.join(DATA_DIR, 'ASSAM-DHEMAJI.csv'),
    os.path.join(DATA_DIR, 'ASSAM-GOALPARA.csv'),
    os.path.join(DATA_DIR, 'ASSAM-LAKHIMPUR.csv'),
    os.path.join(DATA_DIR, 'UP-BALLIA.csv'),
    os.path.join(DATA_DIR, 'UP-BASTI.csv'),
    os.path.join(DATA_DIR, 'UP-GORAKHPUR.csv')
]

# Model parameters
RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'random_state': 42
}

# Train-test split ratio
TEST_SIZE = 0.3

# Random seed for reproducibility
RANDOM_STATE = 42
