import pandas as pd

def load_and_preprocess(file_path):
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()
    print(f"Columns in {file_path}: {data.columns.tolist()}")
    data['Dates'] = pd.to_datetime(data['Dates'], format='%d-%b-%y')

    if 'ACTUAL(mm)' in data.columns and 'NORMAL(mm)' in data.columns:
        data['Flood'] = (data['ACTUAL(mm)'] > data['NORMAL(mm)']).astype(int)
    else:
        raise KeyError(f"Required columns are missing in {file_path}")

    return data

def calculate_flood_percentage(data):
    flood_percentage = (data['Flood'].sum() / len(data)) * 100
    return flood_percentage
