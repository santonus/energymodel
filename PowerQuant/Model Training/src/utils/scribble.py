import os
import sys

import dotenv
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

dotenv.load_dotenv()
project_root = os.getenv('PROJECT_ROOT')
sys.path.append(project_root)

lst_selected_features = [
    'Architecture',
    'avg_comp_lat',
    'avg_glob_lat',
    # 'avg_shar_lat',
    'glob_inst_kernel',
    'glob_load_sm',
    'glob_store_sm',
    'misc_inst_kernel',
    'inst_issue_cycles',
    'cache_penalty',
    'Avg',
]


def load_data(data_path: str = 'data/combined_df.csv') -> pd.DataFrame:
    """Load the preprocessed GPU kernel performance data."""
    project_root = os.getenv('PROJECT_ROOT')
    full_path = os.path.join(project_root, data_path)
    df = pd.read_csv(full_path)
    print(f'Data loaded successfully. Shape: {df.shape}')
    print(f'Columns: {list(df.columns)}')
    return df


def prepare_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare features and target variable from the dataset."""
    # Set target variable
    df = df[lst_selected_features]
    y = np.array(df['Avg'])

    le = LabelEncoder()
    ind = le.fit_transform(df['Architecture'])

    # Remove target and Architecture columns from features
    X = np.array(df.drop(columns=['Avg', 'Architecture']))

    print(f'Features shape: {X.shape}')
    print(f'Target shape: {y.shape}')

    return X, y, ind


df = load_data()
X, y, ind = prepare_data(df)
print(X.shape)
print(y.shape)
print(ind.shape)
