
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset

def load_data(data_path):
    """
    Loads train, test, and sample submission files from the specified path.
    """
    try:
        train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
        test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))
        sample_submission_df = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'))
        return train_df, test_df, sample_submission_df
    except FileNotFoundError as e:
        print(f"Error: Data file not found. {e}")
        raise

def create_features(df):
    """
    Creates time-series based features for the ventilator dataset.
    """
    df = df.copy()
    df['u_in_lag'] = df.groupby('breath_id')['u_in'].shift(1).fillna(0)
    df['u_in_diff'] = df['u_in'] - df['u_in_lag']
    df['u_in_cumsum'] = df.groupby('breath_id')['u_in'].cumsum()
    df['time_step_diff'] = df.groupby('breath_id')['time_step'].diff().fillna(0)
    df['area'] = df['time_step_diff'] * df['u_in']
    return df

def prepare_tensors(train_df, test_df):
    """
    Scales data and converts dataframes into sequence-based PyTorch tensors.
    """
    features = ['R', 'C', 'time_step', 'u_in', 'u_out', 
                'u_in_lag', 'u_in_diff', 'u_in_cumsum', 'area']
    target = 'pressure'

    # Scaling
    scaler = StandardScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    test_df[features] = scaler.transform(test_df[features])

    # Tensor Creation
    def df_to_tensors(df, is_train=True):
        sequences = [g[features].values for _, g in df.groupby('breath_id')]
        X_full = np.array(sequences)
        X_tensor = torch.tensor(X_full, dtype=torch.float32)

        if is_train:
            y_sequences = [g[target].values for _, g in df.groupby('breath_id')]
            y_full = np.array(y_sequences)
            y_tensor = torch.tensor(y_full, dtype=torch.float32).unsqueeze(-1)
            return X_tensor, y_tensor
        return X_tensor

    X_train, y_train = df_to_tensors(train_df, is_train=True)
    X_test = df_to_tensors(test_df, is_train=False)

    return X_train, y_train, X_test

def get_preprocessed_data(data_path):
    """
    Main function to run the full preprocessing pipeline.
    """
    print("--- Starting Data Preprocessing ---")
    
    # Load
    train_df, test_df, sample_submission_df = load_data(data_path)
    print("Data loaded successfully.")
    
    # Feature Engineering
    train_processed = create_features(train_df)
    test_processed = create_features(test_df)
    print("Feature engineering complete.")
    
    # Scaling and Tensor Conversion
    X_train, y_train, X_test = prepare_tensors(train_processed, test_processed)
    print("Data scaling and tensor preparation complete.")
    print(f"Final training data shape: {X_train.shape}")
    print(f"Final test data shape: {X_test.shape}")
    
    return X_train, y_train, X_test, sample_submission_df

