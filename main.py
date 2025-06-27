
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Import from our source directory
from src.config import (
    DATA_PATH, TRANSFORMER_MODEL_PATH, LSTM_MODEL_PATH, SUBMISSION_PATH,
    DEVICE, N_SPLITS, BATCH_SIZE_TRAIN, BATCH_SIZE_INFER,
    LSTM_PARAMS, TRANSFORMER_PARAMS
)
from src.preprocess import get_preprocessed_data
from src.model import VentilatorTransformer, VentilatorLSTM
from src.train import train_model_kfold
from src.evaluate import get_predictions

def run_inference(X_test_full, sample_submission_df, valid_pressures):
    """
    Runs the final inference using the weighted ensemble of trained models.
    """
    print("\n--- Starting Final Inference, Ensembling, and Submission ---")
    
    # DataLoader for Test Set
    test_dataset = TensorDataset(X_test_full)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_INFER, shuffle=False)

    transformer_predictions = []
    lstm_predictions = []

    # Predict with Transformer models
    print("\nPredicting with Transformer models...")
    for fold in range(1, N_SPLITS + 1):
        model_path = os.path.join(TRANSFORMER_MODEL_PATH, f"transformer_model_fold_{fold}.pth")
        if not os.path.exists(model_path): continue
        model = VentilatorTransformer(**TRANSFORMER_PARAMS).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        transformer_predictions.append(get_predictions(model, test_loader, DEVICE))
        print(f"Loaded and predicted with Transformer model from fold {fold}.")

    # Predict with Optimized LSTM models
    print("\nPredicting with Optimized LSTM models...")
    for fold in range(1, N_SPLITS + 1):
        model_path = os.path.join(LSTM_MODEL_PATH, f"lstm_model_fold_{fold}.pth")
        if not os.path.exists(model_path): continue
        model = VentilatorLSTM(**LSTM_PARAMS).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        lstm_predictions.append(get_predictions(model, test_loader, DEVICE))
        print(f"Loaded and predicted with Optimized LSTM model from fold {fold}.")

    # Create Submission File
    if lstm_predictions and transformer_predictions:
        # Weighted Average
        avg_lstm_preds = np.mean(np.stack(lstm_predictions, axis=0), axis=0)
        avg_transformer_preds = np.mean(np.stack(transformer_predictions, axis=0), axis=0)
        ensembled_preds = (0.8 * avg_lstm_preds) + (0.2 * avg_transformer_preds)
        
        # Post-Processing
        if valid_pressures is not None:
            pressure_midpoints = valid_pressures[:-1] + np.diff(valid_pressures) / 2
            flat_preds = ensembled_preds.flatten()
            insertion_indices = np.searchsorted(pressure_midpoints, flat_preds)
            final_preds_flat = valid_pressures[insertion_indices]
        else:
            final_preds_flat = ensembled_preds.flatten()

        # Save to CSV
        submission_df = pd.DataFrame({'id': sample_submission_df['id'], 'pressure': final_preds_flat})
        submission_path = os.path.join(SUBMISSION_PATH, "submission.csv")
        submission_df.to_csv(submission_path, index=False)
        print(f"\n--- Done! Submission file created at: {submission_path} ---")
    else:
        print("\nCould not generate submission. Trained model files not found.")


if __name__ == '__main__':
    # Step 1: Preprocess Data
    X_train, y_train, X_test, sample_submission = get_preprocessed_data(DATA_PATH)
    raw_train_df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    valid_pressures = np.sort(raw_train_df['pressure'].unique())
    
    # Step 2: Train Transformer Models (if they don't exist)
    if not os.path.exists(TRANSFORMER_MODEL_PATH) or not os.listdir(TRANSFORMER_MODEL_PATH):
        transformer_train_config = {'n_folds': N_SPLITS, 'batch_size': BATCH_SIZE_TRAIN, 'learning_rate': 1e-3, 'epochs': 15, 'device': DEVICE}
        train_model_kfold(VentilatorTransformer, TRANSFORMER_PARAMS, transformer_train_config, X_train, y_train, TRANSFORMER_MODEL_PATH, 'transformer')
    else:
        print("\n--- Found existing Transformer models. Skipping training. ---")
    
    # Step 3: Train Optimized LSTM Models (if they don't exist)
    if not os.path.exists(LSTM_MODEL_PATH) or not os.listdir(LSTM_MODEL_PATH):
        lstm_train_config = {'n_folds': N_SPLITS, 'batch_size': BATCH_SIZE_TRAIN, 'learning_rate': 1e-3, 'epochs': 25, 'device': DEVICE}
        train_model_kfold(VentilatorLSTM, LSTM_PARAMS, lstm_train_config, X_train, y_train, LSTM_SAVE_PATH, 'lstm')
    else:
        print("\n--- Found existing Optimized LSTM models. Skipping training. ---")

    # Step 4: Run Inference
    run_inference(X_test, sample_submission, valid_pressures)
