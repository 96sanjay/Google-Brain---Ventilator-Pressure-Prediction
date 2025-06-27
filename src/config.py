
import torch

# --- Core Paths ---
# You can change these paths if your data or models are in different locations.
DATA_PATH = "/home/christmas/machine_learning/ventilator_project/ventilator-pressure-prediction"
TRANSFORMER_MODEL_PATH = "/home/christmas/machine_learning/ventilator_project/ventilator-pressure-prediction/saved_models_transformer"
LSTM_MODEL_PATH = "/home/christmas/machine_learning/ventilator_project/ventilator-pressure-prediction/saved_models_lstm_final"
SUBMISSION_PATH = "." # Save submission.csv in the root project directory

# --- Training & Model Hyperparameters ---
# These settings should match the parameters used to train your saved models.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SPLITS = 5
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_INFER = 128

# Optimized LSTM Hyperparameters
LSTM_PARAMS = {
    "input_dim": 9,      # Corresponds to the number of features
    "lstm_dim": 384,
    "dense_dim": 128,
    "n_layers": 3,
    "dropout": 0.1
}

# Transformer Hyperparameters
TRANSFORMER_PARAMS = {
    "num_features": 9,
    "sequence_length": 80,
    "embed_dim": 64,
    "num_heads": 4,
    "ff_dim": 128
}
