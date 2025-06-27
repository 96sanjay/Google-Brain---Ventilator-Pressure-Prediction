
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

def train_model_kfold(model_class, model_params, train_config, X_train_full, y_train_full, model_save_path, model_name_prefix):
    """
    Trains a given model using k-fold cross-validation.
    
    Args:
        model_class: The model class to instantiate (e.g., VentilatorLSTM).
        model_params (dict): Parameters to initialize the model class.
        train_config (dict): Configuration for training (epochs, lr, etc.).
        X_train_full (Tensor): The full training feature tensor.
        y_train_full (Tensor): The full training target tensor.
        model_save_path (str): Directory to save the trained models.
        model_name_prefix (str): Prefix for the saved model files (e.g., 'lstm').
    """
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    kf = KFold(n_splits=train_config['n_folds'], shuffle=True, random_state=42)
    all_fold_scores = []
    
    print(f"\n--- Starting {model_name_prefix.upper()} 5-Fold Training ---")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
        print(f"\n===== Fold {fold+1}/{train_config['n_folds']} =====")
        start_time = time.time()
        
        # Data splitting
        X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
        y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]
        
        # DataLoaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'])
        
        # Model, Loss, Optimizer
        model = model_class(**model_params).to(train_config['device'])
        loss_fn = nn.L1Loss()
        optimizer = optim.AdamW(model.parameters(), lr=train_config['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
        
        best_val_loss = float('inf')
        
        # Training loop
        for epoch in range(train_config['epochs']):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(train_config['device']), y_batch.to(train_config['device'])
                
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for X_batch_val, y_batch_val in val_loader:
                    y_pred_val = model(X_batch_val.to(train_config['device']))
                    total_val_loss += loss_fn(y_pred_val, y_batch_val.to(train_config['device'])).item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
        
        # Save the final model for this fold
        model_path = os.path.join(model_save_path, f"{model_name_prefix}_model_fold_{fold+1}.pth")
        torch.save(model.state_dict(), model_path)
        
        print(f"Fold {fold+1} Best MAE: {best_val_loss:.4f} | Time: {time.time() - start_time:.2f}s")
        all_fold_scores.append(best_val_loss)

    print(f"\n--- {model_name_prefix.upper()} Training Complete ---")
    print(f"Average MAE over {train_config['n_folds']} folds: {np.mean(all_fold_scores):.4f}")

