
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import torch

def calculate_metrics(y_true, y_pred):
    """
    Calculates and returns a dictionary of regression metrics.
    """
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }
    return metrics

def get_predictions(model, loader, device):
    """
    Gets model predictions for a given data loader.
    """
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for X_batch_test in loader:
            batch_preds = model(X_batch_test[0].to(device))
            all_predictions.append(batch_preds.detach().cpu().numpy())
    
    return np.concatenate(all_predictions, axis=0)


def plot_analysis(y_true_flat, y_pred_flat, model_name="Model"):
    """
    Generates and displays analysis plots for model predictions.
    
    Args:
        y_true_flat (np.array): Flattened array of true target values.
        y_pred_flat (np.array): Flattened array of predicted values.
        model_name (str): Name of the model for plot titles.
    """
    print(f"\n--- Generating Analysis Plots for {model_name} ---")
    
    # 1. Prediction vs. Actuals Plot
    plt.figure(figsize=(10, 10))
    plt.scatter(y_true_flat, y_pred_flat, alpha=0.1, label="Predictions")
    plt.plot([y_true_flat.min(), y_true_flat.max()], [y_true_flat.min(), y_true_flat.max()], 
             '--r', linewidth=2, label="Perfect Prediction")
    plt.title(f"Prediction vs. Actual Values ({model_name} - Validation Set)")
    plt.xlabel("Actual Pressure")
    plt.ylabel("Predicted Pressure")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 2. Error Distribution Plot
    errors = y_pred_flat - y_true_flat
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, edgecolor='k')
    plt.title(f"Distribution of Prediction Errors ({model_name})")
    plt.xlabel("Prediction Error (Predicted - Actual)")
    plt.ylabel("Frequency")
    plt.axvline(x=0, color='r', linestyle='--', label="Zero Error")
    plt.grid(True)
    plt.legend()
    plt.show()

