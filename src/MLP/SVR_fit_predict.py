import os
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from config_data_utils import create_data_loaders, calculate_metrics, save_predictions
import argparse
import torch

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def evaluate_model(model, data_loader, scaler_y=None):
    predictions = []
    actuals = []
    
    for batch_X, batch_y in data_loader:
        batch_X = batch_X.numpy()
        batch_y = batch_y.numpy()
        
        outputs = model.predict(batch_X)
        if scaler_y:
            print("mean of outputs before scaling:", np.mean(outputs))
            print("Scaling outputs")
            outputs = scaler_y.inverse_transform(outputs.reshape(-1, 1)).ravel()
            print("mean of outputs after scaling:", np.mean(outputs))
            # batch_y = scaler_y.inverse_transform(batch_y.reshape(-1, 1)).ravel()
            
        predictions.extend(outputs)
        actuals.extend(batch_y)
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    return calculate_metrics(actuals, predictions)

def get_default_params():
    """Default parameters for SVR"""
    return {
        'C': 50.0,
        'epsilon': 0.1,
        'kernel': 'rbf',
        'gamma': 'scale'
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Train MLP model with specified dataset parameters')
    parser.add_argument('--dataset_name', type=str, default='Omega_processed', help='Name of the dataset')
    parser.add_argument('--y_name', type=str, default='Omega', help='Target variable column name')
    parser.add_argument('--train_label', type=str, default='label', help='Column name for train/val/test split')
    parser.add_argument('--smiles', type=str, default='SMILES', help='SMILES column name')
    parser.add_argument('--other_columns', nargs='+', default=['No'], help='Additional columns to remove')
    parser.add_argument('--save', action='store_true', help='Save predictions to file if specified')
    return parser.parse_args()

def main():
    args = parse_args()
    params = get_default_params()
    print("Using SVR parameters:", params)
    
    # Create data loaders
    data_dict = create_data_loaders(
        dataset_name=args.dataset_name,
        y_name=args.y_name,
        train_label=args.train_label,
        smiles=args.smiles,
        other_columns=args.other_columns,
        batch_size=3000,
        scale_y=1
    )
    
    # Get both scaled and unscaled loaders
    train_loader = data_dict['scaled_loaders']['train']
    val_loader = data_dict['unscaled_loaders']['val']  # Use unscaled for evaluation
    test_loader = data_dict['unscaled_loaders']['test']  # Use unscaled for evaluation
    scaler = data_dict['scaler']
    
    # Get training data (scaled)
    train_data = []
    train_labels = []
    for batch_X, batch_y in train_loader:
        train_data.append(batch_X.numpy())
        train_labels.append(batch_y.numpy())
    
    X_train = np.concatenate(train_data)
    y_train = np.concatenate(train_labels)

    # Train SVR model with scaled data
    model = SVR(**params)
    print("Training SVR model...")
    model.fit(X_train, y_train)
    print("Training completed.")
    

    
    # Modify evaluate_model to use the new prediction function
    for name, loader in [
        ("Train", data_dict['unscaled_loaders']['train']),  # Use unscaled for consistent metrics
        ("Validation", val_loader),
        ("Test", test_loader),
        ("All", data_dict['unscaled_loaders']['all'])
    ]:
        metrics = evaluate_model(model, loader,scaler)
        print(f"\nSVR {name} Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.6f}")
    
    # Save predictions if requested
    if args.save:
        save_predictions(
            data_dict['original_df'],
            model,
            dataset_name=args.dataset_name,
            y_name=args.y_name,
            train_label=args.train_label,
            smiles=args.smiles,
            other_columns=args.other_columns
        )

if __name__ == "__main__":
    main() 