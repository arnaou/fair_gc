import os
import nni
import torch
import numpy as np
from model.model import MLP
from config_data_utils import create_data_loaders, device
import argparse
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def get_default_params(args):
    """Default parameters when not using NNI, matching the search space structure"""
    import json
    import os
    
    config_path = os.path.join(os.path.dirname(__file__), f'config/{args.config_json}')
    
    try:
        with open(config_path, 'r') as f:
            print("using config file:", config_path)
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found. Using hardcoded defaults.")
        return {
            "layer0": {
                "_name": "Dense",
                "size": 256,
                "activation": "leaky_relu"
            },
            "layer1": {
                "_name": "Dense",
                "size": 96,
                "activation": "relu"
            },
            "layer2": {
                "_name": "Dense",
                "size": 96,
                "activation": "leaky_relu"
            },
            "layer3_size": 56,
            "layer3_dropout": 0.001689586605154624,
            "layer3_activation": "leaky_relu",
            "learning_rate": 0.0009080167558708478
        }

def parse_init_json(data):
    """Parse the tree-structured hyperparameters into MLP configuration"""
    params = {
        'layer_sizes': [],
        'dropouts': [],
        'activations': [],
        'learning_rate': data['learning_rate']
    }
    
    # Process first three layers (optional layers without dropout)
    for i in range(3):  # for layers 0-2
        layer_key = f'layer{i}'
        layer_config = data[layer_key]
        
        if layer_config['_name'] == 'Empty':
            continue
        elif layer_config['_name'] == 'Dense':
            params['layer_sizes'].append(int(layer_config['size']))
            params['dropouts'].append(0.0)  # No dropout for layers 0-2
            params['activations'].append(layer_config['activation'])
    
    # Add layer3 (mandatory layer) using flat parameters
    params['layer_sizes'].append(int(data['layer3_size']))
    params['dropouts'].append(data['layer3_dropout'])
    params['activations'].append(data['layer3_activation'])
    
    params['num_layers'] = len(params['layer_sizes'])
    return params

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze synthetic data statistics')
    parser.add_argument('--dataset_name', type=str, default='Omega_processed', help='Name of the dataset')
    parser.add_argument('--y_name', type=str, default='Omega', help='Target variable column name')
    parser.add_argument('--scale_y', type=bool, default=True, help='Scale the target variable')
    parser.add_argument('--batch_size', type=int, default=3200, help='Batch size')
    parser.add_argument('--train_label', type=str, default='label', help='Column name for train/val/test split')
    parser.add_argument('--smiles', type=str, default='SMILES', help='SMILES column name')
    parser.add_argument('--other_columns', nargs='+', default=['No'], help='Additional columns to remove')
    parser.add_argument('--config_json', type=str, default='Omega_params.json', help='Path to the config json file')
    return parser.parse_args()

def analyze_synthetic_data(model, data_dict, args, device):
    """Analyze statistics of synthetic datasets"""
    
    # Get loaders from data_dict
    train_loader = data_dict['scaled_loaders']['train']
    train_loader_unscaled = data_dict['unscaled_loaders']['train']
    scaler = data_dict['scaler']
    
    # Get predictions from reference model with proper scaling
    model.eval()
    train_preds = []
    train_targets = []
    
    print("\nGetting predictions from reference model...")
    with torch.no_grad():
        for batch_X, batch_y_unscaled in train_loader_unscaled:
            batch_X = batch_X.to(device)
            outputs = model(batch_X).squeeze()
            
            # Scale back predictions
            outputs_np = outputs.cpu().numpy()
            if scaler:
                outputs_np = scaler.inverse_transform(outputs_np.reshape(-1, 1)).ravel()
            
            train_preds.extend(outputs_np)
            train_targets.extend(batch_y_unscaled.numpy())
    
    train_preds = np.array(train_preds)
    train_targets = np.array(train_targets)
    
    # Calculate residuals using unscaled values
    train_residuals = train_targets - train_preds
    
    print("\nOriginal Data Statistics:")
    print(f"Targets Mean: {np.mean(train_targets):.4f}, Std: {np.std(train_targets):.4f}")
    print(f"Predictions Mean: {np.mean(train_preds):.4f}, Std: {np.std(train_preds):.4f}")
    print(f"Residuals Mean: {np.mean(train_residuals):.4f}, Std: {np.std(train_residuals):.4f}")
    
    # Initialize storage for synthetic data statistics
    n_boots = 100  # Number of synthetic datasets
    synthetic_stats = []
    
    print(f"\nGenerating {n_boots} synthetic datasets and analyzing their statistics...")
    
    # Create synthetic datasets and analyze their statistics
    for i in range(n_boots):
        # Resample residuals and create synthetic targets
        resampled_residuals = np.random.choice(train_residuals, size=len(train_residuals), replace=True)
        synthetic_targets = train_preds + resampled_residuals
        
        # Record unscaled statistics
        unscaled_mean = np.mean(synthetic_targets)
        unscaled_std = np.std(synthetic_targets)
        
        # Scale the synthetic targets
        if scaler:
            synthetic_targets_scaled = scaler.transform(synthetic_targets.reshape(-1, 1)).ravel()
            scaled_mean = np.mean(synthetic_targets_scaled)
            scaled_std = np.std(synthetic_targets_scaled)
        else:
            scaled_mean = unscaled_mean
            scaled_std = unscaled_std
        
        synthetic_stats.append({
            'iteration': i+1,
            'unscaled_mean': unscaled_mean,
            'unscaled_std': unscaled_std,
            'scaled_mean': scaled_mean,
            'scaled_std': scaled_std
        })
        
        print(f"\nSynthetic Dataset {i+1}:")
        print(f"Unscaled - Mean: {unscaled_mean:.4f}, Std: {unscaled_std:.4f}")
        print(f"Scaled   - Mean: {scaled_mean:.4f}, Std: {scaled_std:.4f}")
    
    # Calculate summary statistics
    unscaled_means = [stat['unscaled_mean'] for stat in synthetic_stats]
    unscaled_stds = [stat['unscaled_std'] for stat in synthetic_stats]
    scaled_means = [stat['scaled_mean'] for stat in synthetic_stats]
    scaled_stds = [stat['scaled_std'] for stat in synthetic_stats]
    
    print("\nSummary of Synthetic Datasets:")
    print("\nUnscaled Statistics:")
    print(f"Mean of means: {np.mean(unscaled_means):.4f} ± {np.std(unscaled_means):.4f}")
    print(f"Mean of stds:  {np.mean(unscaled_stds):.4f} ± {np.std(unscaled_stds):.4f}")
    
    print("\nScaled Statistics:")
    print(f"Mean of means: {np.mean(scaled_means):.4f} ± {np.std(scaled_means):.4f}")
    print(f"Mean of stds:  {np.mean(scaled_stds):.4f} ± {np.std(scaled_stds):.4f}")

def main():
    args = parse_args()
    
    # Get parameters from config file
    params = get_default_params(args)
    print("Using parameters from config:", params)
    
    # Parse the parameters into MLP structure
    mlp_params = parse_init_json(params)
    print("Parsed MLP parameters:", mlp_params)
    
    # Create data loaders
    data_dict = create_data_loaders(
        dataset_name=args.dataset_name,
        y_name=args.y_name,
        train_label=args.train_label,
        smiles=args.smiles,
        other_columns=args.other_columns,
        batch_size=args.batch_size,
        scale_y=args.scale_y
    )
    
    # Load the trained model
    model_path = f'/zhome/32/3/215274/jinlia/project/Adem/model_cache/{args.dataset_name}_best_model.pth'
    print(f"\nLoading model from {model_path}")
    
    # Create model instance (you'll need to use the same architecture as the trained model)
    model = MLP(mlp_params, input_size=data_dict['input_size']).to(device)
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Analyze synthetic data
    analyze_synthetic_data(model, data_dict, args, device)

if __name__ == "__main__":
    print("Starting synthetic data analysis...")
    main()
    print("Analysis completed.") 