import os
import nni
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model.model import MLP
from config_data_utils import create_data_loaders, device, calculate_metrics, save_predictions
import argparse
from torchkeras import summary

import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
import json

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Or suppress all warnings
warnings.filterwarnings("ignore")

# Your code here


# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def evaluate_model(model, data_loader, scaler=None):
    model.eval()
    predictions = []
    actuals = []
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X).squeeze()
            
            # Convert to numpy for scaling
            outputs = outputs.cpu().numpy()
            batch_y = batch_y.cpu().numpy()
            
            # Inverse transform predictions if scaler is provided
            if scaler:
                print("mean of outputs before scaling:", np.mean(outputs))
                outputs = scaler.inverse_transform(outputs.reshape(-1, 1)).ravel()
                print("mean of outputs after scaling:", np.mean(outputs))
                # Don't scale batch_y since we're using unscaled loader
            
            predictions.extend(outputs)
            actuals.extend(batch_y)
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    return calculate_metrics(actuals, predictions)

def train_model1(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=500, dataset_name='hcomb'):
    best_val_loss = float('inf')
    patience = 60
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # train_loss /= (len(train_loader)/train_loader.batch_size)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X).squeeze()
                val_loss += criterion(outputs, batch_y).item()
        
        val_loss /= len(val_loader)
        
        # Report intermediate result to NNI
        # nni.report_intermediate_result(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'/zhome/32/3/215274/jinlia/project/Adem/model_cache/{dataset_name}_best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Report every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
        if patience_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch+1}')
            print(f'Best validation loss: {best_val_loss:.6f}')
            break
    
    return best_val_loss

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
def rmse_loss(y_pred, y_true):
        mse = torch.mean((y_pred - y_true) ** 2)
        return torch.sqrt(mse)

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

def parse_args():
    parser = argparse.ArgumentParser(description='Train MLP model with specified dataset parameters')
    parser.add_argument('--dataset_name', type=str, default='Vc_processed', help='Name of the dataset')
    parser.add_argument('--y_name', type=str, default='Vc', help='Target variable column name')
    parser.add_argument('--scale_y', type=bool, default=True, help='Scale the target variable')
    parser.add_argument('--batch_size', type=int, default=3200, help='Batch size')
    parser.add_argument('--train_label', type=str, default='label', help='Column name for train/val/test split')
    parser.add_argument('--smiles', type=str, default='SMILES', help='SMILES column name')
    parser.add_argument('--other_columns', nargs='+', default=['No'], help='Additional columns to remove')
    parser.add_argument('--save', action='store_true', help='Save predictions to file if specified')
    parser.add_argument('--save_best_model', default=True, help='Save the best model')
    parser.add_argument('--config_json', type=str, default='Vc_params.json', help='Path to the config json file')
    return parser.parse_args()


def get_cpu_model():
    with open('/proc/cpuinfo') as f:
        for line in f:
            if "model name" in line:
                return line.split(":")[1].strip()

def bootstrap_training(model, data_dict, args, device, params):
    """Implements bootstrap training for uncertainty estimation with initialization from reference model"""
    
    # Get loaders from data_dict
    train_loader = data_dict['scaled_loaders']['train']
    val_loader = data_dict['scaled_loaders']['val']
    test_loader = data_dict['scaled_loaders']['test']
    train_loader_unscaled = data_dict['unscaled_loaders']['train']
    val_loader_unscaled = data_dict['unscaled_loaders']['val']
    test_loader_unscaled = data_dict['unscaled_loaders']['test']
    scaler = data_dict['scaler']
    
    # Create directory for this dataset's bootstrap results (with fromRef suffix)
    base_dir = '/zhome/32/3/215274/jinlia/project/Adem'
    results_dir = os.path.join(base_dir, 'bootstrap_results_fromRef', args.dataset_name)
    models_dir = f'{results_dir}/models'
    predictions_dir = f'{results_dir}/predictions'
    
    # Create directories if they don't exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Get reference model state
    reference_state = model.state_dict()
    
    # Get predictions from reference model with proper scaling
    model.eval()
    train_preds = []
    train_targets = []
    
    print("Getting predictions from reference model...")
    with torch.no_grad():
        for batch_X, batch_y_unscaled in train_loader_unscaled:  # Use unscaled loader directly
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
    
    # Initialize storage for bootstrap models and their predictions
    n_boots = 100  # Number of bootstrap samples
    bootstrap_models = []
    
    # Get all metric names from calculate_metrics
    sample_metrics = evaluate_model(model, train_loader, scaler)
    metric_names = list(sample_metrics.keys())
    
    # Initialize bootstrap_metrics with all available metrics
    bootstrap_metrics = {
        'train': {metric: [] for metric in metric_names},
        'val': {metric: [] for metric in metric_names},
        'test': {metric: [] for metric in metric_names}
    }
    
    print(f"Starting bootstrap training with {n_boots} samples...")
    
    # Create synthetic datasets and train bootstrap models
    for i in range(n_boots):
        print(f"\nBootstrap iteration {i+1}/{n_boots}")
        
        # Resample residuals and create synthetic targets
        resampled_residuals = np.random.choice(train_residuals, size=len(train_residuals), replace=True)
        synthetic_targets = train_preds + resampled_residuals
        
        # Scale the synthetic targets if using scaled training
        if scaler:
            synthetic_targets = scaler.transform(synthetic_targets.reshape(-1, 1)).ravel()
        
        # Modified part: Get features from CustomDataset
        train_features = []
        for batch_X, _ in train_loader:
            train_features.append(batch_X)
        train_features = torch.cat(train_features, dim=0)
        
        # Create new data loader with synthetic targets
        synthetic_dataset = torch.utils.data.TensorDataset(
            train_features,
            torch.FloatTensor(synthetic_targets)
        )
        synthetic_loader = torch.utils.data.DataLoader(
            synthetic_dataset, 
            batch_size=train_loader.batch_size,
            shuffle=True
        )
        
        # Create new model and initialize it with reference model weights
        bootstrap_model = MLP(parse_init_json(params), input_size=data_dict['input_size']).to(device)
        bootstrap_model.load_state_dict(reference_state)  # Initialize from reference model
        
        optimizer = optim.Adam(bootstrap_model.parameters(), lr=params['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=10, verbose=True
        )
        
        # Train the bootstrap model
        train_model1(
            bootstrap_model, 
            synthetic_loader, 
            val_loader, 
            rmse_loss, 
            optimizer, 
            scheduler,
            epochs=1000,
            dataset_name=f"{args.dataset_name}_bootstrap_{i}"
        )
        
        # Evaluate model on all datasets
        bootstrap_models.append(bootstrap_model)
        
        # Evaluate on each dataset
        for name, loader in [
            ("train", train_loader_unscaled),
            ("val", val_loader_unscaled),
            ("test", test_loader_unscaled)
        ]:
            metrics = evaluate_model(bootstrap_model, loader, scaler)
            for metric_name, value in metrics.items():
                bootstrap_metrics[name][metric_name].append(value)
            
        # Print progress with all metrics
        print(f"Bootstrap model {i+1} Metrics:")
        for name in ['train', 'val', 'test']:
            metrics_str = ", ".join([f"{metric}={bootstrap_metrics[name][metric][-1]:.4f}" 
                                   for metric in metric_names])
            print(f"{name.capitalize()}: {metrics_str}")
        
        # After training and evaluating metrics:
        if args.save:
            # Save model
            model_path = os.path.join(models_dir, f'model_{i}.pth')
            torch.save(bootstrap_model.state_dict(), model_path)
            
            # Save predictions
            save_predictions(
                data_dict['original_df'],
                bootstrap_model,
                device,
                dataset_name=os.path.join(predictions_dir, f'bootstrap_{i}'),
                y_name=args.y_name,
                train_label=args.train_label,
                smiles=args.smiles,
                other_columns=args.other_columns,
                scaler=scaler
            )
    
    # Calculate and print bootstrap statistics for all metrics
    for dataset in ['train', 'val', 'test']:
        print(f"\n{dataset.capitalize()} Bootstrap Statistics:")
        for metric_name in metric_names:
            metrics_array = np.array(bootstrap_metrics[dataset][metric_name])
            mean = np.mean(metrics_array)
            std = np.std(metrics_array)
            ci_lower = np.percentile(metrics_array, 2.5)
            ci_upper = np.percentile(metrics_array, 97.5)
            
            print(f"{metric_name.upper()}:")
            print(f"  Mean: {mean:.4f}")
            print(f"  Std: {std:.4f}")
            print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Plot bootstrap distributions for all metrics
    fig, axes = plt.subplots(len(metric_names), 1, figsize=(10, 6*len(metric_names)))
    if len(metric_names) == 1:
        axes = [axes]
    
    for idx, metric_name in enumerate(metric_names):
        for dataset in ['train', 'val', 'test']:
            sns.kdeplot(data=bootstrap_metrics[dataset][metric_name], 
                       label=dataset, ax=axes[idx])
        axes[idx].set_xlabel(metric_name.upper())
        axes[idx].set_ylabel('Density')
        axes[idx].set_title(f'Bootstrap {metric_name.upper()} Distributions')
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'bootstrap_metrics_dist.png'))
    plt.close()
    
    # Get predictions and residuals for plotting
    def get_predictions_and_residuals(model, loader_unscaled):  # Only need unscaled loader
        preds = []
        targets = []
        with torch.no_grad():
            for batch_X, batch_y_unscaled in loader_unscaled:
                batch_X = batch_X.to(device)
                outputs = model(batch_X).squeeze()
                
                # Scale back predictions
                outputs_np = outputs.cpu().numpy()
                if scaler:
                    outputs_np = scaler.inverse_transform(outputs_np.reshape(-1, 1)).ravel()
                
                preds.extend(outputs_np)
                targets.extend(batch_y_unscaled.numpy())
        
        preds = np.array(preds)
        targets = np.array(targets)
        return preds, targets, targets - preds

    # Get residuals for all datasets using unscaled loaders
    _, train_targets, train_res = get_predictions_and_residuals(
        model, data_dict['unscaled_loaders']['train'])
    _, val_targets, val_res = get_predictions_and_residuals(
        model, data_dict['unscaled_loaders']['val'])
    _, test_targets, test_res = get_predictions_and_residuals(
        model, data_dict['unscaled_loaders']['test'])

    # Create residuals plot
    res_plot_data = pd.DataFrame()
    res_plot_data['bin'] = (len(train_res) * ['train'] + 
                           len(val_res) * ['val'] + 
                           len(test_res) * ['test'])
    res_plot_data['residuals'] = np.concatenate((train_res.flatten(), 
                                                val_res.flatten(), 
                                                test_res.flatten()))

    # Plot residuals distribution
    f, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(ax=ax, data=res_plot_data[res_plot_data['bin'] == 'train'], 
                x="residuals", label='train', fill=True)
    sns.kdeplot(ax=ax, data=res_plot_data[res_plot_data['bin'] == 'val'], 
                x="residuals", label='val', fill=True)
    sns.kdeplot(ax=ax, data=res_plot_data[res_plot_data['bin'] == 'test'], 
                x="residuals", label='test', fill=True)
    plt.title('Distribution of Residuals')
    plt.xlabel('Residuals (Actual - Predicted)')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'residuals_distribution.png'))
    plt.close()
    
    # After collecting all bootstrap results, add:
    print("\nCreating bootstrap statistics plots...")
    plot_bootstrap_statistics(bootstrap_metrics, results_dir)
    
    print("Performing moment analysis...")
    moment_analysis_plot(bootstrap_metrics, results_dir)
    
    return bootstrap_models, bootstrap_metrics

def plot_bootstrap_statistics(bootstrap_metrics, results_dir):
    """Plot bootstrap statistics for each metric and dataset"""
    metrics = list(bootstrap_metrics['train'].keys())
    datasets = ['train', 'val', 'test']
    
    # Create box plots for all metrics
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 5*len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        data = [bootstrap_metrics[dataset][metric] for dataset in datasets]
        axes[idx].boxplot(data, labels=datasets)
        axes[idx].set_title(f'Bootstrap Distribution of {metric.upper()}')
        axes[idx].set_ylabel(metric.upper())
        axes[idx].grid(True)
        
        # Add individual points for better visualization
        for i, d in enumerate(data, 1):
            axes[idx].scatter([i] * len(d), d, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'bootstrap_statistics_boxplot.png'))
    plt.close()

def moment_analysis_plot(bootstrap_metrics, results_dir):
    """Perform and plot moment analysis on test MAE"""
    test_mae = np.array(bootstrap_metrics['test']['mae'])
    
    def ms_calculator(input_array):
        p = 4  # 1st, 2nd, 3rd and 4th moments
        input_array = input_array.flatten()
        R_r = np.zeros((len(input_array), p))
        R_l = np.zeros((len(input_array), p))

        for i in range(p):
            x = np.abs(input_array) ** (i + 1)
            S = np.cumsum(x)
            M_r = np.maximum.accumulate(x)
            M_l = np.minimum.accumulate(x)
            R_r[:, i] = M_r / S
            R_l[:, i] = M_l / S

        return {'r': R_r, 'l': R_l}

    results = ms_calculator(test_mae)
    x = np.arange(1, len(test_mae) + 1)

    # Plot minimum ratios
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(4):
        ax.semilogx(x, results['l'][:, i], label=f'Moment {i+1}')
    ax.set_xlim(left=1)
    ax.set_title('Minimum Ratios for Different Moments')
    ax.set_xlabel('Number of Models')
    ax.set_ylabel('Ratio')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'moment_analysis_min.png'))
    plt.close()

    # Plot maximum ratios
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(4):
        ax.semilogx(x, results['r'][:, i], label=f'Moment {i+1}')
    ax.set_xlim(left=1)
    ax.set_title('Maximum Ratios for Different Moments')
    ax.set_xlabel('Number of Models')
    ax.set_ylabel('Ratio')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'moment_analysis_max.png'))
    plt.close()

def main():


    cpu_model = get_cpu_model()
    print(f"CPU Model: {cpu_model}")

    args = parse_args()
    
    # Get parameters from NNI if available, otherwise use defaults
    params = nni.get_next_parameter()
    is_nni = True if params else False
    if is_nni:
        print("Using NNI parameters:", params)
    else:
        params = get_default_params(args)
        print("Using default parameters:", params)
    
    # Parse the parameters into MLP structure
    mlp_params = parse_init_json(params)
    print("Parsed MLP parameters:", mlp_params)
    
    # Create data loaders with the parsed arguments
    data_dict = create_data_loaders(
        dataset_name=args.dataset_name,
        y_name=args.y_name,
        train_label=args.train_label,
        smiles=args.smiles,
        other_columns=args.other_columns,
        batch_size=args.batch_size,
        scale_y= args.scale_y
    )
    
    # Get both scaled and unscaled loaders
    train_loader = data_dict['scaled_loaders']['train']
    val_loader = data_dict['scaled_loaders']['val']
    test_loader = data_dict['scaled_loaders']['test']
    input_size = data_dict['input_size']
    scaler = data_dict['scaler']
    
    # Create model with parsed parameters
    model = MLP(mlp_params, input_size=input_size).to(device)
    # summary in cpu
    summary(model, input_shape=(1, input_size), input_dtype=torch.FloatTensor)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    criterion = rmse_loss
    # criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=10, verbose=True
    )
    
    # Train the model using scaled data
    best_val_loss = train_model1(
        model, train_loader, val_loader, criterion, 
        optimizer, scheduler, epochs=1000, dataset_name=args.dataset_name
    )
    
    print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
    
    # Load best model
    model.load_state_dict(torch.load(f'/zhome/32/3/215274/jinlia/project/Adem/model_cache/{args.dataset_name}_best_model.pth'
                                     )) 
                          
    
    print("trained with scaled data where scale_y is", args.scale_y)
    # Evaluate on unscaled data
    print("\nEvaluating on unscaled data:")
    for name, loader in [
        ("Train", data_dict['unscaled_loaders']['train']),
        ("Validation", data_dict['unscaled_loaders']['val']),
        ("Test", data_dict['unscaled_loaders']['test']),
        ("All", data_dict['unscaled_loaders']['all'])
    ]:
        # take the rmse on unscaled validation set as the final result  
        

        metrics = evaluate_model(model, loader, scaler)
        print(f"\nMLP {name} on dataset {args.dataset_name} Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.6f}")
        if name == "Validation":
            final_result = metrics['rmse']
            print("report final result:", final_result)
            if is_nni:
                nni.report_final_result(final_result)
           
    
    # Save predictions if requested
    if args.save:
        save_predictions(
            data_dict['original_df'],
            model,
            device,
            dataset_name=args.dataset_name,
            y_name=args.y_name,
            train_label=args.train_label,
            smiles=args.smiles,
            other_columns=args.other_columns,
            scaler=scaler
        )

    # Add save best model functionality
    if args.save_best_model:
        results_dir = f'data/results/{args.dataset_name}_bootstrap_fromRef'
        os.makedirs(results_dir, exist_ok=True)
        torch.save(model.state_dict(), f'/zhome/32/3/215274/jinlia/project/Adem/data/results/{args.dataset_name}_bootstrap_fromRef/{args.dataset_name}_best_model.pth')

    # Before starting bootstrap analysis, store data_dict

    
    print("\nStarting bootstrap analysis...")
    bootstrap_models, bootstrap_metrics = bootstrap_training(
        model, 
        data_dict,
        args,
        device,
        params
    )
    
    # Save bootstrap results
    if args.save:
        results_dir = f'data/results/{args.dataset_name}_bootstrap_fromRef'
        os.makedirs(results_dir, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for dataset in bootstrap_metrics:
            serializable_metrics[dataset] = {}
            for metric in bootstrap_metrics[dataset]:
                serializable_metrics[dataset][metric] = [float(x) for x in bootstrap_metrics[dataset][metric]]
        
        bootstrap_results = {
            'metrics': serializable_metrics,
            'config': params,
            'dataset': args.dataset_name,
            'statistics': {
                dataset: {
                    metric: {
                        'mean': float(np.mean(bootstrap_metrics[dataset][metric])),
                        'std': float(np.std(bootstrap_metrics[dataset][metric])),
                        'ci_lower': float(np.percentile(bootstrap_metrics[dataset][metric], 2.5)),
                        'ci_upper': float(np.percentile(bootstrap_metrics[dataset][metric], 97.5))
                    }
                    for metric in bootstrap_metrics[dataset]
                }
                for dataset in bootstrap_metrics
            }
        }
        
        with open(os.path.join(results_dir, 'bootstrap_results.json'), 'w') as f:
            json.dump(bootstrap_results, f, indent=2)

    # Plot bootstrap statistics
    plot_bootstrap_statistics(bootstrap_metrics, results_dir)
    moment_analysis_plot(bootstrap_metrics, results_dir)

if __name__ == "__main__":
    main() 