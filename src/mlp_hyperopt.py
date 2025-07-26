##########################################################################################################
#                                                                                                        #
#    Collection of helper function and classes for performing GC-DNN hyperparameter optimization       #
#                                                                                                        #
#                                                                                                        #
#                                                                                                        #
#                                                                                                        #
#    Authors: Adem R.N. Aouichaoui                                                                       #
#    2025/04/23                                                                                          #
#                                                                                                        #
##########################################################################################################

##########################################################################################################
# Import packages and modules
##########################################################################################################
import optuna
import torch
import argparse
from typing import Dict, Any, Tuple
import yaml
import importlib
import os
import joblib
from datetime import datetime
from src.ml_utils import create_pipeline
import json
import numpy as np
import  random
import optunahub
from src.ml_utils import create_model
from sqlalchemy.exc import OperationalError
import time
from src.ml_hyperopt import RetryingStorage, load_config, create_sampler, get_class_from_path
from lightning import seed_everything
import torch.nn.functional as F
from src.models.mlp import build_mlp
import pickle


##########################################################################################################
# Define functions and classes
##########################################################################################################

def mlp_hypopt_parse_arguments():
    """
    function for parsing the command line arguments
    :return:
    """
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for GNN models')
    parser.add_argument('--property', type=str, default='Omega', required=True, help='Tag for the property')
    parser.add_argument('--config_file', type=str, required=True, default='afp_hyperopt_config.yaml',help='Path to the YAML configuration file')
    parser.add_argument('--split_type', type=str, required=False, default='butina_min', help='type of split to be used')
    parser.add_argument('--model', type=str, required=True, default='mlp', help='Model type to optimize (must be defined in config file)')
    parser.add_argument('--metric', type=str, required=False, default='rmse', help='Scoring metric to use (must be defined in config file)')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of optimization trials (uses config default if not specified)' )
    parser.add_argument('--n_jobs', type=int, default=2, help='Number of cores used (uses max if not configured)')
    parser.add_argument('--sampler', type=str, default='auto', help='Sampler to use (uses config default if not specified)')
    parser.add_argument('--path_2_data', type=str, default='data/', required=False, help='Path to the data file')
    parser.add_argument('--path_2_result', type=str, default = 'results/', required=False, help='Path to save the results (metrics and predictions)')
    parser.add_argument('--path_2_model', type=str, required=False, default='models/', help='Path to save the model and eventual check points')
    parser.add_argument('--study_name', type=str, default=None, help='Name of the study for persistence')
    parser.add_argument('--storage', type=str, default=None, help='Database URL for study storage (e.g., sqlite:///optuna.db)')
    parser.add_argument('--no_load_if_exists', action='store_false', dest='load_if_exists', help='Do not load existing study if it exists')
    parser.add_argument('--seed', type=int, default=42, help='Random state for reproducibility')

    return parser.parse_args()

################################
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

################################
def suggest_mlp_parameter(trial: optuna.Trial, name: str, param_config: Dict[str, Any]) -> Any:
    """Suggest a parameter value based on its configuration."""
    param_type = param_config['type']
    if param_type == 'int':
        return trial.suggest_int(
            name,
            param_config['low'],
            param_config['high'],
            step=param_config.get('step', 1)
        )
    elif param_type == 'float':
        return trial.suggest_float(
            name,
            param_config['low'],
            param_config['high'],
            log=param_config.get('log', False)
        )
    else:
        raise ValueError(f"Unsupported parameter type: {param_type}")


################################
def mlp_hyperparameter_optimizer(
        config_path: str,
        model_name: str,
        property_name: str,
        train_loader,
        val_loader,
        study_name: str = None,
        sampler_name: str = None,
        metric_name: str = None,
        storage: str = None,
        n_trials: int = None,
        load_if_exists: bool = True,
        n_jobs: int = -1,
        seed: int = None,
        device: str = None
) -> Tuple[optuna.study.Study, torch.nn.Module, Dict[str, Any], Dict[str, Any]]:
    """
    Generic GNN model optimization using Optuna.

    Args:
        config_path: Path to YAML configuration file
        model_name: Name of the model in config to optimize
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        study_name: Name of the study
        storage: Storage URL for the study
        n_trials: Number of optimization trials
        load_if_exists: Whether to load existing study
        n_jobs: Number of parallel jobs
        seed: Random seed
        device: Device to run on
    """

    # Load configuration
    config = load_config(config_path)
    defaults = config['default_settings']
    model_config = config['models'][model_name]
    param_ranges = model_config['param_ranges']
    train_range = config['training_params']
    # Set random seed if provided
    if seed is not None:
        seed_everything(seed)

    # Setup scoring function
    metric_name = metric_name or defaults['scoring']
    metric_config = config['metric'][metric_name]
    metric_func = get_class_from_path(metric_config['function'])
    direction = metric_config.get('direction', defaults['direction'])

    # Setup sampler
    sampler_name = sampler_name or defaults['sampler']
    sampler_config = config['sampler'][sampler_name]
    sampler = create_sampler(sampler_config, seed=seed)

    # Get number of trials
    n_trials = n_trials or defaults['n_trials']

    # set study name
    study_name = study_name or property_name + '_' + model_name

    # set storage name
    storage = storage or 'sqlite:///optuna_dbs/optuna_' + property_name + '_' + model_name + '.db'
    storage = RetryingStorage(storage)

    def objective(trial: optuna.Trial) -> float:
        # Get model class
        model_class = get_class_from_path(model_config['class'])
        # Build model parameters
        # Initialize model parameters
        model_params = {}
        training_params = {}

        # Keep track of the previous best trial's files
        if not hasattr(objective, 'best_trial_files'):
            objective.best_trial_files = None

         # Add fixed parameters
        model_params.update(model_config.get('fixed_params', {}))

        # Handle MLP structure
        mlp_config = model_config['param_ranges'].get('mlp_config', {})
        if mlp_config:
            # Get number of layers
            n_layers = suggest_mlp_parameter(trial, 'mlp_n_layers', mlp_config['n_layers'])
            dims = []

            # Generate dimensions for each layer
            for i in range(n_layers):
                dim = suggest_mlp_parameter(
                    trial,
                   f'mlp_dim_{i:02d}',  # Use padding to ensure unique names
                    mlp_config['dim_per_layer']
                )
                dims.append(dim)
            model_params['n_hidden'] = dims

        # Add regular model parameters
        for param_name, param_config in param_ranges.items():
            if param_name == 'final_dropout':
                model_params[param_name] = suggest_mlp_parameter(trial, param_name, param_config)

        # Handle training parameters separately
        for param_name, param_config in train_range.items():
            training_params[param_name] = suggest_mlp_parameter(trial, f'train_{param_name}', param_config)

        # Create model and optimizer
        print("Creating model with parameters:", model_params)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model_class(**model_params).to(device)
        optimizer = torch.optim.Adam( model.parameters(), lr=training_params.get('learning_rate', 0.001))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, mode='min',
                                                                factor=training_params.get('lr_reduce', 0.7), patience=5, min_lr=1e-6)

        best_val_loss = float('inf')
        best_state_dict = None
        patience = 30
        patience_counter = 0

        for epoch in range(defaults['max_epochs']):
            # Training
            model.train()
            total_loss = 0
            for batch in train_loader:
                x = batch[0].to(device)
                y = batch[1].to(device)
                optimizer.zero_grad()
                pred = model(x)
                loss = F.mse_loss(pred, y.view(-1, 1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(device)
                    y = batch[1].to(device)
                    pred = model(x)
                    loss = F.mse_loss(pred, y.view(-1, 1))
                    val_loss += loss.item()

            val_loss = val_loss / len(val_loader)
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Convert state dict tensors to lists for serialization
                # Save state dict to a separate file using trial number
                save_path = os.path.join('checkpoints', f'mlp_{property_name}_trial_{trial.number}_best_state.pt')
                torch.save({
                    'state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'epoch': epoch
                }, save_path)

                # Store only the file path in trial user attributes
                trial.set_user_attr('best_state_dict_path', save_path)
                trial.set_user_attr('training_params', training_params)
                patience_counter = 0

            else:
                patience_counter += 1

            if patience_counter >= patience:
                break


            trial.report(val_loss, epoch)


        # Clean up
        del model, optimizer
        torch.cuda.empty_cache()

        # redo the predictions to get it in the right scale and based on the right metric

        return best_val_loss

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=load_if_exists,
        direction=defaults['direction'],
        sampler=sampler,
        )

    # Calculate remaining trials
    existing_trials = len(study.trials)
    if n_trials is not None:
        remaining_trials = n_trials - existing_trials
        n_trials = max(0, remaining_trials)

    print(f"Continuing from existing study with {existing_trials} trials")
    if n_trials > 0:
        print(f"Will run {n_trials} trials")
    else:
        print(f"the desired number of iterations have already been done, consider increasing n_trials")

    # run optimization
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

     # Create best model
    model_class = get_class_from_path(model_config['class'])
    best_params = study.best_params

    # Reconstruct best parameters
    best_model_params = {}
    best_model_params.update(model_config.get('fixed_params', {}))


    # Add optimized parameters
    # Add parameters from param_ranges (standard parameters only)
    for param_name in param_ranges:
        if param_name in best_params:
            if param_name == 'final_dropout':
                best_model_params[param_name] = best_params[param_name]

    # Handle MLP dimensions separately using mlp_config
    if 'mlp_config' in model_config['param_ranges']:  # Use the same path as in objective function
        n_layers = best_params['mlp_n_layers']
        dims = []
        for i in range(n_layers):
            dim_key = f'mlp_dim_{i:02d}'
            if dim_key in best_params:
                dims.append(best_params[dim_key])
        best_model_params['n_hidden'] = dims  # Use the same parameter name as in objective

    # Create final model
    best_model = model_class(**best_model_params).to(device)

    # Get the best trial and load its state dict
    best_trial = study.best_trial
    training_params = best_trial.user_attrs.get('training_params', {})

    # When creating the final model:
    best_trial = study.best_trial
    if 'best_state_dict_path' in best_trial.user_attrs:
        checkpoint = torch.load(best_trial.user_attrs['best_state_dict_path'])
        best_model.load_state_dict(checkpoint['state_dict'])

    return study, best_model, training_params, model_config



def save_mlp_model_package(
        trial_dir: str,
        model_dir: str,
        model: torch.nn.Module,
        model_hyperparameters: Dict[str, Any],
        training_params: Dict[str, Any],
        scaler: Any,
        model_config: Dict[str, Any],
        study: Any = None,
        metric_name: str = "metric",
        additional_info: Dict[str, Any] = None
) -> None:
    """
    Save MLP model and associated data to disk.

    Args:
        trial_dir: Directory to save trial information
        model_dir: Directory to save model files
        model: The trained MLP model
        model_hyperparameters: Model hyperparameters dictionary
        training_params: Training parameters dictionary
        scaler: Data scaler object
        model_config: Model configuration dictionary
        study: Optuna study object (optional)
        metric_name: Name of the optimization metric
        additional_info: Additional information to include in the config
    """
    # Create directories if they don't exist
    os.makedirs(trial_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Save model state in model directory
    model_path = os.path.join(model_dir, "model.pt")
    torch.save(model.state_dict(), model_path)

    # Save scaler in model directory
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Print debug information
    print("Model hyperparameters before cleaning:", model_hyperparameters)

    # Clean model hyperparameters - remove construction-only parameters
    clean_hyperparameters = {}

    # Add fixed parameters
    clean_hyperparameters.update(model_config.get('fixed_params', {}))

    # Reconstruct n_hidden from the best parameters
    if 'mlp_n_layers' in model_hyperparameters:
        n_layers = model_hyperparameters['mlp_n_layers']
        dims = []
        for i in range(n_layers):
            dim_key = f'mlp_dim_{i:02d}'
            if dim_key in model_hyperparameters:
                dims.append(model_hyperparameters[dim_key])
        clean_hyperparameters['n_hidden'] = dims

    # Add remaining valid parameters
    for k, v in model_hyperparameters.items():
        # Skip construction-only parameters, MLP dimensions, and training parameters
        if (k not in ['mlp_n_layers'] and
                not k.startswith('mlp_dim_') and
                not k.startswith('train_')):
            clean_hyperparameters[k] = v

    # Verify all required parameters are present based on your MLP model requirements
    required_params = ['n_in', 'n_out', 'n_hidden']
    missing_params = [param for param in required_params if param not in clean_hyperparameters]
    if missing_params:
        raise ValueError(f"Missing required parameters: {missing_params}")

    # Prepare configuration dictionary
    config = {
        'model_hyperparameters': clean_hyperparameters,
        'training_params': training_params,
        'model_class': model.__class__.__name__,
        'model_module': model.__class__.__module__
    }

    if study is not None:
        config['optimization'] = {
            'study_best_params': study.best_params,
            'study_best_value': float(study.best_value),
            'n_trials': len(study.trials),
            'direction': study.direction.name
        }
        # Save trials as DataFrame
        study_trials = study.trials_dataframe()
        trials_path = os.path.join(trial_dir, "trials.csv")
        study_trials.to_csv(trials_path, index=False)

    if additional_info:
        config.update(additional_info)

    # Save configurations
    trial_config_path = os.path.join(trial_dir, "results.json")
    with open(trial_config_path, 'w') as f:
        json.dump(config, f, indent=4, cls=NumpyEncoder)

    model_config_path = os.path.join(model_dir, "model_config.json")
    model_specific_config = {
        'model_class': config['model_class'],
        'model_module': config['model_module'],
        'model_hyperparameters': clean_hyperparameters,
        'training_params': config['training_params']
    }
    with open(model_config_path, 'w') as f:
        json.dump(model_specific_config, f, indent=4, cls=NumpyEncoder)


def load_mlp_model_package(
        model_dir: str,
        device: str = "cuda"
) -> Dict[str, Any]:
    """
    Load saved MLP model package from model directory.

    Args:
        model_dir: Directory containing saved model
        device: Device to load model to

    Returns:
        Dictionary containing loaded model and components
    """
    # Load configuration
    config_path = os.path.join(model_dir, "model_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Import model class
    #module_path = config['model_module']
    model_name = 'mlp'# config['model_class']
    #module = __import__(module_path, fromlist=[model_name])
    #model_class = getattr(build_mlp)
    #model_class = build_mlp
    # Create model instance with the hyperparameters
    model_params = config['model_hyperparameters']
    model = build_mlp(**model_params)

    # Load model state
    model_path = os.path.join(model_dir, "model.pt")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model state: {e}")
        print("Attempting to load with weights_only=True...")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    model = model.to(device)

    # Load scaler
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return {
        'model': model,
        'config': config,
        'scaler': scaler
    }


def save_best_mlp_model(
        study: Any,
        output_dir: str,
        property_name: str,
        model_class: Any,
        input_dim: int,
        output_dim: int,
        scaler: Any,
        seed: int = 42,
        device: str = None
) -> None:
    """
    Save the best MLP model from an Optuna study.

    Args:
        study: Completed Optuna study
        output_dir: Directory to save model
        property_name: Name of the property being predicted
        model_class: Class of the MLP model
        input_dim: Input dimension
        output_dim: Output dimension
        scaler: Data scaler object
        seed: Random seed
        device: Device to use
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{property_name}_mlp_{timestamp}"
    trial_dir = os.path.join(output_dir, "trials", model_name)
    model_dir = os.path.join(output_dir, "models", model_name)

    # Get best parameters
    best_params = study.best_params

    # Reconstruct model parameters
    model_params = {
        'n_in': input_dim,
        'n_out': output_dim,
    }

    # Add n_hidden from trial parameters
    if 'mlp_n_layers' in best_params:
        n_layers = best_params['mlp_n_layers']
        dims = []
        for i in range(n_layers):
            dim_key = f'mlp_dim_{i:02d}'
            if dim_key in best_params:
                dims.append(best_params[dim_key])
        model_params['n_hidden'] = dims

    # Add dropout parameter if it exists
    if 'final_dropout' in best_params:
        model_params['dropout'] = best_params['final_dropout']

    # Extract training parameters
    training_params = {}
    for k, v in best_params.items():
        if k.startswith('train_'):
            training_params[k.replace('train_', '')] = v

    # Create the best model
    print(f"Creating best model with parameters: {model_params}")
    model = model_class(**model_params).to(device)

    # Load state dict if available
    best_trial = study.best_trial
    if 'best_state_dict_path' in best_trial.user_attrs:
        checkpoint_path = best_trial.user_attrs['best_state_dict_path']
        if os.path.exists(checkpoint_path):
            print(f"Loading best weights from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'])

    # Create mock model config
    model_config = {
        'fixed_params': {
            'n_in': input_dim,
            'n_out': output_dim
        }
    }

    # Save the model package
    save_mlp_model_package(
        trial_dir=trial_dir,
        model_dir=model_dir,
        model=model,
        model_hyperparameters=best_params,
        training_params=training_params,
        scaler=scaler,
        model_config=model_config,
        study=study,
        metric_name="mse",
        additional_info={"timestamp": timestamp}
    )

    print(f"Model saved to {model_dir}")
    return model_dir