########################################################################################################################
#                                                                                                                      #
#    Script helper function for using optuna for GNN hyperparameter optimization                                       #
#                                                                                                                      #
#                                                                                                                      #
#                                                                                                                      #
#                                                                                                                      #
#    Authors: Adem R.N. Aouichaoui                                                                                     #
#    2024/12/03                                                                                                        #
#                                                                                                                      #
########################################################################################################################

##########################################################################################################
# import packages & load arguments
##########################################################################################################
import argparse
import json
import numpy as np
import torch
from typing import Dict, Any, Tuple, Callable
import os
import optuna

from src.ml_hyperopt import RetryingStorage, load_config, create_sampler, get_class_from_path
from lightning import seed_everything
import torch.nn.functional as F
import pickle
from src.features import mol2graph, n_atom_features, n_bond_features




##########################################################################################################
# Define helper function and classes
##########################################################################################################


def gnn_hypopt_parse_arguments():
    """
    function for parsing the command line arguments
    :return:
    """
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for GNN models')
    parser.add_argument('--property', type=str, default='Omega', required=True, help='Tag for the property')
    parser.add_argument('--config_file', type=str, required=True, default='afp_hyperopt_config.yaml',help='Path to the YAML configuration file')
    parser.add_argument('--model', type=str, required=True, default='afp', help='Model type to optimize (must be defined in config file)')
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
def suggest_gnn_parameter(trial: optuna.Trial, name: str, param_config: Dict[str, Any]) -> Any:
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
def afp_hyperparameter_optimizer(
        config_path: str,
        model_name: str,
        property_name: str,
        train_loader,
        val_loader,
        feature_callables: Dict[str, Callable],
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
        feature_callables: Dictionary of callable functions for inferred parameters
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

        # Add inferred parameters first
        for param in model_config.get('inferred_params', []):
            callable_name = param['source']
            if callable_name in feature_callables:
                model_params[param['name']] = feature_callables[callable_name]()
            else:
                raise ValueError(f"Required callable {callable_name} not provided for parameter {param['name']}")

        # Add fixed parameters
        model_params.update(model_config.get('fixed_params', {}))

        # Handle MLP structure
        mlp_config = model_config.get('mlp_config', {})
        if mlp_config:
            # Get number of layers
            n_layers = suggest_gnn_parameter(trial, 'mlp_n_layers', mlp_config['n_layers'])
            dims = []

            # Generate dimensions for each layer
            for i in range(n_layers):
                dim = suggest_gnn_parameter(
                    trial,
                    f'mlp_dim_{i:02d}',  # Use padding to ensure unique names
                    mlp_config['dim_per_layer']
                )
                dims.append(dim)
            model_params['mlp_hidden_dims'] = dims

        # Add regular model parameters
        for param_name, param_config in param_ranges.items():
            model_params[param_name] = suggest_gnn_parameter(trial, param_name, param_config)

        # Handle training parameters separately
        for param_name, param_config in train_range.items():
            training_params[param_name] = suggest_gnn_parameter(trial, f'train_{param_name}', param_config)

        # Create model and optimizer
        print("Creating model with parameters:", model_params)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model_class(**model_params).to(device)
        optimizer = torch.optim.Adam( model.parameters(), lr=training_params.get('learning_rate', 0.001))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, mode='min',
                                                                factor=training_params.get('lr_reduce', 0.7), patience=5, min_lr=1e-6)

        best_val_loss = float('inf')
        best_state_dict = None
        patience = 25
        patience_counter = 0

        for epoch in range(defaults['max_epochs']):
            # Training
            model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = F.mse_loss(pred, batch.y.view(-1, 1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    loss = F.mse_loss(pred, batch.y.view(-1, 1))
                    val_loss += loss.item()

            val_loss = val_loss / len(val_loader)
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Convert state dict tensors to lists for serialization
                # Save state dict to a separate file using trial number
                save_path = os.path.join('/work3/arnaou/checkpoints', f'{property_name}_trial_{trial.number}_best_state.pt')
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
            # if study.pruner is not None:
            #     if trial.should_prune():
            #         raise optuna.TrialPruned()

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
        #pruner=optuna.pruners.PatientPruner(optuna.pruners.SuccessiveHalvingPruner(), patience=25),
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

    for param in model_config.get('inferred_params', []):
        callable_name = param['source']
        if callable_name in feature_callables:
            best_model_params[param['name']] = feature_callables[callable_name]()

    # Add optimized parameters
    # Add parameters from param_ranges (standard parameters only)
    for param_name in param_ranges:
        if param_name in best_params:
            best_model_params[param_name] = best_params[param_name]

    # Handle MLP dimensions separately using mlp_config
    if 'mlp_config' in model_config:
        n_layers = best_params['mlp_n_layers']
        dims = []
        for i in range(n_layers):
            dim_key = f'mlp_dim_{i:02d}'
            if dim_key in best_params:
                dims.append(best_params[dim_key])
        best_model_params['mlp_hidden_dims'] = dims

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

feature_callables = {
    'n_atom_features': n_atom_features,
    'n_bond_features': n_bond_features
}
def save_model_package(
        trial_dir: str,
        model_dir: str,
        model: torch.nn.Module,
        model_hyperparameters: Dict[str, Any],
        training_params: Dict[str, Any],
        scaler: Any,
        model_config: Dict[str, Any],
        study: Any = None,
        metric_name: str = "metric",
        additional_info: Dict[str, Any] = None,
        timestamp: str = None
) -> None:

    # Generate timestamp and base filename


    # Create directories if they don't exist
    os.makedirs(trial_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Save model state in model directory
    model_path = os.path.join(model_dir, "model.pt")
    torch.save(model.state_dict(), model_path)

    # Save scaler in model directory
    scaler_path = os.path.join(model_dir, f"scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Print debug information
    print("Model hyperparameters before cleaning:", model_hyperparameters)

    # Clean model hyperparameters - remove construction-only parameters and individual MLP dimensions
    clean_hyperparameters = {}

    # First, add inferred parameters if they were provided separately
    for param in model_config.get('inferred_params', []):
        param_name = param['name']
        if param_name in ['in_channels', 'edge_dim']:
            clean_hyperparameters[param_name] = feature_callables[param['source']]()

    # Add fixed parameters
    clean_hyperparameters.update(model_config.get('fixed_params', {}))  # This should include out_channels

    # Reconstruct mlp_hidden_dims from the best parameters
    if 'mlp_n_layers' in model_hyperparameters:
        n_layers = model_hyperparameters['mlp_n_layers']
        dims = []
        for i in range(n_layers):
            dim_key = f'mlp_dim_{i:02d}'
            if dim_key in model_hyperparameters:
                dims.append(model_hyperparameters[dim_key])
        clean_hyperparameters['mlp_hidden_dims'] = dims

    # Add remaining valid parameters
    for k, v in model_hyperparameters.items():
        # Skip construction-only parameters, MLP dimensions, and training parameters
        if (k not in ['mlp_n_layers'] and
                not k.startswith('mlp_dim_') and
                not k.startswith('train_')):
            clean_hyperparameters[k] = v

    # Verify all required parameters are present
    required_params = ['in_channels', 'out_channels', 'edge_dim', 'mlp_hidden_dims']
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
    trial_config_path = os.path.join(trial_dir,"results.json")
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




def load_model_package(
        model_dir: str,
        device: str = "cuda"
) -> Dict[str, Any]:
    """
    Load saved model package from model directory.

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
    module_path = config['model_module']
    model_name = config['model_class']
    module = __import__(module_path, fromlist=[model_name])
    model_class = getattr(module, model_name)

    # Create model instance
    if "groupgat" in model_dir.lower():
        model = model_class(config['model_hyperparameters'])
    else:
        model = model_class(**config['model_hyperparameters'])

    # Load model state
    model_path = os.path.join(model_dir, "model.pt")
    model.load_state_dict(torch.load(model_path, weights_only=True))#, weights_only=True)
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


def suggest_megnet_params(trial: optuna.Trial, param_config: Dict[str, Any], model_params: Dict[str, Any]) -> Dict[
    str, Any]:
    """Suggest parameters for MEGNet model."""
    # Handle base parameters
    for param_name, param_config in param_config['param_ranges'].items():
        model_params[param_name] = suggest_gnn_parameter(trial, param_name, param_config)

    # Handle MLP configuration
    if 'mlp_config' in param_config:
        n_layers = suggest_gnn_parameter(trial, 'mlp_n_layers', param_config['mlp_config']['n_layers'])
        dims = []
        for i in range(n_layers):
            dim = suggest_gnn_parameter(
                trial,
                f'mlp_dim_{i:02d}',
                param_config['mlp_config']['dim_per_layer']
            )
            dims.append(dim)
        model_params['mlp_out_hidden'] = dims

    return model_params

def megnet_hyperparameter_optimizer(
        config_path: str,
        model_name: str,
        property_name: str,
        train_loader,
        val_loader,
        feature_callables: Dict[str, Callable],
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
    MEGNet model optimization using Optuna.

    Args:
        config_path: Path to YAML configuration file
        model_name: Name of the model in config to optimize
        property_name: Name of the property being predicted
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        feature_callables: Dictionary of callable functions for inferred parameters
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

    # Set study name
    study_name = study_name or f"megnet_{property_name}_{model_name}"

    # Set storage name
    storage = storage or f'sqlite:///optuna_dbs/optuna_megnet_{property_name}_{model_name}.db'
    storage = RetryingStorage(storage)

    def objective(trial: optuna.Trial) -> float:
        # Get model class (MEGNet)
        model_class = get_class_from_path(model_config['class'])

        # Initialize parameters
        model_params = {}
        training_params = {}

        # Add inferred parameters (node and edge features)
        for param in model_config.get('inferred_params', []):
            callable_name = param['source']
            if callable_name in feature_callables:
                model_params[param['name']] = feature_callables[callable_name]()
            else:
                raise ValueError(f"Required callable {callable_name} not provided for parameter {param['name']}")

        # Add fixed parameters
        model_params.update(model_config.get('fixed_params', {}))

        # Handle MEGNet MLP structure
        mlp_config = model_config.get('mlp_config', {})
        if mlp_config:
            n_layers = suggest_gnn_parameter(trial, 'mlp_n_layers', mlp_config['n_layers'])
            dims = []
            for i in range(n_layers):
                dim = suggest_gnn_parameter(
                    trial,
                    f'mlp_dim_{i:02d}',
                    mlp_config['dim_per_layer']
                )
                dims.append(dim)
            model_params['mlp_out_hidden'] = dims

        # Add regular model parameters
        for param_name, param_config in param_ranges.items():
            model_params[param_name] = suggest_gnn_parameter(trial, param_name, param_config)

        # Handle training parameters
        for param_name, param_config in train_range.items():
            training_params[param_name] = suggest_gnn_parameter(trial, f'train_{param_name}', param_config)

        # Create model and optimizer
        print("Creating MEGNet model with parameters:", model_params)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model_class(**model_params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=training_params.get('learning_rate', 0.001))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=training_params.get('lr_reduce', 0.7),
            patience=5,
            min_lr=1e-6
        )

        best_val_loss = float('inf')
        patience = 25
        patience_counter = 0

        for epoch in range(defaults['max_epochs']):
            # Training
            model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                # MEGNet specific forward pass
                pred = model(batch)
                loss = F.mse_loss(pred, batch.y.view(-1, 1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    # MEGNet specific forward pass
                    pred = model(batch)
                    loss = F.mse_loss(pred, batch.y.view(-1, 1))
                    val_loss += loss.item()

            val_loss = val_loss / len(val_loader)
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join('/work3/arnaou/checkpoints', f'megnet_{property_name}_trial_{trial.number}_best_state.pt')
                torch.save({
                    'state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'epoch': epoch
                }, save_path)
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

        return best_val_loss

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=load_if_exists,
        direction=defaults['direction'],
        sampler=sampler
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
        print(f"The desired number of iterations have already been done, consider increasing n_trials")

    # Run optimization
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    # Create best model
    model_class = get_class_from_path(model_config['class'])
    best_params = study.best_params

    # Reconstruct best parameters
    best_model_params = {}
    best_model_params.update(model_config.get('fixed_params', {}))

    # Add inferred parameters
    for param in model_config.get('inferred_params', []):
        callable_name = param['source']
        if callable_name in feature_callables:
            best_model_params[param['name']] = feature_callables[callable_name]()

    # Add optimized parameters
    for param_name in param_ranges:
        if param_name in best_params:
            best_model_params[param_name] = best_params[param_name]

    # Handle MLP dimensions
    if 'mlp_config' in model_config:
        n_layers = best_params['mlp_n_layers']
        dims = []
        for i in range(n_layers):
            dim_key = f'mlp_dim_{i:02d}'
            if dim_key in best_params:
                dims.append(best_params[dim_key])
        best_model_params['mlp_out_hidden'] = dims

    # Create final model
    best_model = model_class(**best_model_params).to(device)

    # Load best state
    best_trial = study.best_trial
    training_params = best_trial.user_attrs.get('training_params', {})

    if 'best_state_dict_path' in best_trial.user_attrs:
        checkpoint = torch.load(best_trial.user_attrs['best_state_dict_path'], weights_only=True)
        best_model.load_state_dict(checkpoint['state_dict'])

    return study, best_model, training_params, model_config


def save_megnet_package(
        trial_dir: str,
        model_dir: str,
        model: torch.nn.Module,
        model_hyperparameters: Dict[str, Any],
        training_params: Dict[str, Any],
        scaler: Any,
        model_config: Dict[str, Any],
        study: Any = None,
        metric_name: str = "metric",
        additional_info: Dict[str, Any] = None,
        timestamp: str = None
) -> None:
    """
    Save MEGNet model package including model state, hyperparameters, and configuration.

    Args:
        trial_dir: Directory to save trial results
        model_dir: Directory to save model
        model: MEGNet model instance
        model_hyperparameters: Dictionary of model hyperparameters
        training_params: Dictionary of training parameters
        scaler: Data scaler instance
        model_config: Model configuration dictionary
        study: Optuna study instance
        metric_name: Name of metric used
        additional_info: Additional information to save
        timestamp: Timestamp string
    """
    os.makedirs(trial_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Save model state
    model_path = os.path.join(model_dir, "model.pt")
    torch.save(model.state_dict(), model_path)

    # Save scaler
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print("MEGNet hyperparameters before cleaning:", model_hyperparameters)

    # Clean hyperparameters
    clean_hyperparameters = {}

    # Add inferred parameters
    for param in model_config.get('inferred_params', []):
        param_name = param['name']
        if param_name in ['node_in_dim', 'edge_in_dim']:
            clean_hyperparameters[param_name] = feature_callables[param['source']]()

    # Add fixed parameters
    clean_hyperparameters.update(model_config.get('fixed_params', {}))

    # Handle MLP structure
    if 'mlp_n_layers' in model_hyperparameters:
        n_layers = model_hyperparameters['mlp_n_layers']
        dims = []
        for i in range(n_layers):
            dim_key = f'mlp_dim_{i:02d}'
            if dim_key in model_hyperparameters:
                dims.append(model_hyperparameters[dim_key])
        clean_hyperparameters['mlp_out_hidden'] = dims

    # Add remaining parameters
    for k, v in model_hyperparameters.items():
        if (k not in ['mlp_n_layers'] and
                not k.startswith('mlp_dim_') and
                not k.startswith('train_')):
            clean_hyperparameters[k] = v

    # Verify required parameters
    required_params = ['node_in_dim', 'edge_in_dim', 'out_channels', 'node_hidden_dim', 'edge_hidden_dim']
    missing_params = [param for param in required_params if param not in clean_hyperparameters]
    if missing_params:
        raise ValueError(f"Missing required MEGNet parameters: {missing_params}")

    # Prepare configuration
    config = {
        'model_hyperparameters': clean_hyperparameters,
        'training_params': training_params,
        'model_class': 'MEGNet',
        'model_module': 'src.models.megnet'
    }

    if study is not None:
        config['optimization'] = {
            'study_best_params': study.best_params,
            'study_best_value': float(study.best_value),
            'n_trials': len(study.trials),
            'direction': study.direction.name
        }
        # Save trials data
        study_trials = study.trials_dataframe()
        trials_path = os.path.join(trial_dir, "trials.csv")
        study_trials.to_csv(trials_path, index=False)

    if additional_info:
        config.update(additional_info)

    # Save results and config
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



def groupgat_hyperparameter_optimizer(
        config_path: str,
        model_name: str,
        property_name: str,
        train_loader,
        val_loader,
        feature_callables: Dict[str, Callable],
        frag_dim: int,
        study_name: str = None,
        sampler_name: str = None,
        metric_name: str = None,
        storage: str = None,
        n_trials: int = None,
        load_if_exists: bool = True,
        n_jobs: int = -1,
        seed: int = None,
        device: str = None
):
    """
    GroupGAT model optimization using Optuna.

    Args:
        config_path: Path to YAML configuration file
        model_name: Name of the model in config to optimize
        property_name: Name of the property being predicted
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        feature_callables: Dictionary of callable functions for inferred parameters
        frag_dim: Dimension of fragment features
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

    # Set study name and storage
    study_name = study_name or f"groupgat_{property_name}_{model_name}"
    storage = storage or f'sqlite:///optuna_dbs/optuna_groupgat_{property_name}_{model_name}.db'
    storage = RetryingStorage(storage)

    def objective(trial: optuna.Trial) -> float:
        # Get model class
        model_class = get_class_from_path(model_config['class'])

        # Initialize base parameters (non-optimized)
        model_params = {
            'node_in_dim': feature_callables['n_atom_features'](),
            'edge_in_dim': feature_callables['n_bond_features'](),
            'frag_dim': frag_dim,
            'global_features': False
        }

        # Add fixed parameters if any
        model_params.update(model_config.get('fixed_params', {}))

        # Layer 1 parameters
        model_params['L1_hidden_dim'] = suggest_gnn_parameter(trial, 'L1_hidden_dim', param_ranges['L1_hidden_dim'])
        model_params['L1_layers_atom'] = suggest_gnn_parameter(trial, 'L1_layers_atom', param_ranges['L1_layers_atom'])
        model_params['L1_layers_mol'] = suggest_gnn_parameter(trial, 'L1_layers_mol', param_ranges['L1_layers_mol'])
        model_params['L1_dropout'] = suggest_gnn_parameter(trial, 'L1_dropout', param_ranges['L1_dropout'])
        model_params['L1_out_dim'] = suggest_gnn_parameter(trial, 'L1_out_dim', param_ranges['L1_out_dim'])

        # Layer 2 parameters
        model_params['L2_hidden_dim'] = suggest_gnn_parameter(trial, 'L2_hidden_dim', param_ranges['L2_hidden_dim'])
        model_params['L2_layers_atom'] = suggest_gnn_parameter(trial, 'L2_layers_atom', param_ranges['L2_layers_atom'])
        model_params['L2_layers_mol'] = suggest_gnn_parameter(trial, 'L2_layers_mol', param_ranges['L2_layers_mol'])
        model_params['L2_out_dim'] = suggest_gnn_parameter(trial, 'L2_out_dim', param_ranges['L2_out_dim'])
        model_params['L2_dropout'] = suggest_gnn_parameter(trial, 'L2_dropout', param_ranges['L2_dropout'])

        # Layer 3 parameters
        model_params['L3_hidden_dim'] = suggest_gnn_parameter(trial, 'L3_hidden_dim', param_ranges['L3_hidden_dim'])
        model_params['L3_layers_atom'] = suggest_gnn_parameter(trial, 'L3_layers_atom', param_ranges['L3_layers_atom'])
        model_params['L3_layers_mol'] = suggest_gnn_parameter(trial, 'L3_layers_mol', param_ranges['L3_layers_mol'])
        model_params['L3_out_dim'] = suggest_gnn_parameter(trial, 'L3_out_dim', param_ranges['L3_out_dim'])
        model_params['L3_dropout'] = suggest_gnn_parameter(trial, 'L3_dropout', param_ranges['L3_dropout'])

        # MLP and final parameters
        n_mlp_layers = suggest_gnn_parameter(trial, 'n_mlp_layers', param_ranges['n_mlp_layers'])
        mlp_dims = []
        for i in range(n_mlp_layers):
            dim = suggest_gnn_parameter(trial, f'mlp_dim_{i}', param_ranges['mlp_dim'])
            mlp_dims.append(dim)
        model_params['MLP_layers'] = mlp_dims

        model_params['num_heads'] = suggest_gnn_parameter(trial, 'num_heads', param_ranges['num_heads'])
        model_params['final_dropout'] = suggest_gnn_parameter(trial, 'final_dropout', param_ranges['final_dropout'])

        # Handle training parameters
        training_params = {}
        for param_name, param_config in train_range.items():
            training_params[param_name] = suggest_gnn_parameter(trial, f'train_{param_name}', param_config)

        # Pack all parameters into net_params dictionary
        net_params = {
            'node_in_dim': model_params['node_in_dim'],
            'edge_in_dim': model_params['edge_in_dim'],
            'frag_dim': model_params['frag_dim'],
            'global_features': model_params['global_features'],
            'L1_hidden_dim': model_params['L1_hidden_dim'],
            'L1_layers_atom': model_params['L1_layers_atom'],
            'L1_layers_mol': model_params['L1_layers_mol'],
            'L1_dropout': model_params['L1_dropout'],
            'L1_out_dim': model_params['L1_out_dim'],
            'L2_hidden_dim': model_params['L2_hidden_dim'],
            'L2_layers_atom': model_params['L2_layers_atom'],
            'L2_layers_mol': model_params['L2_layers_mol'],
            'L2_dropout': model_params['L2_dropout'],
            'L2_out_dim': model_params['L2_out_dim'],
            'L3_hidden_dim': model_params['L3_hidden_dim'],
            'L3_layers_atom': model_params['L3_layers_atom'],
            'L3_layers_mol': model_params['L3_layers_mol'],
            'L3_dropout': model_params['L3_dropout'],
            'L3_out_dim': model_params['L3_out_dim'],
            'MLP_layers': model_params['MLP_layers'],
            'num_heads': model_params['num_heads'],
            'final_dropout': model_params['final_dropout']
        }
        trial.set_user_attr('training_params', training_params)
        # Create model and optimizer
        print("Creating GroupGAT model with parameters:", model_params)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model_class(net_params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=training_params.get('learning_rate', 0.001))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=training_params.get('lr_reduce', 0.7),
            patience=5,
            min_lr=1e-6
        )

        best_val_loss = float('inf')
        patience = 25
        patience_counter = 0

        for epoch in range(defaults['max_epochs']):
            # Training
            model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                pred = model(batch)
                loss = F.mse_loss(pred, batch.y.view(-1, 1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    pred = model(batch)
                    loss = F.mse_loss(pred, batch.y.view(-1, 1))
                    val_loss += loss.item()

            val_loss = val_loss / len(val_loader)
            scheduler.step(val_loss)

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join('/work3/arnaou/checkpoints', f'groupgat_{property_name}_trial_{trial.number}_best_state.pt')
                torch.save({
                    'state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'epoch': epoch
                }, save_path)
                trial.set_user_attr('best_state_dict_path', save_path)

                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

            trial.report(val_loss, epoch)
        # this has to be out otherwise it take the fina learning rate

        # Clean up
        del model, optimizer
        torch.cuda.empty_cache()

        return best_val_loss

    # Create and run study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=load_if_exists,
        direction=defaults['direction'],
        sampler=sampler
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
        print("The desired number of iterations have already been done, consider increasing n_trials")

    # Run optimization
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    # Create best model
    model_class = get_class_from_path(model_config['class'])
    best_params = study.best_params

    # Reconstruct best parameters
    best_model_params = {
        'node_in_dim': feature_callables['n_atom_features'](),
        'edge_in_dim': feature_callables['n_bond_features'](),
        'frag_dim': frag_dim,
        'global_features': False
    }

    # Add fixed parameters
    best_model_params.update(model_config.get('fixed_params', {}))

    # Add optimized layer parameters
    for param_name in param_ranges:
        if param_name in best_params and not param_name.startswith('mlp_'):
            best_model_params[param_name] = best_params[param_name]

    # Handle MLP layers
    n_mlp_layers = best_params['n_mlp_layers']
    mlp_dims = []
    for i in range(n_mlp_layers):
        dim_key = f'mlp_dim_{i}'
        if dim_key in best_params:
            mlp_dims.append(best_params[dim_key])
    best_model_params['MLP_layers'] = mlp_dims

    # Create final model
    best_model = model_class(best_model_params).to(device)

    # Load best state
    best_trial = study.best_trial
    training_params = best_trial.user_attrs.get('training_params', {})

    if 'best_state_dict_path' in best_trial.user_attrs:
        checkpoint = torch.load(best_trial.user_attrs['best_state_dict_path'])
        best_model.load_state_dict(checkpoint['state_dict'])

    return study, best_model, training_params, model_config, best_model_params




def save_groupgat_package(
        trial_dir: str,
        model_dir: str,
        model: torch.nn.Module,
        net_params: Dict[str, Any],
        training_params: Dict[str, Any],
        scaler: Any,
        model_config: Dict[str, Any],
        study: Any = None,
        metric_name: str = "metric",
        additional_info: Dict[str, Any] = None,
        timestamp: str = None
) -> None:
    """
    Save GroupGAT model package including model state, hyperparameters, and configuration.
    """
    # Create directories if they don't exist
    os.makedirs(trial_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Save model state in model directory
    model_path = os.path.join(model_dir, "model.pt")
    torch.save(model.state_dict(), model_path)

    # Save scaler in model directory
    scaler_path = os.path.join(model_dir, f"scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Print debug information
    print("Model hyperparameters before cleaning:", net_params)

    # Clean model hyperparameters - remove construction-only parameters and individual MLP dimensions
    clean_hyperparameters = {}

    # First, add inferred parameters if they were provided separately
    for param in model_config.get('inferred_params', []):
        param_name = param['name']
        if param_name in ['node_in_dim', 'edge_in_dim']:
            clean_hyperparameters[param_name] = feature_callables[param['source']]()

    # Add fixed parameters
    clean_hyperparameters.update(model_config.get('fixed_params', {}))

    # Add the rest of the parameters from net_params
    for k, v in net_params.items():
        if k not in clean_hyperparameters:  # Don't overwrite inferred or fixed params
            clean_hyperparameters[k] = v

    # Verify all required parameters are present
    required_params = ['node_in_dim', 'edge_in_dim', 'frag_dim', 'global_features',
                      'L1_hidden_dim', 'L2_hidden_dim', 'L3_hidden_dim', 'MLP_layers']
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