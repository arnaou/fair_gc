##########################################################################################################
#                                                                                                        #
#    Script for fitting training graph neural networks                                                   #
#                                                                                                        #
#                                                                                                        #
#                                                                                                        #
#                                                                                                        #
#    Authors: Adem R.N. Aouichaoui                                                                       #
#    2024/12/03                                                                                          #
#                                                                                                        #
##########################################################################################################

##########################################################################################################
# import packages
##########################################################################################################
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.features import  n_atom_features, n_bond_features
from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
from src.training import EarlyStopping
from lightning.pytorch import seed_everything
from src.evaluation import evaluate_gnn
from src.grape.utils import JT_SubGraph, DataSet
from datetime import datetime
from optuna.exceptions import ExperimentalWarning
import warnings
import os
import optuna
from src.gnn_hyperopt import suggest_gnn_parameter
from src.ml_hyperopt import RetryingStorage, load_config, create_sampler, get_class_from_path
from typing import Dict, Any, Tuple, Callable
import pickle
from src.grape.models.GroupGAT import GCGAT_v4pro
# Suppress experimental warnings from Optuna
warnings.filterwarnings('ignore', category=ExperimentalWarning)
import numpy as np
##########################################################################################################
# parsing arguments
##########################################################################################################
parser = argparse.ArgumentParser(description='Hyperparameter optimization for GNN models')
parser.add_argument('--property', type=str, default='Pc', required=False, help='Tag for the property')
parser.add_argument('--config_file', type=str, required=False, default='groupgat_hyperopt_config.yaml', help='Path to the YAML configuration file')
parser.add_argument('--model', type=str, required=False, default='groupgat', help='Model type to optimize (must be defined in config file)')
parser.add_argument('--metric', type=str, required=False, default='rmse', help='Scoring metric to use (must be defined in config file)')
parser.add_argument('--n_trials', type=int, default=45, help='Number of optimization trials (uses config default if not specified)')
parser.add_argument('--n_jobs', type=int, default=2, help='Number of cores used (uses max if not configured)')
parser.add_argument('--sampler', type=str, default='auto', help='Sampler to use (uses config default if not specified)')
parser.add_argument('--path_2_data', type=str, default='data/', required=False, help='Path to the data file')
parser.add_argument('--path_2_result', type=str, default='results/', required=False, help='Path to save the results (metrics and predictions)')
parser.add_argument('--path_2_model', type=str, required=False, default='models/', help='Path to save the model and eventual check points')
parser.add_argument('--study_name', type=str, default=None, help='Name of the study for persistence')
parser.add_argument('--storage', type=str, default=None, help='Database URL for study storage (e.g., sqlite:///optuna.db)')
parser.add_argument('--no_load_if_exists', action='store_false', dest='load_if_exists', help='Do not load existing study if it exists')
parser.add_argument('--seed', type=int, default=42, help='Random state for reproducibility')



args = parser.parse_args()




path_2_data = args.path_2_data+'processed/'+args.property+'/'+args.property+'_processed.xlsx'
seed_everything(args.seed)
##########################################################################################################
#%% Data Loading & Preprocessing
##########################################################################################################
# read the data
df = pd.read_excel(path_2_data)
# split the data
df_train = df[df['label']=='train']
df_val = df[df['label']=='val']
df_test = df[df['label']=='test']
# construct a scaler
y_scaler = StandardScaler()
y_scaler.fit(df_train[args.property].to_numpy().reshape(-1,1))
# define fragmentation object for each of the folds
fragmentation_scheme = "data/MG_plus_reference.csv"
frag_save_path = 'data/processed/'+args.property
print("initializing frag...")
train_fragmentation = JT_SubGraph(scheme=fragmentation_scheme, save_file_path=frag_save_path+'/train_frags.pth')
val_fragmentation = JT_SubGraph(scheme=fragmentation_scheme, save_file_path=frag_save_path+'/val_frags.pth')
test_fragmentation = JT_SubGraph(scheme=fragmentation_scheme, save_file_path=frag_save_path+'/test_frags.pth')
frag_dim = train_fragmentation.frag_dim
print("done.")
# construct the datasets
train_dataset = DataSet(df=df_train, smiles_column='SMILES', target_column=args.property, global_features=None,
                        fragmentation=train_fragmentation, log=True, y_scaler=y_scaler)
val_dataset = DataSet(df=df_val, smiles_column='SMILES', target_column=args.property, global_features=None,
                        fragmentation=val_fragmentation, log=True, y_scaler=y_scaler)
test_dataset = DataSet(df=df_test, smiles_column='SMILES', target_column=args.property, global_features=None,
                        fragmentation=test_fragmentation, log=True, y_scaler=y_scaler)
# construct data loaders
train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

##########################################################################################################
#%% Hyperparameter optimization
##########################################################################################################
#%%
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
                save_path = os.path.join('checkpoints', f'groupgat_{property_name}_trial_{trial.number}_best_state.pt')
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



#%%
# define the callable functions
feature_callables = {
    'n_atom_features': n_atom_features,
    'n_bond_features': n_bond_features
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

study, best_model, train_params, model_config, best_model_params = groupgat_hyperparameter_optimizer(
        config_path = args.config_file,
        model_name=args.model,
        property_name=args.property,
        train_loader = train_loader,
        val_loader = val_loader,
        feature_callables = feature_callables,
        frag_dim = frag_dim,
        study_name= args.study_name,
        sampler_name = args.sampler,
        metric_name = args.metric,
        storage = args.storage,
        n_trials = args.n_trials,
        load_if_exists = args.load_if_exists,
        n_jobs= args.n_jobs,
        seed = args.seed,
        device = 'cuda')


# print the results
print("Best parameters:", study.best_params)
print("Best value:", study.best_value)


##########################################################################################################
# Evaluate results
##########################################################################################################

# perform predictions and evaluate performance
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from src.evaluation import evaluate_gnn
train_pred, train_true, train_metrics = evaluate_gnn(best_model, train_loader, device, y_scaler, tag=args.model)
val_pred, val_true, val_metrics = evaluate_gnn(best_model, val_loader, device, y_scaler, tag=args.model)
test_pred, test_true, test_metrics = evaluate_gnn(best_model, test_loader, device, y_scaler, tag=args.model)

# Print metrics
print("\nTraining Set Metrics:")
for metric, value in train_metrics.items():
    print(f"{metric.upper()}: {value:.4f}")

print("\nValidation Set Metrics:")
for metric, value in val_metrics.items():
    print(f"{metric.upper()}: {value:.4f}")

print("\nTest Set Metrics:")
for metric, value in test_metrics.items():
    print(f"{metric.upper()}: {value:.4f}")

# organize the metrics into a dataframe
metrics = {'train': train_metrics, 'val': val_metrics, 'test': test_metrics}
df_metrics = pd.DataFrame(metrics).T.reset_index()
df_metrics.columns = ['label', 'r2', 'rmse', 'mse', 'mare', 'mae']

# construct dataframe to save the results
df_result = df.copy()
y_true = np.vstack((train_pred, val_true, test_true))
y_pred = np.vstack((train_pred, val_pred, test_pred))
split_index = df_result.columns.get_loc('label') + 1
df_result.insert(split_index, 'pred', y_pred)

# create time stamp for directory
timestamp= datetime.now().strftime('%d%m%Y_%H%M')
# create dir and define path for saving the predictions
prediction_dir = f"{args.path_2_result}/{args.property}/gnn/{args.model}/rmse_{study.best_value:.3g}_{timestamp}"
model_dir = f"{args.path_2_model}/{args.property}/gnn/{args.model}/rmse_{study.best_value:.3g}_{timestamp}"
prediction_path = prediction_dir+'/predictions.xlsx'
os.makedirs(prediction_dir, exist_ok=True)

# save the predictions and metrics
if not os.path.exists(prediction_path):
    with pd.ExcelWriter(prediction_path, mode='w', engine='openpyxl') as writer:
        df_metrics.to_excel(writer, sheet_name='metrics')
        df_result.to_excel(writer, sheet_name='prediction')
else:
    # If the file already exists, append the sheets
    with pd.ExcelWriter(prediction_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        df_metrics.to_excel(writer, sheet_name='metrics')
        df_result.to_excel(writer, sheet_name='prediction')
#%%
import json
from src.gnn_hyperopt import NumpyEncoder
def save_model_package(
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


save_model_package(
        trial_dir = prediction_dir,
        model_dir=model_dir,
        model=best_model,
        net_params=best_model_params,
        training_params=train_params,
        scaler = y_scaler,
        model_config=model_config,
        study = study,
        metric_name = args.metric,
        timestamp = timestamp,
        additional_info = {
            'training_date': datetime.now().strftime('%d%m%Y_%H%M'),
            'dataset_info': 'your_dataset_details',
            'train_performance_metrics': train_metrics,
            'val_performance_metrics': val_metrics,
            'test_performance_metrics': test_metrics
        })


#%%
from src.gnn_hyperopt import load_model_package

loaded = load_model_package(model_dir=model_dir)

best_model2 = loaded['model']
config = loaded['config']
y_scaler = loaded['scaler']

train_pred, train_true, train_metrics = evaluate_gnn(best_model2, train_loader, device, y_scaler, tag=args.model)
val_pred, val_true, val_metrics = evaluate_gnn(best_model2, val_loader, device, y_scaler, tag=args.model)
test_pred, test_true, test_metrics = evaluate_gnn(best_model2, test_loader, device, y_scaler, tag=args.model)

# Print metrics
print("\nTraining Set Metrics:")
for metric, value in train_metrics.items():
    print(f"{metric.upper()}: {value:.4f}")

print("\nValidation Set Metrics:")
for metric, value in val_metrics.items():
    print(f"{metric.upper()}: {value:.4f}")

print("\nTest Set Metrics:")
for metric, value in test_metrics.items():
    print(f"{metric.upper()}: {value:.4f}")