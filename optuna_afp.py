##########################################################################################################
#                                                                                                        #
#    Script for performing hyperparameter optimization of the AFP model                                  #
#    The groups used are based on the Marrero-Gani presented in:                                         #
#    https://doi.org/10.1016/j.fluid.2012.02.010                                                         #
#                                                                                                        #
#                                                                                                        #
#    Authors: Adem R.N. Aouichaoui                                                                       #
#    2024/12/03                                                                                          #
#                                                                                                        #
##########################################################################################################

##########################################################################################################
# import packages & load arguments
##########################################################################################################
import argparse
import os
import warnings

from hyperopt.rand import suggest
from lightning import seed_everything
from optuna.exceptions import ExperimentalWarning
import pandas as pd
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from src.features import mol2graph, n_atom_features, n_bond_features
from torch_geometric.loader import DataLoader
# Suppress experimental warnings from Optuna
warnings.filterwarnings('ignore', category=ExperimentalWarning)


def gnn_hypopt_parse_arguments():
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for GNN models')
    parser.add_argument('--property', type=str, default='Vc', required=False, help='Tag for the property')
    parser.add_argument('--config_file', type=str, required=False, default='gnn_hyperopt_config.yaml',help='Path to the YAML configuration file')
    parser.add_argument('--model', type=str, required=False, default='afp', help='Model type to optimize (must be defined in config file)')
    parser.add_argument('--metric', type=str, required=False, default='rmse', help='Scoring metric to use (must be defined in config file)')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of optimization trials (uses config default if not specified)' )
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

# load arguments
args = gnn_hypopt_parse_arguments()


##########################################################################################################
# Load the data & Preprocessing
##########################################################################################################

# import the data
path_to_data = args.path_2_data+'processed/'+args.property+'/'+args.property+'_processed.xlsx'
df = pd.read_excel(path_to_data)

# split the data
df_train = df[df['label']=='train']
df_val = df[df['label']=='val']
df_test = df[df['label']=='test']
# construct a scaler
y_scaler = StandardScaler()
y_scaler.fit(df_train[args.property].to_numpy().reshape(-1,1))
# construct a column with the mol objects
df_train = df_train.assign(mol=[Chem.MolFromSmiles(i) for i in df_train['SMILES']])
df_val = df_val.assign(mol=[Chem.MolFromSmiles(i) for i in df_val['SMILES']])
df_test = df_test.assign(mol=[Chem.MolFromSmiles(i) for i in df_test['SMILES']])
# construct molecular graphs
train_dataset = mol2graph(df_train, 'mol', args.property, y_scaler=y_scaler)
val_dataset = mol2graph(df_val, 'mol', args.property, y_scaler=y_scaler)
test_dataset = mol2graph(df_test, 'mol', args.property, y_scaler=y_scaler)
# construct data loaders
train_loader = DataLoader(train_dataset, batch_size=600, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=600, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=600, shuffle=False)

#%%
import optuna
from src.optims import get_class_from_path, load_config, create_sampler, RetryingStorage, suggest_gnn_parameter
import numpy as np
import random
import torch
from src.training import EarlyStopping
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Callable

def create_gnn_model(model_class, params):
    """
    Create a GNN model with the given parameters
    """

    # Convert mlp_hidden_dims from tuple to list if present
    if 'mlp_hidden_dims' in params:
        params['mlp_hidden_dims'] = [int(dim) for dim in params['mlp_hidden_dims'].split('_')]

    return model_class(**params)

def evaluate_model(model, loader, device):
    """Evaluate model on given loader"""
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            predictions.extend(out.cpu().numpy())
            targets.extend(batch.y.view(-1,1).cpu().numpy())

    return np.array(predictions), np.array(targets)


def optimize_gnn(
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
        device: str = "cuda"
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
                serializable_state_dict = {
                    k: v.cpu().numpy().tolist()
                    for k, v in model.state_dict().items()
                }
                trial.set_user_attr('best_state_dict', serializable_state_dict)
                trial.set_user_attr('training_params', training_params)
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0

            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

            trial.report(val_loss, epoch)
            # if trial.should_prune():
            #     raise optuna.TrialPruned()

        # Clean up
        del model, optimizer
        torch.cuda.empty_cache()

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

    if hasattr(best_trial, 'user_attrs') and 'best_state_dict' in best_trial.user_attrs:
        # Convert lists back to tensors
        state_dict = {
            k: torch.tensor(v)
            for k, v in best_trial.user_attrs['best_state_dict'].items()
        }
        best_model.load_state_dict(state_dict)

    return study, best_model, training_params, model_config



#%%
feature_callables = {
    'n_atom_features': n_atom_features,
    'n_bond_features': n_bond_features
}


study, best_model, train_params, model_config = optimize_gnn(
    config_path=args.config_file,
    model_name=args.model,
    property_name=args.property,
    study_name=args.study_name,
    feature_callables=feature_callables,
    train_loader= train_loader,
    val_loader= val_loader,
    metric_name=args.metric,
    sampler_name=args.sampler,
    n_trials=args.n_trials,
    storage = args.storage,
    load_if_exists=args.load_if_exists,
    n_jobs = args.n_jobs,
    seed = args.seed,
    device = 'cuda'
)
print("Best parameters:", study.best_params)
print("Best value:", study.best_value)
#%%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from src.evaluation import predict_property
train_pred, train_true, train_metrics = predict_property(best_model, train_loader, device, y_scaler)
val_pred, val_true, val_metrics = predict_property(best_model, val_loader, device, y_scaler)
test_pred, test_true, test_metrics = predict_property(best_model, test_loader, device, y_scaler)

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

# calculate the metrics
metrics = {'train': train_metrics, 'val': val_metrics, 'test': test_metrics}
df_metrics = pd.DataFrame(metrics).T.reset_index()
df_metrics.columns = ['label', 'r2', 'rmse', 'mse', 'mare', 'mae']


# construct dataframe to save the results
df_result = df.copy()

y_true = np.vstack((train_pred, val_true, test_true))
y_pred = np.vstack((train_pred, val_pred, test_pred))
split_index = df_result.columns.get_loc('label') + 1
df_result.insert(split_index, 'pred', y_pred)

prediction_path = (args.path_2_result+'/'+args.property+'/gnn/'+args.model+'/'+args.model+'_'+args.metric+'_'+
                   str(val_metrics[args.metric])+'_predictions.xlsx')
#%%
os.makedirs(args.path_2_result+'/'+args.property+'/gnn/'+args.model+'/', exist_ok=True)

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


import torch
import json
import pickle
import os
import numpy as np
from datetime import datetime
from typing import Dict, Any


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)



#%%
from typing import Dict, Any
def save_model_package(
        trial_dir: str,
        model_dir: str,
        model: torch.nn.Module,
        model_hyperparameters: Dict[str, Any],
        training_params: Dict[str, Any],
        scaler: Any,
        model_config: Dict[str, Any],
        study: Any = None,
        model_name: str = "model",
        metric_name: str = "metric",
        additional_info: Dict[str, Any] = None
) -> None:

    # Generate timestamp and base filename
    timestamp = datetime.now().strftime('%d%m%Y_%H%M')
    best_value = study.best_value if study is not None else 0.0
    base_filename = f"{metric_name}_{best_value:.3g}_{timestamp}"

    # Create directories if they don't exist
    os.makedirs(os.path.join(trial_dir, base_filename), exist_ok=True)
    os.makedirs(os.path.join(model_dir, base_filename), exist_ok=True)

    # Save model state in model directory
    model_path = os.path.join(model_dir,base_filename, "model.pt")
    torch.save(model.state_dict(), model_path)

    # Save scaler in model directory
    scaler_path = os.path.join(model_dir, base_filename, f"scaler.pkl")
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
    trial_config_path = os.path.join(trial_dir, base_filename,"results.json")
    with open(trial_config_path, 'w') as f:
        json.dump(config, f, indent=4, cls=NumpyEncoder)

    model_config_path = os.path.join(model_dir, base_filename,"model_config.json")
    model_specific_config = {
        'model_class': config['model_class'],
        'model_module': config['model_module'],
        'model_hyperparameters': clean_hyperparameters,
        'training_params': config['training_params']
    }
    with open(model_config_path, 'w') as f:
        json.dump(model_specific_config, f, indent=4, cls=NumpyEncoder)


# Saving
trial_path = (args.path_2_result+'/'+args.property+'/gnn/'+args.model)
model_path = args.path_2_model+'/'+args.property+'/gnn/'+args.model
save_model_package(
    trial_dir=trial_path,
    model_dir=model_path,
    model=best_model,
    model_hyperparameters=study.best_params,
    training_params=train_params,
    scaler=y_scaler,
    study=study,
    model_config=model_config,
    metric_name=args.metric,
    additional_info={
        'training_date': datetime.now().strftime('%d%m%Y_%H%M'),
        'dataset_info': 'your_dataset_details',
        'train_performance_metrics': train_metrics,
        'val_performance_metrics': val_metrics,
        'test_performance_metrics': test_metrics
    }
)









# # Loading
# loaded = load_model_package('models/model_001')
# model = loaded['model']
# config = loaded['config']
# scaler = loaded['scaler']
# """














#%%

# def save_results(study, fitted_model, fitted_scaler, config_path, model_name, seed,
#                  metric_name, model_dir='models/', result_dir='results/'):
#     """Save the optimization results, model, and metadata"""
#     # Create directory if it doesn't exist
#     os.makedirs(model_dir, exist_ok=True)
#     os.makedirs(result_dir, exist_ok=True)
#
#     # Generate timestamp
#     timestamp = datetime.now().strftime('%d%m%Y_%H%M')
#
#     # Base filename
#     base_filename = f"{model_name}_{metric_name}_{study.best_value:.3g}_{timestamp}"
#     model_path = os.path.join(model_dir, f"{base_filename}_pipeline.joblib")
#     results_path = os.path.join(result_dir, f"{base_filename}_results.json")
#     trials_path = os.path.join(result_dir, f"{model_name}_{metric_name}_{timestamp}" + "_trials.csv")
#
#     # Create and save pipeline
#     pipeline = create_pipeline(fitted_model, fitted_scaler)
#     # Create trials DataFrame
#     trials_df = study.trials_dataframe()
#     # Get parameter ranges
#     param_ranges = get_parameter_ranges(study)
#
#     # Save optimization results
#     results = {
#         'timestamp': timestamp,
#         'model_name': model_name,
#         'metric_name': metric_name,
#         'best_params': study.best_params,
#         'best_value': study.best_value,
#         'n_trials': len(study.trials),
#         'parameter_ranges': param_ranges,
#         'config_file': config_path,
#         'model_path': model_path,
#         'results_path': results_path,
#         'trials_path': trials_path,
#         'seed': seed,
#     }
#
#     # Save results to JSON
#     with open(results_path, 'w') as f:
#         json.dump(results, f, indent=4)
#     # save the pipeline
#     joblib.dump(pipeline, model_path)
#     # Create trials DataFrame
#     trials_df.to_csv(trials_path, index=False)
#
#     return results


# # Loading
# loaded = ModelSaver.load_model_package('model_package')
# model = loaded['model']
# scaler = loaded['scaler']
# config = loaded['config']  # Contains both model and training parameters