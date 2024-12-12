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
    parser.add_argument('--n_trials', type=int, default=400, help='Number of optimization trials (uses config default if not specified)' )
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of cores used (uses max if not configured)')
    parser.add_argument('--sampler', type=str, default='auto', help='Sampler to use (uses config default if not specified)')
    parser.add_argument('--path_2_data', type=str, default='data/', required=False, help='Path to the data file')
    parser.add_argument('--path_2_result', type=str, default = 'results/gnn/', required=False, help='Path to save the results (metrics and predictions)')
    parser.add_argument('--path_2_model', type=str, required=False, default='models/gnn/', help='Path to save the model and eventual check points')
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
from src.optims import get_class_from_path, load_config, create_sampler, RetryingStorage, create_param_suggest_fn
import numpy as np
import random
import torch
from src.training import EarlyStopping
import torch.nn.functional as F

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

def create_hyperparameter_optimizer(
        config_path: str,
        model_name: str,
        property_name: str,
        train_loader,
        val_loader,
        metric_name: str = None,
        sampler_name: str = None,
        n_trials: int = None,
        study_name: str = None,
        storage: str = None,
        load_if_exists: bool = True,
        n_jobs: int = -1,
        seed: int = None,
):
    """
    Create and run hyperparameter optimization using configuration file

    Args:
        config_path: Path to YAML configuration file
        model_name: Name of the model in config to optimize
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        metric_name: Name of scoring function from config (if None, uses default)
        sampler_name: Name of sampler from config (if None, uses default)
        n_trials: Number of optimization trials (if None, uses default)
    """
    # set no jobs to max

    # Load configuration
    config = load_config(config_path)

    # Get defaults
    defaults = config['default_settings']

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

    # Get model configuration
    model_config = config['models'][model_name]
    model_class = get_class_from_path(model_config['class'])

    # set study name
    study_name = study_name or property_name + '_' + model_name

    # set storage name
    storage = storage or 'sqlite:///optuna_dbs/optuna_' + property_name + '_' + model_name + '.db'
    storage = RetryingStorage(storage)

    # Create parameter suggestion functions
    param_suggest_fns = {
        param_name: create_param_suggest_fn(param_config)
        for param_name, param_config in model_config['param_ranges'].items()
    }

    def objective(trial):
        # Create parameters dictionary using suggestion functions
        params = {
            name: suggest_fn(trial, name)
            for name, suggest_fn in param_suggest_fns.items()
        }

        suggested_params = {
            name: suggest_fn(trial, name)
            for name, suggest_fn in param_suggest_fns.items()
            if name not in ['learning_rate']  # exclude training params
        }

        # Add dynamically inferred parameters
        model_params = {
            'in_channels': n_atom_features(),
            'edge_dim': n_bond_features(),
            'out_channels': 1,  # adjust based on your target property
            **suggested_params
        }

        # set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Create the model
        model = create_gnn_model(model_class, model_params).to(device)

        # Training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=params.get('learning_rate', 0.001))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                                               min_lr=1e-6)
        early_stopping = EarlyStopping(patience=30,  verbose=False)
        best_score = float('-inf')
        best_state_dict = None

        # start the loops
        for epoch in range(params.get('epochs', 100)):
            # model training
            model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                true = batch.y.view(-1,1)
                loss = F.mse_loss(pred, true)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            # model validation
            model.eval()
            val_loss = 0
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                true = batch.y.view(-1,1)
                loss = F.mse_loss(pred, true)
                val_loss += loss.item()

            # print progress
            # print(f'Epoch {epoch + 1:03d}, Train Loss: {total_loss / len(train_loader):.4f}, '
            #       f'Val Loss: {val_loss / len(val_loader):.4f}')

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            current_score = val_loss / len(val_loader)

            if current_score < best_score:
                best_score = current_score
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if early_stopping(current_score):
                print(f"Trial {trial.number} stopped early at epoch {epoch}")
                break


        # Load best model state
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        # final model evaluation
        val_preds, val_targets = evaluate_model(model, val_loader, device)
        final_score = metric_func(val_targets, val_preds)

        # clean up
        del model, optimizer, best_score, scheduler
        torch.cuda.empty_cache()

        return final_score

    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        seed_everything(seed)

    # Setup and run optimization
    study = optuna.create_study(direction=direction, sampler=sampler, study_name=study_name, storage=storage,
                                load_if_exists=load_if_exists)

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

    # Train final best model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_params = study.best_params

    best_model_params = {k: v for k, v in study.best_params.items()
                         if k not in ['learning_rate']}  # exclude training params

    # Add the fixed parameters
    best_model_params.update({
        'in_channels': n_atom_features(),
        'edge_dim': n_bond_features(),
        'out_channels': 1,
    })

    # Convert mlp_hidden_dims string to list if present
    if 'mlp_hidden_dims' in best_model_params:
        if best_model_params['mlp_hidden_dims'] is str:
            best_model_params['mlp_hidden_dims'] = [int(dim) for dim in best_model_params['mlp_hidden_dims'].split('_')]

    best_model = create_gnn_model(model_class, best_model_params).to(device)

    # Train best model with early stopping
    optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params.get('learning_rate', 0.001))
    early_stopping = EarlyStopping(patience=30, verbose=True)
    best_state_dict = None
    best_score = float('-inf') if direction == 'maximize' else float('inf')

    for epoch in range(best_params.get('epochs', 100)):
        # Training phase
        best_model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = best_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = F.mse_loss(out, batch.y.view(-1,1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation phase
        val_predictions, val_targets = evaluate_model(best_model, val_loader, device)
        val_loss = F.mse_loss(torch.from_numpy(val_predictions), torch.from_numpy(val_targets))
        current_score = metric_func(val_targets, val_predictions)

        # print progress
        print(f'Epoch {epoch + 1:03d}, Train Loss: {total_loss / len(train_loader):.4f}, '
              f'Val Loss: {val_loss / len(val_loader):.4f}')

        if current_score < best_score:
            best_score = current_score
            best_state_dict = {k: v.cpu().clone() for k, v in best_model.state_dict().items()}

        if early_stopping(current_score):
            print(f"Best model training stopped early at epoch {epoch}")
            break

    # Load best model state
    if best_state_dict is not None:
        best_model.load_state_dict(best_state_dict)

    return study, best_model


def get_parameter_ranges(study):
    """Extract parameter ranges from completed trials"""
    param_ranges = {}

    # Get all parameter names
    if len(study.trials) > 0:
        param_names = study.trials[0].params.keys()

        for param_name in param_names:
            # Get all values tried for this parameter
            values = [t.params[param_name] for t in study.trials if t.params.get(param_name) is not None]

            if len(values) > 0:
                if isinstance(values[0], (int, float)):
                    param_ranges[param_name] = {
                        'min': min(values),
                        'max': max(values),
                        'type': 'float' if isinstance(values[0], float) else 'int'
                    }
                else:
                    # For categorical parameters
                    param_ranges[param_name] = {
                        'values': sorted(list(set(values))),
                        'type': 'categorical'
                    }

    return param_ranges

#%%

study, best_model = create_hyperparameter_optimizer(
    config_path=args.config_file,
    model_name=args.model,
    property_name=args.property,
    study_name=args.study_name,
    train_loader= train_loader,
    val_loader= val_loader,
    metric_name=args.metric,
    sampler_name=args.sampler,
    n_trials=args.n_trials,
    storage = args.storage,
    load_if_exists=args.load_if_exists,
    n_jobs = args.n_jobs,
    seed = args.seed,
)

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