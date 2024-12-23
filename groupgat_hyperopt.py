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
##########################################################################################################
# parsing arguments
##########################################################################################################
parser = argparse.ArgumentParser(description='Hyperparameter optimization for GNN models')
parser.add_argument('--property', type=str, default='Tc', required=False, help='Tag for the property')
parser.add_argument('--config_file', type=str, required=False, default='groupgat_hyperopt_config.yaml', help='Path to the YAML configuration file')
parser.add_argument('--model', type=str, required=False, default='groupgat', help='Model type to optimize (must be defined in config file)')
parser.add_argument('--metric', type=str, required=False, default='rmse', help='Scoring metric to use (must be defined in config file)')
parser.add_argument('--n_trials', type=int, default=35, help='Number of optimization trials (uses config default if not specified)')
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
) -> Tuple[optuna.study.Study, torch.nn.Module, Dict[str, Any], Dict[str, Any]]:
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

    return study, best_model, training_params, model_config



#%%
# define the callable functions
feature_callables = {
    'n_atom_features': n_atom_features,
    'n_bond_features': n_bond_features
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

study, best_model, train_params, model_config = groupgat_hyperparameter_optimizer(
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

#
# from src.grape.models.AFP import AFP
# from src.grape.models.GroupGAT import GCGAT_v4pro
# net_params = {'node_in_dim': n_atom_features(),
#               'edge_in_dim': n_bond_features(),
#               'frag_dim': frag_dim,
#               'global_features': False,
#               'L1_hidden_dim': 128,
#               'L1_layers_atom': 2,
#               'L1_layers_mol': 2,
#               'L1_dropout': 0.0,
#               'L1_out_dim': 50,
#               'L2_hidden_dim': 155,
#               'L2_layers_atom': 2,
#               'L2_layers_mol': 2,
#               'L2_out_dim': 50,
#               'L2_dropout': 0.0,
#               'L3_hidden_dim': 64,
#               'L3_layers_atom': 2,
#               'L3_layers_mol': 2,
#               'L3_out_dim': 50,
#               'L3_dropout': 0.0,
#               'MLP_layers': [40, 20],
#               'num_heads': 1,
#               'final_dropout': 0.05
#               }
#
#
#
# model = GCGAT_v4pro(net_params).to(device)
#
#
# optimizer = torch.optim.Adam(model.parameters(), lr=0.00526, weight_decay=0.00003250012)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5)
#
# # Initialize early stopping
# early_stopping = EarlyStopping(patience=25, verbose=True)
#
# num_epochs = 150
# best_val_loss = float('inf')
#
#
# for epoch in range(num_epochs):
#     # model training
#     model.train()
#     total_loss = 0
#     for batch in train_loader:
#         batch = batch.to(device)
#         optimizer.zero_grad()
#         pred = model(batch)
#         true = batch.y.view(-1, 1)
#         loss = F.mse_loss(pred, true, reduction='sum')
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     # model validation
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for batch in val_loader:
#             batch = batch.to(device)
#             pred = model(batch)
#             true = batch.y.view(-1, 1)
#             val_loss += F.mse_loss(pred, true, reduction='sum').item()
#
#     # Print progress
#     print(f'Epoch {epoch+1:03d}, Train Loss: {total_loss/len(train_loader):.4f}, '
#           f'Val Loss: {val_loss/len(val_loader):.4f}')
#
#     # Learning rate scheduling
#     scheduler.step(val_loss)
#
#     # Early stopping
#     early_stopping(val_loss, model, path_2_model + '_best.pt')
#     if early_stopping.early_stop:
#         print("Early stopping triggered")
#         break
#
# # Load best model
# model.load_state_dict(torch.load(path_2_model + '_best.pt', weights_only=True))
# model = model.to(device)
#
#
#
# train_pred, train_true, train_metrics = evaluate_gnn(model, train_loader, device, y_scaler, tag='megnet')
# val_pred, val_true, val_metrics = evaluate_gnn(model, val_loader, device, y_scaler, tag='megnet')
# test_pred, test_true, test_metrics = evaluate_gnn(model, test_loader, device, y_scaler, tag='megnet')
#
#
# # Print metrics
# print("\nTraining Set Metrics:")
# for metric, value in train_metrics.items():
#     print(f"{metric.upper()}: {value:.4f}")
#
# print("\nValidation Set Metrics:")
# for metric, value in val_metrics.items():
#     print(f"{metric.upper()}: {value:.4f}")
#
# print("\nTest Set Metrics:")
# for metric, value in test_metrics.items():
#     print(f"{metric.upper()}: {value:.4f}")
#
# #%%
# def save_groupgat_package(
#         trial_dir: str,
#         model_dir: str,
#         model: torch.nn.Module,
#         model_hyperparameters: Dict[str, Any],
#         training_params: Dict[str, Any],
#         scaler: Any,
#         model_config: Dict[str, Any],
#         study: Any = None,
#         metric_name: str = "metric",
#         additional_info: Dict[str, Any] = None,
#         timestamp: str = None
# ) -> None:
#     """
#     Save GroupGAT model package including model state, hyperparameters, and configuration.
#     """
#     os.makedirs(trial_dir, exist_ok=True)
#     os.makedirs(model_dir, exist_ok=True)
#
#     # Save model state
#     model_path = os.path.join(model_dir, "model.pt")
#     torch.save(model.state_dict(), model_path)
#
#     # Save scaler
#     scaler_path = os.path.join(model_dir, "scaler.pkl")
#     with open(scaler_path, 'wb') as f:
#         pickle.dump(scaler, f)
#
#     # Clean hyperparameters
#     clean_hyperparameters = {}
#
#     # Add non-optimized parameters
#     clean_hyperparameters.update({
#         'node_in_dim': model_hyperparameters['node_in_dim'],
#         'edge_in_dim': model_hyperparameters['edge_in_dim'],
#         'frag_dim': model_hyperparameters['frag_dim'],
#         'global_features': model_hyperparameters['global_features']
#     })
#
#     # Add fixed parameters
#     clean_hyperparameters.update(model_config.get('fixed_params', {}))
#
#     # Add layer parameters
#     layer_params = ['L1_hidden_dim', 'L1_layers_atom', 'L1_layers_mol', 'L1_dropout', 'L1_out_dim',
#                    'L2_hidden_dim', 'L2_layers_atom', 'L2_layers_mol', 'L2_dropout', 'L2_out_dim',
#                    'L3_hidden_dim', 'L3_layers_atom', 'L3_layers_mol', 'L3_dropout', 'L3_out_dim',
#                    'num_heads', 'final_dropout']
#
#     for param in layer_params:
#         if param in model_hyperparameters:
#             clean_hyperparameters[param] = model_hyperparameters[param]
#
#     # Handle MLP layers
#     if 'MLP_layers' in model_hyperparameters:
#         clean_hyperparameters['MLP_layers'] = model_hyperparameters['MLP_layers']
#
#     # Prepare configuration
#     config = {
#         'model_hyperparameters': clean_hyperparameters,
#         'training_params': training_params,
#         'model_class': 'GroupGAT',
#         'model_module': 'src.models.groupgat'
#     }
#
#     if study is not None:
#         config