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

from lightning.pytorch import seed_everything
from src.evaluation import evaluate_gnn
from src.grape.utils import JT_SubGraph, DataSet
from datetime import datetime
from optuna.exceptions import ExperimentalWarning
import warnings
import os

from src.gnn_hyperopt import groupgat_hyperparameter_optimizer
from src.gnn_hyperopt import load_model_package, save_groupgat_package

# Suppress experimental warnings from Optuna
warnings.filterwarnings('ignore', category=ExperimentalWarning)
import numpy as np
##########################################################################################################
# parsing arguments
##########################################################################################################
parser = argparse.ArgumentParser(description='Hyperparameter optimization for GNN models')
parser.add_argument('--property', type=str, default='Omega', required=False, help='Tag for the property')
parser.add_argument('--config_file', type=str, required=False, default='groupgat_hyperopt_config.yaml', help='Path to the YAML configuration file')
parser.add_argument('--model', type=str, required=False, default='groupgat', help='Model type to optimize (must be defined in config file)')
parser.add_argument('--metric', type=str, required=False, default='rmse', help='Scoring metric to use (must be defined in config file)')
parser.add_argument('--n_trials', type=int, default=15, help='Number of optimization trials (uses config default if not specified)')
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




path_2_data = args.path_2_data+'processed/'+args.property+'/'+args.property+'_butina_min_processed.xlsx'
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

save_groupgat_package(
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
#python optuna_groupgat_v1.py --property Omega --config_file groupgat_hyperopt_config.yaml --model groupgat --n_trials 20 --path_2_data data/ --path_2_result results/ --path_2_model models/ --seed 42 --n_jobs 3

# loaded = load_model_package(model_dir=model_dir)
#
# best_model2 = loaded['model']
# config = loaded['config']
# y_scaler = loaded['scaler']
#
# train_pred, train_true, train_metrics = evaluate_gnn(best_model2, train_loader, device, y_scaler, tag=args.model)
# val_pred, val_true, val_metrics = evaluate_gnn(best_model2, val_loader, device, y_scaler, tag=args.model)
# test_pred, test_true, test_metrics = evaluate_gnn(best_model2, test_loader, device, y_scaler, tag=args.model)
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