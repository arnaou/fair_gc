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
    parser.add_argument('--property', type=str, default='Tc', required=False, help='Tag for the property')
    parser.add_argument('--config_file', type=str, required=False, default='gnn_hyperopt_config.yaml',help='Path to the YAML configuration file')
    parser.add_argument('--model', type=str, required=False, default='afp', help='Model type to optimize (must be defined in config file)')
    parser.add_argument('--metric', type=str, required=False, default='rmse', help='Scoring metric to use (must be defined in config file)')
    parser.add_argument('--n_trials', type=int, default=60, help='Number of optimization trials (uses config default if not specified)' )
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of cores used (uses max if not configured)')
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
from typing import Dict, Any
import pickle
import torch
import json
from src.gnn_hyperopt import load_model_package


# Loading
#loaded = load_model_package('models/Omega/gnn/afp/rmse_0.149_14122024_1334')
result_folder = {'Omega': 'rmse_0.132_15122024_1739',
                 'Tc': 'rmse_0.0108_15122024_1328',
                 'Pc': 'rmse_0.081_16122024_0136',
                 'Vc': 'rmse_0.00917_15122024_1525'}
loaded = load_model_package('models/'+args.property+'/gnn/afp/'+result_folder[args.property])

best_model = loaded['model']
config = loaded['config']
y_scaler = loaded['scaler']


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from src.evaluation import evaluate_gnn
train_pred, train_true, train_metrics = evaluate_gnn(best_model, train_loader, device, y_scaler)
val_pred, val_true, val_metrics = evaluate_gnn(best_model, val_loader, device, y_scaler)
test_pred, test_true, test_metrics = evaluate_gnn(best_model, test_loader, device, y_scaler)


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