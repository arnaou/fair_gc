import torch
from src.gnn_hyperopt import megnet_hyperparameter_optimizer, save_megnet_package, gnn_hypopt_parse_arguments
import os
import warnings
from optuna.exceptions import ExperimentalWarning
import pandas as pd
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from src.features import mol2graph, n_atom_features, n_bond_features
from torch_geometric.loader import DataLoader
import argparse
import numpy as np
from datetime import datetime



# Suppress experimental warnings from Optuna
warnings.filterwarnings('ignore', category=ExperimentalWarning)


# parse arguments
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

##########################################################################################################
# Hyperparameter optimization
##########################################################################################################

# define the callable functions
feature_callables = {
    'n_atom_features': n_atom_features,
    'n_bond_features': n_bond_features
}

# perform hyperparameter optimization
study, best_model, train_params, model_config = megnet_hyperparameter_optimizer(
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

# save the model and the trials and hyperparameters
from typing import Dict, Any



# Saving
trial_path = (args.path_2_result+'/'+args.property+'/gnn/'+args.model)
model_path = args.path_2_model+'/'+args.property+'/gnn/'+args.model

save_megnet_package(
    trial_dir=prediction_dir,
    model_dir=model_dir,
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

# # python megnet_hyperopt.py --property Tc --config_file megnet_hyperopt_config.yaml --model megnet --n_trials 2500 --path_2_data data/ --path_2_result results/ --path_2_model models/ --seed 42 --n_jobs 3