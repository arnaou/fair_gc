# script for using GC-MLP and to evaluate the model performance
# This script loads the best model from a previous hyperparameter optimization run and evaluates it on the
# training, validation, and test datasets. It prints the evaluation metrics for each dataset.

##########################################################################################################
# Load the data & preprocessing
##########################################################################################################
import os
import sys
# append the src folder
gc_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(gc_dir)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.mlp_hyperopt import mlp_hyperparameter_optimizer, save_mlp_model_package, mlp_hypopt_parse_arguments, load_mlp_model_package
from torch.utils.data import DataLoader, TensorDataset
import warnings
from optuna.exceptions import ExperimentalWarning
from sklearn.exceptions import ConvergenceWarning
from  datetime import datetime
import torch
import argparse
# Suppress experimental warnings from Optuna
warnings.filterwarnings('ignore', category=ExperimentalWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)
from src.evaluation import evaluate_mlp

##########################################################################################################
# parsing arguments
##########################################################################################################
# load arguments
parser = argparse.ArgumentParser(description='Hyperparameter optimization for GNN models')
parser.add_argument('--property', type=str, default='Omega', help='Tag for the property')
parser.add_argument('--path_2_data', type=str, default='data/', help='Path to the data file')
parser.add_argument('--path_2_model', type=str, default='models/', help='Path to save the model and eventual check points')
args = parser.parse_args()


##########################################################################################################
# Load and preprocess data
##########################################################################################################
# define the path
path_to_data = args.path_2_data + 'processed/' + args.property + '/' + args.property + '_butina_min_processed.xlsx'
# read the data
df = pd.read_excel(path_to_data)
# get indices
grp_idx = [str(i) for i in range(1, 425)]
# retrieve indices of available groups
idx_avail = grp_idx
# split the data
df_train = df[df['label']=='train']
df_val = df[df['label']=='val']
df_test = df[df['label']=='test']

# extract feature vectors and targets
X_train = df_train.loc[:,idx_avail].to_numpy()
y_train = {'true':df_train[args.property].to_numpy().reshape(-1,1)}

X_val = df_val.loc[:,idx_avail].to_numpy()
y_val = {'true':df_val[args.property].to_numpy().reshape(-1,1)}

X_test = df_test.loc[:,idx_avail].to_numpy()
y_test = {'true':df_test[args.property].to_numpy().reshape(-1,1)}

# scaling the data
y_scaler = StandardScaler()
y_train['scaled'] = y_scaler.fit_transform(y_train['true'])
y_val['scaled'] = y_scaler.transform(y_val['true'])
y_test['scaled'] = y_scaler.transform(y_test['true'])

# convert dataset into tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train['scaled'], dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val['scaled'], dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test['scaled'], dtype=torch.float32)

# construct tensor dataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# build the data loaders
train_loader = DataLoader(train_dataset, batch_size=600, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=600, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=600, shuffle=False)  # Create test dataloader




##########################################################################################################
# Load the model
##########################################################################################################
# navigate to model path
if args.property == 'Omega':
    model_dir = os.path.join(args.path_2_model,args.property, 'mlp', 'rmse_0.067_26042025_0838') # ok
elif args.property == 'Pc':
    model_dir = os.path.join(args.path_2_model, args.property, 'mlp', 'rmse_0.0573_26042025_0834') #ok
elif args.property == 'Tc':
    model_dir = os.path.join(args.path_2_model, args.property, 'mlp', 'rmse_0.0315_26042025_0837')
elif args.property == 'Vc':
    model_dir = os.path.join(args.path_2_model, args.property, 'mlp', 'rmse_0.0273_26042025_0836')

# load model and parameyers
loaded = load_mlp_model_package(model_dir)
best_model = loaded['model']
config = loaded['config']
y_scaler = loaded['scaler']
# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# perform model evaluation
train_pred, train_true, train_metrics = evaluate_mlp(best_model, train_loader, device, y_scaler)
val_pred, val_true, val_metrics = evaluate_mlp(best_model, val_loader, device, y_scaler)
test_pred, test_true, test_metrics = evaluate_mlp(best_model, test_loader, device, y_scaler)


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