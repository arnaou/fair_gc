########################################################################################################################
#                                                                                                                      #
#    Script for performing hyperparameter optimization of GC-ML model                                                  #
#                                                                                                                      #
#                                                                                                                      #
#                                                                                                                      #
#                                                                                                                      #
#    Authors: Adem R.N. Aouichaoui                                                                                     #
#    2024/12/03                                                                                                        #
#                                                                                                                      #
########################################################################################################################
# python scripts/optuna_mlp.py --property Tc --config_file mlp_hyperopt_config.yaml --model mlp --n_trials 27 --path_2_data data/ --path_2_result results/ --path_2_model models/ --seed 42 --n_jobs 3 --split_type fair_min
##########################################################################################################
# import packages & load arguments
##########################################################################################################
# import packages and modules
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
parser.add_argument('--property', type=str, default='Omega', required=True, help='Tag for the property')
parser.add_argument('--config_file', type=str, required=True, default='mlp_hyperopt_config.yaml',help='Path to the YAML configuration file')
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
args = parser.parse_args()


##########################################################################################################
# Load the data & preprocessing
##########################################################################################################

# import the data
if args.split_type == 'butina_min':
    path_to_data = args.path_2_data+'processed/'+args.property+'/'+args.property+'_butina_min_processed.xlsx'
    df = pd.read_excel(path_to_data)
else:
    print('not yet implemented')
# remove the zero elements
# construct group ids
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
# Hyperparameter optimization
##########################################################################################################

# using optuna to perform hyperparameter optimization
study, best_model,train_params, model_config = mlp_hyperparameter_optimizer(
    config_path=args.config_file,
    model_name=args.model,
    property_name=args.property,
    train_loader=train_loader,
    val_loader=val_loader,
    study_name=args.study_name,
    sampler_name=args.sampler,
    metric_name=args.metric,
    storage=args.storage,
    n_trials=args.n_trials,
    load_if_exists=args.load_if_exists,
    n_jobs = args.n_jobs,
    seed = args.seed,
    device='cuda'
)
# print the results
print("Best parameters:", study.best_params)
print("Best value:", study.best_value)


# perform predictions
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_pred, train_true, train_metrics = evaluate_mlp(best_model, train_loader, device, y_scaler,)
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
prediction_dir = f"{args.path_2_result}/{args.property}/{args.model}/rmse_{study.best_value:.3g}_{timestamp}"
model_dir = f"{args.path_2_model}/{args.property}/{args.model}/rmse_{study.best_value:.3g}_{timestamp}"
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

from datetime import datetime
#%% Saving
trial_path = (args.path_2_result+'/'+args.property+'/'+args.model)
model_path = args.path_2_model+'/'+args.property+'/'+args.model
save_mlp_model_package(
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

#%%




