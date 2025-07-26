########################################################################################################################
#                                                                                                                      #
#    Script for evaluate GC-ML                                               #
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
# import packages and modules
# append the src folder
import os
import sys
gc_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(gc_dir)
import pandas as pd
import numpy as np

from src.splits import find_nonzero_columns
from sklearn.preprocessing import StandardScaler
from src.ml_hyperopt import ml_hypopt_parse_arguments, ml_hyperparameter_optimizer, save_results
from src.ml_utils import predict_new_data, load_model
from src.evaluation import calculate_metrics
import warnings
from optuna.exceptions import ExperimentalWarning
from sklearn.exceptions import ConvergenceWarning
import argparse
import json
from src.ml_utils import create_model
from src.ml_hyperopt import model_selector

# Suppress experimental warnings from Optuna
warnings.filterwarnings('ignore', category=ExperimentalWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# load arguments
parser = argparse.ArgumentParser(description='Hyperparameter optimization')

parser.add_argument('--property', type=str, default='Vc', help='Tag for the property')
parser.add_argument('--model', type=str, default='svr', help='Name of the ML model to evaluate')
parser.add_argument('--path_2_data', type=str, default='data/', help='Path to the data file')
parser.add_argument('--path_2_model', type=str, default='models/', help='Path to save the model and eventual check points')
args = parser.parse_args()
##########################################################################################################
# Load the data & preprocessing
##########################################################################################################

# import the data
path_to_data = args.path_2_data + 'processed/' + args.property + '/' + args.property + '_butina_min_processed.xlsx'
df = pd.read_excel(path_to_data)

# remove the zero elements
# construct group ids
grp_idx = [str(i) for i in range(1, 425)]
# retrieve indices of available groups
#idx_avail = find_nonzero_columns(df, ['SMILES', args.property, 'label', 'No'])
idx_avail = grp_idx
# split the data
df_train = df[df['label']=='train']
df_val = df[df['label']=='val']
df_test = df[df['label']=='test']

# extract feature vectors and targets
#X_train = df_train.loc[:,'1':].to_numpy()
X_train = df_train.loc[:,idx_avail].to_numpy()
y_train = {'true':df_train[args.property].to_numpy().reshape(-1,1)}

#X_val = df_val.loc[:,'1':].to_numpy()
X_val = df_val.loc[:,idx_avail].to_numpy()
y_val = {'true':df_val[args.property].to_numpy().reshape(-1,1)}

#X_test = df_test.loc[:,'1':].to_numpy()
X_test = df_test.loc[:,idx_avail].to_numpy()
y_test = {'true':df_test[args.property].to_numpy().reshape(-1,1)}

# scaling the data
scaler = StandardScaler()
y_train['scaled'] = scaler.fit_transform(y_train['true'])
y_val['scaled'] = scaler.transform(y_val['true'])
y_test['scaled'] = scaler.transform(y_test['true'])

#####################################
# Load model
#####################################
if args.model == 'dt':
    model_folder = {'Omega': "dt_rmse_0.444_24042025_2117_pipeline.joblib",
                     'Tc': "dt_rmse_0.465_24042025_2117_pipeline.joblib",
                     'Pc': "dt_rmse_0.547_24042025_2117_pipeline.joblib",
                     'Vc': "dt_rmse_0.552_24042025_2117_pipeline.joblib"}
elif args.model == 'rf':
    model_folder = {'Omega': "rf_rmse_0.34_24042025_2225_pipeline.joblib",
                     'Tc': "rf_rmse_0.324_25042025_0158_pipeline.joblib",
                     'Pc': "rf_rmse_0.366_25042025_0115_pipeline.joblib",
                     'Vc': "rf_rmse_0.402_24042025_2309_pipeline.joblib"}
elif args.model == 'gb':
    model_folder = {'Omega': "gb_rmse_0.259_24042025_2220_pipeline.joblib",
                     'Tc': "gb_rmse_0.22_24042025_2247_pipeline.joblib",
                     'Pc': "gb_rmse_0.28_24042025_2254_pipeline.joblib",
                     'Vc': "gb_rmse_0.212_24042025_2225_pipeline.joblib"}
elif args.model == 'xgb':
    model_folder = {'Omega': "xgb_rmse_0.277_25042025_0013_pipeline.joblib",
                     'Tc': "xgb_rmse_0.235_24042025_2344_pipeline.joblib",
                     'Pc': "xgb_rmse_0.296_24042025_2300_pipeline.joblib",
                     'Vc': "xgb_rmse_0.245_25042025_0011_pipeline.joblib"}
elif args.model == 'gpr':
    model_folder = {'Omega': "gpr_rmse_0.287_24042025_2129_pipeline.joblib",
                     'Tc': "gpr_rmse_0.163_24042025_2248_pipeline.joblib",
                     'Pc': "gpr_rmse_0.262_24042025_2220_pipeline.joblib",
                     'Vc': "gpr_rmse_0.173_24042025_2203_pipeline.joblib"}
elif args.model == 'svr':
    model_folder = {'Omega': "svr_rmse_0.282_24042025_2200_pipeline.joblib",
                     'Tc': "svr_rmse_0.167_25042025_2141_pipeline.joblib",
                     'Pc': "svr_rmse_0.263_25042025_0919_pipeline.joblib",
                     'Vc': "svr_rmse_0.148_25042025_0248_pipeline.joblib"}
else:
    raise ValueError(f"Model {args.model} is not supported.")

# define the model path
path_2_model = args.path_2_model + '/' + args.property + '/' + args.model + '/' + model_folder[args.property]

# make predictions
_, y_train['pred'] = predict_new_data(path_2_model, X_train)
_, y_val['pred'] = predict_new_data(path_2_model, X_val)
_, y_test['pred'] = predict_new_data(path_2_model, X_test)

# performance metrics
train_metrics =  calculate_metrics(y_train['true'], y_train['pred'])
val_metrics = calculate_metrics(y_val['true'], y_val['pred'])
test_metrics = calculate_metrics(y_test['true'], y_test['pred'])


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
