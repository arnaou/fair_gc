########################################################################################################################
#                                                                                                                      #
#    Script for performing bootstrap uncertainty estimation on GC-ML                                                   #
#       this can be done over multiple properties and ML models                                                        #
#                                                                                                                      #
#                                                                                                                      #
#                                                                                                                      #
#    Authors: Adem R.N. Aouichaoui                                                                                     #
#    2024/12/03                                                                                                        #
#                                                                                                                      #
########################################################################################################################
# python scripts\bootstrap_ml.py --property Pc Vc --n_bootstrap 100 --path_2_data data --path_2_result results --path_2_model models
##########################################################################################################
# import packages & load arguments
##########################################################################################################
import sys
import os
# append the src folder
gc_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(gc_dir)
import pandas as pd
import argparse
import numpy as np
from lightning import seed_everything
import os
from scipy.optimize import least_squares
from src.data import remove_zero_one_sum_rows
from src.gc_tools import retrieve_consts,  Fobj_fog, Fobj_sog, Fobj_tog, predict_gc
from src.splits import find_nonzero_columns, split_indices
import json
from src.evaluation import calculate_metrics
import warnings
from optuna.exceptions import ExperimentalWarning
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

# Suppress experimental warnings from Optuna
warnings.filterwarnings('ignore', category=ExperimentalWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)


##########################################################################################################
# parsing arguments
##########################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--property', type=str, default='Tc', help='tag of the property of interest')
parser.add_argument('--n_bootstrap', type=int, default=100, help='number of bootstrap samples')
parser.add_argument('--path_2_data', type=str, default='../data', help='path to the data')
parser.add_argument('--path_2_result', type=str, default='../results', help='path for storing the results')
parser.add_argument('--path_2_model', type=str, default='../models', help='path for storing the model')
parser.add_argument('--seed', type=int, default=42, help='seed for training')

args = parser.parse_args()
seed_everything(args.seed)

##########################################################################################################
# %% Data Loading and preparation
##########################################################################################################
# construct the path to the data
path_2_data = args.path_2_data+'/processed/'+args.property+'/'+args.property+'_butina_min_processed.xlsx'
# reda the data
df = pd.read_excel(path_2_data)
# construct list of columns indices
columns = [str(i) for i in range(1, 425)]
# remove zero columns
df = remove_zero_one_sum_rows(df, columns)
idx_avail = find_nonzero_columns(df, ['SMILES', args.property, 'label', 'No', 'required', 'superclass'])
# split the indices into 1st, 2nd and 3rd order groups
idx_mg1, idx_mg2, idx_mg3 = split_indices(idx_avail)
# extract the number of available groups in each order
n_mg1 = len(idx_mg1)
n_mg2 = len(idx_mg2)
n_mg3 = len(idx_mg3)
n_pars = n_mg1+n_mg2+n_mg3
# split the data
df_train = df[df['label']=='train']
df_val = df[df['label']=='val']
df_test = df[df['label']=='test']
# extract feature vectors and targets for training
X_train = df_train.loc[:,idx_avail].to_numpy()
y_train = {'true': df_train[args.property].to_numpy()}
# split feature vector based on the orders: training
X1_train = X_train[:, :n_mg1]
X2_train = X_train[:, n_mg1:n_mg1+n_mg2]
X3_train = X_train[:, -n_mg3:]
# extract the validation data
X_val = df_val.loc[:, idx_avail].to_numpy()
y_val = {'true': df_val[args.property].to_numpy().reshape(-1, 1)}
# split feature vector based on the orders: val
X1_val = X_val[:, :n_mg1]
X2_val = X_val[:, n_mg1:n_mg1+n_mg2]
X3_val = X_val[:, -n_mg3:]
# extrac the testing data
X_test = df_test.loc[:, idx_avail].to_numpy()
y_test = {'true': df_test[args.property].to_numpy().reshape(-1, 1)}
# split feature vector based on the orders: training
X1_test = X_test[:, :n_mg1]
X2_test = X_test[:, n_mg1:n_mg1+n_mg2]
X3_test = X_test[:, -n_mg3:]
# retrieve the number of constants and an initial guess for them
n_const, const = retrieve_consts(args.property)
##########################################################################################################
# %% Fitting reference model
##########################################################################################################
# load the reference model
theta = np.load('../models/' + args.property + '/classical/' + args.property + '_step_coefs.npy')
theta0 = theta[~np.isnan(theta)]
# extract the contributions of each order
theta01 = theta0[:n_mg1 + n_const]
theta02 = theta0[n_mg1 + n_const:n_mg1 + n_const + n_mg2]
theta03 = theta0[n_mg1 + n_const + n_mg2:]

# perform predictions
y_train['pred'] = predict_gc(theta0, X_train, args.property).reshape(-1, 1)
y_val['pred'] = predict_gc(theta0, X_val, args.property).reshape(-1, 1)
y_test['pred'] = predict_gc(theta0, X_test, args.property).reshape(-1, 1)
y_pred = np.vstack((y_train['pred'], y_val['pred'], y_test['pred']))
# perform metric calculation
# calculate the metrics
metrics = {'train': calculate_metrics(y_train['true'], y_train['pred']),
           'val': calculate_metrics(y_val['true'], y_val['pred']),
           'test': calculate_metrics(y_test['true'], y_test['pred'])}
df_metrics = pd.DataFrame(metrics).T.reset_index()
df_metrics.columns = ['label', 'r2', 'rmse', 'mse', 'mare', 'mae']
df_metrics.loc[:,'n_boot'] = 0
df_metrics = df_metrics.reindex(columns = ['n_boot','label', 'r2', 'rmse', 'mse', 'mare', 'mae'])

# save the results
df_predictions = df.loc[:,:'required'].copy()
df_predictions.loc[:,0] = y_pred
##########################################################################################################
# %% Construct boostrap and perform model fitting
##########################################################################################################
# construct the path to the model
path_2_result = args.path_2_result + '/' + args.property + '/classical/'
# calculate the errors
err = (y_train['true'] - y_train['pred'].ravel())
# Initialize tensor
err_matrix = np.zeros((len(err), args.n_bootstrap))
# set bootstrap seed
rng = np.random.RandomState(args.seed)
# Populate matrix with random samples from the residual with replacements
for i in range(args.n_bootstrap):
    err_matrix[:, i] = rng.choice(err, size=len(err), replace=True)

# Build synthetic data matrix: prediction + column of res_matrix
synth_data = err_matrix + y_train['pred']

# prepare list for dataframes
y_pred_list = []

# loop over the bootstrap samples
for i in tqdm(range(args.n_bootstrap), ncols=100):
    # extract the target values
    y_train['true'] = synth_data[:, i]
    # perform step-wise parameter fitting
    step_res1 = least_squares(Fobj_fog, theta01, jac='3-point', bounds=(-np.inf, np.inf), loss='soft_l1',
                              args=(X1_train, y_train['true'], args.property, 'res'), verbose=1)

    step_res2 = least_squares(Fobj_sog, theta02, jac='3-point', bounds=(-np.inf, np.inf), loss='soft_l1',
                              args=(X2_train, y_train['true'], args.property, X1_train, step_res1.x, 'res'), verbose=1)

    step_res3 = least_squares(Fobj_tog, theta03, jac='3-point', bounds=(-np.inf, np.inf), loss='soft_l1',
                              args=(
                              X3_train, y_train['true'], args.property, X1_train, step_res1.x, X2_train, step_res2.x,
                              'res'), verbose=1)
    # retrieve the contributions
    theta= np.insert(step_res2.x, 0, step_res1.x)
    theta= np.insert(step_res3.x, 0, theta)
    if args.property != 'Tc':
        theta01 = step_res1.x
        theta02 = step_res2.x
        theta03 = step_res3.x
    # perform predictions
    y_train['pred']  = predict_gc(theta , X_train, args.property)
    y_val['pred'] =  predict_gc(theta, X_val, args.property)
    y_test['pred'] = predict_gc(theta, X_test, args.property)
    y_pred = np.hstack((y_train['pred'], y_val['pred'], y_test['pred']))
    if ~np.isnan(y_pred).any():
        # calculate the metrics and update dataframe
        metrics = {'train': calculate_metrics(y_train['true'], y_train['pred']),
                   'val': calculate_metrics(y_val['true'], y_val['pred']),
                   'test': calculate_metrics(y_test['true'], y_test['pred'])}
        df_0 = pd.DataFrame(metrics).T.reset_index()
        df_0.columns = ['label', 'r2', 'rmse', 'mse', 'mare', 'mae']
        df_0.loc[:, 'n_boot'] = i + 1
        df_0 = df_0.reindex(columns=['n_boot', 'label', 'r2', 'rmse', 'mse', 'mare', 'mae'])

        # update the metric dataframe
        df_metrics = pd.concat([df_metrics, df_0], ignore_index=True)

        # update prediction dataframe
        y_pred_list.append(y_pred.ravel())

        print(df_metrics[df_metrics['label'] == 'test'])

# update the prediction
df0 = pd.DataFrame(y_pred_list).T.reset_index(drop=True)
df0.columns = range(1, len(df0.columns) + 1)

df_predictions = pd.concat([df_predictions, df0], axis=1, ignore_index=False)
##########################################################################################################
# %% Construct ensemble
##########################################################################################################
print(df_metrics[df_metrics['label'] == 'test'])
# Check if the directory exists, if not, create it
path_results = path_2_result + 'bootstrap_predictions.xlsx'
os.makedirs(os.path.dirname(path_results), exist_ok=True)

# Check if the file exists, if not, create it with 'metrics' and 'prediction' sheets
if not os.path.exists(path_results):
    with pd.ExcelWriter(path_results, mode='w', engine='openpyxl') as writer:
        df_metrics.to_excel(writer, sheet_name='metrics')
        df_predictions.to_excel(writer, sheet_name='prediction')
else:
    # If the file already exists, append the sheets
    with pd.ExcelWriter(path_results, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        df_metrics.to_excel(writer, sheet_name='metrics')
        df_predictions.to_excel(writer, sheet_name='prediction')








