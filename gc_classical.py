##########################################################################################################
#                                                                                                        #
#    Script for fitting group-contribution property models                                               #
#    The models used are based on the formulation present in:                                            #
#    https://doi.org/10.1016/j.fluid.2012.02.010                                                         #
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
import numpy as np
from scipy.optimize import least_squares
import os

from src.data import remove_zero_one_sum_rows
from src.gc_tools import retrieve_consts, linearize_gc, Fobj_fog, Fobj_sog, Fobj_tog, predict_gc
from src.splits import find_nonzero_columns, split_indices
from src.evaluation import calculate_metrics


##########################################################################################################
# parsing arguments
##########################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--property', type=str, help='tag of the property of interest')
parser.add_argument('--path_2_data', type=str, help='path to the data')
parser.add_argument('--path_2_result', type=str, help='path for storing the results')
parser.add_argument('--path_2_model', type=str, help='path for storing the model')

args = parser.parse_args()

property_tag = args.property
path_2_data = args.path_2_data
path_2_result = args.path_2_result
path_2_model = args.path_2_model


path_2_data = path_2_data+'/processed/'+property_tag+'/'+property_tag+'_processed.xlsx'
path_2_result = path_2_result+'/classical/'+property_tag+'/'+property_tag+'_result.xlsx'
##########################################################################################################
# Data Loading & Preprocessing
##########################################################################################################
# construct the path do data
#path_2_data = path_2_data + '/processed/' + property_tag + '/' + property_tag + '_processed.xlsx'
# read the data
df = pd.read_excel(path_2_data)
# construct list of columns indices
columns = [str(i) for i in range(1, 425)]
# remove zero columns
df = remove_zero_one_sum_rows(df, columns)
# construct group ids
grp_idx = [str(i) for i in range(1, 425)]
# retrieve indices of available groups
idx_avail = find_nonzero_columns(df, ['SMILES', property_tag, 'label', 'No'])
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
y_train = {'true': df_train[property_tag].to_numpy()}
# extract feature vectors and targets for validation
X_val = df_val.loc[:,idx_avail].to_numpy()
y_val = {'true':df_val[property_tag].to_numpy()}
# extract feature vectors and targets for testing
X_test = df_test.loc[:,idx_avail].to_numpy()
y_test = {'true':df_test[property_tag].to_numpy()}
# # scaling the target
# scaler = StandardScaler()
# y_train['scaled'] = scaler.fit_transform(y_train['true'])
# y_val['scaled'] = scaler.transform(y_val['true'])
# y_test['scaled'] = scaler.transform(y_test['true'])

##########################################################################################################
# GC modelling
##########################################################################################################
# retrieve the number of constants and an initial guess for them
n_const, const = retrieve_consts(property_tag)
# split feature vector based on the orders: training
X1_train = X_train[:, :n_mg1]
X2_train = X_train[:, n_mg1:n_mg1+n_mg2]
X3_train = X_train[:, -n_mg3:]
# split feature vector based on the orders: val
X1_val = X_val[:, :n_mg1]
X2_val = X_val[:, n_mg1:n_mg1+n_mg2]
X3_val = X_val[:, -n_mg3:]
# split feature vector based on the orders: training
X1_test = X_test[:, :n_mg1]
X2_test = X_test[:, n_mg1:n_mg1+n_mg2]
X3_test = X_test[:, -n_mg3:]
# linearize according GC
y_lin = linearize_gc(y_train['true'],const,property_tag)
# fine the initial guess using linear algebra
theta0 = np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(X_train),X_train)),np.transpose(X_train)),y_lin)
# insert the initial guess into the parameter vector
theta0 = np.insert(theta0, 0, const)
# extract the contributions of each order
theta01 = theta0[:n_mg1 + n_const]
theta02 = theta0[n_mg1+n_const:n_mg1+n_const+n_mg2]
theta03 = theta0[n_mg1+n_const+n_mg2:]
# perform step-wise parameter fitting
step_res1 = least_squares(Fobj_fog, theta01, jac='3-point', bounds=(-np.inf, np.inf),loss='soft_l1',
                        args=(X1_train, y_train['true'], property_tag, 'res'), verbose=2)

step_res2 = least_squares(Fobj_sog, theta02, jac='3-point', bounds=(-np.inf, np.inf),loss='soft_l1',
                        args=(X2_train, y_train['true'], property_tag, X1_train, step_res1.x, 'res'), verbose=2)

step_res3 = least_squares(Fobj_tog, theta03, jac='3-point', bounds=(-np.inf, np.inf), loss='soft_l1',
                        args=(X3_train, y_train['true'], property_tag, X1_train, step_res1.x, X2_train, step_res2.x, 'res'), verbose=2)
# retrieve the contributions
theta_step = np.insert(step_res2.x, 0, step_res1.x)
theta_step = np.insert(step_res3.x, 0, theta_step)
theta = {'step': theta_step}

# perform optimization using LM: simultaneous
sim_res = least_squares(Fobj_fog, theta_step, jac='3-point', bounds=(-np.inf, np.inf), loss='soft_l1',
                       args=(X_train, y_train['true'], property_tag, 'res'), verbose=2)
# retrieve the contributions
theta_sim = sim_res.x
theta['sim'] = theta_sim
# perform predictions using the step-wise approach
y_train['step_pred'] = predict_gc(theta['step'] , X_train, property_tag)
y_val['step_pred'] = predict_gc(theta['step'] , X_val, property_tag)
y_test['step_pred'] = predict_gc(theta['step'] , X_test, property_tag)
# perform predictions using the simultaneous approach
y_train['sim_pred'] = predict_gc(theta['sim'] , X_train, property_tag)
y_val['sim_pred'] = predict_gc(theta['sim'] , X_val, property_tag)
y_test['sim_pred'] = predict_gc(theta['sim'] , X_test, property_tag)

# construct dataframe to save the results
df_result = df.copy()
y_true = np.hstack((y_train['true'], y_val['true'], y_test['true']))
y_step_pred = np.hstack((y_train['step_pred'], y_val['step_pred'], y_test['step_pred']))
y_sim_pred = np.hstack((y_train['sim_pred'], y_val['sim_pred'], y_test['sim_pred']))
split_index = df_result.columns.get_loc('label') + 1
df_result.insert(split_index, 'step_pred', y_step_pred)
df_result.insert(split_index + 1, 'sim_pred', y_sim_pred)

# calculate the metrics
metrics = {
    'step': {
        'train' : {},
        'val': {},
        'test': {},
        'all': {}
    },
    'sim': {
        'train': {},
        'val': {},
        'test': {},
        'all': {}
    }
}

metrics['step']['train'] = calculate_metrics(y_train['true'], y_train['step_pred'], n_pars)
metrics['sim']['train'] = calculate_metrics(y_train['true'], y_train['sim_pred'], n_pars)

metrics['step']['val'] = calculate_metrics(y_val['true'], y_val['step_pred'], n_pars)
metrics['sim']['val'] = calculate_metrics(y_val['true'], y_val['sim_pred'], n_pars)

metrics['step']['test'] = calculate_metrics(y_test['true'], y_test['step_pred'], n_pars)
metrics['sim']['test'] = calculate_metrics(y_test['true'], y_test['sim_pred'], n_pars)

metrics['step']['all'] = calculate_metrics(y_true, y_step_pred, n_pars)
metrics['sim']['all'] = calculate_metrics(y_true, y_sim_pred, n_pars)


flat_data = {
    f"{key1}_{key2}": values
    for key1, sub_dict in metrics.items()
    for key2, values in sub_dict.items()
}

# Convert the flattened dictionary to a DataFrame
df_metrics = pd.DataFrame(flat_data).T.reset_index()
df_metrics.columns = ['label', 'r2', 'rmse', 'mse', 'mare', 'mae']

print(df_metrics)


# Check if the directory exists, if not, create it
os.makedirs(os.path.dirname(path_2_result), exist_ok=True)

# Check if the file exists, if not, create it with 'metrics' and 'prediction' sheets
if not os.path.exists(path_2_result):
    with pd.ExcelWriter(path_2_result, mode='w', engine='openpyxl') as writer:
        df_metrics.to_excel(writer, sheet_name='metrics')
        df_result.to_excel(writer, sheet_name='prediction')
else:
    # If the file already exists, append the sheets
    with pd.ExcelWriter(path_2_result, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        df_metrics.to_excel(writer, sheet_name='metrics')
        df_result.to_excel(writer, sheet_name='prediction')


# save the model parameters
pos = [int(i)-1 for i in idx_avail]
nan_arr= np.full(424+n_const, np.nan)
nan_arr[pos] = theta['step'][n_const:]
coefs_lm = np.insert(nan_arr, 0, theta['step'][:n_const])

nan_arr= np.full(424, np.nan)
nan_arr[pos] = theta['sim'][n_const:]
coefs_lms = np.insert(nan_arr, 0, theta['sim'][:n_const])

os.makedirs(os.path.dirname(path_2_model+'/classical/'+property_tag+'/'), exist_ok=True)
np.save(path_2_model+'/classical/'+property_tag+'/'+property_tag+'_step_coefs',coefs_lm, allow_pickle=False)
np.save(path_2_model+'/classical/'+property_tag+'/'+property_tag+'_sim_coefs',coefs_lms, allow_pickle=False)
