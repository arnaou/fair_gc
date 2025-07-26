##########################################################################################################
#                                                                                                        #
#    Script for fitting group-contribution property models                                               #
#    The models used are based on the formulation present in:                                            #
#    https://doi.org/10.1016/j.fluid.2012.02.010                                                         #
#    NB: data are not split here. this is the classical approach using all data                                                                                                    #
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
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt

from src.data import remove_zero_one_sum_rows
from src.gc_tools import retrieve_consts, linearize_gc, Fobj_fog, Fobj_sog, Fobj_tog, predict_gc
from src.splits import find_nonzero_columns, split_indices
from src.evaluation import calculate_metrics, identify_outliers_ecdf


##########################################################################################################
# parsing arguments --property 'Vc' --path_2_data 'data/' --path_2_result 'results/' --path_2_model 'models/'
##########################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--property', type=str, default='Vc', help='tag of the property of interest')
parser.add_argument('--path_2_data', type=str, default='data/', help='path to the data')
parser.add_argument('--path_2_result', type=str, default='results/', help='path for storing the results')
parser.add_argument('--path_2_model', type=str, default='models/', help='path for storing the model')

args = parser.parse_args()

property_tag = args.property
path_2_data = args.path_2_data
path_2_result = args.path_2_result
path_2_model = args.path_2_model


path_2_data = path_2_data+'processed/'+property_tag+'/'+property_tag+'_processed.xlsx'
path_2_result = path_2_result+ property_tag+'/classical/'+property_tag+'_result.xlsx'
path_2_model = path_2_model+property_tag+'/classical/'+property_tag

##########################################################################################################
# Data Loading & Preprocessing
##########################################################################################################
# read the data
df = pd.read_excel(path_2_data)
# construct list of columns indices
columns = [str(i) for i in range(1, 425)]
# remove zero columns
df = remove_zero_one_sum_rows(df, columns)
# construct group ids
grp_idx = [str(i) for i in range(1, 425)]
# retrieve indices of available groups
idx_avail = find_nonzero_columns(df, ['SMILES', property_tag, 'label', 'No', 'required'])
# split the indices into 1st, 2nd and 3rd order groups
idx_mg1, idx_mg2, idx_mg3 = split_indices(idx_avail)
# extract the number of available groups in each order
n_mg1 = len(idx_mg1)
n_mg2 = len(idx_mg2)
n_mg3 = len(idx_mg3)
n_pars = n_mg1+n_mg2+n_mg3
# extract feature vectors and targets
X = df.loc[:,idx_avail].to_numpy()
y = {'true': df[property_tag].to_numpy()}
##########################################################################################################
# GC modelling
##########################################################################################################
#%% retrieve the number of constants and an initial guess for them
n_const, const = retrieve_consts(property_tag)
# split feature vector based on the orders
X1 = X[:, :n_mg1]
X2 = X[:, n_mg1:n_mg1+n_mg2]
X3 = X[:, -n_mg3:]
# linearize according GC
y_lin = linearize_gc(y['true'],const,property_tag)
# fine the initial guess using linear algebra
theta0 = np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(X),X)),np.transpose(X)),y_lin)
# insert the initial guess into the parameter vector
theta0 = np.insert(theta0, 0, const)
# extract the contributions of each order
theta01 = theta0[:n_mg1 + n_const]
theta02 = theta0[n_mg1+n_const:n_mg1+n_const+n_mg2]
theta03 = theta0[n_mg1+n_const+n_mg2:]
# perform step-wise parameter fitting
step_res1 = least_squares(Fobj_fog, theta01, jac='3-point', bounds=(-np.inf, np.inf),loss='soft_l1',
                        args=(X1, y['true'], property_tag, 'res'), verbose=2)
#%%
step_res2 = least_squares(Fobj_sog, theta02, jac='3-point', bounds=(-np.inf, np.inf),loss='soft_l1',
                        args=(X2, y['true'], property_tag, X1, step_res1.x, 'res'), verbose=2)

step_res3 = least_squares(Fobj_tog, theta03, jac='3-point', bounds=(-np.inf, np.inf), loss='soft_l1',
                        args=(X3, y['true'], property_tag, X1, step_res1.x, X2, step_res2.x, 'res'), verbose=2)
# retrieve the contributions
theta_step = np.insert(step_res2.x, 0, step_res1.x)
theta_step = np.insert(step_res3.x, 0, theta_step)
theta = {'step': theta_step}

# perform optimization using LM: simultaneous
sim_res = least_squares(Fobj_fog, theta_step, jac='3-point', bounds=(-np.inf, np.inf), loss='soft_l1',
                       args=(X, y['true'], property_tag, 'res'), verbose=2)
# retrieve the contributions
theta_sim = sim_res.x
theta['sim'] = theta_sim
# perform predictions using the step-wise approach
y['step_pred'] = predict_gc(theta['step'] , X, property_tag)
# perform predictions using the simultaneous approach
y['sim_pred'] = predict_gc(theta['sim'] , X, property_tag)


# construct dataframe to save the results
df_result = df.copy()
split_index = df_result.columns.get_loc('label') + 1
df_result.insert(split_index, 'step_pred', y['step_pred'])
df_result.insert(split_index + 1, 'sim_pred', y['sim_pred'])

# calculate the metrics
# total number of parameters
metrics = {'step': calculate_metrics(y['true'], y['step_pred'], n_pars),
           'sim': calculate_metrics(y['true'], y['sim_pred'], n_pars)}

# Convert the metric dictionary to a DataFrame
df_metrics = pd.DataFrame(metrics).T.reset_index()
df_metrics.columns = ['label', 'r2', 'rmse', 'mse', 'mare', 'mae']

print(df_metrics)

#################################
#save all the results
##################################

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
np.save(path_2_model+'_step_coefs',coefs_lm, allow_pickle=False)
np.save(path_2_model+'_sim_coefs',coefs_lms, allow_pickle=False)


# --property 'Vc' --path_2_data 'data/' --path_2_result 'results/' --path_2_model 'models/'

