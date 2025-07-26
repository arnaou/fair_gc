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
# --property 'Vc' --path_2_data 'data/' --path_2_result 'results/' --path_2_model 'models/'
##########################################################################################################
# import packages
##########################################################################################################
import os
import sys

# append the src folder
gc_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(gc_dir)
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
# parsing arguments --property 'Vc' --path_2_data 'data/' --path_2_result 'results/' --path_2_model 'models/'
##########################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--property', type=str, default='Omega', help='tag of the property of interest')
parser.add_argument('--outlier_treatment', type=bool, default=False, help='should outliers be removed?')
parser.add_argument('--path_2_data', type=str, default='../data/', help='path to the data')
parser.add_argument('--path_2_result', type=str, default='../results/', help='path for storing the results')
parser.add_argument('--path_2_model', type=str, default='../models/', help='path for storing the model')
parser.add_argument('--split_type', type=str, default='random', help='path for storing the model')

args = parser.parse_args()


# define the various paths

path_2_data = args.path_2_data+'processed/'+args.property+'/'+args.property+'_'+args.split_type+'_processed.xlsx'

path_2_result = args.path_2_result+ args.property+'/classical/'+args.property+'_random_result.xlsx'
path_2_model = args.path_2_model+args.property+'/classical/'+args.property+'/'+'random'

##########################################################################################################
# Data Loading & Preprocessing
##########################################################################################################

excel_file = pd.ExcelFile(path_2_data)
sheet_nams = excel_file.sheet_names

#%%
list_test_mae = []
list_test_r2 = []
idx = 0
from operator import itemgetter
# this was done since in some cases the model would be initialized with values that are way too far from the optimal
if args.property == 'Pc':
    sheets = list(itemgetter(0, 2, 3)(sheet_nams))
elif args.property == 'Vc':
    sheets = list(itemgetter(2, 3, 4)(sheet_nams))
elif args.property == 'Omega':
    sheets = list(itemgetter(0,1,4 )(sheet_nams))
else:
    sheets = sheet_nams[:3]
for i in sheets:

    # read the data
    df = pd.read_excel(path_2_data, sheet_name=i)
    # construct list of columns indices
    columns = [str(i) for i in range(1, 425)]
    # remove zero columns
    df = remove_zero_one_sum_rows(df, columns)
    # construct group ids
    grp_idx = [str(i) for i in range(1, 425)]
    # retrieve indices of available groups
    idx_avail = find_nonzero_columns(df, ['SMILES', args.property, 'label', 'No', 'required', 'superclass'])
    # split the indices into 1st, 2nd and 3rd order groups
    idx_mg1, idx_mg2, idx_mg3 = split_indices(idx_avail)
    # extract the number of available groups in each order
    n_mg1 = len(idx_mg1)
    n_mg2 = len(idx_mg2)
    n_mg3 = len(idx_mg3)
    n_pars = n_mg1+n_mg2+n_mg3
    # split the data
    df_train = df[df['label'] == 'train']
    df_val = df[df['label'] == 'val']
    df_test = df[df['label'] == 'test']
    # extract feature vectors and targets for training
    X_train = df_train.loc[:, idx_avail].to_numpy()
    y_train = {'true': df_train[args.property].to_numpy()}
    # extract feature vectors and targets for validation
    X_val = df_val.loc[:, idx_avail].to_numpy()
    y_val = {'true': df_val[args.property].to_numpy()}
    # extract feature vectors and targets for testing
    X_test = df_test.loc[:, idx_avail].to_numpy()
    y_test = {'true': df_test[args.property].to_numpy()}


    ##########################################################################################################
    # GC modelling
    ##########################################################################################################
    # retrieve the number of constants and an initial guess for them
    n_const, const = retrieve_consts(args.property)
    # split feature vector based on the orders: training
    X1_train = X_train[:, :n_mg1]
    X2_train = X_train[:, n_mg1:n_mg1 + n_mg2]
    X3_train = X_train[:, -n_mg3:]
    # split feature vector based on the orders: val
    X1_val = X_val[:, :n_mg1]
    X2_val = X_val[:, n_mg1:n_mg1 + n_mg2]
    X3_val = X_val[:, -n_mg3:]
    # split feature vector based on the orders: training
    X1_test = X_test[:, :n_mg1]
    X2_test = X_test[:, n_mg1:n_mg1 + n_mg2]
    X3_test = X_test[:, -n_mg3:]
    # linearize according GC
    y_lin = linearize_gc(y_train['true'],const,args.property)
    # fine the initial guess using linear algebra
    if idx == 100:
        theta0 = theta_step
    else:
        theta0 = np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(X_train),X_train)),np.transpose(X_train)),y_lin)
        # insert the initial guess into the parameter vector
        theta0 = np.insert(theta0, 0, const)
    # extract the contributions of each order
    theta01 = theta0[:n_mg1 + n_const]
    theta02 = theta0[n_mg1+n_const:n_mg1+n_const+n_mg2]
    theta03 = theta0[n_mg1+n_const+n_mg2:]
    # perform step-wise parameter fitting
    step_res1 = least_squares(Fobj_fog, theta01, jac='3-point', bounds=(-np.inf, np.inf),loss='soft_l1',
                            args=(X1_train, y_train['true'], args.property, 'res'), verbose=0)

    step_res2 = least_squares(Fobj_sog, theta02, jac='3-point', bounds=(-np.inf, np.inf),loss='soft_l1',
                            args=(X2_train, y_train['true'], args.property, X1_train, step_res1.x, 'res'), verbose=0)

    step_res3 = least_squares(Fobj_tog, theta03, jac='3-point', bounds=(-np.inf, np.inf), loss='soft_l1',
                            args=(X3_train, y_train['true'], args.property, X1_train, step_res1.x, X2_train, step_res2.x, 'res'), verbose=0)
    # retrieve the contributions
    theta_step = np.insert(step_res2.x, 0, step_res1.x)
    theta_step = np.insert(step_res3.x, 0, theta_step)
    theta = {'step': theta_step}

    # perform optimization using LM: simultaneous
    # sim_res = least_squares(Fobj_fog, theta_step, jac='3-point', bounds=(-np.inf, np.inf), loss='soft_l1',
    #                        args=(X_train, y_train['true'], args.property, 'res'), verbose=2)
    # retrieve the contributions
    # theta_sim = sim_res.x
    # theta['sim'] = theta_sim
    # perform predictions using the step-wise approach
    y_train['step_pred'] = predict_gc(theta['step'] , X_train, args.property)
    y_val['step_pred'] = predict_gc(theta['step'], X_val, args.property)
    y_test['step_pred'] = predict_gc(theta['step'] , X_test, args.property)
    # perform predictions using the simultaneous approach
    # y_train['sim_pred'] = predict_gc(theta['sim'] , X_train, args.property)
    # y_test['sim_pred'] = predict_gc(theta['sim'] , X_test, args.property)

    # construct dataframe to save the results
    df_result = df.copy()
    y_true = np.hstack((y_train['true'],y_val['true'], y_test['true']))
    y_step_pred = np.hstack((y_train['step_pred'],y_val['step_pred'], y_test['step_pred']))
    # y_sim_pred = np.hstack((y_train['sim_pred'], y_test['sim_pred']))
    split_index = df_result.columns.get_loc('label') + 1
    df_result.insert(split_index, 'step_pred', y_step_pred)
    # df_result.insert(split_index + 1, 'sim_pred', y_sim_pred)

    # calculate the metrics
    metrics = {
        'step': {
            'train' : {},
            #'val': {},
            'test': {},
            'all': {}
        },
        # 'sim': {
        #     'train': {},
        #     'test': {},
        #     'all': {}
        # }
    }

    metrics['step']['train'] = calculate_metrics(y_train['true'], y_train['step_pred'])
    #metrics['sim']['train'] = calculate_metrics(y_train['true'], y_train['sim_pred'])
#
    #metrics['step']['val'] = calculate_metrics(y_val['true'], y_val['step_pred'])
#
    metrics['step']['test'] = calculate_metrics(y_test['true'], y_test['step_pred'])
    # metrics['sim']['test'] = calculate_metrics(y_test['true'], y_test['sim_pred'])

    #metrics['step']['all'] = calculate_metrics(y_true, y_step_pred)
    # metrics['sim']['all'] = calculate_metrics(y_true, y_sim_pred)
    list_test_mae.append(metrics['step']['test']['mae'])
    list_test_r2.append(metrics['step']['test']['r2'])

    flat_data = {
        f"{key1}_{key2}": values
        for key1, sub_dict in metrics.items()
        for key2, values in sub_dict.items()
    }

    # Convert the flattened dictionary to a DataFrame
    df_metrics = pd.DataFrame(flat_data).T.reset_index()
    df_metrics.columns = ['label', 'r2', 'rmse', 'mse', 'mare', 'mae']

    # print the results
    print(df_metrics)

    # Check if the directory exists, if not, create it
    os.makedirs(os.path.dirname(path_2_result), exist_ok=True)

    # Check if the file exists, if not, create it with 'metrics' and 'prediction' sheets
    if not os.path.exists(path_2_result):
        with pd.ExcelWriter(path_2_result, mode='w', engine='openpyxl') as writer:
            #df_metrics.to_excel(writer, sheet_name='metrics'+i)
            df_result.to_excel(writer, sheet_name='prediction'+i)
    else:
        # If the file already exists, append the sheets
        with pd.ExcelWriter(path_2_result, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            #df_metrics.to_excel(writer, sheet_name='metrics'+i)
            df_result.to_excel(writer, sheet_name='prediction'+i)

    idx += 1

    print(f'missed out on {(df_train.loc[:, idx_mg1].sum(axis=0) == 0).sum()} 1st order groups ')
    print(f'missed out on {(df_train.loc[:, idx_mg2].sum(axis=0) == 0).sum()} 2nd order groups ')
    print(f'missed out on {(df_train.loc[:, idx_mg3].sum(axis=0) == 0).sum()} 3rd order groups ')


# # save the model parameters
# pos = [int(i)-1 for i in idx_avail]
# nan_arr= np.full(424+n_const, np.nan)
# nan_arr[pos] = theta['step'][n_const:]
# coefs_lm = np.insert(nan_arr, 0, theta['step'][:n_const])
#
# # nan_arr= np.full(424, np.nan)
# # nan_arr[pos] = theta['sim'][n_const:]
# # coefs_lms = np.insert(nan_arr, 0, theta['sim'][:n_const])
#
#
# print(f'missed out on {(df_train.loc[:,idx_mg1].sum(axis=0)==0).sum()} 1st order groups ')
# print(f'missed out on {(df_train.loc[:,idx_mg2].sum(axis=0)==0).sum()} 2nd order groups ')
# print(f'missed out on {(df_train.loc[:,idx_mg3].sum(axis=0)==0).sum()} 3rd order groups ')
#
#
#
#
# # np.save(path_2_model+'_step_coefs',coefs_lm, allow_pickle=False)
# # np.save(path_2_model+'_sim_coefs',coefs_lms, allow_pickle=False)
# #
# #
# # # saving the contributions
# # # get the tags of the MG
# # with open(args.path_2_data+'MG_groups.txt', 'r') as file:
# #     MG_contribs = [line.strip() for line in file.readlines()]
# # # add the constants
# # MG_contribs  = [f"{args.property}_{i}" for i in range(n_const)] + MG_contribs
# #
# # # Create DataFrame with NaN values
# # df_params = pd.DataFrame({
# #     'groups': MG_contribs,
# #     'value_sim': np.nan,
# #     'value_step': np.nan
# # })
# # # retrieve the indices of the groups
# # idx_mg = list(map(int, idx_avail))
# # # add indices for the constants
# # new_indices = [i + (n_const-1) for i in idx_mg]
# # # construct dataframe for the parameters
# # df_params.loc[new_indices, 'value_sim'] = theta_sim[n_const:]
# # df_params.loc[:n_const-1, 'value_sim'] = theta_sim[:n_const]
# # df_params.loc[new_indices, 'value_step'] = theta_step[n_const:]
# # df_params.loc[:n_const-1, 'value_step'] = theta_step[:n_const]
# # # save the parameters to model folder
# # df_params.to_excel(path_2_model+'_MG_params_split.xlsx')
#
