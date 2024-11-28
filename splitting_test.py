import pandas as pd
import numpy as np

from src.data import ( construct_mg_data, remove_zero_one_sum_rows)
from src.splits import find_minimal_covering_smiles
import os











# define tag
prop_tag = 'HCOM'
# load the data
df = pd.read_excel('data/external/'+prop_tag+'.xlsx', sheet_name='dippr')
df['Const_Value'] = df['Const_Value']/1e6
# extract the smiles of experimental data
lst_smiles = list(df[df['Data_Type'] == 'Experimental']['SMILES'])
# select the experimental data
df = df[df['SMILES'].isin(lst_smiles)].reset_index(drop=True)
# save the smiles for icas fragment generation
file_paths = [
    './data/interim/'+prop_tag+'_mg1.txt',
    './data/interim/'+prop_tag+'_mg2.txt',
    './data/interim/'+prop_tag+'_mg3.txt']

# Check if file exists and write only if it doesn't
for file_path in file_paths:
    if not os.path.exists(file_path):
        print(f"Writing to {file_path}...")
        with open(file_path, 'w') as f:
            f.write('\n'.join(lst_smiles))  # Write the list of strings, one per line
    else:
        print(f"{file_path} already exists, skipping.")

df_mg = construct_mg_data('HCOM', df)


# step0: remove all compounds where the sum of groups is 0 --> could not be segmented
columns = [str(i) for i in range(1, 425)]
df_mg = remove_zero_one_sum_rows(df_mg, columns)
# step 1: ensure that for each dataset, all available groups are available in a subset that is the training set
minimal_set, coverage = find_minimal_covering_smiles(df_mg)
df_train = df_mg[df_mg['SMILES'].isin(minimal_set)]
# step 2: check the percentage of current training data
ratio_train = df_train.shape[0]/df_mg.shape[0]


current_smiles = set(df_train['SMILES'])
available_df = df_mg[~df_mg['SMILES'].isin(current_smiles)].copy()

# step 3: fill up randomly until a quota is reached
target_ratio = 0.70
n_train = int(target_ratio*df_mg.shape[0])
n_train_to_be_added = n_train - df_train.shape[0] #target_size - len(subset_df)
n_val = (available_df.shape[0]-n_train_to_be_added)//2
n_test = available_df.shape[0]-n_train_to_be_added-n_val

#%%
from src.splits import expand_subset

df_train1, df_val1, df_test1 = expand_subset(df_mg, df_train, n_train, method='random', random_seed=42)
df_train2, df_val2, df_test2 = expand_subset(df_mg, df_train, n_train, method='butina', random_seed=42)
#
#
print(len(df_train1.index)+len(df_val1.index)+len(df_test1.index))
print(len(df_train2.index)+len(df_val2.index)+len(df_test2.index))
#%%
from src.model import linearize_gc, Fobj_fog, Fobj_sog, Fobj_tog, predict_gc
from scipy.optimize import least_squares
from src.evaluation import calculate_metrics

from src.splits import (find_nonzero_columns, split_indices)

# perform regression
df_train = df_train2
df_val = df_val2
df_test = df_test2

# guess of constant
n_const = 1
a0 = -100.00

# extract group ids
grp_idx = [str(i) for i in range(1, 425)]

# retrieve indices of available groups
idx_avail = find_nonzero_columns(df_train, ['SMILES', 'Const_Value', 'label', 'No'])
idx_mg1, idx_mg2, idx_mg3 = split_indices(idx_avail)

# split the X and Y for train and eval
X_train = df_train.loc[:, idx_avail].to_numpy() # all data
X1 = df_train.loc[:, idx_mg1].to_numpy() # first order occurrences
X2 = df_train.loc[:, idx_mg2].to_numpy() # second order occurrences
X3 = df_train.loc[:, idx_mg3].to_numpy() # third order occurrences
y_train = df_train.loc[:, 'Const_Value'].to_numpy() # target


X_val = df_val.loc[:, idx_avail].to_numpy()
X_test = df_test.loc[:, idx_avail].to_numpy()
y_val = df_val.loc[:, 'Const_Value'].to_numpy()
y_test = df_test.loc[:, 'Const_Value'].to_numpy()

# extract the number of available groups in each order
n_mg1 = X1.shape[1]
n_mg2 = X2.shape[1]
n_mg3 = X3.shape[1]
n_pars = n_mg1+n_mg2+n_mg3

# linearize according GC
y_lin = linearize_gc(y_train,a0,prop_tag)


# fine the initial guess using linear algebra
theta0 = np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(X_train),X_train)),np.transpose(X_train)),y_lin)


# insert the initial guess into the parameter vector
theta0 = np.insert(theta0, 0, a0)
# extract the contributions of each order
theta01 = theta0[:n_mg1 + n_const]
theta02 = theta0[n_mg1+n_const:n_mg1+n_const+n_mg2]
theta03 = theta0[n_mg1+n_const+n_mg2:]

# perform optimization using LM: step-wise
step_res1 = least_squares(Fobj_fog, theta01, jac='3-point', bounds=(-np.inf, np.inf), loss='soft_l1', f_scale=0.1,args=(X1, y_train, prop_tag, 'res'), verbose=2)

step_res2 = least_squares(Fobj_sog, theta02, jac='3-point', bounds=(-np.inf, np.inf), loss='soft_l1', f_scale=0.1, args=(X2, y_train, prop_tag, X1, step_res1.x, 'res'), verbose=2)

step_res3 = least_squares(Fobj_tog, theta03, jac='3-point', bounds=(-np.inf, np.inf), loss='soft_l1', f_scale=0.1, args=(X3, y_train, prop_tag, X1, step_res1.x, X2, step_res2.x, 'res'), verbose=2)

step_theta = np.insert(step_res2.x, 0, step_res1.x)
step_theta = np.insert(step_res3.x, 0, step_theta)

# perform optimization using LM: simultaneous
sim_res = least_squares(Fobj_fog, step_theta, jac='3-point', bounds=(-np.inf, np.inf), loss='soft_l1', f_scale=0.1, args=(X_train, y_train, prop_tag, 'res'), verbose=2)
sim_theta = sim_res.x

# perform predictions: stepwise
step_y_pred_train = predict_gc(step_theta , X_train, prop_tag)
step_y_pred_val = predict_gc(step_theta , X_val, prop_tag)
step_y_pred_test = predict_gc(step_theta , X_test, prop_tag)

# perform predictions: simultaneous
sim_y_pred_train = predict_gc(sim_theta , X_train, prop_tag)
sim_y_pred_val = predict_gc(sim_theta , X_val, prop_tag)
sim_y_pred_test = predict_gc(sim_theta , X_test, prop_tag)


# calculate the metrics
step_metrics_train = calculate_metrics(y_train, step_y_pred_train, n_pars)
step_metrics_val = calculate_metrics(y_val, step_y_pred_val, n_pars)
step_metrics_test = calculate_metrics(y_test, step_y_pred_test, n_pars)
step_metrics_all = calculate_metrics(np.concatenate((y_train, y_val, y_test), axis=0),
                                     np.concatenate((step_y_pred_train, step_y_pred_val, step_y_pred_test), axis=0))




# calculate the metrics simultaneous
sim_metrics_train = calculate_metrics(y_train, sim_y_pred_train, n_pars)
sim_metrics_val = calculate_metrics(y_val, sim_y_pred_val, n_pars)
sim_metrics_test = calculate_metrics(y_test, sim_y_pred_test, n_pars)
sim_metrics_all = calculate_metrics(np.concatenate((y_train, y_val, y_test), axis=0),
                                     np.concatenate((sim_y_pred_train, sim_y_pred_val, sim_y_pred_test), axis=0))

df_metrics = pd.DataFrame([step_metrics_train, sim_metrics_train, step_metrics_val, sim_metrics_val, step_metrics_test,
                           sim_metrics_test, step_metrics_all, sim_metrics_all])
df_metrics.insert(0, 'label',
                  ['train_step', 'train sim' ,'val_step', 'val sim' ,'test_step', 'test_sim', 'all_step', 'all_sim'])


