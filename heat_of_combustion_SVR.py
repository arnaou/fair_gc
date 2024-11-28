import pandas as pd
import numpy as np
import os
from src.data import (construct_mg_data, remove_zero_one_sum_rows, expand_subset)
from src.splits import (find_minimal_covering_smiles,
                      find_nonzero_columns, split_indices)
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score
from src.evaluation import calculate_metrics
from mealpy.evolutionary_based.GA import BaseGA
from mealpy.swarm_based.PSO import CL_PSO
from mealpy import FloatVar, Problem

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
# step 3: fill up randomly until a quota is reached
target_ratio = 0.70
n_train_tc = int(target_ratio*df_mg.shape[0])


# step 3: fill up randomly until a quota is reached
target_ratio = 0.70
n_train = int(target_ratio*df_mg.shape[0])
df_train, df_val, df_test = expand_subset(df_mg, df_train, n_train, random_seed=42)


df_final = pd.concat([df_train, df_val, df_test], ignore_index=True)


df_final.to_csv('data/processed/'+prop_tag+'.xlsx', index=False)

# retrieve indices of available groups
idx_avail = find_nonzero_columns(df_train, ['SMILES', 'Const_Value', 'label', 'No'])
idx_mg1, idx_mg2, idx_mg3 = split_indices(idx_avail)

#%% extract the data
X_train = df_train.loc[:,'1':].to_numpy()
y_train = df_train['Const_Value'].to_numpy().reshape(-1,1)

X_val = df_val.loc[:,'1':].to_numpy()
y_val = df_val['Const_Value'].to_numpy().reshape(-1,1)

X_test = df_test.loc[:,'1':].to_numpy()
y_test = df_test['Const_Value'].to_numpy().reshape(-1,1)

# scaling the data
scaler = StandardScaler()
y_train = scaler.fit_transform(y_train)
y_val = scaler.transform(y_val)
y_test = scaler.transform(y_test)


# Define the SVR optimization problem
class SVROptimization(Problem):
    def __init__(self, X_train, y_train, X_val, y_val, bounds=None, minmax="min", **kwargs):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, solution):
        try:
            # Round the kernel to nearest integer
            kernel_encoded = round(solution[0])
            kernel_encoded = min(max(kernel_encoded, 0), 3)

            # Get parameters
            c = solution[1]
            gamma = solution[2]
            epsilon = solution[3]

            kernel_decoded = ['linear', 'poly', 'rbf', 'precomputed'][kernel_encoded]

            svr = SVR(C=c, kernel=kernel_decoded, gamma=gamma, epsilon=epsilon)
            svr.fit(self.X_train, self.y_train.ravel())
            y_predict = svr.predict(self.X_val)
            fitness = -r2_score(self.y_val, y_predict)
            return fitness
        except Exception as e:
            return float('inf')


# Define bounds for parameters
bounds = [
    FloatVar(lb=0, ub=3),  # kernel (will be rounded)
    FloatVar(lb=10, ub=100),  # C
    FloatVar(lb=1e-10, ub=1.0),  # gamma
    FloatVar(lb=1e-3, ub=1.0),  # epsilon
]

# Create problem instance
problem = SVROptimization(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    bounds=bounds,
    minmax="min"
)

# Setup GA parameters
epoch = 50
pop_size = 20
pc = 0.95
pm = 0.05

# Create and run GA model
ga_model = BaseGA(epoch, pop_size, pc, pm)
best_solution = ga_model.solve(problem)









# df_metrics = pd.DataFrame([step_metrics_train, sim_metrics_train, step_metrics_val, sim_metrics_val, step_metrics_test,
#                            sim_metrics_test, step_metrics_all, sim_metrics_all])
# df_metrics.insert(0, 'label',
#                   ['train_step', 'train sim' ,'val_step', 'val sim' ,'test_step', 'test_sim', 'all_step', 'all_sim'])

#TODO
# fix the groups fro some reasons it is mixing up things
# redo results fom previous runs

# Source: https://medium.com/@i.v.shedrach/implementation-of-metaheuristic-algorithms-for-hyperparameter-tuning-using-support-vector-d701caf5d8fd