########################################################################################################################
#                                                                                                                      #
#    Script for performing bootstrap uncertainty estimation                                                             #
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
import pandas as pd
import argparse
import numpy as np
from lightning import seed_everything
import os
from src.ml_utils import  create_model
from src.ml_hyperopt import model_selector
from sklearn.preprocessing import StandardScaler
import json
from src.evaluation import calculate_metrics
import warnings
from optuna.exceptions import ExperimentalWarning
from sklearn.exceptions import ConvergenceWarning


# Suppress experimental warnings from Optuna
warnings.filterwarnings('ignore', category=ExperimentalWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)
##########################################################################################################
# parsing arguments --property 'Vc' --path_2_data 'data/' --path_2_result 'results/' --path_2_model 'models/'
##########################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--property', type=str, default='Pc', help='tag of the property of interest')
parser.add_argument('--model', type=str, default='gpr', help='name of ml model')
parser.add_argument('--n_bootstrap', type=int, default=3, help='number of bootstrap samples')
parser.add_argument('--path_2_data', type=str, default='data/', help='path to the data')
parser.add_argument('--path_2_result', type=str, default='results/', help='path for storing the results')
parser.add_argument('--path_2_model', type=str, default='models/', help='path for storing the model')
parser.add_argument('--seed', type=int, default=42, help='seed for training')

args = parser.parse_args()

# list_of_model = ['dt', 'gb', 'gpr', 'rf','svr', 'xgb']
# list_of_props = ['Pc', 'Tc', 'Vc','Omega',]
# for mod in list_of_model:
#     for prop in list_of_props:
#         args.model = mod
#         args.property = prop
#         print(args.property)
#         print(args.model)
##########################################################################################################
#%% Data Loading and preparation
##########################################################################################################
# construct the path to the data
path_2_data = args.path_2_data+'/processed/'+args.property+'/'+args.property+'_processed.xlsx'
# reda the data
df = pd.read_excel(path_2_data)
# split the data
df_train = df[df['label']=='train']
df_val = df[df['label']=='val']
df_test = df[df['label']=='test']
df_train_min = df_train[df_train['required']==True]
idx_avail = [str(i) for i in range(1, 425)]
# extract the X,Y and splits
X_train = df_train.loc[:,idx_avail].to_numpy()
y_train = {'true':df_train[args.property].to_numpy().reshape(-1,1)}

#X_val = df_val.loc[:,'1':].to_numpy()
X_val = df_val.loc[:,idx_avail].to_numpy()
y_val = {'true':df_val[args.property].to_numpy().reshape(-1,1)}

#X_test = df_test.loc[:,'1':].to_numpy()
X_test = df_test.loc[:,idx_avail].to_numpy()
y_test = {'true':df_test[args.property].to_numpy().reshape(-1,1)}

# scale the data
scaler = StandardScaler()
y_train['scaled'] = scaler.fit_transform(y_train['true'])
y_val['scaled'] = scaler.transform(y_val['true'])
y_test['scaled'] = scaler.transform(y_test['true'])

##########################################################################################################
#%% Fitting reference model
##########################################################################################################
# construct the path to the model
path_2_model= args.path_2_result+'/'+args.property+'/'+args.model+'/results.json'

# Read the hyperparameters of the ML model
with open(path_2_model, "r") as file:
    data = json.load(file)

# construct the model
model_class = model_selector(args.model)
params = data['best_params']
model = create_model(model_class, params, seed=args.seed)

# fit the mode
model.fit(X_train, y_train['scaled'].ravel())

# perform prediction
y_train['pred'] = model.predict(X_train)
y_train['pred'] = scaler.inverse_transform(y_train['pred'].reshape(-1, 1))
y_val['pred'] = model.predict(X_val)
y_val['pred'] = scaler.inverse_transform(y_val['pred'].reshape(-1, 1))
y_test['pred'] = model.predict(X_test)
y_test['pred'] = scaler.inverse_transform(y_test['pred'].reshape(-1, 1))

y_true = np.vstack((y_train['true'], y_val['true'], y_test['true']))
y_pred = np.vstack((y_train['pred'], y_val['pred'], y_test['pred']))

# calculate the metrics
metrics = {'train': calculate_metrics(y_train['true'], y_train['pred']),
           'val': calculate_metrics(y_val['true'], y_val['pred']),
           'test': calculate_metrics(y_test['true'], y_test['pred']),
           'all': calculate_metrics(y_true, y_pred)}
df_metrics = pd.DataFrame(metrics).T.reset_index()
df_metrics.columns = ['label', 'r2', 'rmse', 'mse', 'mare', 'mae']
df_metrics.loc[:,'n_boot'] = 0
df_metrics = df_metrics.reindex(columns = ['n_boot','label', 'r2', 'rmse', 'mse', 'mare', 'mae'])

print(df_metrics)

# save the results
df_predictions = df.loc[:,:'required'].copy()
df_predictions.loc[:,0] = y_pred


##########################################################################################################
#%% Construct boostrap and perfom model fitting
##########################################################################################################
        #
        # # construct the path to the model
        # path_2_result = args.path_2_result+'/'+args.property+'/'+args.model+'/'
        #
        # # calculate the errors
        # err = y_train['true'] - y_train['pred']
        #
        # # Initialize tensor
        # err_matrix = np.zeros((len(err), args.n_bootstrap))
        #
        # # Populate matrix with random samples from the residual with replacements
        # for i in range(args.n_bootstrap):
        #     seed_everything(args.seed+i)
        #     err_matrix[:, i] = np.random.choice(err.ravel(), size=len(err), replace=True)
        #
        # # Build synthetic data matrix: prediction + column of res_matrix
        # synth_data = y_train['pred'] + err_matrix
        #
        # # prepare list for daframes
        # y_pred_list = []
        # # loop over the bootstrap samples
        # for i in range(args.n_bootstrap):
        #     # extract the target values
        #     y_train['true'] = synth_data[:,i].reshape(-1, 1)
        #     # scale the targets
        #     scaler = StandardScaler()
        #     y_train['scaled'] = scaler.fit_transform(y_train['true'])
        #     #y_train['scaled'] = scaler.inverse_transform(y_train['true'])
        #     # fit the model
        #     model = create_model(model_class, params, seed=args.seed)
        #     model.fit(X_train, y_train['scaled'].ravel())
        #     # make predictions
        #     y_train['pred'] = model.predict(X_train)
        #     y_train['pred'] = scaler.inverse_transform(y_train['pred'].reshape(-1, 1))
        #     y_val['pred'] = model.predict(X_val)
        #     y_val['pred'] = scaler.inverse_transform(y_val['pred'].reshape(-1, 1))
        #     y_test['pred'] = model.predict(X_test)
        #     y_test['pred'] = scaler.inverse_transform(y_test['pred'].reshape(-1, 1))
        #
        #     y_true = np.vstack((y_train['true'], y_val['true'], y_test['true']))
        #     y_pred = np.vstack((y_train['pred'], y_val['pred'], y_test['pred']))
        #
        #     # calculate the metrics and update dataframe
        #     metrics = {'train': calculate_metrics(y_train['true'], y_train['pred']),
        #                'val': calculate_metrics(y_val['true'], y_val['pred']),
        #                'test': calculate_metrics(y_test['true'], y_test['pred']),
        #                'all': calculate_metrics(y_true, y_pred)}
        #     df_0 = pd.DataFrame(metrics).T.reset_index()
        #     df_0.columns = ['label', 'r2', 'rmse', 'mse', 'mare', 'mae']
        #     df_0.loc[:, 'n_boot'] = i+1
        #     df_0= df_0.reindex(columns = ['n_boot','label', 'r2', 'rmse', 'mse', 'mare', 'mae'])
        #
        #     print(df_0)
        #
        #     # update the metric dataframe
        #     df_metrics = pd.concat([df_metrics, df_0], ignore_index=True)
        #
        #     # update prediction dataframe
        #     #df_predictions.loc[:,i+1] = y_pred
        #     y_pred_list.append(y_pred.ravel())
        #
        #
        # # update the prediction
        # df0 = pd.DataFrame(y_pred_list).T.reset_index(drop=True)
        # df0.columns = range(1, len(df0.columns) + 1)
        #
        # df_predictions = pd.concat([df_predictions, df0], axis=1, ignore_index=False)
        #
        #
        # ##########################################################################################################
        # #%% Construct ensemble
        # ##########################################################################################################
        #
        #
        #
        # # Check if the directory exists, if not, create it
        # path_results = path_2_result+'bootstrap_predictions.xlsx'
        # os.makedirs(os.path.dirname(path_results), exist_ok=True)
        #
        # # Check if the file exists, if not, create it with 'metrics' and 'prediction' sheets
        # if not os.path.exists(path_results):
        #     with pd.ExcelWriter(path_results, mode='w', engine='openpyxl') as writer:
        #         df_metrics.to_excel(writer, sheet_name='metrics')
        #         df_predictions.to_excel(writer, sheet_name='prediction')
        # else:
        #     # If the file already exists, append the sheets
        #     with pd.ExcelWriter(path_results, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        #         df_metrics.to_excel(writer, sheet_name='metrics')
        #         df_predictions.to_excel(writer, sheet_name='prediction')
