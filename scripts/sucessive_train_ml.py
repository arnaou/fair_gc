########################################################################################################################
#                                                                                                                      #
#    Script for performing successive training on increasingly larger training dataset                                 #
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
parser.add_argument('--property', type=str, default='Omega', help='tag of the property of interest')
parser.add_argument('--model', type=str, default='gpr', help='name of ml model')
parser.add_argument('--path_2_data', type=str, default='data/', help='path to the data')
parser.add_argument('--path_2_result', type=str, default='results/', help='path for storing the results')
parser.add_argument('--path_2_model', type=str, default='models/', help='path for storing the model')
parser.add_argument('--seed', type=int, default=42, help='seed for training')

args = parser.parse_args()

list_of_model = ['dt', 'gb', 'gpr', 'rf','svr', 'xgb']
list_of_props = ['Pc', 'Tc', 'Vc','Omega',]
for mod in list_of_model:
    for prop in list_of_props:
        args.model = mod
        args.property = prop
        print(args.property)
        print(args.model)
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
        # construct the indices used for the training with increasing amount
        train_idx = {'1.0': df_train.index.to_list(),
                     '0.0': df_train_min.index.to_list()}
        remaining_idx = [idx for idx in train_idx['1.0'] if idx not in train_idx['0.0']]
        step = 0.05
        fractions = [round(x, 2) for x in np.arange(0, 1 + step, step)]
        for frac in fractions[1:-1]:

            idx_end = remaining_idx[int(frac*len(remaining_idx))]
            train_idx[str(frac)] = [i for i in range(0, idx_end)]
        ##########################################################################################################
        #%% Model construction and training
        ##########################################################################################################
        # construct the path to the model
        path_2_result = args.path_2_result+'/'+args.property+'/'+args.model+'/results.json'

        # Read the hyperparameters of the ML model
        with open(path_2_result, "r") as file:
            data = json.load(file)

        # construct the model
        model_class = model_selector(args.model)
        params = data['best_params']
        model = create_model(model_class, params, seed=None)

        # loop through the data
        metrics = {}
        for frac in fractions:
            frac = str(frac)
            # extract the training data
            X_train = df_train.loc[train_idx[frac],idx_avail].to_numpy()
            y_train = {'true':df_train.loc[train_idx[frac],args.property].to_numpy().reshape(-1,1)}

            # extract the validation data
            X_val = df_val.loc[:,idx_avail].to_numpy()
            y_val = {'true':df_val[args.property].to_numpy().reshape(-1,1)}

            # extrac the testing data
            X_test = df_test.loc[:,idx_avail].to_numpy()
            y_test = {'true':df_test[args.property].to_numpy().reshape(-1,1)}

            # scale the data
            scaler = StandardScaler()
            y_train['scaled'] = scaler.fit_transform(y_train['true'])
            y_val['scaled'] = scaler.transform(y_val['true'])
            y_test['scaled'] = scaler.transform(y_test['true'])

            # fit the mode
            model.fit(X_train, y_train['scaled'].ravel())

            # perform prediction
            y_val['pred'] = model.predict(X_val)
            y_val['pred'] = scaler.inverse_transform(y_val['pred'].reshape(-1, 1))
            y_test['pred'] = model.predict(X_test)
            y_test['pred'] = scaler.inverse_transform(y_test['pred'].reshape(-1,1))

            # calculate the performance metric
            metrics[frac] = {'val': calculate_metrics(y_val['true'], y_val['pred']),
                       'test': calculate_metrics(y_test['true'], y_test['pred'])}


        ##########################################################################################################
        #%% Save the results
        ##########################################################################################################
        # Flatten the dictionary into a list of rows for a DataFrame
        rows = []
        for frac, data in metrics.items():
            for label, metric_values in data.items():
                row = {"frac": float(frac), "label": label}
                row.update(metric_values)  # Add the r2, rmse, etc. to the row
                rows.append(row)

        # Create the DataFrame
        df_metrics = pd.DataFrame(rows)

        # Display the DataFrame
        print(df_metrics)

        # save the results
        df_metrics.to_excel(path_2_result.rsplit('/', 1)[0]+'/successive_training.xlsx')