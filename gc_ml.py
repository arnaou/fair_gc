##########################################################################################################
#                                                                                                        #
#    Script for fitting group-contribution property models using support vector regression (SVR)         #
#    The groups used are based on the Marrero-Gani presented in:                                         #
#    https://doi.org/10.1016/j.fluid.2012.02.010                                                         #
#                                                                                                        #
#                                                                                                        #
#    Authors: Adem R.N. Aouichaoui                                                                       #
#    2024/12/03                                                                                          #
#                                                                                                        #
##########################################################################################################

##########################################################################################################
# import packages & load arguments
##########################################################################################################
# import packages and modules
import pandas as pd
import numpy as np
import os
from src.splits import find_nonzero_columns
from sklearn.preprocessing import StandardScaler
from src.optims import hypopt_parse_arguments, create_hyperparameter_optimizer, save_results
from src.model import predict_new_data
from src.evaluation import calculate_metrics
import warnings
from optuna.exceptions import ExperimentalWarning
from sklearn.exceptions import ConvergenceWarning


# Suppress experimental warnings from Optuna
warnings.filterwarnings('ignore', category=ExperimentalWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# load arguments
args = hypopt_parse_arguments()
##########################################################################################################
# Load the data & preprocessing
##########################################################################################################

# import the data
path_to_data = args.path_2_data+'processed/'+args.property+'/'+args.property+'_processed.xlsx'
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


##########################################################################################################
# Hyperparameter optimization
##########################################################################################################

study, best_model = create_hyperparameter_optimizer(
    config_path=args.config_file,
    model_name=args.model,
    property_name=args.property,
    study_name=args.study_name,
    X_train=X_train,
    y_train=y_train['scaled'],
    X_val=X_val,
    y_val=y_val['scaled'],
    metric_name=args.metric,
    sampler_name=args.sampler,
    n_trials=args.n_trials,
    storage = args.storage,
    load_if_exists=args.load_if_exists,
    n_jobs = -1,
    seed = args.seed,
)

##########################################################################################################
# Evaluate results
##########################################################################################################

# Save results and model
results = save_results(
    study=study,
    fitted_model = best_model,
    fitted_scaler=scaler,
    config_path=args.config_file,
    model_name=args.model,
    metric_name=args.metric or 'default',
    model_dir = args.path_2_model+args.property+'/'+args.model+'/',
    result_dir = args.path_2_result+args.property+'/'+args.model+'/',
    seed=args.seed
)

# predicting using the optimized model:

_, y_train['pred'] = predict_new_data(results['model_path'], X_train)
_, y_val['pred'] = predict_new_data(results['model_path'], X_val)
_, y_test['pred'] = predict_new_data(results['model_path'], X_test)


# construct dataframe to save the results
df_result = df.copy()

y_true = np.vstack((y_train['true'], y_val['true'], y_test['true']))
y_pred = np.hstack((y_train['pred'], y_val['pred'], y_test['pred']))
split_index = df_result.columns.get_loc('label') + 1
df_result.insert(split_index, 'pred', y_pred)

# calculate the metrics
metrics = {'train': calculate_metrics(y_train['true'], y_train['pred']),
           'val': calculate_metrics(y_val['true'], y_val['pred']),
           'test': calculate_metrics(y_test['true'], y_test['pred']),
           'all': calculate_metrics(y_true, y_pred)}
df_metrics = pd.DataFrame(metrics).T.reset_index()
df_metrics.columns = ['label', 'r2', 'rmse', 'mse', 'mare', 'mae']


# Check if the file exists, if not, create it with 'metrics' and 'prediction' sheets
prediction_path =  results['trials_path'].split('trials')[0]+'predictions.xlsx'
if not os.path.exists(prediction_path):
    with pd.ExcelWriter(prediction_path, mode='w', engine='openpyxl') as writer:
        df_metrics.to_excel(writer, sheet_name='metrics')
        df_result.to_excel(writer, sheet_name='prediction')
else:
    # If the file already exists, append the sheets
    with pd.ExcelWriter(prediction_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        df_metrics.to_excel(writer, sheet_name='metrics')
        df_result.to_excel(writer, sheet_name='prediction')

# Print results
print("\nOptimization Results:")
print("-" * 50)
print(f"metric: {args.metric}")
print(f"Best value: {study.best_value:.4f}")
print("\nBest parameters:")
for param, value in study.best_params.items():
    print(f"{param}: {value}")
print(f"\nModel saved to: {results['model_path']}")
print(f"Results saved to: {os.path.dirname(results['results_path'])}")
print(f"Trials saved to: {os.path.dirname(results['trials_path'])}")
print(f"Prediction and metrics saved to: {os.path.dirname(prediction_path)}")

# python gc_ml.py --property Omega --config_file model_config.yaml --model svr --n_trials 2000 --path_2_data data/ --path_2_result results/ --path_2_model models/ --seed 42