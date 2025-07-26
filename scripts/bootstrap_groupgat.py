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
import os
from src.ml_utils import  create_model
from src.ml_hyperopt import model_selector
from sklearn.preprocessing import StandardScaler
import json
from src.evaluation import calculate_metrics
import warnings
from optuna.exceptions import ExperimentalWarning
from sklearn.exceptions import ConvergenceWarning
from src.features import mol2graph
from torch_geometric.loader import DataLoader
from src.gnn_hyperopt import load_model_package, get_class_from_path
import torch
from src.evaluation import evaluate_gnn
import torch.nn.functional as F
from lightning import seed_everything
from rdkit import Chem
from src.grape.utils import JT_SubGraph, DataSet

# Suppress experimental warnings from Optuna
warnings.filterwarnings('ignore', category=ExperimentalWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)
##########################################################################################################
# parsing arguments --property 'Vc' --path_2_data 'data/' --path_2_result 'results/' --path_2_model 'models/'
##########################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--property', type=str, default='Omega', help='tag of the property of interest')
parser.add_argument('--model', type=str, default='groupgat', help='name of ml model')
parser.add_argument('--n_bootstrap', type=int, default=100, help='number of bootstrap samples')
parser.add_argument('--path_2_data', type=str, default='data/', help='path to the data')
parser.add_argument('--path_2_result', type=str, default='results/', help='path for storing the results')
parser.add_argument('--path_2_model', type=str, default='models/', help='path for storing the model')
parser.add_argument('--seed', type=int, default=42, help='seed for training')
parser.add_argument('--n_epochs', type=int, default=500, help='seed for training')

args = parser.parse_args()
seed_everything(args.seed)
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
path_2_data = args.path_2_data+'/processed/'+args.property+'/'+args.property+'_butina_min_processed.xlsx'
# reda the data
df = pd.read_excel(path_2_data)
# split the data
df_train = df[df['label']=='train']
df_val = df[df['label']=='val']
df_test = df[df['label']=='test']
df_train_min = df_train[df_train['required']==True]
idx_avail = [str(i) for i in range(1, 425)]
# construct scaler
y_scaler = StandardScaler()
y_scaler.fit(df_train[args.property].to_numpy().reshape(-1, 1))

# define fragmentation object for each of the folds
fragmentation_scheme = "data/MG_plus_reference.csv"
frag_save_path = 'data/processed/'+args.property
print("initializing frag...")
train_fragmentation = JT_SubGraph(scheme=fragmentation_scheme, save_file_path=frag_save_path+'/train_frags.pth')
val_fragmentation = JT_SubGraph(scheme=fragmentation_scheme, save_file_path=frag_save_path+'/val_frags.pth')
test_fragmentation = JT_SubGraph(scheme=fragmentation_scheme, save_file_path=frag_save_path+'/test_frags.pth')
frag_dim = train_fragmentation.frag_dim
print("done.")
# construct the datasets
train_dataset = DataSet(df=df_train, smiles_column='SMILES', target_column=args.property, global_features=None,
                        fragmentation=train_fragmentation, log=True, y_scaler=y_scaler)
val_dataset = DataSet(df=df_val, smiles_column='SMILES', target_column=args.property, global_features=None,
                        fragmentation=val_fragmentation, log=True, y_scaler=y_scaler)
test_dataset = DataSet(df=df_test, smiles_column='SMILES', target_column=args.property, global_features=None,
                        fragmentation=test_fragmentation, log=True, y_scaler=y_scaler)
# construct data loaders
train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
##########################################################################################################
#%% Fitting reference model
##########################################################################################################
# extract folder name
result_folder = None
if args.model == 'afp':
    result_folder = {'Omega': 'rmse_0.0531_26042025_1536',
                 'Tc': 'rmse_0.0118_26042025_1732',
                 'Pc': 'rmse_0.0258_26042025_1257',
                 'Vc': 'rmse_0.00747_26042025_1240'}
elif args.model == 'groupgat':
    result_folder = {'Omega': 'rmse_0.0461_26042025_0709',
                 'Tc': 'rmse_0.0115_27042025_0205',
                 'Pc': 'rmse_0.0325_25042025_1922',
                 'Vc': 'rmse_0.00761_26042025_0827'}
elif args.model == 'megnet':
    result_folder = {'Omega': 'rmse_0.0778_26042025_0827',
                 'Tc': 'rmse_0.041_26042025_0916',
                 'Pc': 'rmse_0.0669_26042025_1229',
                 'Vc': 'rmse_0.0956_26042025_1719'}


# construct the path to the model
path_2_model = 'models/'+args.property+'/gnn/'+args.model+'/'+result_folder[args.property]

# load the configs
loaded = load_model_package(path_2_model)

# load the model
model = loaded['model']
config = loaded['config']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.eval()
# perform predictions
train_pred, train_true, train_metrics = evaluate_gnn(model, train_loader, device, y_scaler, tag=args.model)
val_pred, val_true, val_metrics = evaluate_gnn(model, val_loader, device, y_scaler, tag=args.model)
test_pred, test_true, test_metrics = evaluate_gnn(model, test_loader, device, y_scaler, tag=args.model)

# calculate metrics
# calculate the performance metric
metrics = {'val': val_metrics,
                 'test': test_metrics}
df_metrics = pd.DataFrame(metrics).T.reset_index()
df_metrics.columns = ['label', 'r2', 'rmse', 'mse', 'mare', 'mae']
df_metrics.loc[:,'n_boot'] = 0
df_metrics = df_metrics.reindex(columns = ['n_boot','label', 'r2', 'rmse', 'mse', 'mare', 'mae'])

print(df_metrics)
#
# save the results
# y_true = np.vstack((train_true, val_true, test_true))
y_pred = np.vstack((train_pred, val_pred, test_pred))
df_predictions = df.loc[:,:'required'].copy()
df_predictions.loc[:,0] = y_pred
#
#
#########################################################################################################
#%% Construct boostrap and perform model fitting
#########################################################################################################

# calculate the errors
err = train_true - train_pred

# Initialize tensor
err_matrix = np.zeros((len(err), args.n_bootstrap))
# set bootstrap seed
rng = np.random.RandomState(args.seed)

# Populate matrix with random samples from the residual with replacements
for i in range(args.n_bootstrap):
    
    err_matrix[:, i] = rng.choice(err.ravel(), size=len(err), replace=True)

# Build synthetic data matrix: prediction + column of res_matrix
synth_data = train_pred + err_matrix

# prepare list for dataframes
y_pred_list = []
# loop over the bootstrap samples
for i in range(args.n_bootstrap):
    # extract the target values
    train_true = synth_data[:,i].reshape(-1, 1)
    # scale the targets
    y_scaler = StandardScaler()
    y_scaler.fit(train_true)
    # update the train set
    train_dataset.target = train_true
    # construct data loaders
    train_loader = DataLoader(train_dataset, batch_size=600, shuffle=False)
    # set up the training configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # construct the model
    model_class = get_class_from_path(config['model_module']+'.'+config['model_class'])
    model = model_class(**config['model_hyperparameters']).to(device)
    config = loaded['config']
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training_params']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=config['training_params']['lr_reduce'], patience=5,
                                                           min_lr=1e-6)

    best_val_loss = float('inf')
    best_state_dict = None
    patience = 25
    patience_counter = 0

    for epoch in range(args.n_epochs):
        # training
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = F.mse_loss(pred, batch.y.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                loss = F.mse_loss(pred, batch.y.view(-1, 1))
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Convert state dict tensors to lists for serialization
            # Save state dict to a separate file using trial number
            save_path = os.path.join('trash/checkpoints', f'{args.property}_boot_best_state.pt')
            torch.save(model.state_dict(), save_path)
            patience_counter = 0


        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break
            # Print progress
        print(f'Epoch {epoch + 1:03d}, Train Loss: {total_loss / len(train_loader):.4f},'f'Val Loss: {val_loss / len(val_loader):.4f}')

    # load the best model
    model.load_state_dict(torch.load(save_path, weights_only=True))
    # set the model in evaluation model
    model.eval()
    # perform predictions
    train_pred, train_true, train_metrics = evaluate_gnn(model, train_loader, device, y_scaler, tag=args.model)
    val_pred, val_true, val_metrics = evaluate_gnn(model, val_loader, device, y_scaler, tag=args.model)
    test_pred, test_true, test_metrics = evaluate_gnn(model, test_loader, device, y_scaler, tag=args.model)
    y_pred = np.vstack((train_pred, val_pred, test_pred))
    # calculate metrics
    # calculate the performance metric
    metrics = {'val': val_metrics,
                     'test': test_metrics}


    df_0 = pd.DataFrame(metrics).T.reset_index()
    df_0.columns = ['label', 'r2', 'rmse', 'mse', 'mare', 'mae']
    df_0.loc[:, 'n_boot'] = i+1
    df_0= df_0.reindex(columns = ['n_boot','label', 'r2', 'rmse', 'mse', 'mare', 'mae'])

    print(df_0)

    # update the metric dataframe
    df_metrics = pd.concat([df_metrics, df_0], ignore_index=True)

    # update prediction dataframe
    #df_predictions.loc[:,i+1] = y_pred
    y_pred_list.append(y_pred.ravel())


# update the prediction
df0 = pd.DataFrame(y_pred_list).T.reset_index(drop=True)
df0.columns = range(1, len(df0.columns) + 1)

df_predictions = pd.concat([df_predictions, df0], axis=1, ignore_index=False)


# ##########################################################################################################
# #%% save the results
# ##########################################################################################################

# construct the path to the model
path_results = args.path_2_result+args.property+'/gnn/'+args.model+'/'+result_folder[args.property]+'/bootstrap_predictions.xlsx'
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
