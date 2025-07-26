########################################################################################################################
#                                                                                                                      #
#    Script for performing bootstrap uncertainty estimation on GC-MLP                                                   #
#       this can be done over multiple properties and ML models                                                        #
#                                                                                                                      #
#                                                                                                                      #
#                                                                                                                      #
#    Authors: Adem R.N. Aouichaoui                                                                                     #
#    2024/12/03                                                                                                        #
#                                                                                                                      #
########################################################################################################################
# python scripts\bootstrap_ml.py --property Pc Vc --model gpr svr --n_bootstrap 100 --path_2_data data --path_2_result results --path_2_model models
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
from src.ml_utils import  create_model
from src.ml_hyperopt import model_selector
from sklearn.preprocessing import StandardScaler
import json
from src.evaluation import calculate_metrics
import warnings
from optuna.exceptions import ExperimentalWarning
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch
from src.mlp_hyperopt import load_mlp_model_package
from src.evaluation import evaluate_mlp
import torch.nn.functional as F
# Suppress experimental warnings from Optuna
warnings.filterwarnings('ignore', category=ExperimentalWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)
from src.models.mlp import build_mlp

##########################################################################################################
# parsing arguments
##########################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--property', type=str, default='Pc', help='tag of the property of interest')
parser.add_argument('--model', type=str, default='mlp', help='name of ml model')
parser.add_argument('--n_bootstrap', type=int, default=100, help='number of bootstrap samples')
parser.add_argument('--path_2_data', type=str, default='data', help='path to the data')
parser.add_argument('--path_2_result', type=str, default='results', help='path for storing the results')
parser.add_argument('--path_2_model', type=str, default='models', help='path for storing the model')
parser.add_argument('--seed', type=int, default=42, help='seed for training')
parser.add_argument('--n_epochs', type=int, default=200, help='seed for training')

args = parser.parse_args()
seed_everything(args.seed)

##########################################################################################################
#%% Data Loading and preparation
##########################################################################################################
# construct the path to the data
path_2_data = args.path_2_data+'/processed/'+args.property+'/'+args.property+'_butina_min_processed.xlsx'
# reda the data
df = pd.read_excel(path_2_data)
# construct group ids
grp_idx = [str(i) for i in range(1, 425)]
# retrieve indices of available groups
idx_avail = grp_idx
# split the data
df_train = df[df['label']=='train']
df_val = df[df['label']=='val']
df_test = df[df['label']=='test']

# extract feature vectors and targets
X_train = df_train.loc[:,idx_avail].to_numpy()
y_train = {'true':df_train[args.property].to_numpy().reshape(-1,1)}

X_val = df_val.loc[:,idx_avail].to_numpy()
y_val = {'true':df_val[args.property].to_numpy().reshape(-1,1)}

X_test = df_test.loc[:,idx_avail].to_numpy()
y_test = {'true':df_test[args.property].to_numpy().reshape(-1,1)}

# scaling the data
y_scaler = StandardScaler()
y_train['scaled'] = y_scaler.fit_transform(y_train['true'])
y_val['scaled'] = y_scaler.transform(y_val['true'])
y_test['scaled'] = y_scaler.transform(y_test['true'])

# convert dataset into tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train['scaled'], dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val['scaled'], dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test['scaled'], dtype=torch.float32)

# construct tensor dataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# build the data loaders
train_loader = DataLoader(train_dataset, batch_size=600, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=600, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=600, shuffle=False)  # Create test dataloader

##########################################################################################################
#%% Fitting reference model
##########################################################################################################
# extract folder name
result_folder = None
if args.model == 'mlp':
    result_folder = {'Omega': 'rmse_0.067_26042025_0838',
                 'Tc': 'rmse_0.0315_26042025_0837',
                 'Pc': 'rmse_0.0573_26042025_0834',
                 'Vc': 'rmse_0.0273_26042025_0836'}



model_path = args.path_2_model+'/'+args.property+'/'+args.model+'/'+result_folder[args.property]
loaded = load_mlp_model_package(model_path)

best_model = loaded['model']
config = loaded['config']
y_scaler = loaded['scaler']


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


best_model.eval()
# perform predictions

train_pred, train_true, train_metrics = evaluate_mlp(best_model, train_loader, device, y_scaler)
val_pred, val_true, val_metrics = evaluate_mlp(best_model, val_loader, device, y_scaler)
test_pred, test_true, test_metrics = evaluate_mlp(best_model, test_loader, device, y_scaler)

# calculate metrics
# calculate the performance metric
metrics = {'val': val_metrics,
                 'test': test_metrics}
df_metrics = pd.DataFrame(metrics).T.reset_index()
df_metrics.columns = ['label', 'r2', 'rmse', 'mse', 'mare', 'mae']
df_metrics.loc[:,'n_boot'] = 0
df_metrics = df_metrics.reindex(columns = ['n_boot','label', 'r2', 'rmse', 'mse', 'mare', 'mae'])

print(df_metrics)

y_pred = np.vstack((train_pred, val_pred, test_pred))
df_predictions = df.loc[:,:'required'].copy()
df_predictions.loc[:,0] = y_pred

#########################################################################################################
#%% Construct boostrap and perform model fitting
#########################################################################################################

# calculate the errors
err = train_true - train_pred

# Initialize tensor
err_matrix = np.zeros((len(err), args.n_bootstrap))
rng = np.random.RandomState(args.seed)
# Populate matrix with random samples from the residual with replacements
for i in range(args.n_bootstrap):

    err_matrix[:, i] = np.random.choice(err.ravel(), size=len(err), replace=True)

# Build synthetic data matrix: prediction + column of res_matrix
synth_data = train_pred + err_matrix

# prepare list for dataframes
y_pred_list = []
# prepare save path for checkpoints:
save_path = os.path.join('checkpoints', f'{args.property}_succ_tr_best_state.pt')

# loop over the bootstrap samples
for i in range(args.n_bootstrap):
    # define the
    best_model = build_mlp(**loaded['config']['model_hyperparameters'])
    best_model.to(device)
    # extract the target values
    y_train['true'] = synth_data[:, i].reshape(-1, 1)
    # scale the targets
    scaler = StandardScaler()
    y_train['scaled'] = scaler.fit_transform(y_train['true'])
    y_val['scaled'] = scaler.transform(y_val['true'])
    y_test['scaled'] = scaler.transform(y_test['true'])
    # convert dataset into tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train['scaled'], dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val['scaled'], dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test['scaled'], dtype=torch.float32)

    # construct tensor dataset
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # build the data loaders
    train_loader = DataLoader(train_dataset, batch_size=600, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=600, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=600, shuffle=False)  # Create test dataloader

    training_params = config.get('training_params')
    optimizer = torch.optim.Adam(best_model.parameters(), lr=training_params.get('learning_rate', 0.001))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=training_params.get('lr_reduce', 0.7), patience=5,
                                                           min_lr=1e-6)

    best_val_loss = float('inf')
    best_state_dict = None
    patience = 30
    patience_counter = 0

    for epoch in range(args.n_epochs):
        # Training
        best_model.train()
        total_loss = 0
        for batch in train_loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            optimizer.zero_grad()
            pred = best_model(x)
            loss = F.mse_loss(pred, y.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        best_model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                y = batch[1].to(device)
                pred = best_model(x)
                loss = F.mse_loss(pred, y.view(-1, 1))
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Convert state dict tensors to lists for serialization
            # Save state dict to a separate file using trial number
            
            torch.save(best_model.state_dict(), save_path)
            patience_counter = 0


        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break
            # Print progress
        print(f'Epoch {epoch + 1:03d}, Train Loss: {total_loss / len(train_loader):.4f},'f'Val Loss: {val_loss / len(val_loader):.4f}')

    # load the best model
    best_model.load_state_dict(torch.load(save_path, weights_only=True))
    # set the model in evaluation model
    best_model.eval()
    # perform predictions
    train_pred, train_true, train_metrics = evaluate_mlp(best_model, train_loader, device, y_scaler)
    val_pred, val_true, val_metrics = evaluate_mlp(best_model, val_loader, device, y_scaler)
    test_pred, test_true, test_metrics = evaluate_mlp(best_model, test_loader, device, y_scaler)
    y_pred = np.vstack((train_pred, val_pred, test_pred))

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

# Check if the directory exists, if not, create it
path_2_result = 'results/'+args.property+'/'+args.model+'/'+result_folder[args.property]

path_results = path_2_result + '/bootstrap_predictions.xlsx'
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

print(path_results)


