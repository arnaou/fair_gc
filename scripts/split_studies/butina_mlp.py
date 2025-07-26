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
from src.models.mlp import build_mlp
# Suppress experimental warnings from Optuna
warnings.filterwarnings('ignore', category=ExperimentalWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

##########################################################################################################
# parsing arguments
##########################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--property', type=str,  default='Omega', help='tag of the property of interest')
parser.add_argument('--model', type=str, default='mlp', help='name of ml model')
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
path_2_data = args.path_2_data+'/processed/'+args.property+'/'+args.property+'_butina_processed.xlsx'
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

# load the model architecture
result_folder = None
if args.model == 'mlp':
    result_folder = {'Omega': 'rmse_0.067_26042025_0838',
                 'Tc': 'rmse_0.0315_26042025_0837',
                 'Pc': 'rmse_0.0573_26042025_0834',
                 'Vc': 'rmse_0.0273_26042025_0836'}

model_path = args.path_2_model+'/'+args.property+'/'+args.model+'/'+result_folder[args.property]
loaded = load_mlp_model_package(model_path)


config = loaded['config']
y_scaler = loaded['scaler']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_model = build_mlp(**loaded['config']['model_hyperparameters'])

training_params = config.get('training_params')
optimizer = torch.optim.Adam(best_model.parameters(), lr=training_params.get('learning_rate', 0.001))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=training_params.get('lr_reduce', 0.7), patience=5,
                                                       min_lr=1e-6)

save_path = os.path.join('checkpoints', f'{args.property}_butina_mlp_state.pt')


best_model.to(device)

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
    print(
        f'Epoch {epoch + 1:03d}, Train Loss: {total_loss / len(train_loader):.4f},'f'Val Loss: {val_loss / len(val_loader):.4f}')

# load the best model
best_model.load_state_dict(torch.load(save_path, weights_only=True))
# set the model in evaluation model
best_model.eval()
train_pred, train_true, train_metrics = evaluate_mlp(best_model, train_loader, device, y_scaler)
val_pred, val_true, val_metrics = evaluate_mlp(best_model, val_loader, device, y_scaler)
test_pred, test_true, test_metrics = evaluate_mlp(best_model, test_loader, device, y_scaler)


print(test_metrics)