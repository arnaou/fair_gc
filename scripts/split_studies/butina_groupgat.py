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

parser = argparse.ArgumentParser()
parser.add_argument('--property', type=str, default='Vc', help='tag of the property of interest')
parser.add_argument('--model', type=str, default='groupgat', help='name of ml model')
parser.add_argument('--path_2_data', type=str, default='data/', help='path to the data')
parser.add_argument('--path_2_result', type=str, default='results/', help='path for storing the results')
parser.add_argument('--path_2_model', type=str, default='models/', help='path for storing the model')
parser.add_argument('--seed', type=int, default=42, help='seed for training')
parser.add_argument('--n_epochs', type=int, default=500, help='seed for training')

args = parser.parse_args()
seed_everything(args.seed)


# construct the path to the data
path_2_data = args.path_2_data+'/processed/'+args.property+'/'+args.property+'_butina_processed.xlsx'
# reda the data
df = pd.read_excel(path_2_data)
# split the data
df_train = df[df['label']=='train']
df_val = df[df['label']=='val']
df_test = df[df['label']=='test']
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


result_folder = None
if args.model == 'afp':
    result_folder = {'Omega': 'rmse_0.132_15122024_1739',
                 'Tc': 'rmse_0.0108_15122024_1328',
                 'Pc': 'rmse_0.081_16122024_0136',
                 'Vc': 'rmse_0.00917_15122024_1525'}
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
config = loaded['config']

# set up the training configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# construct the model
model_class = get_class_from_path(config['model_module']+'.'+config['model_class'])
model = model_class(config['model_hyperparameters']).to(device)
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
        save_path = os.path.join('checkpoints', f'{args.property}_butina_best_state.pt')
        torch.save(model.state_dict(), save_path)
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
model.load_state_dict(torch.load(save_path, weights_only=True))
# set the model in evaluation model
model.eval()
# perform predictions
train_pred, train_true, train_metrics = evaluate_gnn(model, train_loader, device, y_scaler, tag=args.model)
val_pred, val_true, val_metrics = evaluate_gnn(model, val_loader, device, y_scaler, tag=args.model)
test_pred, test_true, test_metrics = evaluate_gnn(model, test_loader, device, y_scaler, tag=args.model)
y_pred = np.vstack((train_pred, val_pred, test_pred))
# calculate metrics
print(test_metrics)
