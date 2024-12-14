##########################################################################################################
#                                                                                                        #
#    Script for performing hyperparameter optimization of the AFP model                                  #
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

import warnings
from optuna.exceptions import ExperimentalWarning
import pandas as pd
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from src.features import mol2graph, n_atom_features, n_bond_features
from torch_geometric.loader import DataLoader
from src.gnn_hyperopt import gnn_hypopt_parse_arguments, NumpyEncoder, gnn_hyperparameter_optimizer, save_model_package
from src.evaluation import evaluate_gnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv
from src.training import EarlyStopping
from torch.nn import Sequential, Linear, ReLU


# Suppress experimental warnings from Optuna
warnings.filterwarnings('ignore', category=ExperimentalWarning)

# load arguments
args = gnn_hypopt_parse_arguments()

##########################################################################################################
# Load the data & Preprocessing
##########################################################################################################

# import the data
path_to_data = args.path_2_data+'processed/'+args.property+'/'+args.property+'_processed.xlsx'
df = pd.read_excel(path_to_data)
# split the data
df_train = df[df['label']=='train']
df_val = df[df['label']=='val']
df_test = df[df['label']=='test']
# construct a scaler
y_scaler = StandardScaler()
y_scaler.fit(df_train[args.property].to_numpy().reshape(-1,1))
# construct a column with the mol objects
df_train = df_train.assign(mol=[Chem.MolFromSmiles(i) for i in df_train['SMILES']])
df_val = df_val.assign(mol=[Chem.MolFromSmiles(i) for i in df_val['SMILES']])
df_test = df_test.assign(mol=[Chem.MolFromSmiles(i) for i in df_test['SMILES']])
# construct molecular graphs
train_dataset = mol2graph(df_train, 'mol', args.property, y_scaler=y_scaler)
val_dataset = mol2graph(df_val, 'mol', args.property, y_scaler=y_scaler)
test_dataset = mol2graph(df_test, 'mol', args.property, y_scaler=y_scaler)
# construct data loaders
train_loader = DataLoader(train_dataset, batch_size=600, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=600, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=600, shuffle=False)
#%%
import torch
import torch.nn.functional as F
from src.models import mpnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net(
#     num_features=n_atom_features(),
#     dim=64,
#     num_message_passing=5,
#     num_mlp_layers=3,
#     hidden_dim=128,
#     set2set_steps=4).to(device)

model = mpnn.MPNN(
    node_dim=n_atom_features(),
    edge_dim=n_bond_features(),
    hidden_dim=64,
    out_dim=1,
    num_message_passing=2,
    message_hidden_dim=100,
    set2set_steps=2,
    mlp_hidden_dims=[64, 32]  # Specify dimensions of MLP layers
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)

# Initialize early stopping
early_stopping = EarlyStopping(patience=25, verbose=True)

num_epochs = 150
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # model training
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        true = batch.y.view(-1, 1)
        loss = F.mse_loss(pred, true, reduction='mean')
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    # model validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model(batch)
            true = batch.y.view(-1, 1)
            val_loss += F.mse_loss(pred, true, reduction='mean').item()

    # Print progress
    print(f'Epoch {epoch+1:03d}, Train Loss: {total_loss/len(train_loader):.4f}, '
          f'Val Loss: {val_loss/len(val_loader):.4f}')

    # Learning rate scheduling
    scheduler.step(val_loss)

    # Early stopping
    early_stopping(val_loss, model, 'test_best.pt')
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

# Load best model
model.load_state_dict(torch.load('test_best.pt', weights_only=True))
model = model.to(device)



train_pred, train_true, train_metrics = evaluate_gnn(model, train_loader, device, y_scaler, tag='mpnn')
val_pred, val_true, val_metrics = evaluate_gnn(model, val_loader, device, y_scaler, tag='mpnn')
test_pred, test_true, test_metrics = evaluate_gnn(model, test_loader, device, y_scaler, tag='mpnn')


# Print metrics
print("\nTraining Set Metrics:")
for metric, value in train_metrics.items():
    print(f"{metric.upper()}: {value:.4f}")

print("\nValidation Set Metrics:")
for metric, value in val_metrics.items():
    print(f"{metric.upper()}: {value:.4f}")

print("\nTest Set Metrics:")
for metric, value in test_metrics.items():
    print(f"{metric.upper()}: {value:.4f}")
