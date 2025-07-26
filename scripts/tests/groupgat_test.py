##########################################################################################################
#                                                                                                        #
#    Script for fitting training graph neural networks                                                   #
#                                                                                                        #
#                                                                                                        #
#                                                                                                        #
#                                                                                                        #
#    Authors: Adem R.N. Aouichaoui                                                                       #
#    2024/12/03                                                                                          #
#                                                                                                        #
##########################################################################################################

##########################################################################################################
# import packages
##########################################################################################################
import argparse
import pandas as pd
from numpy.lib.shape_base import expand_dims
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from src.features import mol2graph, n_atom_features, n_bond_features
from torch_geometric.loader import DataLoader
import torch
from src.models.afp import FlexibleMLPAttentiveFP
import torch.nn.functional as F
from src.training import EarlyStopping
from lightning.pytorch import seed_everything
from src.evaluation import evaluate_gnn
from src.grape.utils import JT_SubGraph, DataSet


##########################################################################################################
# parsing arguments
##########################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--property', type=str, default='Tc', help='tag of the property of interest')
parser.add_argument('--model', type=str, default='afp', help='name of the GNN model')
parser.add_argument('--path_2_data', type=str, default='data/', help='path to the data')
parser.add_argument('--path_2_result', type=str, default='results/', help='path for storing the results')
parser.add_argument('--path_2_model', type=str, default='models/', help='path for storing the model')
parser.add_argument('--seed', type=int, default=42, help='Random state for reproducibility')

args = parser.parse_args()

property_tag = args.property
model_name = args.model
path_2_data = args.path_2_data
path_2_result = args.path_2_result
path_2_model = args.path_2_model
seed = args.seed


path_2_data = path_2_data+'processed/'+property_tag+'/'+property_tag+'_processed.xlsx'
path_2_result = path_2_result+ property_tag+'/gnn/'+model_name+'/'+property_tag+'_result.xlsx'
path_2_model = path_2_model+property_tag+'/gnn/'+model_name+'/'+property_tag

seed_everything(seed)
##########################################################################################################
#%% Data Loading & Preprocessing
##########################################################################################################
# read the data
df = pd.read_excel(path_2_data)
# split the data
df_train = df[df['label']=='train']
df_val = df[df['label']=='val']
df_test = df[df['label']=='test']
# construct a scaler
y_scaler = StandardScaler()
y_scaler.fit(df_train[args.property].to_numpy().reshape(-1,1))
# define fragmentation object for each of the folds
fragmentation_scheme = "data/MG_plus_reference.csv"
frag_save_path = 'data/processed/'+property_tag
print("initializing frag...")
train_fragmentation = JT_SubGraph(scheme=fragmentation_scheme, save_file_path=frag_save_path+'/train_frags.pth')
val_fragmentation = JT_SubGraph(scheme=fragmentation_scheme, save_file_path=frag_save_path+'/val_frags.pth')
test_fragmentation = JT_SubGraph(scheme=fragmentation_scheme, save_file_path=frag_save_path+'/test_frags.pth')
frag_dim = train_fragmentation.frag_dim
print("done.")
# construct the datasets
train_dataset = DataSet(df=df_train, smiles_column='SMILES', target_column=property_tag, global_features=None,
                        fragmentation=train_fragmentation, log=True, y_scaler=y_scaler)
val_dataset = DataSet(df=df_val, smiles_column='SMILES', target_column=property_tag, global_features=None,
                        fragmentation=val_fragmentation, log=True, y_scaler=y_scaler)
test_dataset = DataSet(df=df_test, smiles_column='SMILES', target_column=property_tag, global_features=None,
                        fragmentation=test_fragmentation, log=True, y_scaler=y_scaler)
# construct data loaders
train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

##########################################################################################################
#%% Model Training
##########################################################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from src.grape.models.AFP import AFP
from src.grape.models.GroupGAT import GCGAT_v4pro
net_params = {'node_in_dim': n_atom_features(),
              'edge_in_dim': n_bond_features(),
              'frag_dim': frag_dim,
              'global_features': False,
              'L1_hidden_dim': 128,
              'L1_layers_atom': 2,
              'L1_layers_mol': 2,
              'L1_dropout': 0.0,
              'L1_out_dim': 50,
              'L2_hidden_dim': 155,
              'L2_layers_atom': 2,
              'L2_layers_mol': 2,
              'L2_out_dim': 50,
              'L2_dropout': 0.0,
              'L3_hidden_dim': 64,
              'L3_layers_atom': 2,
              'L3_layers_mol': 2,
              'L3_out_dim': 50,
              'L3_dropout': 0.0,
              'MLP_layers': [40, 20],
              'num_heads': 1,
              'final_dropout': 0.05
              }



model = GCGAT_v4pro(net_params).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.00526, weight_decay=0.00003250012)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5)

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
        loss = F.mse_loss(pred, true, reduction='sum')
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
            val_loss += F.mse_loss(pred, true, reduction='sum').item()

    # Print progress
    print(f'Epoch {epoch+1:03d}, Train Loss: {total_loss/len(train_loader):.4f}, '
          f'Val Loss: {val_loss/len(val_loader):.4f}')

    # Learning rate scheduling
    scheduler.step(val_loss)

    # Early stopping
    early_stopping(val_loss, model, path_2_model + '_best.pt')
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

# Load best model
model.load_state_dict(torch.load(path_2_model + '_best.pt', weights_only=True))
model = model.to(device)



train_pred, train_true, train_metrics = evaluate_gnn(model, train_loader, device, y_scaler, tag='megnet')
val_pred, val_true, val_metrics = evaluate_gnn(model, val_loader, device, y_scaler, tag='megnet')
test_pred, test_true, test_metrics = evaluate_gnn(model, test_loader, device, y_scaler, tag='megnet')


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