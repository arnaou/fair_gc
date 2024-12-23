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
f

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

model = AFP

model(node_in_dim = n_atom_features(), edge_in_dim = n_bond_features(), out_dim = 1,
            hidden_dim = 128, num_layers_atom = 3)


# def __init__(self, node_in_dim: int, edge_in_dim: int, out_dim: int = None, hidden_dim: int = 128,
#              num_layers_atom: int = 3, num_layers_mol: int = 3, dropout: float = 0.0,
#              regressor: bool = True, mlp_out_hidden: Union[int, list] = 512, rep_dropout: float = 0.0,
#              num_global_feats: int = 0):
