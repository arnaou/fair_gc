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
import argparse
import os
import warnings

from hyperopt.rand import suggest
from lightning import seed_everything
from optuna.exceptions import ExperimentalWarning
import pandas as pd
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from src.features import mol2graph, n_atom_features, n_bond_features
from torch_geometric.loader import DataLoader
from src.evaluation import evaluate_gnn
from src.grape.utils import JT_SubGraph, DataSet
import torch
from src.gnn_hyperopt import load_model_package
# Suppress experimental warnings from Optuna
warnings.filterwarnings('ignore', category=ExperimentalWarning)



parser = argparse.ArgumentParser(description='Evaluation of GNN models')
parser.add_argument('--property', type=str, default='Pc', required=False, help='Tag for the property')
parser.add_argument('--model', type=str, required=False, default='groupgat', help='Model type to optimize (must be defined in config file)')
parser.add_argument('--path_2_data', type=str, default='data/', required=False, help='Path to the data file')
parser.add_argument('--path_2_model', type=str, required=False, default='models/', help='Path to save the model and eventual check points')


# load arguments
args = parser.parse_args()


##########################################################################################################
# Load the data & Preprocessing
##########################################################################################################

# import the data
path_to_data = args.path_2_data+'processed/'+args.property+'/'+args.property+'_butina_min_processed.xlsx'

df = pd.read_excel(path_to_data)

# split the data
df_train = df[df['label']=='train']
df_val = df[df['label']=='val']
df_test = df[df['label']=='test']
# construct a scaler
y_scaler = StandardScaler()
y_scaler.fit(df_train[args.property].to_numpy().reshape(-1,1))
if args.model in ['afp', 'megnet']:
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
elif args.model == 'groupgat':
    # define fragmentation object for each of the folds
    fragmentation_scheme = "data/MG_plus_reference.csv"
    frag_save_path = 'data/processed/' + args.property
    print("initializing frag...")
    train_fragmentation = JT_SubGraph(scheme=fragmentation_scheme, save_file_path=frag_save_path + '/train_frags.pth')
    val_fragmentation = JT_SubGraph(scheme=fragmentation_scheme, save_file_path=frag_save_path + '/val_frags.pth')
    test_fragmentation = JT_SubGraph(scheme=fragmentation_scheme, save_file_path=frag_save_path + '/test_frags.pth')
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

# Loading the models
result_folder = None
if args.model == 'afp':
    result_folder = {'Omega': 'rmse_0.0531_26042025_1536',
                     'Tc': 'rmse_0.0118_26042025_1732',
                     'Pc': 'rmse_0.0258_26042025_1257',
                     'Vc': 'rmse_0.00747_26042025_1240'}
elif args.model == 'megnet':
    result_folder = {'Omega': 'rmse_0.0778_26042025_0827',
                     'Tc': 'rmse_0.041_26042025_0916',
                     'Pc': 'rmse_0.0669_26042025_1229',
                     'Vc': 'rmse_0.0956_26042025_1719'}
elif args.model == 'groupgat':
    result_folder = {'Omega': 'rmse_0.0461_26042025_0709',
                     'Tc': 'rmse_0.0115_27042025_0205',
                     'Pc': 'rmse_0.0325_25042025_1922',
                     'Vc': 'rmse_0.00761_26042025_0827'}

# load the model
loaded = load_model_package('models/'+args.property+'/gnn/'+args.model+'/'+result_folder[args.property])
model = loaded['model']
config = loaded['config']
y_scaler = loaded['scaler']


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.eval()
train_pred, train_true, train_metrics = evaluate_gnn(model, train_loader, device, y_scaler, tag=args.model)
val_pred, val_true, val_metrics = evaluate_gnn(model, val_loader, device, y_scaler, tag=args.model)
test_pred, test_true, test_metrics = evaluate_gnn(model, test_loader, device, y_scaler, tag=args.model)


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