import os
import sys

from IPython.core.pylabtools import figsize

# append the src folder
gc_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(gc_dir)

import pandas as pd
import numpy as np
from src.gc_tools import predict_gc
from src.ml_utils import predict_new_data

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch
from src.mlp_hyperopt import load_mlp_model_package

from src.evaluation import evaluate_mlp
from src.gnn_hyperopt import load_model_package
from src.evaluation import evaluate_gnn
from torch_geometric.loader import DataLoader


# load data and prepare
df = pd.read_excel('data/processed/consistency_data.xlsx')
X = df.loc[:,1:].to_numpy()
smiles = df['SMILES'].to_list()

# prepare dicts for predictions
preds = {
    'Tc':
        {'GCM': {},
         'GPR': {},
         'SVR': {},
         'MLP': {},
         'GroupGAT': {}},
    'Pc':
        {'GCM': {},
         'GPR': {},
         'SVR': {},
         'MLP': {},
         'GroupGAT': {}},
    'Vc':
        {'GCM': {},
         'GPR': {},
         'SVR': {},
         'MLP': {},
         'GroupGAT': {}}
}

# perform prediction using GCM
gc = {'Tc': pd.read_excel('models/Tc/classical/Tc_MG_params_split.xlsx', usecols=['value_step']).fillna(0.0).to_numpy(),
      'Pc': pd.read_excel('models/Pc/classical/Pc_MG_params_split.xlsx', usecols=['value_step']).fillna(0.0).to_numpy(),
      'Vc': pd.read_excel('models/Vc/classical/Vc_MG_params_split.xlsx', usecols=['value_step']).fillna(0.0).to_numpy()}

preds['Tc']['GCM'] = predict_gc(gc['Tc'] , X, 'Tc')
preds['Pc']['GCM'] = predict_gc(gc['Pc'] , X, 'Pc')
preds['Vc']['GCM'] = predict_gc(gc['Vc'] , X, 'Vc')

# perform prediction using GPR
gpr = {'Tc':'models/Tc/gpr/gpr_rmse_0.163_24042025_2248_pipeline.joblib',
       'Pc':'models/Pc/gpr/gpr_rmse_0.262_24042025_2220_pipeline.joblib',
       'Vc':'models/Vc/gpr/gpr_rmse_0.173_24042025_2203_pipeline.joblib'}


preds['Tc']['GPR'] = predict_new_data(gpr['Tc'], X)[1]
preds['Pc']['GPR'] = predict_new_data(gpr['Pc'], X)[1]
preds['Vc']['GPR'] = predict_new_data(gpr['Vc'], X)[1]

# perform prediction using SVR
svr = {'Tc':'models/Tc/svr/svr_rmse_0.167_25042025_2141_pipeline.joblib',
       'Pc':'models/Pc/svr/svr_rmse_0.263_25042025_0919_pipeline.joblib',
       'Vc':'models/Vc/svr/svr_rmse_0.148_25042025_0248_pipeline.joblib'}


preds['Tc']['SVR'] = predict_new_data(svr['Tc'], X)[1]
preds['Pc']['SVR'] = predict_new_data(svr['Pc'], X)[1]
preds['Vc']['SVR'] = predict_new_data(svr['Vc'], X)[1]

#%%
# perform predictions using MLP
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.zeros(X_tensor.shape[0], dtype=torch.float32)
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=600, shuffle=False)

# a = load_mlp_model_package('models/Tc/mlp/rmse_0.0315_26042025_0837/')
mlp = {'Tc': load_mlp_model_package('models/Tc/mlp/rmse_0.0315_26042025_0837/')['model'],
       'Pc': load_mlp_model_package('models/Pc/mlp/rmse_0.0573_26042025_0834/')['model'],
       'Vc': load_mlp_model_package('models/Vc/mlp/rmse_0.0273_26042025_0836/')['model']}

scaler = {'Tc':load_mlp_model_package('models/Tc/mlp/rmse_0.0315_26042025_0837/')['scaler'],
          'Pc':load_mlp_model_package('models/Pc/mlp/rmse_0.0573_26042025_0834/')['scaler'],
          'Vc':load_mlp_model_package('models/Vc/mlp/rmse_0.0273_26042025_0836/')['scaler']}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
preds['Tc']['MLP'], _, _ = evaluate_mlp(mlp['Tc'], loader, device, scaler['Tc'])
preds['Pc']['MLP'], _, _ = evaluate_mlp(mlp['Pc'], loader, device, scaler['Pc'])
preds['Vc']['MLP'], _, _ = evaluate_mlp(mlp['Vc'], loader, device, scaler['Vc'])
#%%
from src.grape.utils import JT_SubGraph, DataSet
# predict using GroupGAT
# define fragmentation object for each of the folds
fragmentation_scheme = "data/MG_plus_reference.csv"
frag_save_path = {'Tc':'data/processed/Tc_frags.pth',
                  'Pc':'data/processed/Pc_frags.pth',
                  'Vc':'data/processed/Vc_frags.pth'}

print("initializing frag...")
fragmentation = {'Tc': JT_SubGraph(scheme=fragmentation_scheme, save_file_path=frag_save_path['Tc']),
                 'Pc': JT_SubGraph(scheme=fragmentation_scheme, save_file_path=frag_save_path['Pc']),
                 'Vc': JT_SubGraph(scheme=fragmentation_scheme, save_file_path=frag_save_path['Vc']),
}

model_folder = { 'Tc': 'models/Tc/gnn/groupgat/rmse_0.0115_27042025_0205',
                 'Pc': 'models/Pc/gnn/groupgat/rmse_0.0325_25042025_1922',
                 'Vc': 'models/Vc/gnn/groupgat/rmse_0.00761_26042025_0827'}
loaded = {'Tc':load_model_package(model_folder['Tc']),
          'Pc':load_model_package(model_folder['Pc']),
          'Vc':load_model_package(model_folder['Vc'])}

dataset = {'Tc':DataSet(df=df, smiles_column='SMILES', target_column='nC', global_features=None, fragmentation=fragmentation['Tc'], log=True, y_scaler=loaded['Tc']['scaler']),
           'Pc':DataSet(df=df, smiles_column='SMILES', target_column='nC', global_features=None, fragmentation=fragmentation['Pc'], log=True, y_scaler=loaded['Pc']['scaler']),
           'Vc':DataSet(df=df, smiles_column='SMILES', target_column='nC', global_features=None, fragmentation=fragmentation['Vc'], log=True, y_scaler=loaded['Vc']['scaler']),}


loader = {'Tc':DataLoader(dataset['Tc'], batch_size=1000, shuffle=False),
          'Pc':DataLoader(dataset['Pc'], batch_size=1000, shuffle=False),
          'Vc':DataLoader(dataset['Vc'], batch_size=1000, shuffle=False)}

frag_dim = fragmentation['Tc'].frag_dim
#%%
Tc_model = loaded['Tc']['model']
Pc_model = loaded['Pc']['model']
Vc_model = loaded['Vc']['model']
Tc_model.eval()
Pc_model.eval()
Vc_model.eval()
preds['Tc']['GroupGAT'] = evaluate_gnn(Tc_model, loader['Tc'], device, loaded['Tc']['scaler'], tag='groupgat')[0]
preds['Pc']['GroupGAT'] = evaluate_gnn(Pc_model, loader['Pc'], device, loaded['Pc']['scaler'], tag='groupgat')[0]
preds['Vc']['GroupGAT'] = evaluate_gnn(Vc_model, loader['Vc'], device, loaded['Vc']['scaler'], tag='groupgat')[0]

#%%
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 15})
plt.rcParams['image.cmap'] = 'Paired'
plt.rcParams['axes.facecolor'] = ('whitesmoke')
# Create the plot
plt.figure(figsize=(10, 6))

property = 'Tc'

# Plot each line with different color and label
plt.plot(df['nC'], preds[property]['GCM'], label='GCM')
plt.plot(df['nC'], preds[property]['GPR'], label='GPR')
plt.plot(df['nC'], preds[property]['SVR'], label='SVR')
plt.plot(df['nC'], preds[property]['MLP'], label='MLP')
plt.plot(df['nC'], preds[property]['GroupGAT'], label='GroupGAT')

# Add labels and title
plt.xlabel('number of Carbon atoms')
plt.ylabel('Tc [K]')
#plt.ylabel('Pc [bar]')
#plt.ylabel('Vc [L/mol]')
#plt.title('Trend of the Tc of linear alkanes')
#plt.title('Trend of the Pc of linear alkanes')
plt.title(f'Trend of the {property} of linear alkanes')
plt.xlim([0, 200])
plt.ylim([150, 2500])

# Add legend
plt.legend()

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Show plot
plt.tight_layout()
#plt.savefig('plot_Pc.png')
plt.savefig(f'plot_{property}.png')
#plt.savefig('plot_Tc.png')S
plt.show()


