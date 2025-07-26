import argparse
from prettytable import PrettyTable
from src.gnn_hyperopt import load_model_package

parser = argparse.ArgumentParser()
parser.add_argument('--property', type=str, default='Omega', help='tag of the property of interest')
parser.add_argument('--model', type=str, default='groupgat', help='name of ml model')
parser.add_argument('--path_2_model', type=str, default='models/', help='path for storing the model')


args = parser.parse_args()

def count_parameters(model, printing=False):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    if printing == True:
        print(table)
        print(f"Total Trainable Params: {total_params}")
    return total_params

result_folder = {'Omega': 'rmse_0.0461_26042025_0709',
             'Tc': 'rmse_0.0115_27042025_0205',
             'Pc': 'rmse_0.0325_25042025_1922',
             'Vc': 'rmse_0.00761_26042025_0827'}

# construct the path to the model
path_2_model = 'models/'+args.property+'/gnn/'+args.model+'/'+result_folder[args.property]

# load the configs
loaded = load_model_package(path_2_model)

# load the model
model = loaded['model']
config = loaded['config']

count_parameters(model, printing=True)