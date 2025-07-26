import time

import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import json
import os
import urllib
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--property', type=str, default='Pc', help='tag of the property of interest')
args = parser.parse_args()


path_2_data = 'data/processed/'+args.property+'/'+args.property+'_butina_min_processed.xlsx'
# read the data
df = pd.read_excel(path_2_data)


# get list of unique smiles
smi_list = df['SMILES'].unique()

# get Inchikey
mol_list = []
for smi in smi_list:
    try:
        mol_list.append(Chem.MolFromSmiles(smi))
    except:
        mol_list.append('')

inchikey_rdkit = []
for mol in mol_list:
    try:
        inchikey_rdkit.append(Chem.inchi.MolToInchiKey(mol))
    except:
        inchikey_rdkit.append('')

# download classification using inchikey
path_folder = 'data/processed/'+args.property+'/'+args.property+'_classes'
if not os.path.exists(path_folder):
    os.makedirs(path_folder)

missing_keys = False
path_report = 'missing_keys.txt'
report = open(path_report, 'w')


def print_report(string, file=report):
    file.write('\n' + string)

for i in tqdm(range(len(inchikey_rdkit))):
    key = inchikey_rdkit[i]
    print(key)
    url = 'https://cfb.fiehnlab.ucdavis.edu/entities/'+str(key)+'.json'
    try:
        with urllib.request.urlopen(url) as webpage:
            data = json.loads(webpage.read().decode())

        with open(path_folder + '/' + str(i) + '.json', 'w') as f:
            json.dump(data, f)
    except:
        print_report(str(i) + '    ' + str(key))
        missing_keys = True
        pass

    time.sleep(math.ceil(len(inchikey_rdkit)/12/60))

report.close()

if missing_keys:
    print('Some InChikeys were not available. Please check "Missing_ichikeys.txt" file.')
else:
    os.remove(path_report)
