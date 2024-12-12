import pandas as pd
from rdkit import Chem

# load the data
property = 'Omega'
df = pd.read_excel('data/processed/'+property+'/'+property+'_processed.xlsx')

smis = df['SMILES']
formal_charges = []
num_Hs = []
atom_symbs = []
degress = []
#%%
for smi in smis:
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        formal_charges.append(atom.GetFormalCharge())
        atom_symbs.append(atom.GetSymbol())
        num_Hs.append(atom.GetTotalNumHs())
        degress.append(atom.GetTotalDegree())

print(set(num_Hs))
print(set(formal_charges))
print(set(atom_symbs))
print(set(degress))