########################################################################################################################
#                                                                                                                      #
#    Script for checking the available node features in order to reduce the size of features                           #
#                                                                                                                      #
#                                                                                                                      #
#                                                                                                                      #
#                                                                                                                      #
#    Authors: Adem R.N. Aouichaoui                                                                                     #
#    2024/12/03                                                                                                        #
#                                                                                                                      #
########################################################################################################################

##########################################################################################################
# import packages & load arguments
##########################################################################################################
import pandas as pd
from rdkit import Chem


##########################################################################################################
# Load the data
##########################################################################################################
# define the tag
property = 'Omega'
# load the data
df = pd.read_excel('data/processed/'+property+'/'+property+'_processed.xlsx')
# extract the smiles
smis = df['SMILES']
# initialize lists for features
formal_charges = []
num_Hs = []
atom_symbs = []
degress = []

# loop through the molecules and extract the features
for smi in smis:
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        formal_charges.append(atom.GetFormalCharge())
        atom_symbs.append(atom.GetSymbol())
        num_Hs.append(atom.GetTotalNumHs())
        degress.append(atom.GetTotalDegree())

# print uniques values for each feature
print(set(num_Hs))
print(set(formal_charges))
print(set(atom_symbs))
print(set(degress))