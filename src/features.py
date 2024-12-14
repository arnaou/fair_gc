##########################################################################################################
#                                                                                                        #
#    Collection of helper function and classes for featurization of molecular graphs for GNN             #
#                                                                                                        #
#                                                                                                        #
#                                                                                                        #
#                                                                                                        #
#    Authors: Adem R.N. Aouichaoui                                                                       #
#    2024/12/03                                                                                          #
#                                                                                                        #
##########################################################################################################

##########################################################################################################
# Importing packages
##########################################################################################################
from rdkit import Chem
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from rdkit.Chem import Descriptors

##########################################################################################################
# User Defined features
##########################################################################################################

# list of possible atoms
possible_atom_list = ['C','N','O','Cl','S','F','Br','I','Si','P','H']

# list of possible hybridization
possible_hybridization = [Chem.rdchem.HybridizationType.S,
                          Chem.rdchem.HybridizationType.SP,
                          Chem.rdchem.HybridizationType.SP2,
                          Chem.rdchem.HybridizationType.SP3
                          ]

# list of possible chirality centers
possible_chiralities =[Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                       Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                       Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW]

# list of possible bounds
possible_num_bonds = [0, 1, 2, 3, 4, 5, 6]

# list of possible charges
possible_formal_charge = [0, 1, -1]

# list of possible hydrogen attached
possible_num_Hs  = [0, 1, 2, 3, 4]

possible_stereo  = [Chem.rdchem.BondStereo.STEREONONE,
                    Chem.rdchem.BondStereo.STEREOANY,
                    Chem.rdchem.BondStereo.STEREOZ,
                    Chem.rdchem.BondStereo.STEREOE,
                    Chem.rdchem.BondStereo.STEREOCIS,
                    Chem.rdchem.BondStereo.STEREOTRANS]

##########################################################################################################
# Helper function
##########################################################################################################

def one_of_k_encoding(x, allowable_set):
    """
    function that performs one-hot encoding
    :param x: the feature
    :param allowable_set: list of allowed states
    :return:
    """
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """
    function that encodes unknown states as the last element in a feature vector
    :param x: the state
    :param allowable_set: the allowable states
    :return:
    """

    if x not in allowable_set:
        x = allowable_set[-1]                               # Specify as Unknown
    return list(map(lambda s: x == s, allowable_set))

##########################################################################################################
# Featurizer Functions
##########################################################################################################

def atom_features(atom):
    """
    Function for featurizing the atoms
    :param atom:
    :return:
    """
    # obtain the symbol of the atom
    Symbol = atom.GetSymbol()

    # Construct the features
    Type_atom = one_of_k_encoding(Symbol, possible_atom_list)
    Ring_atom = [atom.IsInRing()]
    Aromaticity = [atom.GetIsAromatic()]
    Hybridization = one_of_k_encoding(atom.GetHybridization(), possible_hybridization)
    Degrees = one_of_k_encoding(atom.GetTotalDegree(), possible_num_bonds)#one_of_k_encoding(len(atom.GetNeighbors()), possible_num_bonds)
    Formal_charge = one_of_k_encoding(atom.GetFormalCharge(), possible_formal_charge)
    num_Hs = one_of_k_encoding(atom.GetTotalNumHs(), possible_num_Hs)
    Type_chirality = one_of_k_encoding(atom.GetChiralTag(), possible_chiralities)
    atom_mass = [atom.GetMass() * 0.01]

    # Construct list of features
    results = (Type_atom + Ring_atom + Aromaticity + Hybridization +  Degrees + Formal_charge + num_Hs +
               Type_chirality + atom_mass)

    return np.array(results).astype(np.float32)



def get_bond_pair(mol):
    """
    Function for constructing the bonds
    :param mol:
    :return:
    """
    bonds = mol.GetBonds()
    res = [[], []]
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
    return res


def bond_features(bond):
    """
    Function for featurizing the bonds
    :param bond:
    :return:
    """

    # obtain the bond type
    bt = bond.GetBondType()
    # obtain the stereo isomery
    type_stereo = one_of_k_encoding(bond.GetStereo(), possible_stereo)

    # Construct the bond features
    bond_feats = [bt == Chem.rdchem.BondType.SINGLE,
                  bt == Chem.rdchem.BondType.DOUBLE,
                  bt == Chem.rdchem.BondType.TRIPLE,
                  bt == Chem.rdchem.BondType.AROMATIC,
                  bond.GetIsConjugated(),
                  bond.IsInRing()] + type_stereo
    return np.array(bond_feats).astype(np.float32)

def graph_features(mol):
    """
    Function for featurizing the graph with overall molecular information
    :param mol:
    :return:
    """
    # calculate the molecular weight
    mw = Descriptors.ExactMolWt(mol)

    return mw

def mol2graph(df, mol_column, target, global_feats=None, y_scaler=None):
    """
    Function that construct a molecular graph to be used for GNNs
    :param df: dataframe
    :param mol_column: column with mol objects
    :param global_feat: columns for global feature e.g. temperature or pressure
    :param target: column name with the target values
    :param y_scaler: scikit-learn scaler
    :return:
    """
    # initialize the list of graphs
    graphs = []
    # make a list of the mols
    mols = df[mol_column].tolist()
    # exact the target values
    ys = df[target].tolist()
    # iterate over each mol and construct the features
    for mol, y in zip(mols, ys):
        # retrieve the atom and bon
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()

        # construct the node features/atom features
        node_f = [atom_features(atom) for atom in atoms]

        # construct the edges features/bond features
        edge_index = get_bond_pair(mol)
        edge_attr = []

        for bond in bonds:
            edge_attr.append(bond_features(bond))
            edge_attr.append(bond_features(bond))

        # convert the information into tensors
        node_feats = torch.tensor(np.array(node_f), dtype=torch.float)
        edge_idx = torch.tensor(np.array(edge_index), dtype=torch.long)
        edge_feats = torch.tensor(np.array(edge_attr), dtype=torch.float)
        if global_feats is not None:
            global_feats = torch.tensor(global_feats, dtype=torch.float)

        # construct the data
        graph = Data(x=node_feats, edge_index=edge_idx, edge_attr=edge_feats)

        # scale the data
        if y_scaler is not None:
            y = np.array(y).reshape(-1, 1)
            y = y_scaler.transform(y).astype(np.float32)
            graph.y = torch.tensor(y[0], dtype=torch.float)
        else:
            graph.y = torch.tensor(y, dtype=torch.float)

        # append all graphs
        graphs.append(graph)

    return graphs


def n_atom_features():
    """
    Function for counting the atom/node features
    :return:
    """
    atom = Chem.MolFromSmiles('CC').GetAtomWithIdx(0)
    return len(atom_features(atom))


def n_bond_features():
    """
    function for counting the edge/bond features
    :return:
    """
    bond = Chem.MolFromSmiles('CC').GetBondWithIdx(0)
    return len(bond_features(bond))

