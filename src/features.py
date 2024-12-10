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

# import packages
from rdkit import Chem
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch_geometric.loader import DataLoader




possible_atom_list = ['C','N','O','Cl','S','F','Br','I','Si','P', 'H']

possible_hybridization = [Chem.rdchem.HybridizationType.S,
                          Chem.rdchem.HybridizationType.SP,
                          Chem.rdchem.HybridizationType.SP2,
                          Chem.rdchem.HybridizationType.SP3]

possible_chiralities =[Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                       Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                       Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW]

possible_num_bonds = [0, 1, 2, 3, 4]

possible_formal_charge = [0, 1, -1]

possible_num_Hs  = [0, 1, 2, 3, 4]

possible_stereo  = [Chem.rdchem.BondStereo.STEREONONE,
                    Chem.rdchem.BondStereo.STEREOANY,
                    Chem.rdchem.BondStereo.STEREOZ,
                    Chem.rdchem.BondStereo.STEREOE,
                    Chem.rdchem.BondStereo.STEREOCIS,
                    Chem.rdchem.BondStereo.STEREOTRANS
                    ]


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]                               # Specify as Unknown
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    '''
    Get atom features
    '''
    Symbol = atom.GetSymbol()

    # Features
    Type_atom = one_of_k_encoding(Symbol, possible_atom_list)
    Ring_atom = [atom.IsInRing()]
    Aromaticity = [atom.GetIsAromatic()]
    Hybridization = one_of_k_encoding(atom.GetHybridization(), possible_hybridization)
    Bonds_atom = one_of_k_encoding(len(atom.GetNeighbors()), possible_num_bonds)
    Formal_charge = one_of_k_encoding(atom.GetFormalCharge(), possible_formal_charge)
    num_Hs = one_of_k_encoding(atom.GetTotalNumHs(), possible_num_Hs)
    Type_chirality = one_of_k_encoding(atom.GetChiralTag(), possible_chiralities)

    # Merge features in a list
    results = Type_atom + Ring_atom + Aromaticity + Hybridization + \
              Bonds_atom + Formal_charge + num_Hs + Type_chirality

    return np.array(results).astype(np.float32)


def get_bond_pair(mol):
    bonds = mol.GetBonds()
    res = [[], []]
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
    return res


def bond_features(bond):
    '''
    Get bond features
    '''
    bt = bond.GetBondType()

    type_stereo = one_of_k_encoding(bond.GetStereo(), possible_stereo)

    # Features
    bond_feats = [bt == Chem.rdchem.BondType.SINGLE,
                  bt == Chem.rdchem.BondType.DOUBLE,
                  bt == Chem.rdchem.BondType.TRIPLE,
                  bt == Chem.rdchem.BondType.AROMATIC,
                  bond.GetIsConjugated(),
                  bond.IsInRing()] + type_stereo
    return np.array(bond_feats).astype(np.float32)


def mol2torchdata(df, mol_column, target, y_scaler=None):
    '''
    Takes a molecule and return a graph
    '''
    graphs = []
    mols = df[mol_column].tolist()
    ys = df[target].tolist()
    for mol, y in zip(mols, ys):
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()

        # Information on nodes
        node_f = [atom_features(atom) for atom in atoms]

        # Information on edges
        edge_index = get_bond_pair(mol)
        edge_attr = []

        for bond in bonds:
            edge_attr.append(bond_features(bond))
            edge_attr.append(bond_features(bond))

        # Store all information in a graph
        nodes_info = torch.tensor(np.array(node_f), dtype=torch.float)
        edges_indx = torch.tensor(np.array(edge_index), dtype=torch.long)
        edges_info = torch.tensor(np.array(edge_attr), dtype=torch.float)

        graph = Data(x=nodes_info, edge_index=edges_indx, edge_attr=edges_info)

        if y_scaler is not None:
            y = np.array(y).reshape(-1, 1)
            y = y_scaler.transform(y).astype(np.float32)
            graph.y = torch.tensor(y[0], dtype=torch.float)
        else:
            # y = y.astype(np.float32)
            graph.y = torch.tensor(y, dtype=torch.float)

        graphs.append(graph)

    return graphs


def n_atom_features():
    atom = Chem.MolFromSmiles('CC').GetAtomWithIdx(0)
    return len(atom_features(atom))


def n_bond_features():
    bond = Chem.MolFromSmiles('CC').GetBondWithIdx(0)
    return len(bond_features(bond))


# class GraphScaler:
#     def __init__(self, scale_type='standard', node_features=True, edge_features=True, targets=True):
#         """
#         Initialize scalers for different components of the graph data.
#
#         Args:
#             scale_type: 'standard' for StandardScaler, 'minmax' for MinMaxScaler
#             node_features: Whether to scale node features
#             edge_features: Whether to scale edge features
#             targets: Whether to scale target values
#         """
#         self.scale_type = scale_type
#         self.node_features = node_features
#         self.edge_features = edge_features
#         self.targets = targets
#
#         # Initialize appropriate scalers
#         scaler_class = StandardScaler if scale_type == 'standard' else MinMaxScaler
#         self.node_scaler = scaler_class() if node_features else None
#         self.edge_scaler = scaler_class() if edge_features else None
#         self.target_scaler = scaler_class() if targets else None
#
#     def fit(self, dataset):
#         """
#         Fit scalers to the dataset.
#
#         Args:
#             dataset: PyG dataset or list of PyG Data objects
#         """
#         if self.node_features:
#             # Collect all node features across the dataset
#             node_features = torch.cat([data.x for data in dataset], dim=0).numpy()
#             self.node_scaler.fit(node_features)
#
#         if self.edge_features:
#             # Collect all edge features across the dataset
#             edge_features = torch.cat([data.edge_attr for data in dataset if hasattr(data, 'edge_attr')], dim=0).numpy()
#             self.edge_scaler.fit(edge_features)
#
#         if self.targets:
#             # Collect all targets across the dataset
#             targets = torch.cat([data.y.reshape(-1, 1) for data in dataset], dim=0).numpy()
#             self.target_scaler.fit(targets)
#
#     def transform(self, data):
#         """
#         Transform a single PyG Data object.
#
#         Args:
#             data: PyG Data object
#
#         Returns:
#             transformed_data: Transformed PyG Data object
#         """
#         transformed_data = Data()
#
#         # Copy all attributes from original data
#         for key in data.keys:
#             transformed_data[key] = data[key]
#
#         # Transform node features
#         if self.node_features and hasattr(data, 'x'):
#             transformed_data.x = torch.FloatTensor(
#                 self.node_scaler.transform(data.x.numpy())
#             )
#
#         # Transform edge features
#         if self.edge_features and hasattr(data, 'edge_attr'):
#             transformed_data.edge_attr = torch.FloatTensor(
#                 self.edge_scaler.transform(data.edge_attr.numpy())
#             )
#
#         # Transform targets
#         if self.targets and hasattr(data, 'y'):
#             transformed_data.y = torch.FloatTensor(
#                 self.target_scaler.transform(data.y.reshape(-1, 1)).flatten()
#             )
#
#         return transformed_data
#
#     def fit_transform(self, dataset):
#         """
#         Fit scalers to dataset and transform it.
#
#         Args:
#             dataset: PyG dataset or list of PyG Data objects
#
#         Returns:
#             transformed_dataset: List of transformed PyG Data objects
#         """
#         self.fit(dataset)
#         return [self.transform(data) for data in dataset]
#
#     def inverse_transform_targets(self, targets):
#         """
#         Inverse transform scaled target values.
#
#         Args:
#             targets: Tensor of scaled target values
#
#         Returns:
#             original_scale_targets: Tensor of original scale target values
#         """
#         if not self.targets:
#             return targets
#
#         return torch.FloatTensor(
#             self.target_scaler.inverse_transform(targets.reshape(-1, 1))
#         ).flatten()
#
#
# # Example usage:
# def scale_dataset(train_dataset, val_dataset, test_dataset):
#     """
#     Scale training, validation and test datasets.
#
#     Args:
#         train_dataset: Training dataset (PyG Dataset or list of Data objects)
#         val_dataset: Validation dataset
#         test_dataset: Test dataset
#
#     Returns:
#         scaled_train_dataset: Scaled training dataset
#         scaled_val_dataset: Scaled validation dataset
#         scaled_test_dataset: Scaled test dataset
#         scaler: Fitted GraphScaler object
#     """
#     # Initialize scaler
#     scaler = GraphScaler(
#         scale_type='standard',
#         node_features=True,
#         edge_features=True,
#         targets=True
#     )
#
#     # Fit on training data and transform all datasets
#     scaled_train_dataset = scaler.fit_transform(train_dataset)
#     scaled_val_dataset = [scaler.transform(data) for data in val_dataset]
#     scaled_test_dataset = [scaler.transform(data) for data in test_dataset]
#
#     return scaled_train_dataset, scaled_val_dataset, scaled_test_dataset, scaler