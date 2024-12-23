






##########################################################################################################
# Load packages and modules
##########################################################################################################
from rdkit import Chem
import torch
import pandas as pd
import os
from torch_geometric.data import Data, HeteroData, Batch
from torch_geometric.utils import subgraph
import numpy as np
from collections.abc import Sequence
from typing import Union, Callable,  Any
import pickle
from numpy import ndarray
from torch_geometric.utils import dense_to_sparse
from rdkit.Chem import rdmolops, MolFromSmiles, MolToSmiles
from src.features import atom_features, bond_features, get_bond_pair
from torch import Tensor

##########################################################################################################
#%% JT utils
##########################################################################################################

def graph_2_frag(smiles, origin_graph, JT_subgraph):
    mol = Chem.MolFromSmiles(smiles)
    frag_graph_list, motif_graph, atom_mask, frag_flag = JT_subgraph.fragmentation(origin_graph, mol)
    return frag_graph_list, motif_graph, atom_mask, frag_flag


def find_edge_ids(edge_index, src_nodes, dst_nodes):
    # This assumes edge_index is 2 * num_edges
    edge_ids = []
    for src, dst in zip(src_nodes, dst_nodes):
        # Find indices where the source and destination match the provided nodes
        mask = (edge_index[0] == src) & (edge_index[1] == dst)
        edge_ids.extend(mask.nonzero(as_tuple=False).squeeze(1).tolist())
    return edge_ids


def add_edge(data, edge):
    """
    Add an edge to a PyTorch Geometric graph.
    Takes:
        data (torch_geometric.data.Data): The graph data object.
        edge (tuple) (src, dst)
    Returns:
        torch_geometric.data.Data: The updated graph data object with the new edge added.
    """
    # get the source and target nodes
    idx1, idx2 = edge
    new_edge = torch.tensor([[idx1], [idx2]], dtype=torch.long)

    if data.edge_index.device != new_edge.device:
        new_edge = new_edge.to(data.edge_index.device)

    data.edge_index = torch.cat([data.edge_index, new_edge], dim=1)
    return data


def remove_edges(data, to_remove: list[tuple[int, int]]):
    """
    Takes: PyG data object, list of pairs of nodes making edges to remove.
    Returns: Data with specified edges removed, including edge attributes.
    """
    edge_index = data.edge_index
    edge_attr = data.edge_attr

    # List to store indices of edges to keep
    keep_indices = []

    for i in range(edge_index.size(1)):
        src, tgt = edge_index[0, i].item(), edge_index[1, i].item()

        if (src, tgt) not in to_remove and (tgt, src) not in to_remove:  # removes both directions
            keep_indices.append(i)

    keep_indices = torch.tensor(keep_indices, dtype=torch.long)
    # filter edges and attr over mask
    new_edge_index = edge_index[:, keep_indices]
    new_edge_attr = edge_attr[keep_indices] if edge_attr is not None else None

    new_data = Data(x=data.x, edge_index=new_edge_index, edge_attr=new_edge_attr, num_nodes=data.num_nodes)
    return new_data


class JT_SubGraph(object):
    def __init__(self, scheme, save_file_path=None, verbose=True):
        # path = os.path.join('./env',
        #                     scheme + '.csv')  # change to your needs TODO: load from yaml or larger config of script where called
        data_from = os.path.realpath(scheme)
        df = pd.read_csv(data_from)
        pattern = df[['First-Order Group', 'SMARTs', 'Priority']].values.tolist()
        self.patterns = sorted(pattern, key=lambda x: x[2], reverse=False)
        self.frag_name_list = [x[0] for x in self.patterns]
        self.frag_dim = len(self.frag_name_list)
        self.save_file_path = save_file_path
        self.verbose = verbose

    def fragmentation(self, graph, mol, check_metadata=False):
        """
        Parameters:
        - graph: The input graph to fragment.
        - mol: The RDKit molecule object.
        - save_file_path: Optional; path to save/load the fragmentation result.
        Currently that logic is implemented in the `_prepare_frag` method of DataSet class (TODO: change this)
        - check_metadata: Optional; if True, checks fragment metadata before returning a loaded file.

        Returns:
        - frag_graph_list: List of fragment graphs (subgraphs resulting of fragmentation).
        - motif_graph: The "motif graph" (junction tree), encoding connectivity between fragments
        - atom_mask: for each fragment, a mask of atoms in the original molecule.
        - frag_flag: Fragment flags identifying fragments to nodes in the motif graph.
        """
        num_atoms = mol.GetNumAtoms()

        frag_graph, frag_flag, atom_mask, idx_tuples, frag_features = self.compute_fragments(mol, graph, num_atoms)
        num_motifs = atom_mask.shape[0]

        edge_index = torch.tensor([], dtype=torch.long)
        motif_graph = Data(edge_index=edge_index, )
        motif_graph.num_nodes = num_motifs
        _, idx_tuples, motif_graph = self.build_adjacency_motifs(atom_mask, idx_tuples, motif_graph)

        if frag_features.ndim == 1:
            frag_features = frag_features.reshape(-1, 1).transpose()

        motif_graph.x = torch.Tensor(frag_features)
        motif_graph.atom_mask = torch.Tensor(atom_mask)

        # if necessary
        edge_features = graph.edge_attr  # Assuming edge_attr stores the features
        add_edge_feats_ids_list = []

        for _, item in enumerate(idx_tuples):  # TODO: maybe needs error handling?
            es = find_edge_ids(graph.edge_index, [item[0]], [item[1]])
            add_edge_feats_ids_list.append(es)
        add_edge_feats_ids_list[:] = [i for sublist in add_edge_feats_ids_list for i in sublist]

        if num_atoms != 1:
            # Assuming a procedure to handle the features as necessary
            motif_edge_features = edge_features[add_edge_feats_ids_list, :]  # da same
            motif_graph.edge_attr = motif_edge_features
            frag_graph_list = self.rebuild_frag_graph(frag_graph, motif_graph, mol)
        else:
            frag_graph_list = self.rebuild_frag_graph(frag_graph, motif_graph, mol)

        return frag_graph_list, motif_graph, atom_mask, frag_flag

    def compute_fragments(self, mol, graph, num_atoms):
        clean_edge_index = graph.edge_index
        # graph.edge_index = add_self_loops(graph.edge_index)[0] # might make it slower: TODO: investigate #this part changes the self loops
        pat_list = []
        mol_size = mol.GetNumAtoms()
        num_atoms = mol.GetNumAtoms()
        for line in self.patterns:
            pat = Chem.MolFromSmarts(line[1])
            pat_list.append(list(mol.GetSubstructMatches(pat)))
            # if pat_list[-1] != []:
            # print("Pattern: ", line, " found in molecule")
        atom_idx_list = list(range(num_atoms))
        hit_ats = {}
        frag_flag = []  # List[str], len := #fragments
        prior_set = set()
        adj_mask = []
        atom_mask = []
        frag_features = []
        k = 0

        for idx, line in enumerate(self.patterns):
            key = line[0]
            frags = pat_list[idx]
            # print(frags)
            # remove all the nodes in the frag that might appear multiple times until they appear
            for i, item in enumerate(frags):
                item_set = set(item)  # set(int)
                new_frags = frags[:i] + frags[i + 1:]
                left_set = set(sum(new_frags, ()))
                if not item_set.isdisjoint(left_set):
                    frags = new_frags

            for frag in frags:  # frag:tuple in frags:List[Tuples]
                frag_set = set(frag)
                if not prior_set.isdisjoint(frag_set) or not frag_set:
                    continue
                ats = frag_set
                adjacency_origin = Chem.rdmolops.GetAdjacencyMatrix(mol)
                adj_mask.append(adjacency_origin.copy())
                atom_mask.append(torch.zeros((mol_size,)))
                frag_features.append(torch.tensor([float(key == s) for s in self.frag_name_list], dtype=torch.float))

                if key not in hit_ats.keys():
                    hit_ats[key] = np.asarray(list(ats))
                else:
                    hit_ats[key] = np.vstack((hit_ats[key], np.asarray(list(ats))))
                ignores = list(set(atom_idx_list) - set(ats))
                adj_mask[k][ignores, :] = 0
                adj_mask[k][:, ignores] = 0
                atom_mask[k][list(ats)] = 1
                frag_flag.append(key)
                k += 1
                prior_set.update(ats)

        # unknown fragments:
        unknown_ats = list(set(atom_idx_list) - prior_set)
        for i, at in enumerate(unknown_ats):
            if k == 0:
                if num_atoms == 1:
                    adjacency_origin = Chem.rdmolops.GetAdjacencyMatrix(mol)
                adj_mask = adjacency_origin
                atom_mask = np.zeros((1, mol_size))
            else:
                # adjacency_origin = Chem.rdmolops.GetAdjacencyMatrix(m)[np.newaxis, :, :]
                adj_mask.append(adjacency_origin.copy())
                atom_mask.append(torch.zeros((mol_size,)))
            if 'unknown' not in hit_ats.keys():
                hit_ats['unknown'] = np.asarray(at)
            else:
                hit_ats['unknown'] = np.append(hit_ats['unknown'],
                                               np.asarray(at))  # stack all unknown atoms into 1 thing
            ignores = list(set(atom_idx_list) - set([at]))

            if num_atoms != 1:
                adj_mask[k][ignores, :] = 0
                adj_mask[k][:, ignores] = 0

            atom_mask[k][at] = 1
            frag_flag.append('unknown')
            if num_atoms != 1:
                frag_features.append(np.asarray(list(map(lambda s: float('unknown' == s), self.frag_name_list))))
            else:
                frag_features = np.asarray(
                    list(map(lambda s: float('unknown' == s), self.frag_name_list)))  # convert to PyG
            k += 1
            # should be modified to only vstack at the end instead of in all the complex conditions
        #### end of preprocessing #####

        if k > 0:
            frag_features = np.asarray(frag_features)
            adj_mask = np.asarray(adj_mask)
            atom_mask = np.asarray(atom_mask)

        adjacency_fragments = adj_mask.sum(axis=0)

        idx1, idx2 = (adjacency_origin - adjacency_fragments).nonzero()

        idx_tuples = list(zip(idx1.tolist(), idx2.tolist()))  # the tuples to remove?
        # if bigraph is wanted it should be setup here
        frag_graph = remove_edges(graph, idx_tuples)
        graph.edge_index = clean_edge_index  # set the edge index back. Quick fix TODO: find a better way to count self loops instead
        return frag_graph, frag_flag, atom_mask, idx_tuples, frag_features

    def build_adjacency_motifs(self, atom_mask, idx_tuples, motif_graph):
        k = atom_mask.shape[0]
        duplicate_bond = []
        adjacency_motifs = np.zeros((k, k)).astype(int)
        motif_edge_begin = list(map(lambda x: self.atom_locate_frag(atom_mask, x[0]), idx_tuples))
        motif_edge_end = list(map(lambda x: self.atom_locate_frag(atom_mask, x[1]), idx_tuples))

        # eliminate duplicate bond in triangle substructure
        for idx1, idx2 in zip(motif_edge_begin, motif_edge_end):
            if adjacency_motifs[idx1, idx2] == 0:
                adjacency_motifs[idx1, idx2] = 1
                add_edge(motif_graph, (idx1, idx2))
            else:
                rm_1 = self.frag_locate_atom(atom_mask, idx1)
                rm_2 = self.frag_locate_atom(atom_mask, idx2)
                if isinstance(rm_1, int):
                    rm_1 = [rm_1]
                if isinstance(rm_2, int):
                    rm_2 = [rm_2]
                for i in rm_1:
                    for j in rm_2:
                        duplicate_bond.extend([tup for tup in idx_tuples if tup == (i, j)])
        if duplicate_bond:
            idx_tuples.remove(duplicate_bond[0])
            idx_tuples.remove(duplicate_bond[2])
        return adjacency_motifs, idx_tuples, motif_graph

    def atom_locate_frag(self, atom_mask, atom):
        return atom_mask[:, atom].tolist().index(1)

    def frag_locate_atom(self, atom_mask, frag):
        return atom_mask[frag, :].nonzero()[0].tolist()

    def rebuild_frag_graph(self, frag_graph, motif_graph, mol=None):
        if frag_graph.x is None:
            print("FRAG GRAPF X IS NONE !!!")
        num_motifs = motif_graph.num_nodes
        frag_graph_list = []

        for idx_motif in range(num_motifs):
            # Get the indices of nodes in this motif
            coord = motif_graph.atom_mask[idx_motif:idx_motif + 1, :].nonzero(as_tuple=True)[1]
            idx_list = coord.tolist()

            # Create new fragment graph as a subgraph of the original
            new_graph_edge_index, new_graph_edge_attr = subgraph(
                idx_list, frag_graph.edge_index, edge_attr=frag_graph.edge_attr, relabel_nodes=True,
                num_nodes=frag_graph.num_nodes,
            )

            new_node_features = frag_graph.x[idx_list] if frag_graph.x is not None else None

            new_frag_graph = Data(
                edge_index=new_graph_edge_index,
                edge_attr=new_graph_edge_attr,
                num_nodes=len(idx_list),
                x=new_node_features
                # explicitly passing nodes. TODO: unit test to make sure feats match with origin graph
            )
            frag_graph_list.append(new_frag_graph)

        return frag_graph_list

##########################################################################################################
#%% DataLoader utils
##########################################################################################################

class DataLoad(object):
    """The basic datasets-loading class. It holds the root as well as the raw  and processed files directories.

    """
    def __int__(self, root = None):
        if root is None:
            root = './data'
        self.root = root

    @property
    def raw_dir(self):
        """

        Returns: path

        """
        return os.path.join(self.root, 'raw')

    @property
    def processed_dir(self):
        """

        Returns: path

        """
        return os.path.join(self.root, 'processed')


def construct_dataset(smiles: list[str], target: Union[list[int], list[float], ndarray],
                      atom_features_fn: Callable[[object], Any] = atom_features,
                      bond_features_fn: Callable[[object], Any] = bond_features,
                      global_features=None) -> list[Data]:
    """Constructs a dataset out of the smiles and target lists based on the feature lists provided. The dataset will be
    a list of torch geometric Data objects, using their conventions.

    Parameters
    ------------
    smiles : list of str
        Smiles that are featurized and passed into a PyG DataSet.
    target: list of int or list of float or ndarray
        Array of values that serve as the graph 'target'.
    atom_features_fn : callable
        Function for featurizing the atoms
    bond_features_fn : callable
        Function for featurizing the Bonds
    global_features
        A list of global features matching the length of the SMILES or target. Default: None

    Returns
    --------
    list of Data
        list of Pytorch-Geometric Data objects

    """

    atom_featurizer = atom_features_fn

    bond_featurizer = bond_features_fn

    data = []

    for (smile, i) in zip(smiles, range(len(smiles))):

        # construct mol object
        mol = MolFromSmiles(smile)  # get into rdkit object

        # retrieve the atom and bon
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()

        # construct the node features/atom features
        node_f = [atom_features(atom) for atom in atoms]

        # construct the edges features/bond features
        edge_index = torch.tensor(get_bond_pair(mol))

        edge_attr = []

        for bond in bonds:
            edge_attr.append(bond_features(bond))
            edge_attr.append(bond_features(bond))

        # convert the information into tensors
        node_feats = torch.tensor(np.array(node_f), dtype=torch.float)
        edge_feats = torch.tensor(np.array(edge_attr), dtype=torch.float)

        # Prepare the target `y_value`
        y_value = torch.tensor(target[i], dtype=torch.float32)
        if y_value.ndim == 0:
            y_value = y_value.unsqueeze(0)  # Ensure `y_value` is at least 1-dimensional

        data_temp = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_feats, y=y_value)

        # TODO: Might need to be tested on multidim global feats
        if global_features is not None:
            global_feat_value = torch.tensor(global_features[i], dtype=torch.float32)
            if global_feat_value.ndim == 0:
                global_feat_value = global_feat_value.unsqueeze(0)  # Ensure it's at least 1-dimensional
            data_temp['global_feats'] = global_feat_value
        else:
            data_temp['global_feats'] = None
        # ------------------------------------------

        data.append(data_temp)

    return data  # actual PyG graphs

class DataSet(DataLoad):
    """A class that takes a path to a pickle file or a list of smiles and targets. The datasets is stored in
    Pytorch-Geometric Data instances and be accessed like an array.

    Heavily inspired by the PyTorch-Geometric Dataset class, especially indices and index_select.

    Parameters
    ------------
    file_path: str
        The path to a pickle file that should be loaded and the datasets therein used.
    smiles: list of str
        List of smiles to be made into a graph.
    target: list of in or float
        List of target values that will act as the graphs 'y'.
    allowed_atoms: list of str
        List of allowed atom symbols.
    only_organic: bool
        Checks if a molecule is ``organic`` counting the number of ``C`` atoms. If set to True, then molecules with less
        than one carbon will be discarded. Default: True
    atom_feature_list: list of str
        List of features to be applied. Default: All implemented features.
    bond_feature_list: list of str
        List of features that will be applied. Default: All implemented features.
    log: bool
        Decides if the filtering output and other outputs will be shown.
    fragmentation: instance of grape_chem.utils.junction_tree_utils.Fragmentation
        for now, a function that performs fragmentation
    custom_split: list (optional)
        The custom split array calculated on the dataset before filtering.
        it's passed in here to avoid recomputing the split after filtering. Default: None
    target_columns: list of str
        In the case of multi dimensional targets, columns that contain them. Default: None

    """

    def __init__(self, file_path: str = None, df: pd.DataFrame = None, smiles_column: str = None, target_column: str = None,
                 global_features: Union[list[float], ndarray] = None,  log: bool = False, root: str = None,
                 indices: list[int] = None, fragmentation=None, y_scaler=None):

        assert (file_path is not None) or (df is not None)

        super().__int__(root)

        allow_dupes = global_features is not None  # in the case of global features, there will be multiple observations for the same molecule

        if df is None:
            with open(file_path, 'rb') as handle:
                try:
                    df = pd.read_pickle(handle)
                    print('Loaded dataset.')
                except:
                    raise ValueError('A dataset is stored as a DataFrame.')

        # noinspection PyTypeChecker
        self.smiles = np.array(df[smiles_column])



        # noinspection PyTypeChecker
        self.target = np.array(df[target_column])

        # scale the data
        if y_scaler is not None:
            self.target = self.target.reshape(-1, 1)
            self.target = y_scaler.transform(self.target).astype(np.float32)

        self.global_features = None

        if global_features is not None:
            self.global_features = np.array(df[global_features])

        self.graphs = construct_dataset(smiles=self.smiles,
                                        target=self.target,
                                        global_features=self.global_features)


        for g in self.graphs:
            assert g.edge_index.shape[1] == g.edge_attr.shape[
                0], "Mismatch between edge_index and edge_attr dimensions"


        self._indices = indices
        self.num_node_features = self.graphs[0].num_node_features
        self.num_edge_features = self.graphs[0].num_edge_features
        self.data_name = None




        if fragmentation is not None:
            self.fragmentation = fragmentation
            self._prepare_frag_data()


    def save_dataset(self, filename: str = None):
        """Loads the dataset (specifically the smiles, target, global features and graphs) to a pandas DataFrame and
        dumps it into a pickle file. Saving and loading the same dataset is about 20x faster than recreating it from
        scratch.

        Parameters
        ------------
        filename: str
            Name of the datasets file. Default: 'dataset'


        """
        if filename is None:
            if self.data_name is None:
                filename = 'dataset'
            else:
                filename = self.data_name

        path = self.processed_dir

        if not os.path.exists(path):
            os.makedirs(path)

        df_save = pd.DataFrame(
            {'smiles': self.smiles, 'target': self.target, 'global_features': self.global_features,
             'graphs': self.graphs})

        path = os.path.join(path, filename + '.pickle')

        with open(path, 'wb') as handle:
            pickle.dump(df_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'File saved at: {path}')


    def _prepare_frag_data(self, log_progress=True, log_every=100):
        """
        return a list of frag_graphs and motif_graphs based on the fragmentation
        passed-in when initializing the datatset

        If the file path exists, it will attempt to load the data from there.
        Else it will compute fragmentation and attempt to save to the file path.
        TODO: could be made static
        """
        assert self.fragmentation is not None, 'fragmentation scheme and method must not be none to prepare frag data'
        if self.fragmentation.save_file_path is not None:
            try:
                if os.path.exists(self.fragmentation.save_file_path):
                    with open(self.fragmentation.save_file_path, 'rb') as saved_frags:
                        frag_data = pickle.load(saved_frags)
                        if len(frag_data) != len(self.smiles):
                            raise ValueError(
                                f"Fragmentation data does not match the number of smiles in the dataset. ")
                        for i in range(len(self.smiles)):
                            frag_graphs, motif_graph = frag_data[i]
                            self.graphs[i].frag_graphs = frag_graphs
                            self.graphs[i].motif_graphs = motif_graph
                        print(f"successfully loaded saved fragmentation at {self.fragmentation.save_file_path}")
                        return
            except Exception as e:
                print(
                    f"Warning: Could not load fragmentation data from {self.fragmentation.save_file_path}. Reason: {e}")

        ## - Actual computation - ##
        print("beginning fragmentation...")
        frag_data = []
        for i, s in enumerate(self.smiles):
            if log_progress:
                if (i + 1) % log_every == 0:
                    print(
                        'Currently performing fragmentation on molecule {:d}/{:d}'.format(i + 1, len(self.smiles)))
            frag_graphs, motif_graph, _, _ = graph_2_frag(s, self.graphs[i], self.fragmentation)
            if hasattr(motif_graph, 'atom_mask'):
                del motif_graph.atom_mask  # hacky, but necessary to avoid issues with pytorch geometric
            if frag_graphs is not None:
                self.graphs[i].frag_graphs = Batch.from_data_list(
                    frag_graphs)  # empty lists could cause issues. consider replacing with None in case list empty
                self.graphs[i].motif_graphs = motif_graph
                frag_data.append((self.graphs[i].frag_graphs, self.graphs[i].motif_graphs))
            else:
                frag_data.append((None, None))
                # ^to keep consistent with the length of the dataset, in case it contains singletons which won't produce frag and motif
        ## --- ##

        if self.fragmentation.save_file_path is not None:
            with open(self.fragmentation.save_file_path, 'wb') as save_frags:
                pickle.dump(frag_data, save_frags)

    def indices(self):
        return range(len(self.graphs)) if self._indices is None else self._indices

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return self.graphs[idx]
        else:
            return self.index_select(idx)

    def __iter__(self):
        for i in range(len(self.graphs)):
            yield self.graphs[i]


    def index_select(self, idx: object):
        r"""Creates a subset of the dataset from specified indices :obj:`idx`.
        Indices :obj:`idx` can be a slicing object, *e.g.*, :obj:`[2:5]`, a
        list, a tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type
        long or bool.

        Modified from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/dataset.html#Dataset

        Parameters
        ----------
            idx: obj
                Index list of datasets objects to retrieve.

        Returns
        -------
        list
            Python list of datasets objects.

        """

        indices = self.indices()

        if isinstance(idx, slice):
            start, stop, step = idx.start, idx.stop, idx.step
            # Allow floating-point slicing, e.g., dataset[:0.9]
            if isinstance(start, float):
                start = round(start * len(self))
            if isinstance(stop, float):
                stop = round(stop * len(self))
            idx = slice(start, stop, step)

            indices = indices[idx]

        elif isinstance(idx, Tensor) and idx.dtype == torch.long:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Tensor) and idx.dtype == torch.bool:
            idx = idx.flatten().nonzero(as_tuple=False)
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == bool:
            idx = idx.flatten().nonzero()[0]
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            indices = [indices[i] for i in idx]

        else:
            raise IndexError(
                f"Only slices (':'), list, tuples, torch.tensor and "
                f"np.ndarray of dtype long or bool are valid indices (got "
                f"'{type(idx).__name__}')")

        return [self[i] for i in indices]

