#!/Users/patrickwang/opt/anaconda3/bin/python
import sys
from torch_geometric.utils import to_undirected, to_networkx
from torch_geometric.data import Data
from cut_utils import solve_gurobi_maxclique
from torch_geometric.datasets import TUDataset

# the inflection point of node size where we split "big" and "small" for each dataset name
# check https://docs.google.com/spreadsheets/d/1e3rWa1f4L9ps8QpD3490XXpCaFpXke9UgzwrCOvnubw/edit?usp=sharing for node size progressions
DATASET_NAME_SIZE_INFLECTION_POINTS = {
    'Fingerprint': 12,
    'Cuneiform': 26,
    'IMDB-MULTI': 27,
    'IMDB-BINARY': 38,
    'ENZYMES': 57,
    'MSRC_21': 100,
    'COLLAB': 133,
    'PROTEINS': 170,
    'REDDIT-BINARY': 555,
    'DD': 753,
    'REDDIT-MULTI-12K': 955,
    'REDDIT-MULTI-5K': 1339,
    'FIRSTMM_DB': 2257,
}

# ordered from smallest to biggest in terms of subjective inflection point
ALL_DATASET_NAMES = [name for name, size in sorted(DATASET_NAME_SIZE_INFLECTION_POINTS.items(), key=(lambda data: (data[1], data[0])))]

DATASET_NORM_SAMPLE_EVERY = {
    'IMDB-MULTI': 1,
    'IMDB-BINARY': 1,
    'ENZYMES': 1,
    'COLLAB': 60, # 903 total
    'REDDIT-BINARY': 32, # 329 total
}

DATASET_BIG_SAMPLE_EVERY = {
    'IMDB-MULTI': 1,
    'IMDB-BINARY': 1,
    'ENZYMES': 1,
    'COLLAB': 80, # 484 total
    'REDDIT-BINARY': 30, # 351 total
}

def is_big(dataset_name):
    return DATASET_NAME_SIZE_INFLECTION_POINTS[dataset_name] >= 200

def get_dataset(dataset_name):
    return TUDataset(root='datasets/', name=dataset_name)

def graphs_from_dataset(dataset):
    graphs = []

    for data in dataset:
        my_graph = to_networkx(Data(x=data.x, edge_index = data.edge_index)).to_undirected()
        graphs.append(my_graph)

    return graphs

def print_node_size_progression(dataset):
    node_sizes = []

    for data in dataset:
        my_graph = to_networkx(Data(x=data.x, edge_index = data.edge_index)).to_undirected()
        num_nodes = my_graph.number_of_nodes()
        node_sizes.append(num_nodes)

    print('\n'.join([f'{i} {size}' for i, size in enumerate(sorted(node_sizes))]))

if __name__ == '__main__':
    # dataset_name = ALL_DATASET_NAMES[int(sys.argv[1])]
    dataset_name = sys.argv[1]
    dataset = get_dataset(dataset_name)
    print_node_size_progression(dataset)
