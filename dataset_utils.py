#!/Users/patrickwang/opt/anaconda3/bin/python
import sys
from torch_geometric.utils import to_undirected, to_networkx
from torch_geometric.data import Data
from cut_utils import solve_gurobi_maxclique

# ordered from smallest to biggest in terms of subjective inflection point
ALL_DATASET_NAMES = ['IMDB-MULTI', 'IMDB-BINARY', 'ENZYMES', 'COLLAB', 'REDDIT-BINARY']

# the inflection point of node size where we split "big" and "small" for each dataset name
# check https://docs.google.com/spreadsheets/d/1e3rWa1f4L9ps8QpD3490XXpCaFpXke9UgzwrCOvnubw/edit?usp=sharing for node size progressions
DATASET_NAME_SIZE_INFLECTION_POINTS = {
    'IMDB-MULTI': 27,
    'IMDB-BINARY': 38,
    'ENZYMES': 57,
    'COLLAB': 133,
    'REDDIT-BINARY': 555,
}

DATASET_NORM_SAMPLE_EVERY = {
    'IMDB-MULTI': 1,
    'IMDB-BINARY': 1,
    'ENZYMES': 1,
    'COLLAB': 25, # 4516 total
    'REDDIT-BINARY': 160, # 1649 total
}

DATASET_BIG_SAMPLE_EVERY = {
    'IMDB-MULTI': 1,
    'IMDB-BINARY': 1,
    'ENZYMES': 1,
    'COLLAB': 20, # 484 total
    'REDDIT-BINARY': 30, # 351 total
}

def print_node_size_progression(dataset):
    node_sizes = []

    for data in dataset:
        my_graph = to_networkx(Data(x=data.x, edge_index = data.edge_index)).to_undirected()
        num_nodes = my_graph.number_of_nodes()
        node_sizes.append(num_nodes)

    print('\n'.join([f'{i} {size}' for i, size in enumerate(sorted(node_sizes))]))

if __name__ == '__main__':
    # dataset_name = ALL_DATASET_NAMES[int(sys.argv[1])]
    dataset_name = 'IMDB-MULTI'
    dataset = get_dataset(dataset_name)
    print_node_size_progression(dataset)
