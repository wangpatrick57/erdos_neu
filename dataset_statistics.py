#!/Users/patrickwang/opt/anaconda3/bin/python
import sys
from run_clique import get_dataset, ALL_DATASET_NAMES
from torch_geometric.utils import to_undirected, to_networkx
from torch_geometric.data import Data
from  cut_utils import solve_gurobi_maxclique

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
