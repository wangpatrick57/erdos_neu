#!/Users/patrickwang/opt/anaconda3/bin/python
import sys
from run_clique import get_dataset, ALL_DATASET_NAMES
from torch_geometric.utils import to_undirected, to_networkx
from torch_geometric.data import Data
from  cut_utils import solve_gurobi_maxclique

def get_gurobi_ground_truth(testdata, sample_every=1):
    testdata = testdata.index_select(range(0, len(testdata), sample_every))
    test_data_clique = []

    for data in testdata:
        my_graph = to_networkx(Data(x=data.x, edge_index = data.edge_index)).to_undirected()

        start_time = time.time()
        cliqno, nodes = solve_gurobi_maxclique(my_graph)
        end_time = time.time()

        data.clique_number = cliqno
        test_data_clique.append(data)

    return test_data_clique

def check_highest_degree_max_clique(graph):
    highest_deg_node = sorted(graph.degree(), key=(lambda data : (-data[1], data[0])))[0][0]
    cliqno, nodes = solve_gurobi_maxclique(graph)
    return nodes[highest_deg_node] == 1.0

def check_hdmc_on_dataset(dataset):
    num_hdmc = 0

    for data in dataset:
        my_graph = to_networkx(Data(x=data.x, edge_index = data.edge_index)).to_undirected()

        if check_highest_degree_max_clique(my_graph):
            num_hdmc += 1

    return num_hdmc

if __name__ == '__main__':
    dataset_name = ALL_DATASET_NAMES[int(sys.argv[1])]
    dataset = get_dataset(dataset_name)
    num_hdmc = check_hdmc_on_dataset(dataset)
    print(f'{num_hdmc} / {len(dataset)} graphs with the highest degree node in the maximum clique')
