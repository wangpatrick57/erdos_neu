#!/Users/patrickwang/opt/anaconda3/bin/python
import sys
from dataset_utils import *
from torch_geometric.utils import to_undirected, to_networkx
from torch_geometric.data import Data
from cut_utils import solve_gurobi_maxclique

def check_highest_degree_max_clique(graph):
    nodes_sorted_by_deg = sorted(graph.degree(), key=(lambda data : (-data[1], data[0])))

    if len(nodes_sorted_by_deg) == 0:
        return None

    highest_deg_node = nodes_sorted_by_deg[0][0]
    cliqno, nodes = solve_gurobi_maxclique(graph)
    return nodes[highest_deg_node] == 1.0

def check_hdmc_on_dataset(dataset, progressive=False, stop_at=None):
    num_hdmc = 0
    num_not_hdmc = 0
    graphs = graphs_from_dataset(dataset)

    if progressive:
        graphs.sort(key=(lambda graph: graph.number_of_nodes()))

    for graph in graphs:
        res = check_highest_degree_max_clique(graph)

        if res == True:
            num_hdmc += 1
        elif res == False:
            num_not_hdmc += 1
        else:
            assert res == None

        if progressive:
            print(f'{num_hdmc} {num_hdmc + num_not_hdmc}')

        if stop_at != None and num_hdmc + num_not_hdmc >= stop_at:
            break

    return num_hdmc, num_not_hdmc

if __name__ == '__main__':
    dataset_name = sys.argv[1]
    stop_at = int(sys.argv[2])
    dataset = get_dataset(dataset_name)
    progressive = is_big(dataset_name)
    num_hdmc, num_not_hdmc = check_hdmc_on_dataset(dataset, progressive=progressive, stop_at=stop_at)
    print(f'{dataset_name}: {num_hdmc} / {num_not_hdmc + num_hdmc}')
