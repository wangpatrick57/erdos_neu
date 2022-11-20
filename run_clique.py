#!/Users/patrickwang/opt/anaconda3/bin/python
import torch
import torch.nn.functional as F
from torch.nn import Linear
from itertools import product
import time
from torch import tensor
from torch.optim import Adam
from torch.optim import SGD
from math import ceil
from torch.nn import Linear
from torch.distributions import categorical
from torch.distributions import Bernoulli
import torch.nn
from matplotlib import pyplot as plt
from torch_geometric.utils import convert as cnv
from torch_geometric.utils import sparse as sp
from torch_geometric.data import Data
from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx
from torch.utils.data.sampler import RandomSampler
from torch.nn.functional import gumbel_softmax
from torch.distributions import relaxed_categorical
import myfuncs
from torch_geometric.nn.inits import uniform
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Parameter
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.nn import GINConv, GATConv, global_mean_pool, NNConv, GCNConv
from torch.nn import Parameter
from torch.nn import Sequential as Seq, Linear, ReLU, LeakyReLU
from torch_geometric.nn import MessagePassing
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Batch
from torch_scatter import scatter_min, scatter_max, scatter_add, scatter_mean
from torch import autograd
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.utils import softmax, add_self_loops, remove_self_loops, segregate_self_loops, remove_isolated_nodes, contains_isolated_nodes, add_remaining_self_loops
from torch_geometric.utils import dropout_adj, to_undirected, to_networkx
from torch_geometric.utils import is_undirected
from cut_utils import get_diracs
import scipy
import scipy.io
from matplotlib.lines import Line2D
from torch_geometric.utils.convert import from_scipy_sparse_matrix
import GPUtil
from networkx.algorithms.approximation import max_clique
import pickle
from torch_geometric.nn import SplineConv, global_mean_pool, DataParallel
from torch_geometric.data import DataListLoader, DataLoader
from random import shuffle
from networkx.algorithms.approximation import max_clique
from networkx.algorithms import graph_clique_number
from networkx.algorithms import find_cliques
from torch_geometric.nn.norm import graph_size_norm
from torch_geometric.datasets import TUDataset
import visdom
from visdom import Visdom
import numpy as np
import matplotlib.pyplot as plt
from  cut_utils import solve_gurobi_maxclique
import gurobipy as gp
from gurobipy import GRB
from models import clique_MPNN
from torch_geometric.nn.norm.graph_size_norm import GraphSizeNorm
from modules_and_utils import decode_clique_final, decode_clique_final_speed

def set_random_seeds():
    torch.manual_seed(1)
    np.random.seed(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_all_dataset_names():
    return ['IMDB-BINARY']

def get_dataset(dataset_name):
    return TUDataset(root='datasets/', name=dataset_name)

def split_dataset(dataset):
    num_trainpoints = int(np.floor(0.6*len(dataset)))
    num_valpoints = int(np.floor(num_trainpoints/3))
    num_testpoints = len(dataset) - (num_trainpoints + num_valpoints)

    traindata= dataset[0:num_trainpoints]
    valdata = dataset[num_trainpoints:num_trainpoints + num_valpoints]
    testdata = dataset[num_trainpoints + num_valpoints:]

    return traindata, valdata, testdata

def train_model(dataset, traindata, valdata):
    numlayers = 5
    net = clique_MPNN(dataset, numlayers, 32, 32, 1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr_decay_step_size = 5
    lr_decay_factor = 0.95

    epochs = 3
    net.train()
    retdict = {}
    edge_drop_p = 0.0
    edge_dropout_decay = 0.90

    # this loops through all combinations of hyperparameters
    b_sizes = [32, 64]
    l_rates = [0.001]
    depths = [4]
    coefficients = [4.]
    rand_seeds = [66]
    widths = [64]

    for batch_size, learning_rate, numlayers, penalty_coeff, r_seed, hidden_1 in product(b_sizes, l_rates, depths, coefficients, rand_seeds, widths):
        torch.manual_seed(r_seed)

        train_loader = DataLoader(traindata, batch_size, shuffle=True)
        val_loader =  DataLoader(valdata, batch_size, shuffle=False)

        receptive_field= numlayers + 1
        val_losses = []
        cliq_dists = []

        #hidden_1 = 128
        hidden_2 = 1

        net = clique_MPNN(dataset,numlayers, hidden_1, hidden_2 ,1)
        net.to(device).reset_parameters()
        optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=0.00000)

        for epoch in range(epochs):
            totalretdict = {}
            count=0
            if epoch % 5 == 0:
                edge_drop_p = edge_drop_p*edge_dropout_decay
                print("Edge_dropout: ", edge_drop_p)

            if epoch % 10 == 0:
                penalty_coeff = penalty_coeff + 0.
                print("Penalty_coefficient: ", penalty_coeff)

            #learning rate schedule
            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_decay_factor * param_group['lr']

            #show currrent epoch and GPU utilizationss
            print('Epoch: ', epoch)
            GPUtil.showUtilization()

            net.train()
            for data in train_loader:
                count += 1
                optimizer.zero_grad(),
                data = data.to(device)
                data_prime = get_diracs(data, 1, sparse = True, effective_volume_range=0.15, receptive_field = receptive_field)

                data = data.to('cpu')
                data_prime = data_prime.to(device)

                retdict = net(data_prime, None, penalty_coeff)

                for key,val in retdict.items():
                    if "sequence" in val[1]:
                        if key in totalretdict:
                            totalretdict[key][0] += val[0].item()
                        else:
                            totalretdict[key] = [val[0].item(),val[1]]

                if epoch > 2:
                        retdict["loss"][0].backward()
                        #reporter.report()

                        torch.nn.utils.clip_grad_norm_(net.parameters(),1)
                        optimizer.step()
                        del(retdict)

            if epoch > -1:
                for key,val in totalretdict.items():
                    if "sequence" in val[1]:
                        val[0] = val[0]/(len(train_loader.dataset)/batch_size)
                del data_prime

if __name__ == '__main__':
    dataset = get_dataset(get_all_dataset_names()[0])
    traindata, valdata, testdata = split_dataset(dataset)
    train_model(dataset, traindata, valdata)
