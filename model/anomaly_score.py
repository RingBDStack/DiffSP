import torch
import networkx as nx
import numpy as np
from sklearn.linear_model import LinearRegression
# import pytz
# from datetime import datetime
import sys
import os
sys.path.append(os.path.abspath('./'))
# from parameter import parameter_parser
# from utils.loader import load_attack_data, load_non_attack_data
from torch_geometric.utils import to_networkx
import traceback


def get_feature(G, remove_isolate=False):
    # Feature dictionary in the format {node i's id: [Ni, Ei, Wi, λw,i]}
    featureDict = {}
    nodelist = list(G.nodes)
    for ite in nodelist:
        featureDict[ite] = []
        # The number of node i's neighbors
        Ni = G.degree(ite)
        featureDict[ite].append(Ni)
        # The set of node i's neighbors
        iNeighbor = list(G.neighbors(ite))
        # The number of edges in egonet i
        Ei = 0
        # Sum of weights in egonet i
        Wi = 0
        # The principal eigenvalue (the maximum eigenvalue with abs) of egonet i's weighted adjacency matrix
        Lambda_w_i = 0
        Ei += Ni
        egonet = nx.Graph()
        for nei in iNeighbor:
            Wi += 1  # Edge weight is set to 1
            egonet.add_edge(ite, nei, weight=1)
        iNeighborLen = len(iNeighbor)
        for it1 in range(iNeighborLen):
            for it2 in range(it1+1, iNeighborLen):
                # If it1 is in it2's neighbor list
                if iNeighbor[it1] in list(G.neighbors(iNeighbor[it2])):
                    Ei += 1
                    Wi += 1  # Edge weight is set to 1
                    egonet.add_edge(iNeighbor[it1], iNeighbor[it2], weight=1)
        # try:
        #     egonet_adjacency_matrix = nx.adjacency_matrix(egonet).todense()
        # except:
        #     print(egonet, ite)
        #     exit()
                # Create adjacency matrix if egonet is not empty
        if len(egonet.nodes) > 0:
            egonet_adjacency_matrix = nx.adjacency_matrix(egonet).todense()
            eigenvalue, eigenvector = np.linalg.eig(egonet_adjacency_matrix)
            eigenvalue.sort()
            Lambda_w_i = max(abs(eigenvalue[0]), abs(eigenvalue[-1]))
        else:
            Lambda_w_i = 0
        featureDict[ite].append(Ei)
        featureDict[ite].append(Wi)
        featureDict[ite].append(Lambda_w_i)

        if remove_isolate and (Ei == 0 or Ni == 0):
            print(f"Node {ite} has Ei={Ei} and Ni={Ni}. Removing this node.")
            del featureDict[ite]

    return featureDict


# feature dictionary which format is {node i's id:Ni, Ei, Wi, λw,i}
def regression(featureDict):
    N = []
    E = []
    for key in featureDict.keys():
        N.append(featureDict[key][0])
        E.append(featureDict[key][1])
    #E=CN^α => log on both sides => logE=logC+αlogN
    #regard as y=b+wx to do linear regression
    #here the base of log is 2
    y_train = np.log2(E)
    y_train = np.array(y_train)
    y_train = y_train.reshape(len(E), 1)
    x_train = np.log2(N)
    x_train = np.array(x_train)
    x_train = x_train.reshape(len(N), 1)
    model = LinearRegression()
    print('Fitting....')

    # Debug prints
    # if np.any(np.isinf(E)) or np.any(np.isnan(E)):
    #     print("x_train contains infinity or NaN values.")
    # if np.any(np.isinf(E)) or np.any(np.isnan(E)):
    #     print("y_train contains infinity or NaN values.")


    model.fit(x_train, y_train)
    print('Over....')

    # w = model.coef_[0][0]
    # b = model.intercept_[0]
    # C = 2**b
    # alpha = w
    
    # outlineScoreDict = {}
    # for key in featureDict.keys():
    #     yi = featureDict[key][1]
    #     xi = featureDict[key][0]
    #     outlineScore = (max(yi, C*(xi**alpha))/min(yi, C*(xi**alpha)))*np.log(abs(yi-C*(xi**alpha))+1)
    #     outlineScoreDict[key] = outlineScore
    # return outlineScoreDict
    return model


def dataset_to_feature(dataset_list: list):
    global_node_id = 0
    combined_featureDict = {}
    for dataset in dataset_list:
        for graph in dataset:
            G = to_networkx(graph, to_undirected=True)
            G_featureDict = get_feature(G, remove_isolate=True)
    
            for node_id, features in G_featureDict.items():
                combined_featureDict[global_node_id] = features
                global_node_id += 1
    
    return combined_featureDict


def eval(model, dataset, dataset_attack):
    epsilon = 1e-10
    w = model.coef_[0][0]
    b = model.intercept_[0]
    C = 2**b
    alpha = w

    attack_score = []
    normal_score = []
    for i in range(len(dataset)):
        graph = dataset[i]
        graph_attack = dataset_attack[i]

        edge_index = graph.edge_index
        edge_index_attack = graph_attack.edge_index

        edges = set(map(tuple, edge_index.T.tolist()))
        edges_attack = set(map(tuple, edge_index_attack.T.tolist()))

        # Find the difference between the sets
        attacked_edges = edges.symmetric_difference(edges_attack)

        # Extract nodes from the attacked edges
        attacked_nodes = set()
        for edge in attacked_edges:
            attacked_nodes.update(edge)

        G = to_networkx(graph_attack)
        featureDict = get_feature(G)

        for key in featureDict.keys():
            yi = featureDict[key][1]
            xi = featureDict[key][0]
            try:
                outlineScore = (max(yi, C * (xi ** alpha)) / (min(yi, C * (xi ** alpha)) + epsilon)) * np.log(abs(yi - C * (xi ** alpha)) + 1)
            except:
                print(xi, yi, C, alpha)
                print(traceback.format_exc())
                exit()

            if key in attacked_nodes:
                attack_score.append(outlineScore)
            else:
                normal_score.append(outlineScore)
    print(np.mean(normal_score), np.mean(attack_score))


def get_batch_link_anomaly_score(config, batch_adj_matrix, flags, C, alpha, epsilon=1e-10):
    batch_size, max_n, _ = batch_adj_matrix.size()

    batch_score_adj = torch.zeros_like(batch_adj_matrix)
    batch_score_node = torch.zeros((batch_size, max_n))

    for b in range(batch_size):
        # Extract the graph for the current batch element
        adj_matrix = batch_adj_matrix[b]
        node_flags = flags[b]

        G = nx.from_numpy_array(adj_matrix.detach().cpu().numpy())
        
        # Filter out nodes not in the current graph
        nodes_to_remove = [node for node in G.nodes if node_flags[node] == 0]
        G.remove_nodes_from(nodes_to_remove)

        # Get feature dictionary
        featureDict = get_feature(G)

        # Get node scores
        outlineScoreDict = {}
        for key in featureDict.keys():
            yi = featureDict[key][1]
            xi = featureDict[key][0]
            try:
                outlineScore = (max(yi, C * (xi ** alpha)) / (min(yi, C * (xi ** alpha)) + epsilon)) * np.log(abs(yi - C * (xi ** alpha)) + 1)
            except:
                print(f"Batch {b}, Node {key}, xi: {xi}, yi: {yi}, C: {C}, alpha: {alpha}")
                print(traceback.format_exc())
                exit()
            outlineScoreDict[key] = outlineScore

        num_nodes = G.number_of_nodes()

        # Get node score matrix
        node_scores = torch.tensor(list(outlineScoreDict.values()))

        min_score_node = torch.min(node_scores)
        max_score_node = torch.max(node_scores)
        normalized_node_scores = (node_scores - min_score_node) / (max_score_node - min_score_node)
        for index, node in enumerate(outlineScoreDict.keys()):
            batch_score_node[b][node] = normalized_node_scores[index]

        # Get adj score matrix
        score_adj = torch.zeros((max_n, max_n))

        for s in range(num_nodes):
            for t in range(num_nodes):
                if s != t:  # Avoid self-loops and non-existent edges
                    score_s = outlineScoreDict[s]
                    score_t = outlineScoreDict[t]
                    score_adj[s][t] = (score_s + score_t) / 2  # or any other function to combine scores

        relevant_scores = score_adj[:num_nodes, :num_nodes].flatten()
        min_score = torch.min(relevant_scores)
        max_score = torch.max(relevant_scores)
        normalized_scores = (relevant_scores - min_score) / (max_score - min_score)

        normalized_score_adj = torch.zeros((max_n, max_n))
        normalized_score_adj[:num_nodes, :num_nodes] = normalized_scores.reshape(num_nodes, num_nodes)

        batch_score_adj[b] = normalized_score_adj
    
    return batch_score_adj.to(config.device), batch_score_node.to(config.device)

def get_dataset_link_anomaly_score(model, dataset):
    epsilon = 1e-10
    w = model.coef_[0][0]
    b = model.intercept_[0]
    C = 2**b
    alpha = w

    adj_score_list = []

    for i in range(len(dataset)):
        graph = dataset[i]

        G = to_networkx(graph)
        featureDict = get_feature(G)

        # get node score
        outlineScoreDict = {}
        for key in featureDict.keys():
            yi = featureDict[key][1]
            xi = featureDict[key][0]
            try:
                outlineScore = (max(yi, C * (xi ** alpha)) / (min(yi, C * (xi ** alpha)) + epsilon)) * np.log(abs(yi - C * (xi ** alpha)) + 1)
                # if outlineScore <= 0:
                #     print(xi, yi, key); exit()
            except:
                print(i, xi, yi, C, alpha)
                print(traceback.format_exc())
                exit()
            outlineScoreDict[key] = outlineScore

        # get edge score
        num_nodes = graph.x.size(0)
        score_adj = torch.zeros((num_nodes, num_nodes))

        for s in range(num_nodes):
            for t in range(num_nodes):
                if s != t:  # Avoid self-loops
                    score_s = outlineScoreDict[s]
                    score_t = outlineScoreDict[t]
                    score_adj[s][t] = (score_s + score_t) / 2  # or any other function to combine scores
        
        # normarlize edge score
        scores = score_adj.flatten()

        min_score = torch.min(scores)
        max_score = torch.max(scores)
        normalized_scores = (scores - min_score) / (max_score - min_score + epsilon)
        normalized_score_adj = normalized_scores.reshape(score_adj.shape)
    
        adj_score_list.append(normalized_score_adj)

        # print(adj_score_list)
    
    return adj_score_list

# from datetime import datetime
# import pytz
# from parameter import parameter_parser
# if __name__ == "__main__":
#     config = parameter_parser()
#     dataname = config.data.dataset
#     time = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%b-%d-%H:%M:%S')
#     config.time = time
#     # ------------
#     if dataname == "MUTAG":
#         data_version = "Jul-26-20:36:05"
#     elif dataname == "IMDB-BINARY":
#         data_version = "Jul-26-20:43:20"
#     elif dataname == "PROTEINS":
#         data_version = "Jul-26-20:59:58"

#     # ------------
#     from utils.loader import load_attack_data, load_non_attack_data
#     train_dataset, eval_dataset, test_dataset = load_non_attack_data(config, data_version)
#     train_dataset_attack, eval_dataset_attack, test_dataset_attack = load_attack_data(config, data_version)

#     # ------------
#     train_featureDict = dataset_to_feature([train_dataset_attack, eval_dataset_attack])

#     # ------------
#     regress = regression(train_featureDict)
    
#     # ------------ test whether the outline score corresonding to anomaly ndoe ------------
#     eval(regress, test_dataset, test_dataset_attack)

#     # ------------
#     get_dataset_link_anomaly_score(regress, test_dataset_attack)
