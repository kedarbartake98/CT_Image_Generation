import networkx as nx
import numpy as np
import random
import torch
from sklearn.model_selection import StratifiedKFold

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0


def load_data(dataset, degree_as_tag):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('/content/drive/My Drive/pgnn/dataset/dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

    #add labels and edge_mat       
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    #Extracting unique tag labels   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)

def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list

    
def getNodesDict(patient_seg_dict, patient_seg_cur_dict):
    n = 12
    gap = 24
    
    '''Creating static graph structure for all instances'''
    patient_seg_adj_dict = {}
    for i in range(n + gap):
        patient_seg_adj_dict[i] = []
    
    for i in range(n):
        for j in range(n):
            if i!=j:
                patient_seg_adj_dict[i].append(j)
    
    add_co = n
    for i in range(n):
        patient_seg_adj_dict[i].append(add_co)
        patient_seg_adj_dict[add_co].append(i)
        patient_seg_adj_dict[add_co].append(add_co + 1)
        patient_seg_adj_dict[add_co + 1].append(add_co)
        add_co+=1
        if i+1 != n:
            patient_seg_adj_dict[i+1].append(add_co)
            patient_seg_adj_dict[add_co].append(i+1)
            add_co+=1
        else:
            patient_seg_adj_dict[0].append(add_co)
            patient_seg_adj_dict[add_co].append(0)
    
    '''Calculating nodes coordinates for the graphs'''
    patient_seg_nodes_dict = {}
    for patient in patient_seg_dict.keys():
        patient_seg_nodes_dict[patient] = {}
        for imageposno in patient_seg_dict[patient].keys():
            patient_seg_nodes_dict[patient][imageposno] = [{},{},{},{},{},{}]
            curve = patient_seg_dict[patient][imageposno][0]
            curve_curvs = patient_seg_cur_dict[patient][imageposno][0]
            length_seg = len(curve) / n
            length_gap = len(curve) / gap
            max_curv_index_first = 0
            prev_max_curv_index = - length_gap
            add_co = n
            counter = -1
            for idx0, idx1 in [[length_seg*i, length_seg*(i+1)] for i in range(n)]:
                counter += 1
                if idx1 != length_seg*n:
                    idx0 = max([idx0 , prev_max_curv_index + length_gap])
                    max_curv_index = curve_curvs.index(max(curve_curvs[idx0 : idx1]))
                    if not (max_curv_index >= idx0 and max_curv_index < idx1):
                         max_curv_index = (idx0 + idx1) / 2
                    if counter == 0:
                        max_curv_index_first = max_curv_index
                    else:
                        index1 = prev_max_curv_index + (max_curv_index - prev_max_curv_index)/3
                        index2 = prev_max_curv_index + 2*(max_curv_index - prev_max_curv_index)/3
                        patient_seg_nodes_dict[patient][imageposno][0][add_co] = curve[index1]
                        add_co+=1
                        patient_seg_nodes_dict[patient][imageposno][0][add_co] = curve[index2]    
                        add_co+=1
                    patient_seg_nodes_dict[patient][imageposno][0][counter] = curve[max_curv_index]
                    prev_max_curv_index = max_curv_index
                else:
                    idx0 = max([idx0 , prev_max_curv_index + length_gap])
                    idx1 = min([len(curve) , len(curve) + (max_curv_index_first - length_gap)])
                    max_curv_index = curve_curvs.index(max(curve_curvs[idx0 : idx1]))
                    if not (max_curv_index >= idx0 and max_curv_index < idx1):
                         max_curv_index = (idx0 + idx1) / 2
                    index1 = prev_max_curv_index + (max_curv_index - prev_max_curv_index)/3
                    index2 = prev_max_curv_index + 2*(max_curv_index - prev_max_curv_index)/3
                    patient_seg_nodes_dict[patient][imageposno][0][add_co] = curve[index1]
                    add_co+=1
                    patient_seg_nodes_dict[patient][imageposno][0][add_co] = curve[index2]    
                    add_co+=1
                    patient_seg_nodes_dict[patient][imageposno][0][counter] = curve[max_curv_index]
    
                    last_gap = len(curve) + max_curv_index_first - max_curv_index
                    index1 = max_curv_index + last_gap/3
                    index2 = max_curv_index + 2*last_gap/3
                    if index1 >= len(curve):
                        index1 = index1 - len(curve)
                    if index2 >= len(curve):
                        index2 = index2 - len(curve)
                    patient_seg_nodes_dict[patient][imageposno][0][add_co] = curve[index1]
                    add_co+=1
                    patient_seg_nodes_dict[patient][imageposno][0][add_co] = curve[index2] 
                    
    return patient_seg_nodes_dict



