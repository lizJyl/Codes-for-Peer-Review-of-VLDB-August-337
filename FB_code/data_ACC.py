import numpy as np
import scipy.sparse as sp
import torch
import numpy.random as npr
import random
import networkx as nx
import os

np.set_printoptions(threshold=np.inf)
npr.seed(170624)
random.seed(170624)


def load_data(path="./facebook/", ego=0):
    ego_feat = np.genfromtxt("{}{}.egofeat".format(path, ego), dtype=np.dtype(str))
    feat = np.genfromtxt("{}{}.feat".format(path, ego), dtype=np.dtype(str))
    ego_feat = ego_feat.astype(np.int32)
    feat = feat.astype(np.int32)
    features = np.vstack((ego_feat, feat[:, 1:]))
    node_map = {}
    node_map[int(ego)] = 0
    for i in range(feat.shape[0]):
        node_map[feat[i, 0]] = i + 1
    print(node_map)
    edges = np.genfromtxt("{}{}.edges".format(path, ego), dtype=np.dtype(str))
    edges = edges.astype(np.int32)
    Edges = []

    for i in range(feat.shape[0]):
        Edges.append((node_map[int(ego)], i + 1))
    for e in edges:
        Edges.append((node_map[e[0]], node_map[e[1]]))
    # Edges.append((0,0))
    # for i in range(feat.shape[0]):
    #     Edges.append((i+1,i+1))
    G = nx.Graph()
    G.add_edges_from(Edges)
    A = nx.adjacency_matrix(G).todense()
    A = np.array(A, dtype=np.float32)

    # print(A)


    f=open("{}ACC/{}.graph".format(path, ego),'w')
    f2 = open("{}ACC/{}.node".format(path, ego), 'w')
    # np.savetxt("{}{}.attr".format(path, ego), inputs, fmt='%d')

    for i in range(A.shape[0]):
        edge=[]
        edge.append(i)
        for j in range(A.shape[0]):
            if A[i][j] ==1:
                edge.append(j)
        print("edge",edge)
        np.savetxt(f,[edge],fmt='%d',delimiter=' ', newline='\n',)

        nfeat=[]
        nfeat.append(i)
        nfeat.append(i)
        for j in range(features.shape[1]):
            if features[i][j]==1:
                nfeat.append(j)
        print("feat",nfeat)
        np.savetxt(f2, [nfeat], fmt='%d', delimiter=' ', newline='\n', )

        # degree[i] = G.degree[i] / A.shape[0]
    f2.close()
    f.close()



load_data()