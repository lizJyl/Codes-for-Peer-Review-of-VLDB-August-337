import numpy as np
import scipy.sparse as sp
import torch
import numpy.random as npr
import random
import networkx as nx
import os
from numpy import dot
from numpy.linalg import norm


np.set_printoptions(threshold=np.inf)
npr.seed(170624)
random.seed(170624)


def fnormalize( mx):
    """Row-normalize sparse matrix"""

    mx = mx.transpose(0, 1)
    print("mx shape", mx.shape)
    rowsum = mx.sum(1)
    # rowsum = rowsum[:,np.newaxis]
    rowsum[rowsum == 0] = 1
    # print("rowsum shape", rowsum.shape)
    print("rowsum", rowsum[:24])
    mx = mx / rowsum[:, np.newaxis]
    mx = mx.transpose(0, 1)
    return mx

def load_data(path="../facebook/ATC/"):
    ego = 414
    ego = 686
    ego = 348
    ego = 0
    ego = 3437
    ego = 1912
    # ego = 1684
    # ego = 107

    data_dir = '../facebook/'

    ego_feat = np.genfromtxt("{}{}.egofeat".format(data_dir, ego), dtype=np.dtype(str))
    feat = np.genfromtxt("{}{}.feat".format(data_dir, ego), dtype=np.dtype(str))
    ego_feat = ego_feat.astype(np.int32)
    feat = feat.astype(np.int32)
    features = np.vstack((ego_feat, feat[:, 1:]))
    features = fnormalize(features)

    node_map = {}
    node_map[int(ego)] = 0
    for i in range(feat.shape[0]):
        node_map[feat[i, 0]] = i

    edges = np.genfromtxt("{}{}.edges".format(data_dir, ego), dtype=np.dtype(str))
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



    Adj = A

    for i in range(features.shape[0]):
        Adj[i, i] = 0
        A[i, i] = 1



    # feats = features

    file = open("{}result/{}ATC_com.txt".format(path, ego), 'r')
    val_list = file.readlines()
    # val_list=val_list[:,0:-1]
    lists = []
    for string in val_list:
        #	string=string.split('\n')
        string = string.strip().split(' ')
        # print(string[1:])
        # if(len(string)<6):
        #     continue
        string= np.array(string)
        lists.append(string)
    # print("list", lists)
    ACC_result = np.array(lists)
    # ACC_result=lists
    file.close()
    # print("ACC_result",ACC_result)

    file = open("{}{}.query.com".format(path, ego), 'r')
    val_list = file.readlines()
    # val_list=val_list[:,0:-1]
    lists = []
    for string in val_list:
        #	string=string.split('\n')
        string = string.strip().split(' ')
        # print(string[1:])
        # if(len(string)<6):
        #     continue
        string = np.array(string)
        lists.append(string[:])
    ground_truth = np.array(lists)
    # ground_truth=lists
    file.close()
    # print("ground_truth", ground_truth)

    # ACC_result= np.genfromtxt("{}ACC/result/{}.result".format(path, ego), dtype=np.dtype(str))
    # ground_truth = np.genfromtxt("{}1nodeQuery/{}.query.com".format(path, ego), dtype=np.dtype(str))


    # ACC_result = ACC_result.astype(np.int32)
    # ground_truth = ground_truth.astype(np.int32)
    print("ground_truth.shape[0] ",ground_truth.shape[0])
    union=0
    ints=0
    predict=0
    true=0

    precision=0
    recall=0
    f1=0
    jac=0

    j=2

    all_cos=0
    all_dens=0
    count=0
    all_size=0

    var_cos=[]
    var_dens=[]
    var_size=[]
    for i in range(ground_truth.shape[0]):
        # print(i,j)
        truth=set()
        for k in range(ground_truth[i].shape[0]):
            truth.add(int(ground_truth[i][k]))
        result=set()

        cos=0
        cos_count=0
        comm=np.zeros((Adj.shape[0]), dtype=np.int32)
        for k in range(ACC_result[j].shape[0]):
            result.add(int(ACC_result[j][k]))
            comm[int(ACC_result[j][k])]=1
            for k2 in range(ACC_result[j].shape[0]):
                cos = cos + dot(features[int(ACC_result[j][k])],features[int(ACC_result[j][k2])])/(norm(features[int(ACC_result[j][k])])*norm(features[int(ACC_result[j][k2])]))

                cos_count = cos_count + 1

        output_edges = Adj[comm == 1][:, comm == 1].sum()

        if cos_count!=0:
            all_cos = all_cos+cos/cos_count
            var_cos.append(cos/cos_count)
        else:
            var_cos.append(0)
        if comm.sum()!=0:
            all_dens = all_dens+output_edges / (comm.sum()*(comm.sum()-1))
            var_dens.append(output_edges / (comm.sum()*(comm.sum()-1)))
            print('density in output ', output_edges / comm.sum(),output_edges / (comm.sum()*(comm.sum()-1)))
        else:
            var_dens.append(0)
        count = count+1
        all_size=all_size+comm.sum()
        var_size.append(comm.sum())






        ints=len(truth&result)
        union=len(truth|result)
        predict=len(result)
        true=len(truth)
        precision+=ints/predict
        recall+=ints/true
        f1+=2*precision*recall/(precision+recall)
        jac+=ints/union
        j=j+2
        # print("&", len(truth&result))
        # print("|", len(truth|result))
        print("truth",(truth))
        print("result",(result))

    print("union",union)
    print("intersection", ints)

    print("\npredict", predict)
    print("true", true)

    print("\n\naverage size, dense, cos",all_size/count,all_dens/count,all_cos/count)
    print("average size, dense, cos", np.var(var_size), np.var(var_dens), np.var(var_cos))

    # precision=ints/predict
    # recall=ints/true
    # f1=2*precision*recall/(precision+recall)
    # jac=ints/union
    print("\n\nF1=", f1/count)
    print("\nprecision=", precision/count)
    print("recall", recall/count)
    # print("\nF1=", f1)
    print("\njac", jac/count)
    # print("i=",i)
    # print("j=", j)




load_data()