import numpy as np
import scipy.sparse as sp
import torch
import numpy.random as npr
import random

np.set_printoptions(threshold=np.inf)
npr.seed(170624)
random.seed(170624)


def load_data(path="./facebook/", ego=698):
    ego_feat = np.genfromtxt("{}{}.egofeat".format(path, ego), dtype=np.dtype(str))
    feat = np.genfromtxt("{}{}.feat".format(path, ego), dtype=np.dtype(str))
    features = np.vstack((ego_feat, feat[:, 1:]))
    # print(features[0:3,:])
    ego_node = int(ego)
    node = np.array(np.vstack((ego_node, feat[:, 0:1])))
    n_node = node.shape[0]
    print("node num",n_node)
    # node_map={j:i for i,j in enumerate(node)}
    node_map = {}
    inv_node_map = {}
    for i in range(node.shape[0]):
        node_map[int(node[i])] = i
        inv_node_map[i] = int(node[i])

    file = open("{}{}.circles".format(path, ego), 'r')
    val_list = file.readlines()
    # val_list=val_list[:,0:-1]
    lists = []
    for string in val_list:
        #	string=string.split('\n')
        string = string.strip().split('\t')
        # if(len(string)<6):
        #     continue
        # print(string[1:])
        lists.append(string[1:])
    circles = np.array(lists)
    file.close()
    print("community num",circles.shape[0])
    # for i in range(circles.shape[0]):
    # 	print(circles[i][0])
    inputs = []
    features=features.astype(int)
    All_feats_sum=features.sum(0)
    print("feat shape", features.shape)
    print("All_feats_sum",All_feats_sum)
    # sort = np.argsort(All_feats_sum)
    # # print("All_feats_sum sort", sort)
    # all_sort=sort[-5:]
    # print("All_feats_sum sort", all_sort)
    # All_feats_sort=np.zeros(features.shape[1], dtype=np.int32)
    # All_feats_sort[all_sort]=1
    # print("All_feats_sum sort", All_feats_sort)
    for i in range(circles.shape[0]):
        input = np.zeros(features.shape[1], dtype=np.int32)
        for j in circles[i]:
            input=input+features[int(node_map[int(j)])]
        sort = np.argsort(input)
        # print("All_feats_sum sort", sort)
        sort = sort[-5:]
        print("feats_sum sort", i,sort)
        all_sort=(np.argsort(All_feats_sum-input))[-3:]
        print("All_feats_sum sort", all_sort)
        sort=set(sort)-set(all_sort)
        sort=np.array(list(sort))
        print("feats_sum sort", i, sort)
        feats_sort = np.zeros(features.shape[1], dtype=np.int32)
        feats_sort[sort] = 1
        inputs.append(feats_sort[np.newaxis, :])
    inputs = np.concatenate(inputs, axis=0)
    print(inputs.shape, inputs.dtype)
    # print("com_feats 0", inputs[0])
    # sort=np.argsort(inputs[0])
    # print("com_feats 0", sort)
    np.savetxt("{}{}.attr".format(path, ego), inputs, fmt='%d')
    #
    # labels = []
    # for i in range(circles.shape[0]):
    #     for k in range(len(circles[i]) // 3):
    #         output = np.zeros(n_node)
    #         for j in circles[i]:
    #             output[node_map[int(j)]] = 1
    #         labels.append(output[np.newaxis, :])
    # labels = np.concatenate(labels, axis=0)
    #
    # np.savetxt("{}{}.labels".format(path, ego), labels, fmt='%d')


load_data()