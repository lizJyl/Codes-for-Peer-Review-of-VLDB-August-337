import numpy as np
import networkx as nx
import os
import torch
import random
# import kcore
import itertools
from copy import deepcopy
# from sets import Set
# from math import log
import scipy.sparse as sp
from torch.utils.data.dataset import Dataset
import numpy.random as npr
import normalization as Norm
# from torch.utils.data import DataLoader
#import matplotlib.pyplot as plt



class GraphDataset(Dataset):
    def __init__(self, data_dir='./data/facebook/', phase='train',ego=0, method='ATC'):
        super(GraphDataset, self).__init__()

        data_dir='./data/facebook/'

        graph_name = [ego]
        feats = {}
        adjs = {}
        node_idx = {}
        samples_dis = {}
        samples_in = {}
        samples_out = {}
        samples_att = {}
        egos = []

        # for name in graph_name:
        for ego in graph_name:
            ego_feat = np.genfromtxt("{}{}.egofeat".format(data_dir, ego), dtype=np.dtype(str))
            feat = np.genfromtxt("{}{}.feat".format(data_dir, ego), dtype=np.dtype(str))
            ego_feat = ego_feat.astype(np.int32)
            feat = feat.astype(np.int32)
            features = np.vstack((ego_feat, feat[:, 1:]))
            print("all feature num", features.sum(),features.sum()/features.shape[0])
            # print("1 feature shape", features.shape)
            features = self.fnormalize(features)

            # print("2 feature shape", features.shape)

            node_map = {}
            node_map[int(ego)] = 0
            for i in range(feat.shape[0]):
                node_map[feat[i,0]] = i

            edges = np.genfromtxt("{}{}.edges".format(data_dir, ego), dtype=np.dtype(str))
            edges = edges.astype(np.int32)
            Edges = []

            for i in range(feat.shape[0]):
                Edges.append((node_map[int(ego)], i+1))
            for e in edges:
                Edges.append((node_map[e[0]], node_map[e[1]]))
            # Edges.append((0,0))
            # for i in range(feat.shape[0]):
            #     Edges.append((i+1,i+1))
            G = nx.Graph()
            G.add_edges_from(Edges)
            A = nx.adjacency_matrix(G).todense()
            A = np.array(A, dtype=np.float32)

            degree=np.zeros(A.shape[0], dtype=np.float32)
            for i in range(A.shape[0]):
                degree[i]=G.degree[i]/A.shape[0]
            # print("size of degree",degree)
            core_G=nx.core_number(G)
            core = np.zeros(A.shape[0], dtype=np.float32)
            for i in range(A.shape[0]):
                core[i] = core_G[i]

            # print("size of core",core)
            core=core/core.max()
            # print("Norm core", core)

            cluster_G=nx.clustering(G)
            cluster = np.zeros(A.shape[0], dtype=np.float32)
            for i in range(A.shape[0]):
                cluster[i] = cluster_G[i]

            # print("size of cluster",cluster)
            cluster = cluster / cluster.max()
            # print("Norm cluster", cluster)

            triangle_G = nx.clustering(G)
            triangle = np.zeros(A.shape[0], dtype=np.float32)
            for i in range(A.shape[0]):
                triangle[i] = triangle_G[i]

            # print("size of cluster",cluster)
            triangle = triangle / triangle.max()
            # print("Norm cluster", cluster)



            Adj=A

            for i in range(features.shape[0]):
                Adj[i,i]=0
                A[i, i] = 1

            # A=self.normalize(A)
            A=Norm.normalized_adjacency(A)
            # A=Norm.aug_normalized_adjacency(A)

            # print('!!!!!You make a change here A[i,i]=1 in Loader.py line 71!!!!!')
            # for i in range(A.shape[0]): # residual
            #     A[i, i] = 1

            feats[int(ego)] = features
            adjs[int(ego)] = A
            node_idx[int(ego)] = node_map

            ##### all zoro



            if method=='ACC_QfN':
                cur_in = np.genfromtxt("{}ACC_QfN/{}.sample".format(data_dir, ego), dtype=np.dtype(str))
            elif method=="ACC":
                cur_in = np.genfromtxt("{}1nodeQuery/{}.sample".format(data_dir, ego), dtype=np.dtype(str))
            elif method == "ATC_QfN":
                cur_in = np.genfromtxt("{}ATC_QfN/{}.sample".format(data_dir, ego), dtype=np.dtype(str))
            elif method == "ATC":
                cur_in = np.genfromtxt("{}ATC/{}.sample".format(data_dir, ego), dtype=np.dtype(str))
            else:
                cur_in = np.genfromtxt("{}{}.samples".format(data_dir, ego), dtype=np.dtype(str))
            # cur_in = np.genfromtxt("{}CTC/{}.sample".format(data_dir, ego), dtype=np.dtype(str))
            # cur_in = np.genfromtxt("{}ATC/{}.sample".format(data_dir, ego), dtype=np.dtype(str))
            # cur_in = np.genfromtxt("{}ATC_illQ/{}.sample".format(data_dir, ego), dtype=np.dtype(str))
            # cur_in = np.genfromtxt("{}ATC_randomQ/{}.sample".format(data_dir, ego), dtype=np.dtype(str))

            if False: #Fasle zhengchang    True all zero
                cur_in= np.zeros(cur_in.shape, dtype=np.float32)
                cur_in_dis = np.zeros(cur_in.shape, dtype=np.float32)
            else:

                cur_in = cur_in.astype(np.int32)
                print('cur_in', cur_in.shape)

                dis = nx.shortest_path(G)
                all_dis = np.zeros(A.shape, dtype=np.int32)
                for i in range(A.shape[0]):
                    for j in range(A.shape[1]):
                        # all_dis[i,j] = len(dis[i][j])-1
                        if i==j:
                            all_dis[i, j]=0
                        else:
                            if j in dis[i]:
                                assert len(dis[i][j])>1
                                all_dis[i,j] = len(dis[i][j])-1
                            else:
                                all_dis[i,j] = A.shape[0] # we set specially here

                cur_in_dis = np.zeros(cur_in.shape, dtype=np.float32)
                for i in range(cur_in.shape[0]):
                    source = []
                    for j in range(cur_in.shape[1]):
                        if cur_in[i,j] == 1:
                            source.append(all_dis[j:j+1])
                    source = np.concatenate(source, axis=0)
                    assert source.shape[0] == cur_in[i].sum()
                    for j in range(cur_in.shape[1]):
                        cur_in_dis[i,j] = source[:,j].min() #mean()

                ## normalization ###
                # print("distance ",cur_in_dis)
                for j in range(cur_in_dis.shape[0]):
                    max_val = (cur_in_dis[j] * (cur_in_dis[j] < A.shape[0])).max()
                    # print("max distance ", max_val)
                    assert max_val > 0., (max_val)
                    for i in range(cur_in_dis[j].shape[0]):
                        if cur_in_dis[j][i] == A.shape[0]:  # unreachable:
                            cur_in_dis[j][i] = 0.
                        else:
                            cur_in_dis[j][i] = 1. - cur_in_dis[j][i] / (max_val + 1)
                    for k in range(cur_in.shape[1]):
                        if cur_in[j][k]==1:
                            cur_in_dis[j][k] = 1
                    # print("distance ", cur_in_dis)


            if method=='ACC_QfN':
                cur_out = np.genfromtxt("{}ACC_QfN/{}.label".format(data_dir, ego), dtype=np.dtype(str))
            elif method=="ACC":
                cur_out = np.genfromtxt("{}1nodeQuery/{}.label".format(data_dir, ego), dtype=np.dtype(str))
            elif method == "ATC_QfN":
                cur_out = np.genfromtxt("{}ATC_QfN/{}.label".format(data_dir, ego), dtype=np.dtype(str))
            elif method == "ATC":
                cur_out = np.genfromtxt("{}ATC/{}.label".format(data_dir, ego), dtype=np.dtype(str))
            else:
                cur_out = np.genfromtxt("{}{}.labels".format(data_dir, ego), dtype=np.dtype(str))
            # cur_out = np.genfromtxt("{}CTC/{}.label".format(data_dir, ego), dtype=np.dtype(str))
            # cur_out = np.genfromtxt("{}ATC/{}.label".format(data_dir, ego), dtype=np.dtype(str))
            # cur_out = np.genfromtxt("{}ATC_illQ/{}.label".format(data_dir, ego), dtype=np.dtype(str))
            # cur_out = np.genfromtxt("{}ATC_randomQ/{}.label".format(data_dir, ego), dtype=np.dtype(str))





            cur_out = cur_out.astype(np.int32)
            print("label shape", cur_out.shape)

            savelable = {}
            count = 0
            countlab = []
            for label in cur_out:
                # print("1 label  shape",label.shape)
                key_label = tuple(label)
                if key_label not in savelable:
                    savelable[key_label] = count
                    print(count, label.sum())
                    countlab.append([count, label.sum()])
                    count += 1
            print("label class", len(savelable))
            np.savetxt("{}{}.labelnum".format(data_dir, ego), countlab)





            file = open("{}{}.circles".format(data_dir, ego), 'r')
            val_list = file.readlines()
            lists = []
            for string in val_list:
                string = string.strip().split('\t')
                if (len(string) < 6):
                    continue
                lists.append(string[1:])
            circles = np.array(lists)
            # file.close()

            if method == 'ACC_QfN':
                cur_att = np.genfromtxt("{}ACC_QfN/{}.query.attr".format(data_dir, ego), dtype=np.dtype(str))
                cur_att = cur_att.astype(np.int32)
                attr = cur_att
            elif method == "ATC_QfN":
                cur_att = np.genfromtxt("{}ATC_QfN/{}.sample.attri".format(data_dir, ego), dtype=np.dtype(str))
                cur_att = cur_att.astype(np.int32)
                attr = cur_att
            else:
                cur_att = np.genfromtxt("{}{}.attr".format(data_dir, ego), dtype=np.dtype(str))
                cur_att = cur_att.astype(np.int32)
                attr = []
                for i in range(circles.shape[0]):
                    if len(circles[i]) < 5:
                        continue
                    for k in range(len(circles[i]) // 2 - 1):
                        attr.append(cur_att[i][np.newaxis, :])
                        # attr.append(np.zeros(features.shape[1])[np.newaxis, :]/features.shape[1]) # noattribute
                attr = np.concatenate(attr, axis=0)
                print("attri shape", attr.shape)
            # cur_att = np.genfromtxt("{}ATC_illQ/{}.sample.attri".format(data_dir, ego), dtype=np.dtype(str))
            # cur_att = np.genfromtxt("{}ATC_randomQ/{}.sample.attri".format(data_dir, ego), dtype=np.dtype(str))




            if phase == 'train':
                for i in range(len(cur_in_dis)):
                    if i % 7 == 0 or i % 7 == 5 or i % 7 == 6:
                    # if i % 6 == 0:# or i % 7 == 5 or i % 7 == 6:
                        if int(ego) not in samples_in:
                            samples_in[int(ego)] = []
                            samples_out[int(ego)] = []
                            samples_dis[int(ego)]=[]
                            samples_att[int(ego)] = []
                        samples_in[int(ego)].append(cur_in[i:i+1])
                        samples_dis[int(ego)].append(cur_in_dis[i:i + 1])
                        samples_out[int(ego)].append(cur_out[i:i+1])
                        samples_att[int(ego)].append(attr[i:i + 1])
            if phase == 'eval':
                for i in range(len(cur_in_dis)):
                    if i % 7 == 1 or i % 7 == 4 :
                    # if i % 6 == 1 or i % 6 == 4:
                        if int(ego) not in samples_in:
                            samples_in[int(ego)] = []
                            samples_out[int(ego)] = []
                            samples_dis[int(ego)] = []
                            samples_att[int(ego)] = []
                        samples_dis[int(ego)].append(cur_in_dis[i:i + 1])
                        samples_in[int(ego)].append(cur_in[i:i+1])
                        samples_out[int(ego)].append(cur_out[i:i+1])
                        samples_att[int(ego)].append(attr[i:i + 1])
            if phase == 'test':
                for i in range(len(cur_in_dis)):
                    if i % 7 == 2 or i % 7 == 3 :
                    # if i % 6 == 2 or i % 6 == 3 or i % 6 == 5:
                        if int(ego) not in samples_in:
                            samples_in[int(ego)] = []
                            samples_out[int(ego)] = []
                            samples_dis[int(ego)] = []
                            samples_att[int(ego)] = []
                        samples_dis[int(ego)].append(cur_in_dis[i:i + 1])
                        samples_in[int(ego)].append(cur_in[i:i+1])
                        samples_out[int(ego)].append(cur_out[i:i+1])
                        samples_att[int(ego)].append(attr[i:i + 1])
            samples_in[int(ego)] = np.concatenate(samples_in[int(ego)], 0)
            samples_dis[int(ego)] = np.concatenate(samples_dis[int(ego)], 0)
            samples_out[int(ego)] = np.concatenate(samples_out[int(ego)], 0)
            samples_att[int(ego)] = np.concatenate(samples_att[int(ego)], 0)
            # assert 1<0, (cur_in.shape, samples_in[int(ego)].shape)

        self.feats = feats
        self.adjs = adjs

        self.Adj=Adj
        # self.circle=circles
        # self.node_map=node_map
        # self.attr=cur_att
        # self.all_dis=all_dis
        # self.max_select_num=3
        self.node_idx = node_idx
        self.samples_in = samples_in
        self.samples_dis = samples_dis
        self.samples_out = samples_out
        self.samples_att = samples_att
        self.egos = egos
        self.degree=degree
        self.core = core
        self.cluster = cluster
        self.triangle = triangle
        self.ego=ego
        self.phase=phase
        self.savelable=savelable


    def __len__(self):
        # return len(self.circle)
        # return 9
        return self.samples_in[self.ego].shape[0]

    def __getitem__(self, item):
        # ego = self.egos[item]
        ego = self.ego
       
        cur_dis = self.samples_dis[ego].copy()[item, :]
        cur_in = self.samples_in[ego].copy()[item,:]
        cur_out = self.samples_out[ego].copy()[item,:]
        cur_att = self.samples_att[ego].copy()[item, :]

        degree=self.degree[:,np.newaxis]
        core = self.core[:, np.newaxis]
        cluster = self.cluster[:, np.newaxis]
        triangle=self.triangle[:, np.newaxis]

        cur_in = cur_in[:,np.newaxis] # BN1
        cur_dis = cur_dis[:, np.newaxis]
        cur_out = cur_out[:,np.newaxis] # BN1
        cur_att = cur_att[:, np.newaxis]  # BN1


        cur_adj = self.adjs[ego].copy()
        feats = self.feats[ego].copy()

        #
        input = cur_in
        input = np.concatenate((input, cur_dis), axis=1)  # BN(D+1)

        # return torch.FloatTensor(cur_feats), torch.FloatTensor(cur_out), torch.FloatTensor(cur_adj)
        return input, cur_att,cur_adj,feats, cur_out, self.Adj
        # return input, cur_att,cur_adj,feats, cur_out,self.Adj#self.savelable.copy()

    def fnormalize(self, mx):
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

    def normalized_adjacency(self,adj):
        # adj = sp.coo_matrix(adj)
        row_sum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        return (d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt))



    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        # print(rowsum)
        r_inv = 1 / rowsum
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

        # rowsum = mx.sum(1)
        # rowsum = rowsum[:,np.newaxis]
        # rowsum[rowsum==0] = 1
        # mx = mx/rowsum
        # return mx

        
        
class WebKBDataset(Dataset):
    def __init__(self, data_dir='./data/WebKB/', phase='train',file = "cornell",method="QD-GCN"):
        super(WebKBDataset, self).__init__()


        data_dir='./data/WebKB/'



        feat = np.genfromtxt("{}{}.content".format(data_dir, file), dtype=np.dtype(str))

        node_name = feat[:, 0]
        node_map = {}
        i = 0
        for j in node_name:
            node_map[j] = i
            i = i + 1

        # print(node_map)

        fedge = open("{}{}.cites".format(data_dir, file), "r")
        val_list = fedge.readlines()
        # val_list=val_list[:,0:-1]
        edges = []
        for string in val_list:
            if (file == "cora" or file == "citeseer"):
                string = string.strip().split('\t')
            else:
                string = string.strip().split(' ')
            # print(len(string))
            if (len(string) < 2):
                continue
            edges.append([node_map[string[0]], node_map[string[1]]])

        fedge.close()

        A = np.zeros((node_name.shape[0], node_name.shape[0]), dtype=np.int32)
        for e in edges:
            # print(e[1])
            A[e[0], e[1]] = 1
            A[e[1], e[0]] = 1
        # for i in range(A.shape[0]):
        #     A[i, i] = 1
        A = np.array(A, dtype=np.float32)

        feature = feat[:, 1:-1].astype(np.int32)
        labelname=feat[:,-1]
        print("all feature num", feature.sum(),feature.sum()/A.shape[0])
        feature = self.fnormalize(feature)

        Edges = []

        for i in range(feat.shape[0]):
            Edges.append((i,i))
        for e in edges:
            Edges.append((e[0], e[1]))
        G = nx.Graph()
        G.add_edges_from(Edges)


        degree=np.zeros(A.shape[0], dtype=np.float32)
        for i in range(A.shape[0]):
            degree[i]=G.degree[i]/A.shape[0]


        Adj=A

        for i in range(A.shape[0]):
            A[i, i] = 1
            Adj[i][i]=0

        # A=self.normalize(A)
        A=Norm.normalized_adjacency(A)
        print("A",A.shape)

        print("{}{}/xxx/{}.sample".format(data_dir, file, file))
        if(method=="onenode"):
            cur_in = np.genfromtxt("{}{}/ACC/{}.sample".format(data_dir, file, file), dtype=np.dtype(str))
        elif (method == "ATC"):
            cur_in = np.genfromtxt("{}{}/ATC/{}.samples".format(data_dir, file, file), dtype=np.dtype(str))
        elif (method == "ATC_QfN"):
            cur_in = np.genfromtxt("{}{}/ATC_QfromNode/{}.samples".format(data_dir, file, file), dtype=np.dtype(str))
        elif (method == "ACC_QfN"):
            cur_in = np.genfromtxt("{}{}/ACC_QfN/{}.sample".format(data_dir, file, file), dtype=np.dtype(str))
        else:
            # laode multi nodes query
            cur_in = np.genfromtxt("{}{}/{}.samples".format(data_dir, file, file), dtype=np.dtype(str))

        cur_in = cur_in.astype(np.int32)
        print('sample', cur_in.shape,cur_in.sum(1))

        dis = nx.shortest_path(G)
        all_dis = np.zeros(A.shape, dtype=np.int32)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                # all_dis[i,j] = len(dis[i][j])-1
                if i==j:
                    all_dis[i, j]=0
                else:
                    if j in dis[i]:
                        assert len(dis[i][j])>1
                        all_dis[i,j] = len(dis[i][j])-1
                    else:
                        all_dis[i,j] = A.shape[0] # we set specially here

        cur_in_dis = np.zeros(cur_in.shape, dtype=np.float32)
        for i in range(cur_in.shape[0]):
            source = []
            for j in range(cur_in.shape[1]):
                if cur_in[i,j] == 1:
                    source.append(all_dis[j:j+1])
            source = np.concatenate(source, axis=0)
            assert source.shape[0] == cur_in[i].sum()
            for j in range(cur_in.shape[1]):
                cur_in_dis[i,j] = source[:,j].min() #mean()

        ## normalization ###
        # print("distance ",cur_in_dis)
        for j in range(cur_in_dis.shape[0]):
            max_val = (cur_in_dis[j] * (cur_in_dis[j] < A.shape[0])).max()
            # print("max distance ", max_val)
            if(max_val==0):
                print("max_val=0")
            for i in range(cur_in_dis[j].shape[0]):
                if cur_in_dis[j][i] == A.shape[0]:  # unreachable:
                    cur_in_dis[j][i] = 0.
                else:
                    cur_in_dis[j][i] = 1. - cur_in_dis[j][i] / (max_val + 1)
            for k in range(cur_in.shape[1]):
                if cur_in[j][k]==1:
                    cur_in_dis[j][k] = 1
            # print("distance ", cur_in_dis)

        if (method == "onenode"):
            cur_out = np.genfromtxt("{}{}/ACC/{}.lable".format(data_dir, file, file), dtype=np.dtype(str))
        elif(method=="ATC"):
            cur_out = np.genfromtxt("{}{}/ATC/{}.lables".format(data_dir, file, file), dtype=np.dtype(str))
        elif(method == "ATC_QfN"):
            cur_out = np.genfromtxt("{}{}/ATC_QfromNode/{}.lables".format(data_dir, file, file), dtype=np.dtype(str))
        elif (method == "ACC_QfN"):
            cur_out = np.genfromtxt("{}{}/ACC_QfN/{}.lable".format(data_dir, file, file), dtype=np.dtype(str))
        else:
            cur_out = np.genfromtxt("{}{}/{}.lables".format(data_dir, file, file), dtype=np.dtype(str))

        cur_out = cur_out.astype(np.int32)
        print("label shape", cur_out.shape)

        savelable = {}
        count=0
        countlab=[]
        for label in cur_out:
            # print("1 label  shape",label.shape)
            key_label = tuple(label)
            if key_label not in savelable:
                for k in range(label.shape[0]):
                    if label[k]==1:
                        break
                savelable[key_label] = count
                print(count,label.sum(),labelname[k])
                countlab.append([count,label.sum()])
                count += 1
        print("label class",len(savelable))
        np.savetxt("{}/{}.labelnum".format(data_dir, file), countlab)




        if (method == "onenode"):
            attr = np.genfromtxt("{}{}/ACC/{}.attr".format(data_dir, file, file), dtype=np.dtype(str))
        else:
            if (method == "QD-GCN"):
                attr = np.genfromtxt("{}{}/{}.attrs".format(data_dir, file, file), dtype=np.dtype(str))
            elif(method=="ATC"):
                attr = np.genfromtxt("{}{}/ATC/{}.attrs".format(data_dir, file, file), dtype=np.dtype(str))
            elif (method == "ATC_QfN"):
                attr = np.genfromtxt("{}{}/ATC_QfromNode/{}.attrs".format(data_dir, file, file), dtype=np.dtype(str))
            elif (method == "ACC_QfN"):
                attr = np.genfromtxt("{}{}/ACC_QfN/{}.attr".format(data_dir, file, file), dtype=np.dtype(str))
            else:
                attr = np.zeros((cur_out.shape[0], feature.shape[1]), dtype=np.int32)
	
	######    no attribute query
        attr = np.zeros((cur_out.shape[0], feature.shape[1]), dtype=np.int32)
	
        attr = attr.astype(np.int32)
        print("attr shape", attr.shape)

        samples_in=[]
        samples_out=[]
        samples_dis=[]
        samples_att=[]
        if phase == 'train':
            for i in range(len(cur_in_dis)):
                if i % 7 == 0 or i % 7 == 5 or i % 7 == 6:
                    samples_in.append(cur_in[i:i+1])
                    samples_dis.append(cur_in_dis[i:i + 1])
                    samples_out.append(cur_out[i:i+1])
                    samples_att.append(attr[i:i + 1])
        if phase == 'eval':
            for i in range(len(cur_in_dis)):
                if i % 7 == 1 or i % 7 == 4 :
                    samples_dis.append(cur_in_dis[i:i + 1])
                    samples_in.append(cur_in[i:i+1])
                    samples_out.append(cur_out[i:i+1])
                    samples_att.append(attr[i:i + 1])
        if phase == 'test':
            for i in range(len(cur_in_dis)):
                if i % 7 == 2 or i % 7 == 3 :
                    samples_dis.append(cur_in_dis[i:i + 1])
                    samples_in.append(cur_in[i:i+1])
                    samples_out.append(cur_out[i:i+1])
                    samples_att.append(attr[i:i + 1])
        samples_in = np.concatenate(samples_in, 0)
        samples_dis = np.concatenate(samples_dis, 0)
        samples_out = np.concatenate(samples_out, 0)
        samples_att = np.concatenate(samples_att, 0)
        # assert 1<0, (cur_in.shape, samples_in[int(ego)].shape)

        self.feats = feature
        self.adjs = A
        self.Adj=Adj
        # self.circle=circles
        # self.node_map=node_map
        # self.attr=cur_att
        # self.all_dis=all_dis
        # self.max_select_num=3
        # self.node_idx = node_idx
        self.samples_in = samples_in
        self.samples_dis = samples_dis
        self.samples_out = samples_out
        self.samples_att = samples_att
        self.degree=degree
        # self.core = core
        # self.cluster = cluster
        # self.triangle = triangle
        self.phase=phase
        self.savelable=savelable
        # print('self.savelable', savelable)
        # print('==============')


    def __len__(self):
        # return len(self.circle)
        # return 9
        return self.samples_in.shape[0]

    def __getitem__(self, item):

        cur_dis = self.samples_dis.copy()[item, :]
        cur_in = self.samples_in.copy()[item,:]
        cur_out = self.samples_out.copy()[item,:]
        cur_att = self.samples_att.copy()[item, :]

        # degree=self.degree[:,np.newaxis]
        # core = self.core[:, np.newaxis]
        # cluster = self.cluster[:, np.newaxis]
        # triangle=self.triangle[:, np.newaxis]

        cur_in = cur_in[:,np.newaxis] # BN1
        cur_dis = cur_dis[:, np.newaxis]
        cur_out = cur_out[:,np.newaxis] # BN1
        cur_att = cur_att[:, np.newaxis]  # BN1
        cur_adj = self.adjs.copy()
        feats = self.feats.copy()
        # input=np.concatenate((self.feats[ego].copy(), cur_dis), axis=1)  # BN(D+2)


        # input = cur_in
        # input=np.concatenate((input, cur_in), axis=1) # BN(D+1)

        # input=cur_dis
        # input = np.concatenate((input, cur_dis), axis=1)  # BN(D+1)
        #
        input = cur_in
        input = np.concatenate((input, cur_dis), axis=1)  # BN(D+1)

        # return torch.FloatTensor(cur_feats), torch.FloatTensor(cur_out), torch.FloatTensor(cur_adj)
        return input, cur_att,cur_adj,feats, cur_out,self.Adj#self.savelable.copy()

    def fnormalize(self, mx):
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

    def normalized_adjacency(self,adj):
        # adj = sp.coo_matrix(adj)
        row_sum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        return (d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt))



    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        # print(rowsum)
        r_inv = 1 / rowsum
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

        # rowsum = mx.sum(1)
        # rowsum = rowsum[:,np.newaxis]
        # rowsum[rowsum==0] = 1
        # mx = mx/rowsum
        # return mx
