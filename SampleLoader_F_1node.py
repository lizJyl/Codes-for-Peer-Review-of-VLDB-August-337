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

class Com_Split(Dataset):
    def __init__(self, data_dir='/apdcephfs/private_lizyljiang/CommunitySearch/data/facebook/', phase='train',ego=0, method='ATC'):
        super(Com_Split, self).__init__()

        if(data_dir=='facebook'):
            data_dir='/apdcephfs/private_lizyljiang/CommunitySearch/data/facebook/'
        if(data_dir=='twitter'):
            data_dir='/apdcephfs/private_lizyljiang/CommunitySearch/data/Twitter/'
        
        # graph_name = set()
        # for name in os.listdir(data_dir):
        #     graph_name.add(name.split('.')[0])
        # ego=3437
        graph_name = [ego]
        # graph_name = sorted(list(graph_name))
        # if phase == 'train':
        #   graph_name = graph_name[:7]
        # elif phase == 'eval':
        #   graph_name = graph_name[7:]
        # else:
        #   assert 1<0
        feats = {}
        adjs = {}
        node_idx = {}
        samples_dis = {}
        samples_in = {}
        samples_out = {}
        samples_att = {}
        samples_source = {}
        egos = []

        # for name in graph_name:
        for ego in graph_name:
            ego_feat = np.genfromtxt("{}{}.egofeat".format(data_dir, ego), dtype=np.dtype(str))
            feat = np.genfromtxt("{}{}.feat".format(data_dir, ego), dtype=np.dtype(str))
            ego_feat = ego_feat.astype(np.int32)
            feat = feat.astype(np.int32)
            # print('!!!!!You make a change here feat.rand() in Loader.py line 45!!!!!')
            # print(feat.shape, ego_feat.shape)
            # feat_v2 = npr.rand(feat.shape[0], feat.shape[1])
            # features = np.vstack((ego_feat, feat_v2[:, 1:]))
            # features = npr.rand(1046, 3)
            # print("0 feature shape", feat.shape)
            features = np.vstack((ego_feat, feat[:, 1:]))
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

            # for i in range(features.shape[0]):
            #    Adj[i,i]=0
            #    A[i, i] = 1

            # A=self.normalize(A)
            # A=Norm.normalized_adjacency(A)
            # A=Norm.aug_normalized_adjacency(A)
            # A=Norm.bingge_norm_adjacency(A)
            #A=Norm.gcn(A)
            #A=Norm.normalized_laplacian(A)
            #A=Norm.laplacian(A)
            #A=Norm.random_walk_laplacian(A)
            #A=Norm.random_walk(A)
            # A=Norm.aug_random_walk(A)
            A=Norm.i_norm(A)


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
                cur_in = np.genfromtxt("{}ComSplit/ATC_QfN/{}.sample".format(data_dir, ego), dtype=np.dtype(str))
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
                batch_source = []
                max_q = 0
                for i in range(cur_in.shape[0]):
                    source = []
                    for j in range(cur_in.shape[1]):
                        if cur_in[i,j] == 1:
                            source.append(all_dis[j:j+1])
                    source = np.concatenate(source, axis=0) # [Q,D]
                    assert source.shape[0] == cur_in[i].sum()
                    for j in range(cur_in.shape[1]):
                        cur_in_dis[i,j] = source[:,j].min() #mean()
                    batch_source.append(source)
                    max_q = max(max_q, source.shape[0])
                    # print('shape', i, source.shape)
                self.max_q = max_q
                # assert 1<0, (len(batch_source), batch_source[0].shape, cur_in.shape, cur_in_dis.shape)

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
                cur_out = np.genfromtxt("{}ComSplit/ATC_QfN/{}.label".format(data_dir, ego), dtype=np.dtype(str))
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
            file.close()

            if method == 'ACC_QfN':
                cur_att = np.genfromtxt("{}ACC_QfN/{}.query.attr".format(data_dir, ego), dtype=np.dtype(str))
                cur_att = cur_att.astype(np.int32)
                attr = cur_att
            elif method == "ATC_QfN":
                cur_att = np.genfromtxt("{}ComSplit/ATC_QfN/{}.sample.attri".format(data_dir, ego), dtype=np.dtype(str))
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




            # cur_in = cur_in_dis# only distance to represent selected nodes
            # num=cur_in.shape[0]
            # if phase == 'train': # tot 164
            #     # samples_in[int(ego)] = cur_in[43:44]
            #     # samples_out[int(ego)] = cur_out[43:44]
            #     samples_in[int(ego)] = cur_in[num//5*1:num//5*4]
            #     samples_out[int(ego)] = cur_out[num//5*1:num//5*4]
            #     # samples_in[int(ego)] = cur_in[:1]
            #     # samples_out[int(ego)] = cur_out[:1]
            # elif phase == 'eval':
            #     samples_in[int(ego)] = cur_in[:num//5*1]
            #     samples_out[int(ego)] = cur_out[:num//5*1]
            #     # samples_in[int(ego)] = cur_in[num//5*3:num//5*4]
            #     # samples_out[int(ego)] = cur_out[num//5*3:num//5*4]
            # elif phase == 'test':
            #     # samples_in[int(ego)] = cur_in[:num//5*3]
            #     # samples_out[int(ego)] = cur_out[:num//5*3]
            #     samples_in[int(ego)] = cur_in[num//5*4:]
            #     samples_out[int(ego)] = cur_out[num//5*4:]
            # else:
            #     assert 1<0, phase


            # assert 1<0
            
            sum_com = 0
            sum_com_test=0
            sum_com_train=0
            allcount_train=0
            allcount_test=0
            allcount=0

            for i in range(circles.shape[0]):
                if (i % 2 == 0):
                    sum_com_train=sum_com_train+ len(circles[i])
                if (i % 2 == 1):
                    sum_com_test=sum_com_test+ len(circles[i])
                sum_com = sum_com + len(circles[i])

            if phase == 'train':
                for i in range(circles.shape[0]):
                    print('circle i ', i, len(circles[i]),'-----count train',allcount_train,'-----allcount_test',allcount_test)
                    if allcount > 199:
                        break
                    # count = 0
                    count_train=0
                    count_test=0
                    for k in range(len(circles[i]) * 200):
                        if allcount > 199:
                            break   
                        if i % 2 == 0 and allcount_train > 99:
                            break
                        if i % 2 == 1 and allcount_test > 99:
                            break
                        # if count > (200 * len(circles[i]) / sum_com + 1):
                            # break
                        if i % 2 == 0 and count_train > (100 * len(circles[i]) / sum_com_train + 1):
                            break
                        if i % 2 == 1 and count_test > (100 * len(circles[i]) / sum_com_test + 1):
                            break
                        if (i % 2 == 0):
                            allcount_train = allcount_train + 1
                            count_train = count_train + 1
                            if int(ego) not in samples_in:
                                samples_in[int(ego)] = []
                                samples_out[int(ego)] = []
                                samples_dis[int(ego)]=[]
                                samples_att[int(ego)] = []
                                samples_source[int(ego)] = []
                            samples_in[int(ego)].append(cur_in[allcount:allcount+1])
                            samples_dis[int(ego)].append(cur_in_dis[allcount:allcount + 1])
                            samples_out[int(ego)].append(cur_out[allcount:allcount+1])
                            samples_att[int(ego)].append(attr[allcount:allcount + 1])
                            samples_source[int(ego)].append(batch_source[allcount])
                        if (i % 2 == 1):
                            allcount_test = allcount_test + 1
                            count_test = count_test + 1
                        # count = count + 1
                        allcount = allcount + 1
            if phase == 'test':
                for i in range(circles.shape[0]):
                    print('circle i ', i, len(circles[i]),'-----count train',allcount_train,'-----allcount_test',allcount_test)
                    if allcount > 199:
                        break
                    # count = 0
                    count_train=0
                    count_test=0
                    # print('circle i ', i, len(circles[i]),'-----count train',allcount_train,'------allcount_test',allcount_test)
                    for k in range(len(circles[i]) * 200):
                        if allcount > 199:
                            break   
                        if i % 2 == 0 and allcount_train > 99:
                            break
                        if i % 2 == 1 and allcount_test > 99:
                            break
                        # if count > (200 * len(circles[i]) / sum_com + 1):
                            # break
                        if i % 2 == 0 and count_train > (100 * len(circles[i]) / sum_com_train + 1):
                            break
                        if i % 2 == 1 and count_test > (100 * len(circles[i]) / sum_com_test + 1):
                            break
                        if (i % 2 == 1):
                            allcount_test = allcount_test + 1
                            count_test = count_test + 1
                            if int(ego) not in samples_in:
                                samples_in[int(ego)] = []
                                samples_out[int(ego)] = []
                                samples_dis[int(ego)] = []
                                samples_att[int(ego)] = []
                                samples_source[int(ego)] = []
                            samples_dis[int(ego)].append(cur_in_dis[allcount:allcount + 1])
                            samples_in[int(ego)].append(cur_in[allcount:allcount+1])
                            samples_out[int(ego)].append(cur_out[allcount:allcount+1])
                            samples_att[int(ego)].append(attr[allcount:allcount + 1])
                            samples_source[int(ego)].append(batch_source[allcount])
                        if (i % 2 == 0):
                            allcount_train = allcount_train + 1
                            count_train = count_train + 1
                        allcount = allcount + 1
            samples_in[int(ego)] = np.concatenate(samples_in[int(ego)], 0)
            samples_dis[int(ego)] = np.concatenate(samples_dis[int(ego)], 0)
            samples_out[int(ego)] = np.concatenate(samples_out[int(ego)], 0)
            samples_att[int(ego)] = np.concatenate(samples_att[int(ego)], 0)
            # assert 1<0, (cur_in.shape, samples_in[int(ego)].shape)
            print('OUT    -----count train',allcount_train,'-----allcount_test',allcount_test)
            print('samples_in', samples_in[int(ego)].shape[0])

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
        self.smaples_source = samples_source
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
        # id = npr.choice(self.circle.shape[0], 1, replace=False)[0]
        # cmts = self.circle[id]
        # print("train number", self.samples_in[self.ego].shape[0])
        # node_map=self.node_map

        #
        # degree = self.degree[:, np.newaxis]
        # core = self.core[:, np.newaxis]
        # cluster = self.cluster[:, np.newaxis]
        # triangle = self.triangle[:, np.newaxis]

        # all_dis = self.all_dis
        #
        # sample_node_num = min(self.max_select_num, len(cmts) // 2)
        # if sample_node_num == 0 and len(cmts) == 1:
        #     sample_node_num = 1
        # else:
        #     assert sample_node_num > 0, (sample_node_num, len(cmts))
        # i = 0
        # while 1:
        #     #    for i in range(100):
        #     i += 1
        #     if (i % 100) == 99 or len(cmts)<5:
        #         id = npr.choice(self.circle.shape[0], 1, replace=False)[0]
        #         cmts = self.circle[id]
        #         sample_node_num = min(self.max_select_num, len(cmts) // 2)
        #         if sample_node_num == 0 and len(cmts) == 1:
        #             sample_node_num = 1
        #         else:
        #             assert sample_node_num > 0, (sample_node_num, len(cmts))
        #         sample_node_num = random.randint(1, sample_node_num)
        #     sample_nodes = npr.choice(cmts, sample_node_num, replace=False)
        #     sample_nodes= sample_nodes.astype(int)
        #     # print("sample nodes sum",sample_nodes.sum(),sample_nodes.sum()%5 )
        #     if self.phase == 'train':
        #         if ((sample_nodes.sum() % 5) < 3):
        #             break
        #     elif self.phase == 'eval':
        #         if ((sample_nodes.sum() % 5) == 3):  # and (sample_nodes.sum() % 50) < 40):
        #             break
        #     elif self.phase == 'test':
        #         if ((sample_nodes.sum() % 5) == 4):
        #             break
        #
        # # sample_node_num = random.randint(1, sample_node_num)
        # # sample_nodes = npr.choice(cmts, sample_node_num, replace=False)
        #
        # cur_input = np.zeros(self.adjs[ego].shape[0], dtype=np.float32)
        # cur_adj = self.adjs[ego].copy()
        # cur_out = np.zeros(self.adjs[ego].shape[0], dtype=np.float32)
        #
        # for k in sample_nodes:
        #     # for k in cmts:
        #     cur_input[node_map[k]] = 1
        #
        # cur_in_dis = np.zeros(cur_input.shape, dtype=np.float32)
        # source = []
        # for j in range(cur_input.shape[0]):
        #     if cur_input[j] == 1:
        #         source.append(all_dis[j:j + 1])
        # source = np.concatenate(source, axis=0)
        # assert source.shape[0] == cur_input.sum()
        # for j in range(cur_input.shape[0]):
        #     cur_in_dis[j] = source[:, j].min()  # mean()
        #
        # # for j in range(cur_in_dis.shape[0]):
        # # print("dis",cur_in_dis)
        # max_val = (cur_in_dis * (cur_in_dis < cur_adj.shape[0])).max()
        # # print("max distance ", max_val)
        # # assert max_val > 0., (max_val)
        # if max_val == 0:
        #     max_val = cur_in_dis.shape[0] - 1
        # for i in range(cur_in_dis.shape[0]):
        #     if cur_in_dis[i] == cur_adj.shape[0]:  # unreachable:
        #         cur_in_dis[i] = 0.
        #     else:
        #         cur_in_dis[i] = 1. - cur_in_dis[i] / (max_val + 1)
        # for k in range(cur_input.shape[0]):
        #     if cur_input[k] == 1:
        #         cur_in_dis[k] = 1
        # # print("distance ", cur_in_dis)
        #
        # for k in cmts:
        #     cur_out[node_map[int(k)]] = 1
        #
        # cur_adj = self.adjs[ego].copy()
        # feats = self.feats[ego].copy()
        # input = np.concatenate((feats, cur_in_dis[:, np.newaxis]), axis=1)  # BN(D+2)
        # input = np.concatenate((input, cur_input[:, np.newaxis]), axis=1)  # BN(D+1)
        #
        # cur_att = self.attr[id].copy()[:, np.newaxis]
        # print("att",cur_att.shape)
        # if len(self.samples_in[ego]) < self.batch_size:
        #   cur_in = self.samples_in[ego].copy()
        #   cur_out = self.samples_out[ego].copy()
        # else:
        #   rand_idx = npr.choice(self.samples_in[ego].shape[0], self.batch_size, replace=False)
        #   cur_in = self.samples_in[ego].copy()[rand_idx,:]
        #   cur_out = self.samples_out[ego].copy()[rand_idx,:]

            # cur_in = self.samples_in[ego].copy()[0:1,:]
            # cur_out = self.samples_out[ego].copy()[0:1,:]
        cur_dis = self.samples_dis[ego].copy()[item, :]
        cur_in = self.samples_in[ego].copy()[item,:]
        cur_out = self.samples_out[ego].copy()[item,:]
        cur_att = self.samples_att[ego].copy()[item, :]
        cur_source = self.smaples_source[ego][item].copy()
        cur_source_pad = np.zeros((self.max_q, cur_source.shape[1]), dtype=cur_source.dtype)
        cur_source_pad[:cur_source.shape[0], :cur_source.shape[1]] = cur_source
        cur_source = cur_source_pad.copy()
        # assert 1<0, (cur_dis.shape, cur_source.shape, type(cur_dis), type(cur_source))
        # print('cur', cur_dis.shape, cur_in.shape, cur_out.shape, cur_att.shape, cur_source.shape)
        # assert 1<0, (self.samples_dis[ego].shape, item, ego)

        degree=self.degree[:,np.newaxis]
        core = self.core[:, np.newaxis]
        cluster = self.cluster[:, np.newaxis]
        triangle=self.triangle[:, np.newaxis]

        cur_in = cur_in[:,np.newaxis] # BN1
        cur_dis = cur_dis[:, np.newaxis]
        cur_out = cur_out[:,np.newaxis] # BN1
        cur_att = cur_att[:, np.newaxis]  # BN1
        # flag = False # true no feat;    false: with feat
        # if flag:
        #     cur_in = np.concatenate((cur_in, cur_att), axis=0)
        # else:
        #     cur_feats = self.feats[ego].copy()
        #     # cur_feats = np.repeat(cur_feats, cur_in.shape[0], axis=0) # BND
        #     # print('shape', cur_feats.shape, cur_in.shape)
        #     # cur_in = np.repeat(cur_in, cur_feats.shape[1], axis=1)
        #
        #     # cur_feats = np.concatenate((cur_feats, degree), axis=1)  # BN(D+2)
        #     # cur_feats = np.concatenate((cur_feats, core), axis=1)  # BN(D+2)
        #     # cur_feats = np.concatenate((cur_feats, cluster), axis=1)  # BN(D+2)
        #     # cur_feats = np.concatenate((cur_feats, triangle), axis=1)  # BN(D+2)
        #     cur_feats = np.concatenate((cur_feats, cur_in), axis=1) # BN(D+1)
        #     cur_feats = np.concatenate((cur_feats, cur_dis), axis=1)  # BN(D+2)

        cur_adj = self.adjs[ego].copy()
        feats = self.feats[ego].copy()
        # input=np.concatenate((self.feats[ego].copy(), cur_dis), axis=1)  # BN(D+2)


        # input = cur_in
        # input=np.concatenate((input, cur_in), axis=1) # BN(D+1)

        # input=cur_dis
        # input = np.concatenate((input, cur_dis), axis=1)  # BN(D+1)
        #
        input = cur_in
        input = np.concatenate((input, cur_dis), axis=1)  # BN(D+1)

        # return torch.FloatTensor(cur_feats), torch.FloatTensor(cur_out), torch.FloatTensor(cur_adj)
        return input, cur_att,cur_adj,feats, cur_out, self.Adj, cur_source

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


class GraphDataset(Dataset):
    def __init__(self, data_dir='ComSplit/ATC_QfN/', phase='train',ego=0, method='ATC'):
        super(GraphDataset, self).__init__()

        if(data_dir=='facebook'):
            data_dir='ComSplit/ATC_QfN/'
        if(data_dir=='twitter'):
            data_dir='/apdcephfs/private_lizyljiang/CommunitySearch/data/Twitter/'
        
        # graph_name = set()
        # for name in os.listdir(data_dir):
        #     graph_name.add(name.split('.')[0])
        # ego=3437
        graph_name = [ego]
        # graph_name = sorted(list(graph_name))
        # if phase == 'train':
        # 	graph_name = graph_name[:7]
        # elif phase == 'eval':
        # 	graph_name = graph_name[7:]
        # else:
        # 	assert 1<0
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
            # print('!!!!!You make a change here feat.rand() in Loader.py line 45!!!!!')
            # print(feat.shape, ego_feat.shape)
            # feat_v2 = npr.rand(feat.shape[0], feat.shape[1])
            # features = np.vstack((ego_feat, feat_v2[:, 1:]))
            # features = npr.rand(1046, 3)
            # print("0 feature shape", feat.shape)
            features = np.vstack((ego_feat, feat[:, 1:]))
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




            # cur_in = cur_in_dis# only distance to represent selected nodes
            # num=cur_in.shape[0]
            # if phase == 'train': # tot 164
            #     # samples_in[int(ego)] = cur_in[43:44]
            #     # samples_out[int(ego)] = cur_out[43:44]
            #     samples_in[int(ego)] = cur_in[num//5*1:num//5*4]
            #     samples_out[int(ego)] = cur_out[num//5*1:num//5*4]
            #     # samples_in[int(ego)] = cur_in[:1]
            #     # samples_out[int(ego)] = cur_out[:1]
            # elif phase == 'eval':
            #     samples_in[int(ego)] = cur_in[:num//5*1]
            #     samples_out[int(ego)] = cur_out[:num//5*1]
            #     # samples_in[int(ego)] = cur_in[num//5*3:num//5*4]
            #     # samples_out[int(ego)] = cur_out[num//5*3:num//5*4]
            # elif phase == 'test':
            #     # samples_in[int(ego)] = cur_in[:num//5*3]
            #     # samples_out[int(ego)] = cur_out[:num//5*3]
            #     samples_in[int(ego)] = cur_in[num//5*4:]
            #     samples_out[int(ego)] = cur_out[num//5*4:]
            # else:
            #     assert 1<0, phase


            # assert 1<0
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
        # id = npr.choice(self.circle.shape[0], 1, replace=False)[0]
        # cmts = self.circle[id]
        # # print("id", id)
        # node_map=self.node_map

        #
        # degree = self.degree[:, np.newaxis]
        # core = self.core[:, np.newaxis]
        # cluster = self.cluster[:, np.newaxis]
        # triangle = self.triangle[:, np.newaxis]

        # all_dis = self.all_dis
        #
        # sample_node_num = min(self.max_select_num, len(cmts) // 2)
        # if sample_node_num == 0 and len(cmts) == 1:
        #     sample_node_num = 1
        # else:
        #     assert sample_node_num > 0, (sample_node_num, len(cmts))
        # i = 0
        # while 1:
        #     #    for i in range(100):
        #     i += 1
        #     if (i % 100) == 99 or len(cmts)<5:
        #         id = npr.choice(self.circle.shape[0], 1, replace=False)[0]
        #         cmts = self.circle[id]
        #         sample_node_num = min(self.max_select_num, len(cmts) // 2)
        #         if sample_node_num == 0 and len(cmts) == 1:
        #             sample_node_num = 1
        #         else:
        #             assert sample_node_num > 0, (sample_node_num, len(cmts))
        #         sample_node_num = random.randint(1, sample_node_num)
        #     sample_nodes = npr.choice(cmts, sample_node_num, replace=False)
        #     sample_nodes= sample_nodes.astype(int)
        #     # print("sample nodes sum",sample_nodes.sum(),sample_nodes.sum()%5 )
        #     if self.phase == 'train':
        #         if ((sample_nodes.sum() % 5) < 3):
        #             break
        #     elif self.phase == 'eval':
        #         if ((sample_nodes.sum() % 5) == 3):  # and (sample_nodes.sum() % 50) < 40):
        #             break
        #     elif self.phase == 'test':
        #         if ((sample_nodes.sum() % 5) == 4):
        #             break
        #
        # # sample_node_num = random.randint(1, sample_node_num)
        # # sample_nodes = npr.choice(cmts, sample_node_num, replace=False)
        #
        # cur_input = np.zeros(self.adjs[ego].shape[0], dtype=np.float32)
        # cur_adj = self.adjs[ego].copy()
        # cur_out = np.zeros(self.adjs[ego].shape[0], dtype=np.float32)
        #
        # for k in sample_nodes:
        #     # for k in cmts:
        #     cur_input[node_map[k]] = 1
        #
        # cur_in_dis = np.zeros(cur_input.shape, dtype=np.float32)
        # source = []
        # for j in range(cur_input.shape[0]):
        #     if cur_input[j] == 1:
        #         source.append(all_dis[j:j + 1])
        # source = np.concatenate(source, axis=0)
        # assert source.shape[0] == cur_input.sum()
        # for j in range(cur_input.shape[0]):
        #     cur_in_dis[j] = source[:, j].min()  # mean()
        #
        # # for j in range(cur_in_dis.shape[0]):
        # # print("dis",cur_in_dis)
        # max_val = (cur_in_dis * (cur_in_dis < cur_adj.shape[0])).max()
        # # print("max distance ", max_val)
        # # assert max_val > 0., (max_val)
        # if max_val == 0:
        #     max_val = cur_in_dis.shape[0] - 1
        # for i in range(cur_in_dis.shape[0]):
        #     if cur_in_dis[i] == cur_adj.shape[0]:  # unreachable:
        #         cur_in_dis[i] = 0.
        #     else:
        #         cur_in_dis[i] = 1. - cur_in_dis[i] / (max_val + 1)
        # for k in range(cur_input.shape[0]):
        #     if cur_input[k] == 1:
        #         cur_in_dis[k] = 1
        # # print("distance ", cur_in_dis)
        #
        # for k in cmts:
        #     cur_out[node_map[int(k)]] = 1
        #
        # cur_adj = self.adjs[ego].copy()
        # feats = self.feats[ego].copy()
        # input = np.concatenate((feats, cur_in_dis[:, np.newaxis]), axis=1)  # BN(D+2)
        # input = np.concatenate((input, cur_input[:, np.newaxis]), axis=1)  # BN(D+1)
        #
        # cur_att = self.attr[id].copy()[:, np.newaxis]
        # print("att",cur_att.shape)
        # if len(self.samples_in[ego]) < self.batch_size:
        # 	cur_in = self.samples_in[ego].copy()
        # 	cur_out = self.samples_out[ego].copy()
        # else:
        # 	rand_idx = npr.choice(self.samples_in[ego].shape[0], self.batch_size, replace=False)
        # 	cur_in = self.samples_in[ego].copy()[rand_idx,:]
        # 	cur_out = self.samples_out[ego].copy()[rand_idx,:]

            # cur_in = self.samples_in[ego].copy()[0:1,:]
            # cur_out = self.samples_out[ego].copy()[0:1,:]
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
        # flag = False # true no feat;    false: with feat
        # if flag:
        #     cur_in = np.concatenate((cur_in, cur_att), axis=0)
        # else:
        #     cur_feats = self.feats[ego].copy()
        #     # cur_feats = np.repeat(cur_feats, cur_in.shape[0], axis=0) # BND
        #     # print('shape', cur_feats.shape, cur_in.shape)
        #     # cur_in = np.repeat(cur_in, cur_feats.shape[1], axis=1)
        #
        #     # cur_feats = np.concatenate((cur_feats, degree), axis=1)  # BN(D+2)
        #     # cur_feats = np.concatenate((cur_feats, core), axis=1)  # BN(D+2)
        #     # cur_feats = np.concatenate((cur_feats, cluster), axis=1)  # BN(D+2)
        #     # cur_feats = np.concatenate((cur_feats, triangle), axis=1)  # BN(D+2)
        #     cur_feats = np.concatenate((cur_feats, cur_in), axis=1) # BN(D+1)
        #     cur_feats = np.concatenate((cur_feats, cur_dis), axis=1)  # BN(D+2)

        cur_adj = self.adjs[ego].copy()
        feats = self.feats[ego].copy()
        # input=np.concatenate((self.feats[ego].copy(), cur_dis), axis=1)  # BN(D+2)


        # input = cur_in
        # input=np.concatenate((input, cur_in), axis=1) # BN(D+1)

        # input=cur_dis
        # input = np.concatenate((input, cur_dis), axis=1)  # BN(D+1)
        #
        input = cur_in
        input = np.concatenate((input, cur_dis), axis=1)  # BN(D+1)

        # return torch.FloatTensor(cur_feats), torch.FloatTensor(cur_out), torch.FloatTensor(cur_adj)
        return input, cur_att,cur_adj,feats, cur_out, self.Adj

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

class EmailDataset(Dataset):
    def __init__(self, edge_files='/apdcephfs/private_lizyljiang/CommunitySearchdata/Email/email-Eu-core.txt', id_files='/apdcephfs/private_lizyljiang/CommunitySearchdata/Email/email-Eu-core-department-labels.txt',
                 phase='train', thr=10, max_select_num=3):
        super(EmailDataset, self).__init__()
        G = nx.Graph()
        Edges = []

        edges = np.genfromtxt(edge_files, dtype=np.dtype(str))
        edges = edges.astype(np.int32)
        for e in edges:
            # if (e[0]!=e[1]):
            Edges.append((e[0],e[1]))

        G.add_edges_from(Edges)

        G.remove_edges_from(nx.selfloop_edges(G))



        A = nx.adjacency_matrix(G).todense()
        A = np.array(A, dtype=np.float32)

        print("A.shape",A.shape)

        degree = np.zeros(A.shape[0], dtype=np.float32)
        # print("degree", G.degree)
        print("degree[0]", G.degree[0])
        for i in range(A.shape[0]):
            # if i in G.degree:
            degree[i] = G.degree[i] / A.shape[0]
            # else:
            #     degree[i] =0
        print("size of degree",degree)
        core_G = nx.core_number(G)
        core = np.zeros(A.shape[0], dtype=np.float32)
        for i in range(A.shape[0]):
            if i in core_G:
                core[i] = core_G[i]
            else:
                core[i] =0

        # print("size of core",core)
        core = core / core.max()
        # print("Norm core", core)

        cluster_G = nx.clustering(G)
        cluster = np.zeros(A.shape[0], dtype=np.float32)
        for i in range(A.shape[0]):
            if i in cluster_G:
                cluster[i] = cluster_G[i]
            else:
                cluster[i] = 0

        # print("size of cluster",cluster)
        cluster = cluster / cluster.max()
        # print("Norm cluster", cluster)

        triangle_G = nx.triangles(G)
        triangle = np.zeros(A.shape[0], dtype=np.float32)
        for i in range(A.shape[0]):
            if i in triangle_G:
                triangle[i] = triangle_G[i]
            else:
                triangle[i] =0

        # print("size of triangle",triangle)
        triangle = triangle / triangle.max()
        # print("Norm cluster", cluster)

        dis = nx.shortest_path(G)
        all_dis = np.zeros(A.shape, dtype=np.int32)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                # all_dis[i,j] = len(dis[i][j])-1
                if i == j:
                    all_dis[i, j] = 0
                else:
                    if j in dis[i]:
                        assert len(dis[i][j]) > 1
                        all_dis[i, j] = len(dis[i][j]) - 1
                    else:
                        all_dis[i, j] = A.shape[0]  # we set specially here







        # nx.draw(G, with_labels=True, edge_color='b', node_color='g', node_size=1000)
        # plt.show()
        # assert 1<0



        # print((A.transpose(0, 1) != A).sum() == 0)
        # print(A.transpose(0, 1) == A)
        for i in range(A.shape[0]):
            A[i,i] = 1

        A = self.normalize(A)

        idx = np.genfromtxt(id_files, dtype=np.dtype(str))
        idx = idx.astype(np.int32)
        cmt = {}
        for node in idx:
            if node[1] not in cmt:
                cmt[node[1]] = []
            cmt[node[1]].append(node[0])
        all_ids = [k for k in sorted(cmt.keys())]

        cmt_ids =all_ids
        # if phase == 'train':
        #     cmt_ids = []
        #     for i in range(len(all_ids)):
        #         if i % 7 ==0 or i % 7 == 5 or i % 7 == 6:
        #             cmt_ids.append(all_ids[i])
        #         else:
        #             del cmt[all_ids[i]]
        # elif phase == 'eval':
        #     cmt_ids = []
        #     for i in range(len(all_ids)):
        #         if i % 7 == 1 or i % 7 == 4:
        #             cmt_ids.append(all_ids[i])
        #         else:
        #             del cmt[all_ids[i]]
        # elif phase == 'test':
        #     cmt_ids = []
        #     for i in range(len(all_ids)):
        #         if i % 7 == 2 or i % 7 == 3:
        #             cmt_ids.append(all_ids[i])
        #         else:
        #             del cmt[all_ids[i]]
        # else:
        #     cmt_ids = None
        #     assert 1<0, phase

        self.cmt_ids = cmt_ids
        self.cmt = cmt
        self.A = A
        self.max_select_num = max_select_num
        self.phase = phase

        self.degree = degree
        self.core = core
        self.cluster = cluster
        self.triangle = triangle
        self.all_dis=all_dis

        random.seed(20170624)
        npr.seed(20170624)

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        # print(rowsum)
        r_inv = 1 / rowsum
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def __len__(self):
        return len(self.cmt_ids)

    def __getitem__(self, item):
        # id = self.cmt_ids[item]
        # print("id",id)
        # id = self.cmt_ids[0]
        id = npr.choice(self.cmt_ids, 1, replace=False)[0]
        cmts = self.cmt[id]
        # print("id", id)

        degree = self.degree[:, np.newaxis]
        core = self.core[:, np.newaxis]
        cluster = self.cluster[:, np.newaxis]
        triangle = self.triangle[:, np.newaxis]

        all_dis= self.all_dis


        sample_node_num = min(self.max_select_num, len(cmts)//2)
        if sample_node_num == 0 and len(cmts) == 1:
            sample_node_num = 1
        else:
            assert sample_node_num > 0, (sample_node_num, len(cmts))
        i=0
        while 1:
        #    for i in range(100):
            i+=1
            sample_nodes = npr.choice(cmts, sample_node_num, replace=False)
            #print("sample nodes sum",sample_nodes.sum(),sample_nodes.sum()%5 )
            if self.phase == 'train':
                if ((sample_nodes.sum() % 5) < 3):
                    break
            elif self.phase == 'eval':
                if ((sample_nodes.sum() % 5) == 3):# and (sample_nodes.sum() % 50) < 40):
                    break
            elif self.phase == 'test':
                if ((sample_nodes.sum() % 5) == 4):
                    break
            if (i%100)==0:
                id = npr.choice(self.cmt_ids, 1, replace=False)[0]
                cmts = self.cmt[id]
                sample_node_num = min(self.max_select_num, len(cmts) // 2)
                if sample_node_num == 0 and len(cmts) == 1:
                    sample_node_num = 1
                else:
                    assert sample_node_num > 0, (sample_node_num, len(cmts))
                sample_node_num = random.randint(1, sample_node_num)
        # sample_node_num = random.randint(1, sample_node_num)
        # sample_nodes = npr.choice(cmts, sample_node_num, replace=False)

        cur_input = np.zeros(self.A.shape[0], dtype=np.float32)
        cur_adj = self.A.copy()
        cur_label = np.zeros(self.A.shape[0], dtype=np.float32)

        for k in sample_nodes:
        # for k in cmts:
            cur_input[k] = 1

        cur_in_dis = np.zeros(cur_input.shape, dtype=np.float32)
        source = []
        for j in range(cur_input.shape[0]):
            if cur_input[j] == 1:
                source.append(all_dis[j:j + 1])
        source = np.concatenate(source, axis=0)
        assert source.shape[0] == cur_input.sum()
        for j in range(cur_input.shape[0]):
            cur_in_dis[j] = source[:, j].min()  # mean()

        # for j in range(cur_in_dis.shape[0]):
        # print("dis",cur_in_dis)
        max_val = (cur_in_dis * (cur_in_dis < cur_adj.shape[0])).max()
        # print("max distance ", max_val)
        # assert max_val > 0., (max_val)
        if max_val==0:
            max_val=cur_in_dis.shape[0]-1
        for i in range(cur_in_dis.shape[0]):
            if cur_in_dis[i] == cur_adj.shape[0]:  # unreachable:
                cur_in_dis[i] = 0.
            else:
                cur_in_dis[i] = 1. - cur_in_dis[i] / (max_val + 1)
        for k in range(cur_input.shape[0]):
            if cur_input[k] == 1:
                cur_in_dis[k] = 1
        # print("distance ", cur_in_dis)

        for k in cmts:
            cur_label[k] = 1
        cur_feats = cur_input[:, np.newaxis]
        # cur_feats = cur_in_dis[:, np.newaxis]
        cur_feats = np.concatenate((cur_feats, cur_in_dis[:, np.newaxis]), axis=1)  # BN(D+2)
        cur_feats = np.concatenate((cur_feats, degree), axis=1)  # BN(D+2)
        cur_feats = np.concatenate((cur_feats, core), axis=1)  # BN(D+2)
        # cur_feats = np.concatenate((cur_feats, cluster), axis=1)  # BN(D+2)
        cur_feats = np.concatenate((cur_feats, triangle), axis=1)  # BN(D+2)


        # cur_input -= 0.5
        return cur_feats, cur_adj, cur_label

class DBLPDataset(Dataset):
    def __init__(self, edge_files='/apdcephfs/private_lizyljiang/CommunitySearchdata/DBLP/com-dblp.ungraph.txt',
                 cmt_files='/apdcephfs/private_lizyljiang/CommunitySearchdata/DBLP/com-dblp.top5000.cmty.txt',
                 phase='train', thr=10, max_select_num=3):
        super(DBLPDataset, self).__init__()
        # G = nx.Graph()
        # Edges = []
        #
        # edges = np.genfromtxt(edge_files, dtype=np.dtype(str))
        # edges = edges.astype(np.int32)
        # for e in edges:
        # 	Edges.append((e[0], e[1]))
        #
        # G.add_edges_from(Edges)
        #
        # A = nx.adjacency_matrix(G).todense()
        # A = np.array(A, dtype=np.float32)
        print("load DBLP dateset",phase)
        node=set()
        edges = np.genfromtxt(edge_files, dtype=np.int32)
        # edges = edges.astype(np.int32)
        for e in edges:
            node.add(e[0])
            node.add(e[1])
        # node=set(node)
        node_map = {j: i for i, j in enumerate(node)}

        print("node size",len(node))

        # print(node_map)
        # assert 1<0

        print("build DBLP Adj",phase)
        # A = np.zeros((len(node), len(node)), dtype=np.int32)
        # for e in edges:
        # 	A[node_map[e[0]], node_map[e[1]]] = 1
        # 	A[node_map[e[1]], node_map[e[0]]] = 1
        # # for i in range(317080):
        # # 	A[i, i] = 1
        # A = np.array(A, dtype=np.float32)

        # for i in range(A.shape[0]):
        # 	A[i,i] = 1

        # A = self.normalize(A)
        edges_map = np.array(list(map(node_map.get, edges.flatten())),
                         dtype=np.int32).reshape(edges.shape)

        adj = sp.coo_matrix((np.ones(edges_map.shape[0]), (edges_map[:, 0], edges_map[:, 1])),
                            # ones  all 1 vector size m(edge size)
                            shape=(len(node), len(node)),
                            dtype=np.float32)

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        A = self.SPnormalize(adj)

        Edges = []
        for e in edges:
            Edges.append((node_map[e[0]], node_map[e[1]]))
        #     Edges.append((node_map[e[1]], node_map[e[0]]))
        # # Edges.append((0, 0))
        # for i in range(adj.shape[0]):
        #     Edges.append((i , i ))
        G = nx.Graph()
        G.add_edges_from(Edges)


        print("load DBLP community set",phase)

        file = open(cmt_files, 'r')
        cmt_list = file.readlines()
        # val_list=val_list[:,0:-1]
        cmt = []
        for string in cmt_list:
            #	string=string.split('\n')
            string = string.strip().split('\t')
            # print(string[1:])
            cmt.append(string[1:])
        file.close()

        print("cmt size", len(cmt))

        # print(cmt[0][0],int(cmt[0][0]))
        # assert 1<0

        cmts = {}
        for i in range(len(cmt)):
            cmts[i] = []
            # print("cmt i size",len(cmt[i]))
            for j in range(len(cmt[i])):
                cmts[i].append(node_map[int(cmt[i][j])])
        # all_ids = [k for k in sorted(cmt.keys())]
        all_cmt=[k for k in sorted(cmts.keys())]
        # if phase == 'train':
        #     cmt_ids = []
        #     for i in range(len(all_cmt)):
        #         if i % 5 == 0 or i % 5 == 3 or i % 5 == 4:
        #             cmt_ids.append(all_cmt[i])
        #         else:
        #             del cmts[all_cmt[i]]
        # elif phase == 'eval':
        #     cmt_ids = []
        #     for i in range(len(all_cmt)):
        #         if i % 5 == 1 :
        #             cmt_ids.append(all_cmt[i])
        #         else:
        #             del cmts[all_cmt[i]]
        # elif phase == 'test':
        #     cmt_ids = []
        #     for i in range(len(all_cmt)):
        #         if i % 5 == 2 :
        #             cmt_ids.append(all_cmt[i])
        #         else:
        #             del cmts[all_cmt[i]]
        # else:
        #     cmt_ids = None
        #     assert 1 < 0, phase

        self.cmt_ids = all_cmt
        self.G=G
        # self.cmt_ids = cmt_ids
        self.cmt = cmts
        self.A = A
        self.max_select_num = max_select_num
        self.phase = phase
        random.seed(20170624)
        npr.seed(20170624)

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        # print(rowsum)
        r_inv = 1 / rowsum
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def SPnormalize(self,mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.  # 1/0   -> 0
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)  # matrix *
        return mx

    def sparse_mx_to_torch_sparse_tensor(self,sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)  # .row   .col   .data for not zero data
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))  # vstck() bingjie two matrix  [.row
        # .col]
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

        # translate to torch's matrix tensor
        # shape    matrix size
        # indices  not zero location
        # values   not zero value


    def __len__(self):
        return len(self.cmt_ids)
        #return 8

    def __getitem__(self, item):
        # id = self.cmt_ids[item]
        # print("id",id)
        # id = self.cmt_ids[0]
        id = npr.choice(self.cmt_ids, 1, replace=False)[0]
        cmts = self.cmt[id]
        #print("id", id)

        sample_node_num = min(self.max_select_num, len(cmts) // 2)
        if sample_node_num == 0 and len(cmts) == 1:
            sample_node_num = 1
        else:
            assert sample_node_num > 0, (sample_node_num, len(cmts))
        sample_node_num = random.randint(1, sample_node_num)
        # sample_nodes = npr.choice(cmts, sample_node_num, replace=False)
        i=0
        while 1:
        #    for i in range(100):
            i+=1
            sample_nodes = npr.choice(cmts, sample_node_num, replace=False)
            #print("sample nodes sum",sample_nodes.sum(),sample_nodes.sum()%5 )
            if self.phase == 'train':
                if ((sample_nodes.sum() % 5) < 3):
                    break
            elif self.phase == 'eval':
                if ((sample_nodes.sum() % 5) == 3):# and (sample_nodes.sum() % 50) < 40):
                    break
            elif self.phase == 'test':
                if ((sample_nodes.sum() % 5) == 4):
                    break
            if (i%100)==0:
                id = npr.choice(self.cmt_ids, 1, replace=False)[0]
                cmts = self.cmt[id]
                sample_node_num = min(self.max_select_num, len(cmts) // 2)
                if sample_node_num == 0 and len(cmts) == 1:
                    sample_node_num = 1
                else:
                    assert sample_node_num > 0, (sample_node_num, len(cmts))
                sample_node_num = random.randint(1, sample_node_num)
        cur_adj = self.A.copy()
       # print("out")

        cur_input = np.zeros(cur_adj.shape[0], dtype=np.float32)
        # cur_adj = self.A.copy()
        cur_label = np.zeros(cur_adj.shape[0], dtype=np.float32)
        source=[]
        for k in sample_nodes:
            if True:
                dis = nx.shortest_path(self.G, source=k)
                all_dis = np.zeros(cur_adj.shape[0], dtype=np.int32)
                for i in range(cur_adj.shape[0]):
                    if i==k:
                        all_dis[i]=0
                    else:
                        if i in dis:
                            assert len(dis[i]) > 1
                            all_dis[i] = len(dis[i]) - 1
                        else:
                            all_dis[i] = cur_adj.shape[0]
                source.append(all_dis[np.newaxis])
            cur_input[k] = 1
        if True:
            source = np.concatenate(source, axis=0)
            # assert 1<0, (source.shape)
            #print("source.shape[0],cur_input.sum()",source.shape[0],cur_input.sum())
            assert source.shape[0] == cur_input.sum()
            cur_in_dis = np.zeros(cur_input.shape, dtype=np.float32)
            for i in range(cur_input.shape[0]):
                cur_in_dis[i] = source[:, i].min()  # or mean()
    
            max_val = (cur_in_dis * (cur_in_dis < cur_input.shape)).max()
            # print("max distance ", max_val)
            assert max_val > 0., (max_val, sample_nodes)
            for i in range(cur_in_dis.shape[0]):
                if cur_in_dis[i] == cur_input.shape:  # unreachable:
                    cur_in_dis[i] = 0.
                elif cur_in_dis[i]==0:
                    cur_in_dis[i]=1.
                else:
                    cur_in_dis[i] = 1. - cur_in_dis[i] / (max_val + 1)

        # print("dis max" , cur_in_dis.max())
        #
        # print("dis min", cur_in_dis.min())


        for k in cmts:
            cur_label[k] = 1

        cur_adj = self.sparse_mx_to_torch_sparse_tensor(cur_adj)
        #cur_input = torch.FloatTensor(cur_input)
        cur_input = torch.FloatTensor(cur_in_dis)
        cur_label = torch.FloatTensor(cur_label)
        # cur_input = torch.LongTensor(cur_input)
        # cur_label = torch.LongTensor(cur_label)

        # cur_input -= 0.5
        return cur_input, cur_adj, cur_label

class SPcoraDataset(Dataset):
    def __init__(self, edge_files='/apdcephfs/private_lizyljiang/CommunitySearchdata/cora/cora/cora.cites',
                 id_files='/apdcephfs/private_lizyljiang/CommunitySearchdata/cora/cora/cora.content',
                 phase='train', thr=10, max_select_num=3):
        super(SPcoraDataset, self).__init__()

        print("load cora dateset", phase)

        node = set()
        edges = np.genfromtxt(edge_files, dtype=np.int32)
        for e in edges:
            node.add(e[0])
            node.add(e[1])
        node_map = {j: i for i, j in enumerate(node)}

        print("node size", len(node))
        #
        # print("build cora Adj", phase)
        #
        # edges_map = np.array(list(map(node_map.get, edges.flatten())),
        # 					 dtype=np.int32).reshape(edges.shape)
        # adj = sp.coo_matrix((np.ones(edges_map.shape[0]), (edges_map[:, 0], edges_map[:, 1])),
        # 					shape=(len(node), len(node)),
        # 					dtype=np.float32)
        # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # A = self.SPnormalize(adj)


        A = np.zeros((len(node), len(node)), dtype=np.int32)
        for e in edges:
            A[node_map[e[0]], node_map[e[1]]] = 1
            A[node_map[e[1]], node_map[e[0]]] = 1
        for i in range(A.shape[0]):
            A[i,i] = 1
        A = np.array(A, dtype=np.float32)

        A = self.normalize(A)

        print("adj shape", A.shape)


        ### get dis using nx ###
        # print('!!!add dis in loader.py line 724 !!!')
        use_dis = True
        if use_dis:
            Edges = []
            for e in edges:
                Edges.append((node_map[e[0]], node_map[e[1]]))
            G = nx.Graph()
            G.add_edges_from(Edges)
            dis = nx.shortest_path(G)
            all_dis = np.zeros(A.shape, dtype=np.int32)
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    if i==j:
                        all_dis[i, j]=0
                    else:
                        if j in dis[i]:
                            assert len(dis[i][j])>1
                            all_dis[i,j] = len(dis[i][j])-1
                        else:
                            all_dis[i,j] = A.shape[0] # we set specially here
            # assert 1<0, (all_dis.shape, all_dis.max(), all_dis.mean())
            # assert 1<0, all_dis[100,100]
        else:
            all_dis = None

        # # print('all_dis', all_dis.max(), all_dis.min(), all_dis.mean(), all_dis[10][10])
        # # assert 1<0
        # all_dis[all_dis==-1] = A.shape[0]
        ### end ###



        idx_features_labels = np.genfromtxt(id_files,dtype=np.dtype(str))
        feature=idx_features_labels[:, 1:-1].astype(np.int32)

        # print('feature', feature.sum())
        # print('Warning!!!!   remove the fnormalize(feature) in the init of loader.py line-832')
        # assert 1<0, feature.shape
        # feature = self.normalize(feature)
        feature = self.fnormalize(feature)
        # print('feature_norm', (feature>0.).sum())

        # print("feature",feature.shape)

        labels = self.encode_onehot(idx_features_labels[:, -1])
        node_label=idx_features_labels[:,0]
        node_label=node_label.astype(np.int32)

        # print("node_label",node_label[0:10])
        # assert 1<0

        map_feature = feature.copy().astype(np.int32)
        for i in range(idx_features_labels.shape[0]):
            map_feature[node_map[int(idx_features_labels[i][0])]] = feature[i].astype(np.int32)

        cmt = {}
        for node in node_label:
            if labels[node_map[node]] not in cmt:
                cmt[labels[node_map[node]]] = []

            cmt[labels[node_map[node]]].append(node_map[node])
        all_ids = [k for k in sorted(cmt.keys())]

        # print("cmt",cmt)
        # assert 1 < 0
        cmt_ids=all_ids

        # if phase == 'train':
        #     cmt_ids = []
        #     for i in range(len(all_ids)):
        #         if i % 7 == 0 or i % 7 == 3 or i % 7 == 6:
        #         # if i == 0:
        #             cmt_ids.append(all_ids[i])
        #         else:
        #             del cmt[all_ids[i]]
        # elif phase == 'eval':
        #     cmt_ids = []
        #     for i in range(len(all_ids)):
        #         if i % 7 == 1 or i % 7 == 4:
        #         # if i % 7 == 0 or i % 7 == 5 or i % 7 == 6:
        #         # if i == 0:
        #             cmt_ids.append(all_ids[i])
        #         else:
        #             del cmt[all_ids[i]]
        # elif phase == 'test':
        #     cmt_ids = []
        #     for i in range(len(all_ids)):
        #         if i % 7 == 2 or i % 7 == 5:
        #         # if i % 7 == 0 or i % 7 == 5 or i % 7 == 6:
        #         # if i == 0:
        #             cmt_ids.append(all_ids[i])
        #         else:
        #             del cmt[all_ids[i]]
        # else:
        #     cmt_ids = None
        #     assert 1 < 0, phase


        if phase == 'train':
            self.cmt_ids = cmt_ids
        else:
            self.cmt_ids = cmt_ids

        print("len in sample laoder", len(self.cmt_ids))

        # self.cmt_ids = cmt_ids
        # print('phase, cmt_ids', phase, cmt_ids)
        # for i in cmt_ids:
        #     print(i, len(cmt[i]), A.shape)
        # assert 1<0
        self.cmt = cmt
        self.A = A
        self.max_select_num = max_select_num
        self.map_feature = map_feature
        self.all_dis = all_dis
        self.phase = phase
        random.seed(20170624)
        npr.seed(20170624)

    def encode_onehot(self, labels):
        classes = set(labels)
        print("class size",len(classes))
        # classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
        # 				enumerate(classes)}
        classes_dict = {c: i for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot

    def fnormalize(self, mx):
        """colum-normalize sparse matrix"""
        mx=mx.transpose(0,1)
        print("mx shape", mx.shape)
        rowsum = mx.sum(1)
        # rowsum = rowsum[:,np.newaxis]
        rowsum[rowsum == 0] = 1
        # print("rowsum shape", rowsum.shape)
        print("rowsum", rowsum)
        mx = mx / rowsum[:,np.newaxis]
        mx = mx.transpose(0,1)
        # assert (mx.sum(0).astype(np.uint8)!=1).sum() == 0, (mx.sum(0), mx.sum(1), mx.shape)
        return mx

    def normalize(self, mx):
        """Row-normalize matrix"""
        print("mx shape", mx.shape)
        rowsum = mx.sum(1)
        # rowsum = rowsum[:,np.newaxis]
        rowsum[rowsum == 0] = 1
        # print("rowsum shape", rowsum.shape)
        print("rowsum", rowsum)
        mx = mx / rowsum[:,np.newaxis]
        # assert (mx.sum(1).astype(np.uint8)!=1).sum()==0, (mx.sum(1)) # all 0
        return mx

    def SPnormalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.  # 1/0   -> 0
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)  # matrix *
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)  # .row   .col   .data for not zero data
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))  # vstck() bingjie two matrix  [.row
        # .col]
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    # translate to torch's matrix tensor
    # shape    matrix size
    # indices  not zero location
    # values   not zero value

    def __len__(self):
        # print("len in sample laoder", len(self.cmt_ids))
        if self.phase == 'train':
            return len(self.cmt_ids)*10
        else:
            return len(self.cmt_ids)
            # return len(self.cmt_ids)*10

    def __getitem__(self, item):
        # id = self.cmt_ids[item]
        # print("id",id)
        # id = self.cmt_ids[0]
        # id = npr.choice(self.cmt_ids, 1, replace=False)[0]
        # cmts = self.cmt[id].copy()
        # cmt_id, node_id = item%3, item//3
        cmts = self.cmt[item%7].copy()
        # if self.phase == 'train':
        #     # id = self.cmt_ids[item]
        #     id = self.cmt_ids[cmt_id]
        # else:
        #     # id = npr.choice(self.cmt_ids, 1, replace=False)[0]
        #     # id = self.cmt_ids[item]
        #     id = self.cmt_ids[cmt_id]
        # assert item < len(self.cmt_ids), (item, self.cmt_ids, self.phase)
        # id = self.cmt_ids[item]
        # cmts = self.cmt[id].copy()

        # if self.all_dis is not None:
        #     new_cmts = []
        #     for i in cmts:
        #         if self.all_dis[i].max() < 0.: ## unreachable to others

        # print("id", id)
        # print("cmt",cmts)

        sample_node_num = min(self.max_select_num, len(cmts) // 2)
        # assert len(cmts) >= 2
        if sample_node_num == 0 and len(cmts) == 1:
            sample_node_num = 1
        else:
            assert sample_node_num > 0, (sample_node_num, len(cmts))
        fix_one_input = False
        if fix_one_input:
            # sample_nodes = npr.choice(cmts, sample_node_num, replace=False)
            sample_nodes = cmts[0:1]
            # print('item', sample_nodes)
            # assert 1<0, (sample_nodes, cmts)
        else:
            # sample_node_num = random.randint(1, sample_node_num)
            sample_node_num = 3
            while 1:
                sample_nodes = npr.choice(cmts, sample_node_num, replace=False)
                # print("sample nodes sum",sample_nodes.sum() )
                if self.phase == 'train':
                    if ((sample_nodes.sum() % 5) < 3):
                        break
                elif self.phase == 'eval':
                    if ((sample_nodes.sum() % 5) == 3):
                        break
                elif self.phase == 'test':
                    if (sample_nodes.sum() % 5)==4:
                        break
            # sample_nodes = cmts[node_id:node_id+1]
            # sample_nodes = cmts[:3]
            # print('sample_node', sample_node_num, sample_nodes)

        cur_adj = self.A.copy()
        # print("1", cur_adj.todense()[0:5, 0:5])

        # print("torch size",cur_adj.shape[0])
        # assert 1 < 0
        # print('cmt_id', id)
        # print('sample_node', sample_node_num)

        cur_input = np.zeros(cur_adj.shape[0], dtype=np.float32)
        # cur_adj = self.A.copy()
        cur_label = np.zeros(cur_adj.shape[0], dtype=np.float32)

        for k in sample_nodes:
            # for k in cmts:
            cur_input[k] = 1

        for k in cmts:
            cur_label[k] = 1
        # cnt = 0
        # for i in range(self.map_feature.shape[0]):
        #     for j in range(i):
        #         if (self.map_feature[i]!=self.map_feature[j]).sum() == 0:
        #             if cur_label[i] != cur_label[j]:
        #                 cnt += 1
        #                 print('diff', i, j)
        # assert cnt == 0, cnt
        # print('cur_input', cur_input, cur_input.sum())
        # print('cur_label', cur_label, cur_label.sum())
        # print("label-",cur_label[:25])

        ## add dis ##
        if self.all_dis is not None:
            cur_in_dis = np.zeros(cur_input.shape, dtype=np.float32)
            source = []
            for i in range(cur_input.shape[0]):
                if cur_input[i] == 1:
                    source.append(self.all_dis[i:i+1])
            source = np.concatenate(source, axis=0)
            # assert 1<0, (source.shape)
            assert source.shape[0] == cur_input.sum()
            for i in range(cur_input.shape[0]):
                cur_in_dis[i] = source[:,i].min()#or mean()
            # assert 1<0, (cur_in_dis.shape, cur_in_dis.max(), cur_in_dis.mean(), cur_in_dis.min(),
            #              (cur_in_dis<=1e-7).sum(), cur_input.sum())
            ### normalization ###
            # print("distance ",cur_in_dis)
            max_val = (cur_in_dis*(cur_in_dis<cur_input.shape)).max()
            # print("max distance ", max_val)
            assert max_val > 0., (max_val, sample_nodes)
            for i in range(cur_in_dis.shape[0]):
                if cur_in_dis[i] == cur_input.shape:  # unreachable:
                    cur_in_dis[i] = 0.
                elif cur_in_dis[i] == 0:
                    cur_in_dis[i] = 1.
                else:
                    cur_in_dis[i] = 1. - cur_in_dis[i] / (max_val + 1)
            # for k in sample_nodes:
            #     # for k in cmts:
            #     cur_in_dis[k] = 0
            # print("distance ", cur_in_dis)
            # assert 1<0, (cur_in_dis.shape, cur_in_dis.max(), cur_in_dis.mean(), cur_in_dis.min(),
            #              (cur_in_dis<=1e-7).sum(), cur_input.sum(), (cur_in_dis>1-1e-7).sum())
        else:
            cur_in_dis = None
        # cur_input = cur_in_dis
        # fg_dis = []
        # bg_dis = []
        # for i in range(cur_label.shape[0]):
        # 	if cur_label[i] == 1:
        # 		fg_dis.append(cur_in_dis[i])
        # 	else:
        # 		bg_dis.append(cur_in_dis[i])

        #print('dis', item, np.mean(fg_dis), np.min(fg_dis), np.max(fg_dis), np.mean(bg_dis), np.min(bg_dis), np.max(bg_dis))
        ## end ##

        # print(self.map_feature.shape, cur_input.shape)
        ## add repeat ##
        # print('shape', cur_input.shape, self.map_feature.shape)
        # cur_input = cur_input[:,np.newaxis]
        # assert 1<0
        #cur_input = np.repeat(cur_input, self.map_feature.shape[1], axis=1)
        #cur_input = np.concatenate((self.map_feature, cur_input), axis=1)
        ## end ##
        # assert 1<0, (self.map_feature.shape, cur_input.shape, cur_in_dis.shape)
        if self.all_dis is None:
            cur_input = np.concatenate((self.map_feature, cur_input[:,np.newaxis]*10), axis=1)
        else:
            # cur_input = np.concatenate((self.map_feature, cur_input[:, np.newaxis] * 10,
            #                             cur_in_dis[:,np.newaxis]*10), axis=1)
            cur_input = np.concatenate((self.map_feature, cur_in_dis[:,np.newaxis]), axis=1) # only distance
        # print('cur_input_add_feat', cur_input.sum(), cur_input.shape)

        # cur_adj = self.sparse_mx_to_torch_sparse_tensor(cur_adj)
        cur_adj = torch.FloatTensor(cur_adj)
        cur_input = torch.FloatTensor(cur_input)
        cur_label = torch.FloatTensor(cur_label)

        # cur_input -= 0.5
        return cur_input, cur_adj, cur_label

class PhilDataset(Dataset):
    def __init__(self, edge_files='/apdcephfs/private_lizyljiang/CommunitySearchdata/phil/phil.edges', id_files='data/phil/phil.communities',
                 phase='train', thr=10, max_select_num=3):
        super(PhilDataset, self).__init__()

        featname = np.genfromtxt("/apdcephfs/private_lizyljiang/CommunitySearchdata/phil/phil.featnames", dtype=np.dtype(str))
        # feat = feat.astype(np.int32)
        print("featname", featname[2])
        print("featname shape", featname.shape)
        featmap= {}
        for i in range(featname.shape[0]):
            featmap[int(featname[i][0])] = i
        print("len featmap",len(featmap))
        print("featmap ",featname[2][0],featmap[int(featname[2][0])])
        # assert 1 < 0

        nodefeat = np.genfromtxt("/apdcephfs/private_lizyljiang/CommunitySearchdata/phil/phil.nodefeat", dtype=np.dtype(str))
        nodefeat = nodefeat.astype(np.int32)
        print("nodefeat 2", nodefeat[2])
        feats=np.zeros((1508, featname.shape[0]), dtype=np.int32)

        for f in nodefeat:
            feats[f[0],featmap[f[1]]]=1
        print("feats 1467, sum ",feats[1467],feats[1467].sum())

        # assert 1 < 0

        G = nx.Graph()
        Edges = []

        A = np.zeros((1508, 1508), dtype=np.int32)

        edges = np.genfromtxt(edge_files, dtype=np.dtype(str))
        edges = edges.astype(np.int32)
        for e in edges:
            # if (e[0]!=e[1]):
            # print(e[0],e[1])
            A[e[0],e[1]]=1
            A[e[1],e[0]] = 1
            Edges.append((e[0],e[1]))







        # A = nx.adjacency_matrix(G).todense()
        # A = np.array(A, dtype=np.float32)

        for i in range(A.shape[0]):
            A[i,i] = 1
            Edges.append((i,i))

        print("A.shape", A.shape)
        G.add_edges_from(Edges)

        A = self.normalize(A)

        print("A.shape",A.shape)

        G.remove_edges_from(nx.selfloop_edges(G))

        degree = np.zeros(A.shape[0], dtype=np.float32)
        # print("size of degree", G.degree)
        print("degree[2]", G.degree[2])
        for i in range(A.shape[0]):
            # print(i,(i in G.degree))
            # if i in G.degree:
            #     print(i)
            degree[i] = G.degree[i]# / A.shape[0]
            # else:
                # print("no",i)
                # degree[i] =0
        degree = degree / degree.max()
        print("Norm degree[2]",degree[2])


        core_G = nx.core_number(G)
        core = np.zeros(A.shape[0], dtype=np.float32)
        for i in range(A.shape[0]):
            if i in core_G:
                core[i] = core_G[i]
            else:
                core[i] =0

        # print("size of core",core)
        print("core[2]", core[2])
        core = core / core.max()
        print("Norm core[2]", core[2])
        # print("Norm core", core)

        cluster_G = nx.clustering(G)
        cluster = np.zeros(A.shape[0], dtype=np.float32)
        for i in range(A.shape[0]):
            if i in cluster_G:
                cluster[i] = cluster_G[i]
            else:
                cluster[i] = 0

        print("cluster",cluster[2])
        cluster = cluster / cluster.max()
        print("Norm cluster", cluster[2])

        triangle_G = nx.triangles(G)
        triangle = np.zeros(A.shape[0], dtype=np.float32)
        for i in range(A.shape[0]):
            if i in triangle_G:
                triangle[i] = triangle_G[i]
            else:
                triangle[i] =0

        print("triangle",triangle[2])
        triangle = triangle / triangle.max()
        print("Norm triangle", triangle[2])

        dis = nx.shortest_path(G)
        print("dis len",len(dis))
        all_dis = np.zeros(A.shape, dtype=np.int32)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                # all_dis[i,j] = len(dis[i][j])-1
                if i == j:
                    all_dis[i, j] = 0
                else:
                    if j in dis[i]:
                        assert len(dis[i][j]) > 1
                        all_dis[i, j] = len(dis[i][j]) - 1
                    else:
                        all_dis[i, j] = A.shape[0]  # we set specially here
        print("dis[1,2]", all_dis[1,2])












        file = open(id_files, 'r')
        cmt_list = file.readlines()
        # val_list=val_list[:,0:-1]
        cmt = []
        for string in cmt_list:
            #	string=string.split('\n')
            string = string.strip().split('\t')
            # print(string[0])
            if (len(string)>5):
                cmt.append(string[1:])
        file.close()

        print("cmt size", len(cmt))

        # print(cmt[0][0],int(cmt[0][0]))
        # assert 1<0

        cmts = {}
        for i in range(len(cmt)):
            cmts[i] = []
            # print("cmt i size",len(cmt[i]))
            for j in range(len(cmt[i])):
                cmts[i].append(int(cmt[i][j]))
        # all_ids = [k for k in sorted(cmt.keys())]
        all_cmt = [k for k in sorted(cmts.keys())]

        # if phase == 'train':
        #     cmt_ids = []
        #     for i in range(len(all_ids)):
        #         if i % 7 ==0 or i % 7 == 5 or i % 7 == 6:
        #             cmt_ids.append(all_ids[i])
        #         else:
        #             del cmt[all_ids[i]]
        # elif phase == 'eval':
        #     cmt_ids = []
        #     for i in range(len(all_ids)):
        #         if i % 7 == 1 or i % 7 == 4:
        #             cmt_ids.append(all_ids[i])
        #         else:
        #             del cmt[all_ids[i]]
        # elif phase == 'test':
        #     cmt_ids = []
        #     for i in range(len(all_ids)):
        #         if i % 7 == 2 or i % 7 == 3:
        #             cmt_ids.append(all_ids[i])
        #         else:
        #             del cmt[all_ids[i]]
        # else:
        #     cmt_ids = None
        #     assert 1<0, phase

        self.cmt_ids = all_cmt
        self.cmt = cmts
        self.A = A
        self.feats=feats
        self.max_select_num = max_select_num
        self.phase = phase

        self.degree = degree
        self.core = core
        self.cluster = cluster
        self.triangle = triangle
        self.all_dis=all_dis

        random.seed(20170624)
        npr.seed(20170624)

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        # print(rowsum)
        r_inv = 1 / rowsum
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def __len__(self):
        return 160
        # return len(self.cmt_ids)

    def __getitem__(self, item):
        # id = self.cmt_ids[item]
        # print("id",id)
        # id = self.cmt_ids[0]
        id = npr.choice(self.cmt_ids, 1, replace=False)[0]
        cmts = self.cmt[id]
        # print("id", id)

        degree = self.degree[:, np.newaxis]
        core = self.core[:, np.newaxis]
        cluster = self.cluster[:, np.newaxis]
        triangle = self.triangle[:, np.newaxis]

        all_dis= self.all_dis


        sample_node_num = min(self.max_select_num, len(cmts)//2)
        if sample_node_num == 0 and len(cmts) == 1:
            sample_node_num = 1
        else:
            assert sample_node_num > 0, (sample_node_num, len(cmts))
        i=0
        while 1:
        #    for i in range(100):
            i+=1
            sample_nodes = npr.choice(cmts, sample_node_num, replace=False)
            #print("sample nodes sum",sample_nodes.sum(),sample_nodes.sum()%5 )
            if self.phase == 'train':
                if ((sample_nodes.sum() % 5) < 3):
                    break
            elif self.phase == 'eval':
                if ((sample_nodes.sum() % 5) == 3):# and (sample_nodes.sum() % 50) < 40):
                    break
            elif self.phase == 'test':
                if ((sample_nodes.sum() % 5) == 4):
                    break
            if (i%100)==0:
                id = npr.choice(self.cmt_ids, 1, replace=False)[0]
                cmts = self.cmt[id]
                sample_node_num = min(self.max_select_num, len(cmts) // 2)
                if sample_node_num == 0 and len(cmts) == 1:
                    sample_node_num = 1
                else:
                    assert sample_node_num > 0, (sample_node_num, len(cmts))
                sample_node_num = random.randint(1, sample_node_num)
        # sample_node_num = random.randint(1, sample_node_num)
        # sample_nodes = npr.choice(cmts, sample_node_num, replace=False)

        cur_input = np.zeros(self.A.shape[0], dtype=np.float32)
        cur_adj = self.A.copy()
        cur_label = np.zeros(self.A.shape[0], dtype=np.float32)

        for k in sample_nodes:
        # for k in cmts:
            cur_input[k] = 1

        cur_in_dis = np.zeros(cur_input.shape, dtype=np.float32)
        source = []
        for j in range(cur_input.shape[0]):
            if cur_input[j] == 1:
                source.append(all_dis[j:j + 1])
        source = np.concatenate(source, axis=0)
        assert source.shape[0] == cur_input.sum()
        for j in range(cur_input.shape[0]):
            cur_in_dis[j] = source[:, j].min()  # mean()

        # for j in range(cur_in_dis.shape[0]):
        # print("dis",cur_in_dis)
        max_val = (cur_in_dis * (cur_in_dis < cur_adj.shape[0])).max()
        # print("max distance ", max_val)
        # assert max_val > 0., (max_val)
        if max_val==0:
            max_val=cur_in_dis.shape[0]-1
        for i in range(cur_in_dis.shape[0]):
            if cur_in_dis[i] == cur_adj.shape[0]:  # unreachable:
                cur_in_dis[i] = 0.
            else:
                cur_in_dis[i] = 1. - cur_in_dis[i] / (max_val + 1)
        for k in range(cur_input.shape[0]):
            if cur_input[k] == 1:
                cur_in_dis[k] = 1
        # print("distance ", cur_in_dis)

        for k in cmts:
            cur_label[k] = 1
        cur_feats = np.concatenate((self.feats.copy(), cur_in_dis[:, np.newaxis]), axis=1)  # BN(D+2)
        # cur_feats = np.concatenate((self.feats.copy(), cur_input[:, np.newaxis]), axis=1)  # BN(D+2)
        cur_feats = np.concatenate((cur_feats, cur_input[:, np.newaxis]), axis=1)  # BN(D+2)
        # cur_feats = np.concatenate((cur_feats, degree), axis=1)  # BN(D+2)
        # cur_feats = np.concatenate((cur_feats, core), axis=1)  # BN(D+2)
        # cur_feats = np.concatenate((cur_feats, cluster), axis=1)  # BN(D+2)
        # cur_feats = np.concatenate((cur_feats, triangle), axis=1)  # BN(D+2)


        # cur_input -= 0.5
        return cur_feats, cur_adj, cur_label

class WebKBDataset(Dataset):
    def __init__(self, data_dir='/apdcephfs/private_lizyljiang/CommunitySearch/data/WebKB/', phase='train',file = "cornell",method="QD-GCN"):
        super(WebKBDataset, self).__init__()



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
        # print("size of degree",degree)
        # G.remove_edges_from(nx.selfloop_edges(G))
        # core_G=nx.core_number(G)
        # core = np.zeros(A.shape[0], dtype=np.float32)
        # for i in range(A.shape[0]):
        #     core[i] = core_G[i]
        #
        # # print("size of core",core)
        # core=core/core.max()
        # # print("Norm core", core)
        #
        # cluster_G=nx.clustering(G)
        # cluster = np.zeros(A.shape[0], dtype=np.float32)
        # for i in range(A.shape[0]):
        #     cluster[i] = cluster_G[i]
        #
        # # print("size of cluster",cluster)
        # cluster = cluster / cluster.max()
        # # print("Norm cluster", cluster)
        #
        # triangle_G = nx.clustering(G)
        # triangle = np.zeros(A.shape[0], dtype=np.float32)
        # for i in range(A.shape[0]):
        #     triangle[i] = triangle_G[i]
        #
        # # print("size of cluster",cluster)
        # triangle = triangle / triangle.max()
        # # print("Norm cluster", cluster)

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


# class DBLPBatchDataset(DBLPDataset):
#     def get_adjacent_matrix(self):
#         return self.sparse_mx_to_torch_sparse_tensor(self.A)
#
#     def get_node_num(self):
#         return self.A.shape[0]
#
#     def __getitem__(self, item):
#         # id = npr.choice(self.cmt_ids, 1, replace=False)[0]
#         # cmts = self.cmt[id]
#         cmts = self.cmt[item]
#         # print("id", id)
#
#         sample_node_num = min(self.max_select_num, len(cmts) // 2)
#         if sample_node_num == 0 and len(cmts) == 1:
#             sample_node_num = 1
#         else:
#             assert sample_node_num > 0, (sample_node_num, len(cmts))
#         sample_node_num = random.randint(1, sample_node_num)
#         sample_nodes = npr.choice(cmts, sample_node_num, replace=False)
#
#         node_num = self.get_node_num()
#         cur_input = np.zeros(node_num, dtype=np.float32)
#         cur_label = np.zeros(node_num, dtype=np.float32)
#
#         for k in sample_nodes:
#             # for k in cmts:
#             cur_input[k] = 1
#         for k in cmts:
#             cur_label[k] = 1
#
#         cur_input = torch.LongTensor(cur_input)
#         cur_label = torch.LongTensor(cur_label)
#
#         # cur_input -= 0.5
#         return cur_input, cur_label
#
#
# class SPEmailDataset(Dataset):
#     def __init__(self, edge_files='data/Email/email-Eu-core.txt', id_files='data/Email/email-Eu-core-department-labels.txt',
#                  phase='train', thr=10, max_select_num=3):
#         super(SPEmailDataset, self).__init__()
#
#
#         print("load Email dateset", phase)
#
#         node = set()
#         edges = np.genfromtxt(edge_files, dtype=np.int32)
#         # edges = edges.astype(np.int32)
#         for e in edges:
#             node.add(e[0])
#             node.add(e[1])
#         # node=set(node)
#         node_map = {j: i for i, j in enumerate(node)}
#
#         print("node size", len(node))
#
#         # print(node_map)
#         # assert 1<0
#
#         print("build Email Adj", phase)
#         # print("edges",edges[0:10,:])
#         edges_map = np.array(list(map(node_map.get, edges.flatten())),
#                              dtype=np.int32).reshape(edges.shape)
#         # print("edges map", edges_map[0:10,:])
#         adj = sp.coo_matrix((np.ones(edges_map.shape[0]), (edges_map[:, 0], edges_map[:, 1])),
#                             shape=(len(node), len(node)),
#                             dtype=np.float32)
#         # print("1", adj.todense()[0:10,0:10])
#         # assert 1 < 0
#         adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#
#
#         print("adj shape",adj.shape)
#
#
#         A = self.SPnormalize(adj)
#         A=A.todense()
#         # G = nx.Graph()
#         # Edges = []
#         #
#         # edges = np.genfromtxt(edge_files, dtype=np.dtype(str))
#         # edges = edges.astype(np.int32)
#         # for e in edges:
#         # 	Edges.append((e[0], e[1]))
#         #
#         # G.add_edges_from(Edges)
#         #
#         #
#         # A = nx.adjacency_matrix(G).todense()
#         # A = np.array(A, dtype=np.float32)
#         #
#         #
#         # A = self.normalize(A)
#
#
#
#
#         idx = np.genfromtxt(id_files, dtype=np.dtype(str))
#         idx = idx.astype(np.int32)
#         cmt = {}
#         for node in idx:
#             if node[1] not in cmt:
#                 cmt[node[1]] = []
#             cmt[node[1]].append(node[0])
#         all_ids = [k for k in sorted(cmt.keys())]
#
#         if phase == 'train':
#             cmt_ids = []
#             for i in range(len(all_ids)):
#                 if i % 7 ==0 or i % 7 == 5 or i % 7 == 6:
#                     cmt_ids.append(all_ids[i])
#                 else:
#                     del cmt[all_ids[i]]
#         elif phase == 'eval':
#             cmt_ids = []
#             for i in range(len(all_ids)):
#                 if i % 7 == 1 or i % 7 == 4:
#                     cmt_ids.append(all_ids[i])
#                 else:
#                     del cmt[all_ids[i]]
#         elif phase == 'test':
#             cmt_ids = []
#             for i in range(len(all_ids)):
#                 if i % 7 == 2 or i % 7 == 3:
#                     cmt_ids.append(all_ids[i])
#                 else:
#                     del cmt[all_ids[i]]
#         else:
#             cmt_ids = None
#             assert 1<0, phase
#
#         self.cmt_ids = cmt_ids
#         self.cmt = cmt
#         self.A = A
#         self.max_select_num = max_select_num
#         random.seed(20170624)
#         npr.seed(20170624)
#
#     def normalize(self, mx):
#         """Row-normalize sparse matrix"""
#         print("mx shape",mx.shape)
#         rowsum = mx.sum(1)
#         # rowsum = rowsum[:,np.newaxis]
#         rowsum[rowsum==0] = 1
#         print("rowsum shape", rowsum.shape)
#         mx = mx/rowsum
#         return mx
#
#     def SPnormalize(self, mx):
#         """Row-normalize sparse matrix"""
#         rowsum = np.array(mx.sum(1))
#         r_inv = np.power(rowsum, -1).flatten()
#         r_inv[np.isinf(r_inv)] = 0.  # 1/0   -> 0
#         r_mat_inv = sp.diags(r_inv)
#         mx = r_mat_inv.dot(mx)  # matrix *
#         return mx
#
#     def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
#         """Convert a scipy sparse matrix to a torch sparse tensor."""
#         sparse_mx = sparse_mx.tocoo().astype(np.float32)  # .row   .col   .data for not zero data
#         indices = torch.from_numpy(
#             np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))  # vstck() bingjie two matrix  [.row
#         # .col]
#         values = torch.from_numpy(sparse_mx.data)
#         shape = torch.Size(sparse_mx.shape)
#         return torch.sparse.FloatTensor(indices, values, shape)
#
#     # translate to torch's matrix tensor
#     # shape    matrix size
#     # indices  not zero location
#     # values   not zero value
#
#     def __len__(self):
#         return len(self.cmt_ids)
#
#     def __getitem__(self, item):
#         # id = self.cmt_ids[item]
#         # print("id",id)
#         # id = self.cmt_ids[0]
#         id = npr.choice(self.cmt_ids, 1, replace=False)[0]
#         cmts = self.cmt[id]
#         # print("id", id)
#
#         sample_node_num = min(self.max_select_num, len(cmts) // 2)
#         if sample_node_num == 0 and len(cmts) == 1:
#             sample_node_num = 1
#         else:
#             assert sample_node_num > 0, (sample_node_num, len(cmts))
#         sample_node_num = random.randint(1, sample_node_num)
#         sample_nodes = npr.choice(cmts, sample_node_num, replace=False)
#
#         cur_adj = self.A.copy()
#         # print("1", cur_adj.todense()[0:5, 0:5])
#
#
#         # print("torch size",cur_adj.shape[0])
#         # assert 1 < 0
#
#         cur_input = np.zeros(cur_adj.shape[0], dtype=np.float32)
#         # cur_adj = self.A.copy()
#         cur_label = np.zeros(cur_adj.shape[0], dtype=np.float32)
#
#         for k in sample_nodes:
#             # for k in cmts:
#             cur_input[k] = 1
#         for k in cmts:
#             cur_label[k] = 1
#
#         # cur_adj = self.sparse_mx_to_torch_sparse_tensor(cur_adj)
#         cur_adj = torch.FloatTensor(cur_adj)
#         cur_input = torch.FloatTensor(cur_input)
#         cur_label = torch.FloatTensor(cur_label)
#
#         # cur_input -= 0.5
#         return cur_input, cur_adj, cur_label
