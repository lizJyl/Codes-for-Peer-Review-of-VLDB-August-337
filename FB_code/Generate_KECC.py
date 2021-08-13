import numpy as np
import scipy.sparse as sp
import torch
import numpy.random as npr
import random
import networkx as nx

np.set_printoptions(threshold=np.inf)
npr.seed(170624)
random.seed(170624)

def load_data(path="./facebook/",ego=3980):
	ego_feat=np.genfromtxt("{}{}.egofeat".format(path,ego),dtype=np.dtype(str))
	feat=np.genfromtxt("{}{}.feat".format(path,ego),dtype=np.dtype(str))

	features=np.vstack((ego_feat,feat[:,1:]))
	#print(features[0:3,:])
	ego_node=int(ego)
	node=np.array(np.vstack((ego_node,feat[:,0:1])))
	n_node=node.shape[0]
	print(n_node)
	# node_map={j:i for i,j in enumerate(node)}
	node_map = {}
	# inv_node_map = {}
	for i in range(node.shape[0]):
		node_map[int(node[i])] = i
		# inv_node_map[i] = int(node[i])



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

	f = open("../baseline/CommunitySearch/kecc/facebook/{}/edges.txt".format(ego), 'w')

	edge = []
	edge.append([A.shape[0], G.number_of_edges()])
	for i in range(A.shape[0]):
		for j in range(A.shape[0]):
			if A[i][j] == 1  and i < j:
				edge.append([i,j])
		# print("edge", edge)
	np.savetxt(f, edge, fmt='%d', delimiter=' ', newline='\n', )



	f.close()

load_data()