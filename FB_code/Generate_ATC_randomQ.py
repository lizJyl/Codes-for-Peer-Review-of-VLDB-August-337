import numpy as np
import scipy.sparse as sp
import torch
import numpy.random as npr
import random
import networkx as nx

np.set_printoptions(threshold=np.inf)
npr.seed(170624)
random.seed(170624)

def load_data(path="../facebook/"):

	ego =414
	# ego = 686
	ego = 348
	ego = 0
	ego = 3437
	ego = 1912
	ego = 1684
	ego = 107


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

	file = open("{}{}.circles".format(path,ego), 'r')
	val_list = file.readlines()
	#val_list=val_list[:,0:-1]
	lists =[]
	for string in val_list:
	#	string=string.split('\n')
		string=string.strip().split('\t')
		#print(string[1:])
		if(len(string)<6):
			continue
		lists.append(string[1:])
	circles=np.array(lists)
	file.close()
	# print(circles.shape[0])
	# for i in range(circles.shape[0]):
	# 	print(circles[i][0])
	inputs = []
	# qeury_node=[]
	querynum=0
	count=0
	for i in range(circles.shape[0]):
		if len(circles[i])<5:
			continue
		for k in range(len(circles[i])//2-1):
			if (querynum % 7 == 2 or querynum % 7 == 3):
				count=count+1
			querynum=querynum+1

	queryatt = np.genfromtxt("{}{}.attr".format(path, ego), dtype=np.dtype(str))
	queryatt = queryatt.astype(np.int32)


	f1 = open("{}/ATC_randomQ/{}.query.node".format(path, ego), 'w')
	np.savetxt(f1, [count], fmt='%d')
	querynum=0
	count=0
	allattr = []
	for i in range(circles.shape[0]):
		if len(circles[i])<5:
			continue
		for k in range(len(circles[i])//2-1):
			# print('k', k, circles.shape[0]//3, len(circles[i]))


			randnum_node=random.randint(1,3)
			select_cols = npr.choice(len(circles[i]), randnum_node, replace=False)#random.randint(1,3)
			input=np.zeros(n_node, dtype=np.int32)
			qeury_n=[]
			query_att=[]
			qeury_num = []

			if (querynum % 7 == 2 or querynum % 7 == 3):
				randnum_att = random.randint(2, 5)
				select_att = npr.choice(features.shape[1], randnum_att, replace=False)
				att_test = np.zeros(features.shape[1], dtype=np.int32)
				# print("features.shape[1]",features.shape[1])
				for j in select_att:
					att_test[j] = 1
					query_att.append(j)
				allattr.append(att_test[np.newaxis, :])
			else:
				allattr.append(queryatt[i][np.newaxis, :])

			for j in select_cols:
				# print('select_cols', j, i, circles[i][j])
				input[node_map[int(circles[i][j])]] = 1
				if(querynum%7==2 or querynum%7==3):
					qeury_n.append(node_map[int(circles[i][j])])

			if (querynum % 7 == 2 or querynum % 7 == 3):
				qeury_num.append(randnum_node)
				qeury_num.append(randnum_att)
				np.savetxt(f1, [qeury_num], fmt='%d')
				np.savetxt(f1, [qeury_n], fmt='%d')
				np.savetxt(f1, [query_att], fmt='%d')
				count=count+1
			querynum = querynum + 1
			inputs.append(input[np.newaxis, :])
			# print(input.shape, 'input')
			# assert 1<0
	inputs = np.concatenate(inputs, axis=0)
	print("query size",inputs.shape, inputs.dtype)
	np.savetxt("{}/ATC_randomQ/{}.sample".format(path,ego), inputs, fmt='%d')
	f1.close()

	allattr = np.concatenate(allattr, axis=0)
	print("allattr size", allattr.shape, allattr.dtype)
	np.savetxt("{}/ATC_randomQ/{}.sample.attri".format(path, ego), allattr, fmt='%d')


	labels=[]
	querynum=0
	count=0
	f = open("{}/ATC_randomQ/{}.query.com".format(path, ego), 'w')
	for i in range(circles.shape[0]):
		if len(circles[i])<5:
			continue
		for k in range(len(circles[i])//2-1):
			com = []
			output=np.zeros(n_node)
			for j in circles[i]:
				output[node_map[int(j)]]=1
				com.append(node_map[int(j)])
			labels.append(output[np.newaxis,:])
			if (querynum % 7 == 2 or querynum % 7 == 3):
				np.savetxt(f, [com], fmt='%d', delimiter=' ', newline='\n', )
				count = count + 1
			querynum = querynum + 1
	print("ground truth size", len(labels))
	labels = np.concatenate(labels, axis=0)
	f.close()
	np.savetxt("{}/ATC_randomQ/{}.label".format(path,ego), labels, fmt='%d')

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

	f = open("{}/ATC_randomQ/{}.graph".format(path, ego), 'w')
	f2 = open("{}ATC_randomQ/{}.keywords".format(path, ego), 'w')
	edge = []
	for i in range(A.shape[0]):
		for j in range(A.shape[0]):
			if A[i][j] == 1:
				edge.append([i,j])
		# print("edge", edge)
		nfeat = []
		# nfeat.append(i)
		nfeat.append(i)
		num = 0
		for j in range(features.shape[1]):
			if features[i][j] == 1:
				num = num + 1
		nfeat.append(num)
		for j in range(features.shape[1]):
			if features[i][j] == 1:
				nfeat.append(j)
		# print("feat", nfeat)
		np.savetxt(f2, [nfeat], fmt='%d', delimiter=' ', newline='\n', )

	# degree[i] = G.degree[i] / A.shape[0]
	f2.close()

	np.savetxt(f, edge, fmt='%d', delimiter=' ', newline='\n', )



	f.close()

load_data()