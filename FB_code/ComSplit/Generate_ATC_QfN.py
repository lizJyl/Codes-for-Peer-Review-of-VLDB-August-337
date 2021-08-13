import numpy as np
import scipy.sparse as sp
import torch
import numpy.random as npr
import random
import networkx as nx

np.set_printoptions(threshold=np.inf)
npr.seed(170624)
random.seed(170624)

def load_data(path="/apdcephfs/private_lizyljiang/CommunitySearch/data/facebook/"):

	ego =414
	# ego = 686
	# ego = 348
	# ego = 0
	# ego = 3437
	# ego = 1912
	# ego = 1684
	# ego = 107


	ego_feat=np.genfromtxt("{}{}.egofeat".format(path,ego),dtype=np.dtype(str))
	feat=np.genfromtxt("{}{}.feat".format(path,ego),dtype=np.dtype(str))

	features=np.vstack((ego_feat,feat[:,1:]))
	features = features.astype(int)
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
	allattr=[]
	Qcount=0
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

	for i in range(circles.shape[0]):
		if allcount > 199:
			break
		# count = 0
		count_train=0
		count_test=0
		print('circle i ', i, len(circles[i]))
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
			if (i % 2 == 1):
				allcount_test = allcount_test + 1
				count_test = count_test + 1
				Qcount=Qcount+1
			# count = count + 1
			allcount = allcount + 1


	print('test query num ', Qcount)


	f1 = open("{}ComSplit/ATC_QfN/{}.query.node".format(path, ego), 'w')
	np.savetxt(f1, [Qcount], fmt='%d')
	querynum=0
	# count=0
	allcount=0
	allcount_train=0
	allcount_test=0
	for i in range(circles.shape[0]):
		if allcount > 199:
			break
		# count = 0
		count_train=0
		count_test=0
		print('circle i ', i, len(circles[i]))
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

			randnum=random.randint(1,3)
			select_cols = npr.choice(len(circles[i]), randnum, replace=False)#random.randint(1,3)
			input=np.zeros(n_node, dtype=np.int32)
			qeury_n=[]
			qeury_num = []

			att = np.zeros(features.shape[1], dtype=np.int32)

			for j in select_cols:
				# print('select_cols', j, i, circles[i][j])
				input[node_map[int(circles[i][j])]] = 1
				att = att + features[int(node_map[int(circles[i][j])])]
				if(i % 2 == 1):
					qeury_n.append(node_map[int(circles[i][j])])
			sort = np.argsort(att)
			sort = sort[-5:]
			feats_sort = np.zeros(features.shape[1], dtype=np.int32)
			feats_sort[sort] = 1
			allattr.append(feats_sort[np.newaxis, :])

			if (i % 2 == 1):
				qeury_num.append(randnum)
				qeury_num.append(len(sort))
				np.savetxt(f1, [qeury_num], fmt='%d')
				np.savetxt(f1, [qeury_n], fmt='%d')
				np.savetxt(f1, [sort], fmt='%d')
			if (i % 2 == 0):
				allcount_train = allcount_train + 1
				count_train = count_train + 1
			if (i % 2 == 1):
				allcount_test = allcount_test + 1
				count_test = count_test + 1
			allcount = allcount + 1
			querynum = querynum + 1
			inputs.append(input[np.newaxis, :])
			# print(input.shape, 'input')
			# assert 1<0
	inputs = np.concatenate(inputs, axis=0)
	print("query size",inputs.shape, inputs.dtype)
	np.savetxt("{}ComSplit/ATC_QfN/{}.sample".format(path,ego), inputs, fmt='%d')
	f1.close()

	allattr = np.concatenate(allattr, axis=0)
	print("allattr size", allattr.shape, allattr.dtype)
	np.savetxt("{}ComSplit/ATC_QfN/{}.sample.attri".format(path, ego), allattr, fmt='%d')



	labels=[]
	querynum=0
	allcount=0
	allcount_train=0
	allcount_test=0
	f = open("{}ComSplit/ATC_QfN/{}.query.com".format(path, ego), 'w')
	for i in range(circles.shape[0]):
		if allcount > 199:
			break
		count_train=0
		count_test=0
		print('circle i ', i, len(circles[i]))
		for k in range(len(circles[i]) * 200):
			if allcount > 199:
				break
			if i % 2 == 0 and allcount_train > 99:
				break
			if i % 2 == 1 and allcount_test > 99:
				break
			if i % 2 == 0 and count_train > (100 * len(circles[i]) / sum_com_train + 1):
				break
			if i % 2 == 1 and count_test > (100 * len(circles[i]) / sum_com_test + 1):
				break
			com = []
			output=np.zeros(n_node)
			for j in circles[i]:
				output[node_map[int(j)]]=1
				com.append(node_map[int(j)])
			labels.append(output[np.newaxis,:])
			if (i % 2 == 1):
				np.savetxt(f, [com], fmt='%d', delimiter=' ', newline='\n', )
			if (i % 2 == 0):
				allcount_train = allcount_train + 1
				count_train = count_train + 1
			if (i % 2 == 1):
				allcount_test = allcount_test + 1
				count_test = count_test + 1
			allcount = allcount + 1
			querynum = querynum + 1
	print("ground truth size", len(labels))
	labels = np.concatenate(labels, axis=0)
	f.close()
	np.savetxt("{}ComSplit/ATC_QfN/{}.label".format(path,ego), labels, fmt='%d')

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

	f = open("{}ComSplit/ATC_QfN/{}.graph".format(path, ego), 'w')
	f2 = open("{}ComSplit/ATC_QfN/{}.keywords".format(path, ego), 'w')
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