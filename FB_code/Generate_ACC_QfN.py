import numpy as np
import scipy.sparse as sp
import torch
import numpy.random as npr
import random

np.set_printoptions(threshold=np.inf)
npr.seed(170624)
random.seed(170624)

def load_data(path="../facebook/",ego=0):
	ego = 414
	ego = 686
	ego = 348
	ego = 0
	ego = 3437
	ego = 1912
	ego = 1684
	ego = 107

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
	sum_com=0
	for i in range(circles.shape[0]):
		sum_com=sum_com+len(circles[i])


	inputs = []
	allattr=[]
	qeury_node=[]
	count = 0
	allcount = 0
	labels = []
	f = open("{}/ACC_QfN/{}.query.com".format(path, ego), 'w')
	f2 = open("{}/ACC_QfN/{}.query.feat".format(path, ego), 'w')
	for i in range(circles.shape[0]):
		if allcount > 349:
			break
		count = 0
		print('circle i ',i, len(circles[i]))
		for k in range(len(circles[i])*200):
			if allcount > 349:
				break
			if count > (350* len(circles[i]) / sum_com  + 1):
				break
			# print('k', k, circles.shape[0]//3, len(circles[i]))
			select_cols = npr.choice(len(circles[i]), 1, replace=False)#random.randint(1,3)
			input=np.zeros(n_node, dtype=np.int32)
			for j in select_cols:
				# print('select_cols', j, i, circles[i][j])
				input[node_map[int(circles[i][j])]] = 1
				qeury_node.append(node_map[int(circles[i][j])])
				attr=features[node_map[int(circles[i][j])]]
				allattr.append(features[node_map[int(circles[i][j])]][np.newaxis, :])
			att = []
			for j in range(attr.shape[0]):
				if attr[j] == 1:
					att.append(j)
			np.savetxt(f2, [att], fmt='%d', delimiter=' ', newline='\n', )
			inputs.append(input[np.newaxis,:])

			com = []
			output = np.zeros(n_node)
			for j in circles[i]:
				output[node_map[int(j)]] = 1
				com.append(node_map[int(j)])
			labels.append(output[np.newaxis, :])
			np.savetxt(f, [com], fmt='%d', delimiter=' ', newline='\n', )

			count = count + 1
			allcount = allcount + 1
			# print(input.shape, 'input')
			# assert 1<0
	f2.close()
	f.close()
	print("ground truth size", len(labels))
	labels = np.concatenate(labels, axis=0)
	np.savetxt("{}/ACC_QfN/{}.label".format(path, ego), labels, fmt='%d')
	inputs = np.concatenate(inputs, axis=0)

	allattr=np.concatenate(allattr, axis=0)
	print("query size",inputs.shape, inputs.dtype)
	np.savetxt("{}/ACC_QfN/{}.sample".format(path,ego), inputs, fmt='%d')
	np.savetxt("{}/ACC_QfN/{}.query.node".format(path, ego), qeury_node, fmt='%d')
	np.savetxt("{}/ACC_QfN/{}.query.attr".format(path, ego), allattr, fmt='%d')





load_data()