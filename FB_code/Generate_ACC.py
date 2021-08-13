import numpy as np
import scipy.sparse as sp
import torch
import numpy.random as npr
import random

np.set_printoptions(threshold=np.inf)
npr.seed(170624)
random.seed(170624)

def load_data(path="./facebook/",ego=0):
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
	qeury_node=[]
	for i in range(circles.shape[0]):
		if len(circles[i])<5:
			continue
		for k in range(len(circles[i])//2-1):
			# print('k', k, circles.shape[0]//3, len(circles[i]))
			select_cols = npr.choice(len(circles[i]), 1, replace=False)#random.randint(1,3)
			input=np.zeros(n_node, dtype=np.int32)
			for j in select_cols:
				# print('select_cols', j, i, circles[i][j])
				input[node_map[int(circles[i][j])]] = 1
				qeury_node.append(node_map[int(circles[i][j])])
			inputs.append(input[np.newaxis,:])
			# print(input.shape, 'input')
			# assert 1<0
	inputs = np.concatenate(inputs, axis=0)
	print("query size",inputs.shape, inputs.dtype)
	np.savetxt("{}/1nodeQuery/{}.sample".format(path,ego), inputs, fmt='%d')
	np.savetxt("{}/1nodeQuery/{}.query.node".format(path, ego), qeury_node, fmt='%d')


	labels=[]
	f = open("{}/1nodeQuery/{}.query.com".format(path, ego), 'w')
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
			np.savetxt(f, [com], fmt='%d', delimiter=' ', newline='\n', )
	print("ground truth size", len(labels))
	labels = np.concatenate(labels, axis=0)
	f.close()
	np.savetxt("{}/1nodeQuery/{}.label".format(path,ego), labels, fmt='%d')

	queryatt = np.genfromtxt("{}{}.attr".format(path, ego), dtype=np.dtype(str))
	queryatt = queryatt.astype(np.int32)
	f2 = open("{}/1nodeQuery/{}.query.feat".format(path, ego), 'w')
	for i in range(circles.shape[0]):
		if len(circles[i])<5:
			continue
		for k in range(len(circles[i])//2-1):
			att=[]
			for j in range(queryatt.shape[1]):
				if queryatt[i][j]==1:
					att.append(j)
			np.savetxt(f2, [att], fmt='%d', delimiter=' ', newline='\n', )

	f2.close()

load_data()