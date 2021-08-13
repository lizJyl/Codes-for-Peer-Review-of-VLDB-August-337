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


def load_data(path="../facebook/ATC_QfN/"):
    ego = 414
    ego = 686
    ego = 348
    # ego = 0
    # ego = 3437
    # ego = 1912
    # ego = 1684
    # ego = 107

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

    j=2
    for i in range(ground_truth.shape[0]):
        # print(i,j)
        truth=set()
        for k in range(ground_truth[i].shape[0]):
            truth.add(int(ground_truth[i][k]))
        result=set()
        for k in range(ACC_result[j].shape[0]):
            result.add(int(ACC_result[j][k]))
        ints=ints+len(truth&result)
        union=union+len(truth|result)
        predict=predict+len(result)
        true=true+len(truth)
        j=j+2
        # print("&", len(truth&result))
        # print("|", len(truth|result))
        print("truth",(truth))
        print("result",(result))

    print("union",union)
    print("intersection", ints)

    print("\npredict", predict)
    print("true", true)

    precision=ints/predict
    recall=ints/true
    f1=2*precision*recall/(precision+recall)
    jac=ints/union
    print("\n\nF1=", f1)
    print("\nprecision=", precision)
    print("recall", recall)
    # print("\nF1=", f1)
    print("\njac", jac)
    # print("i=",i)
    # print("j=", j)




load_data()