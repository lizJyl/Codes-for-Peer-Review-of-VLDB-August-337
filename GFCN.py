import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, FC, Attention, SPGraphConvolution,SelfLoop



class GFCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        # self.gc1 = GraphConvolution(nfeat, nclass)
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc1_v2 = GraphConvolution(1, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        # self.gc1.weight.data.normal_(0, 0.001)
        # self.gc1.bias.data.zero_()
        # self.gc1_v2.weight.data.normal_(0, 0.001)
        # self.gc1_v2.bias.data.zero_()
        # self.gc2.weight.data.normal_(0, 0.001)
        # self.gc2.bias.data.zero_()
        # self.gc3.weight.data.normal_(0, 0.001)
        # self.gc3.bias.data.zero_()
        self.dropout = dropout

    def forward(self, x, adj, training=True):
        # adj = adj.to_dense()
        # adj = adj.unsqueeze(0)
        # # x = x.unsqueeze(0)
        # print("a shape 1",x.shape)
        #print('shape', x.shape, adj.shape, self.gc1, self.gc1_v2)
        # x = F.relu(self.gc1(x, adj)  )
        # x = F.relu(self.gc1(x[:,:,:-1], adj)  ) + F.relu(self.gc1_v2(x[:,:,-1:],adj))*10
        # print("a shape 2", x.shape)
        # assert 1<0
        # x = F.dropout(x, self.dropout, training=training)
        # x = F.relu(self.gc2(x, adj) +x )
        # x = F.dropout(x, self.dropout, training=training)
        # x = self.gc3(x, adj)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=training)
        x = self.gc3(x, adj)
        # print("weight max,min,mean ",self.gc3.weight.data.max(), self.gc3.weight.data.min(), self.gc3.weight.data.mean())
        # print("grad max,min,mean ", self.gc3.weight.grad_fn.max(), self.gc3.weight.grad_fn.min(),
        #       self.gc3.weight.grad_fn.mean())

        return F.sigmoid(x) #F.log_softmax(x, dim=1)

class GCN_BN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_BN, self).__init__()

        # self.gc1 = GraphConvolution(nfeat, nclass)
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        # self.gc1.weight.data.normal_(0, 0.001)
        # self.gc1.bias.data.zero_()
        # self.gc1_v2.weight.data.normal_(0, 0.001)
        # self.gc1_v2.bias.data.zero_()
        # self.gc2.weight.data.normal_(0, 0.001)
        # self.gc2.bias.data.zero_()
        # self.gc3.weight.data.normal_(0, 0.001)
        # self.gc3.bias.data.zero_()
        self.dropout = dropout

    def forward(self, x, adj, training=True):
        x = self.gc1(x, adj)
        x = self.bn1(x.transpose(-1,-2)).transpose(-1,-2)
        x = F.relu(x)
        # x = F.dropout(x, self.dropout, training=training)
        # y = x
        x = self.gc2(x, adj)
        x = self.bn2(x.transpose(-1,-2)).transpose(-1,-2)
        x = F.relu(x)
        # x = x+y
        # x = F.dropout(x, self.dropout, training=training)
        x = self.gc3(x, adj)
        # print("weight max,min,mean ",self.gc3.weight.data.max(), self.gc3.weight.data.min(), self.gc3.weight.data.mean())
        # print("grad max,min,mean ", self.gc3.weight.grad_fn.max(), self.gc3.weight.grad_fn.min(),
        #       self.gc3.weight.grad_fn.mean())

        return F.sigmoid(x) #F.log_softmax(x, dim=1)

class ResGCN_BN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(ResGCN_BN, self).__init__()

        # self.gc1 = GraphConvolution(nfeat, nclass)
        self.gc1 = GraphConvolution(2, nhid)
        self.fd1 = GraphConvolution(nfeat, nhid)
        self.sl1fa = SelfLoop(nfeat, nhid)
        self.fc1 = GraphConvolution(1, nhid)
        self.sl1s = SelfLoop(2, nhid)
        self.sl1f = SelfLoop(1, nhid)
        self.sl1 = SelfLoop(nhid, nhid)
        self.res1 = SelfLoop(nfeat, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn11 = nn.BatchNorm1d(nhid)
        self.bn12 = nn.BatchNorm1d(nhid)
        self.bn13 = nn.BatchNorm1d(nhid)
        self.cnd1 = SelfLoop(3 * nhid, nhid)

        self.gc2 = GraphConvolution(nhid, nhid)
        self.fc2 = GraphConvolution(nhid, nhid)
        self.sl2s = SelfLoop(nhid, nhid)
        self.sl2f = SelfLoop(nhid, nhid)
        self.sl2 = SelfLoop(nhid, nhid)
        self.res2 = SelfLoop(nhid, nhid)
        self.fd2 = GraphConvolution(nhid, nhid)
        self.sl2fa = SelfLoop(nhid, nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.bn21 = nn.BatchNorm1d(nhid)
        self.bn22 = nn.BatchNorm1d(nhid)
        self.bn23 = nn.BatchNorm1d(nhid)
        self.cnd2 = SelfLoop(3 * nhid, nhid)

        self.gc3 = GraphConvolution(nhid, nclass)
        self.fc3 = GraphConvolution(nhid, nclass)
        self.sl3s = SelfLoop(nhid, nclass)
        self.sl3f = SelfLoop(nhid, nclass)
        self.sl3 = SelfLoop(nhid, nclass)
        self.fd3 = GraphConvolution(nhid, nclass)
        self.sl3fa = SelfLoop(nhid, nclass)
        self.res3 = SelfLoop(nhid, nclass)
        self.bn3 = nn.BatchNorm1d(nclass)
        self.cnd3 = SelfLoop(3 * nclass, nclass)


        self.dropout = dropout
        self.dropout = dropout

    def forward(self, x, att, adj, feat, training=True):
        y1 = x
        x1 = self.gc1(x, adj) + self.sl1s(x)
        x2 = self.fc1(att, feat)
        x3=self.fd1(feat,adj)+ self.sl1fa(feat)
        # x = x1 + x2 +x3 # +self.sl1(x2)
        x= torch.cat([x1,x2],2)
        x=torch.cat([x,x3],2)
        x=self.cnd1(x)

        x = self.bn1(x.transpose(-1, -2)).transpose(-1, -2)
        # x2 = self.bn1(x2.transpose(-1, -2)).transpose(-1, -2)
        x = F.relu(x)
        # x = F.dropout(x, self.dropout, training=training)
        y = x
        x2 = feat.transpose(-1, -2).matmul(x) + self.sl1f(att)
        x2 = self.bn12(x2.transpose(-1, -2)).transpose(-1, -2)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, self.dropout, training=training)

        x = F.dropout(x, self.dropout, training=training)

        # x1=x+self.sl1s(y1)
        x1=x
        # x1 = self.bn11(x1.transpose(-1, -2)).transpose(-1, -2)
        # x1 = F.relu(x1)
        # x1 = F.dropout(x1, self.dropout, training=training)

        x3=self.bn13(x3.transpose(-1, -2)).transpose(-1, -2)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, self.dropout, training=training)


        y1=x1
        x1 = self.gc2(x1, adj) + self.sl2s(x1)
        y2 = x2
        x2 = self.fc2(x2, feat)
        x3 = self.fd2(x3, adj) + self.sl2fa(x3)
        # x = x + x2  # +self.sl2(x2)
        x = torch.cat([x1, x2], 2)
        x = torch.cat([x, x3], 2)
        x = self.cnd2(x)

        x = self.bn2(x.transpose(-1, -2)).transpose(-1, -2)
        x = F.relu(x)
        # x = F.dropout(x, self.dropout, training=training)
        y = x
        x2 = feat.transpose(-1, -2).matmul(x) + self.sl2f(y2)
        x2 = self.bn21(x2.transpose(-1, -2)).transpose(-1, -2)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, self.dropout, training=training)

        x = F.dropout(x, self.dropout, training=training)

        x1=x
        # x1=x+ self.sl2s(y1)
        # x1 = self.bn21(x1.transpose(-1, -2)).transpose(-1, -2)
        # x1 = F.relu(x1)
        # x1 = F.dropout(x1, self.dropout, training=training)
        x3 = self.bn23(x3.transpose(-1, -2)).transpose(-1, -2)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, self.dropout, training=training)



        x1 = self.gc3(x1, adj) + self.sl3s(x1)  # +self.sl3(y)
        # x2 = self.sl3(x2)
        x2 = self.fc3(x2, feat)
        x3 = self.fd3(x3, adj) + self.sl3fa(x3)
        # x = x + x2#self.sl3(x2)
        x = torch.cat([x1, x2], 2)
        x = torch.cat([x, x3], 2)
        # print("concate shape", x.shape)
        x = self.cnd3(x)


        # x = self.bn3(x.transpose(-1, -2)).transpose(-1, -2)
        # x1 = F.relu(x1)  # +self.res1(y)
        # x2 = F.relu(x2)

        # x = F.dropout(x, self.dropout, training=training)

        return F.sigmoid(x) ,x # F.log_softmax(x, dim=1)


class QD_GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(QD_GCN, self).__init__()

        self.graph_encoder1 = GraphConvolution(nfeat, nhid)
        self.structure_encoder1 = GraphConvolution(2, nhid)
        self.attribute_encoder1 = GraphConvolution(1, nhid)
        self.self_GE1= SelfLoop(nfeat, nhid)
        self.self_SE1 = SelfLoop(2, nhid)
        self.self_AE1 = SelfLoop(1, nhid)
        # self.sl1 = SelfLoop(nhid, nhid)
        # self.res1 = SelfLoop(nfeat, nhid)
        self.bn_GE1 = nn.BatchNorm1d(nhid)
        self.bn_SE1 = nn.BatchNorm1d(nhid)
        self.bn_AE1 = nn.BatchNorm1d(nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.cnd1 = SelfLoop(3 * nhid, nhid)

        self.graph_encoder2 = GraphConvolution(nhid, nhid)
        self.structure_encoder2 = GraphConvolution(nhid, nhid)
        self.attribute_encoder2 = GraphConvolution(nhid, nhid)
        self.self_GE2 = SelfLoop(nhid, nhid)
        self.self_SE2 = SelfLoop(nhid, nhid)
        self.self_AE2 = SelfLoop(nhid, nhid)
        # self.res2 = SelfLoop(nhid, nhid)
        # self.sl2fa = SelfLoop(nhid, nhid)
        self.bn_GE2 = nn.BatchNorm1d(nhid)
        self.bn_SE2 = nn.BatchNorm1d(nhid)
        self.bn_AE2 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.cnd2 = SelfLoop(3 * nhid, nhid)

        self.graph_encoder3 = GraphConvolution(nhid, nclass)
        self.structure_encoder3 = GraphConvolution(nhid, nclass)
        self.attribute_encoder3 = GraphConvolution(nhid, nclass)
        self.self_GE3 = SelfLoop(nhid, nclass)
        self.self_SE3 = SelfLoop(nhid, nclass)
        self.self_AE3 = SelfLoop(nhid, nclass)
        # self.sl3fa = SelfLoop(nhid, nclass)
        # self.res3 = SelfLoop(nhid, nclass)
        # self.bn3 = nn.BatchNorm1d(nclass)
        self.cnd3 = SelfLoop(3 * nclass, nclass)


        self.dropout = dropout

      

    def forward(self, node_input, att_input, adj, Fadj, feat, training=True):
        # y1 = x
        model1=self.graph_encoder1(feat,adj)+ self.self_GE1(feat)
        model2 = self.structure_encoder1(node_input, adj) + self.self_SE1(node_input)
        model3 = self.attribute_encoder1(att_input, Fadj)
        
        # x = x1 + x2 +x3 # +self.sl1(x2)
        model= torch.cat([model1,model2],2)
        model=torch.cat([model,model3],2)
        model=self.cnd1(model)

        model = self.bn1(model.transpose(-1, -2)).transpose(-1, -2)
        model = F.relu(model)
        model = F.dropout(model, self.dropout, training=training)

        ##############################################################################################################

        model1=self.bn_GE1(model1.transpose(-1, -2)).transpose(-1, -2)
        model1 = F.relu(model1)
        model1 = F.dropout(model1, self.dropout, training=training)
        model1=self.graph_encoder2(model1,adj)+ self.self_GE2(model1)

        # model_SE = self.bn_SE1(model2.transpose(-1, -2)).transpose(-1, -2)
        # model_SE = F.relu(model_SE)
        # model_SE = F.dropout(model_SE, self.dropout, training=training)
        # model2 = self.structure_encoder2(model, adj) + self.self_SE2(model_SE)
        model2 = self.structure_encoder2(model, adj) + self.self_SE2(model)

        model3 = Fadj.transpose(-1, -2).matmul(model) + self.self_AE1(att_input)
        model_AE = self.bn_AE1(model3.transpose(-1, -2)).transpose(-1, -2)
        model_AE = F.relu(model_AE)
        model_AE = F.dropout(model_AE, self.dropout, training=training)
        # model3 = self.attribute_encoder2(model3, Fadj)
        model3 = self.attribute_encoder2(model_AE, Fadj)


        model= torch.cat([model1,model2],2)
        model=torch.cat([model,model3],2)
        model=self.cnd2(model)

        model = self.bn2(model.transpose(-1, -2)).transpose(-1, -2)
        model = F.relu(model)
        model = F.dropout(model, self.dropout, training=training)

        ################################################################################################################

        model1=self.bn_GE2(model1.transpose(-1, -2)).transpose(-1, -2)
        model1 = F.relu(model1)
        model1 = F.dropout(model1, self.dropout, training=training)
        model1=self.graph_encoder3(model1,adj)+ self.self_GE3(model1)

        # model_SE = self.bn_SE2(model2.transpose(-1, -2)).transpose(-1, -2)
        # model_SE = F.relu(model_SE)
        # model_SE = F.dropout(model_SE, self.dropout, training=training)
        # model2 = self.structure_encoder3(model, adj) + self.self_SE3(model_SE)
        model2 = self.structure_encoder3(model, adj) + self.self_SE3(model)

        model3 = Fadj.transpose(-1, -2).matmul(model) + self.self_AE2(model_AE)
        model3 = self.bn_AE2(model3.transpose(-1, -2)).transpose(-1, -2)
        model3 = F.relu(model3)
        model3 = F.dropout(model3, self.dropout, training=training)
        model3 = self.attribute_encoder3(model3, Fadj)


        model= torch.cat([model1,model2],2)
        model=torch.cat([model,model3],2)
        model=self.cnd3(model)

        # model = self.bn1(model.transpose(-1, -2)).transpose(-1, -2)
        # model = F.relu(model)
        # model = F.dropout(model, self.dropout, training=training)


        return torch.sigmoid(model) ,model # F.log_softmax(x, dim=1)

class QD_GCN_alloutput(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(QD_GCN_alloutput, self).__init__()

        self.graph_encoder1 = GraphConvolution(nfeat, nhid)
        self.structure_encoder1 = GraphConvolution(2, nhid)
        self.attribute_encoder1 = GraphConvolution(1, nhid)
        self.self_GE1= SelfLoop(nfeat, nhid)
        self.self_SE1 = SelfLoop(2, nhid)
        self.self_AE1 = SelfLoop(1, nhid)
        # self.sl1 = SelfLoop(nhid, nhid)
        # self.res1 = SelfLoop(nfeat, nhid)
        self.bn_GE1 = nn.BatchNorm1d(nhid)
        self.bn_SE1 = nn.BatchNorm1d(nhid)
        self.bn_AE1 = nn.BatchNorm1d(nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.cnd1 = SelfLoop(3 * nhid, nhid)

        self.graph_encoder2 = GraphConvolution(nhid, nhid)
        self.structure_encoder2 = GraphConvolution(nhid, nhid)
        self.attribute_encoder2 = GraphConvolution(nhid, nhid)
        self.self_GE2 = SelfLoop(nhid, nhid)
        self.self_SE2 = SelfLoop(nhid, nhid)
        self.self_AE2 = SelfLoop(nhid, nhid)
        # self.res2 = SelfLoop(nhid, nhid)
        # self.sl2fa = SelfLoop(nhid, nhid)
        self.bn_GE2 = nn.BatchNorm1d(nhid)
        self.bn_SE2 = nn.BatchNorm1d(nhid)
        self.bn_AE2 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.cnd2 = SelfLoop(3 * nhid, nhid)

        self.graph_encoder3 = GraphConvolution(nhid, nclass)
        self.structure_encoder3 = GraphConvolution(nhid, nclass)
        self.attribute_encoder3 = GraphConvolution(nhid, nclass)
        self.self_GE3 = SelfLoop(nhid, nclass)
        self.self_SE3 = SelfLoop(nhid, nclass)
        self.self_AE3 = SelfLoop(nhid, nclass)
        # self.sl3fa = SelfLoop(nhid, nclass)
        # self.res3 = SelfLoop(nhid, nclass)
        # self.bn3 = nn.BatchNorm1d(nclass)
        self.cnd3 = SelfLoop(3 * nclass, nclass)


        self.dropout = dropout

      

    def forward(self, node_input, att_input, adj, Fadj, feat, training=True):
        # y1 = x
        model1=self.graph_encoder1(feat,adj)+ self.self_GE1(feat)
        model2 = self.structure_encoder1(node_input, adj) + self.self_SE1(node_input)
        model3 = self.attribute_encoder1(att_input, Fadj)
        
        # x = x1 + x2 +x3 # +self.sl1(x2)
        model= torch.cat([model1,model2],2)
        model=torch.cat([model,model3],2)
        model=self.cnd1(model)

        model = self.bn1(model.transpose(-1, -2)).transpose(-1, -2)
        model = F.relu(model)
        model = F.dropout(model, self.dropout, training=training)

        ##############################################################################################################

        model1=self.bn_GE1(model1.transpose(-1, -2)).transpose(-1, -2)
        model1 = F.relu(model1)
        model1 = F.dropout(model1, self.dropout, training=training)
        model1=self.graph_encoder2(model1,adj)+ self.self_GE2(model1)
        model11 = model1
        # model_SE = self.bn_SE1(model2.transpose(-1, -2)).transpose(-1, -2)
        # model_SE = F.relu(model_SE)
        # model_SE = F.dropout(model_SE, self.dropout, training=training)
        # model2 = self.structure_encoder2(model, adj) + self.self_SE2(model_SE)
        model2 = self.structure_encoder2(model, adj) + self.self_SE2(model)
        model22 = model2
        model3 = Fadj.transpose(-1, -2).matmul(model) + self.self_AE1(att_input)
        model_AE = self.bn_AE1(model3.transpose(-1, -2)).transpose(-1, -2)
        model_AE = F.relu(model_AE)
        model_AE = F.dropout(model_AE, self.dropout, training=training)
        # model3 = self.attribute_encoder2(model3, Fadj)
        model3 = self.attribute_encoder2(model_AE, Fadj)
        model33 = model3

        model= torch.cat([model1,model2],2)
        model=torch.cat([model,model3],2)
        model=self.cnd2(model)
        model_ = model

        model = self.bn2(model.transpose(-1, -2)).transpose(-1, -2)
        model = F.relu(model)
        model = F.dropout(model, self.dropout, training=training)

        ################################################################################################################

        model1=self.bn_GE2(model1.transpose(-1, -2)).transpose(-1, -2)
        model1 = F.relu(model1)
        model1 = F.dropout(model1, self.dropout, training=training)
        model1=self.graph_encoder3(model1,adj)+ self.self_GE3(model1)

        # model_SE = self.bn_SE2(model2.transpose(-1, -2)).transpose(-1, -2)
        # model_SE = F.relu(model_SE)
        # model_SE = F.dropout(model_SE, self.dropout, training=training)
        # model2 = self.structure_encoder3(model, adj) + self.self_SE3(model_SE)
        model2 = self.structure_encoder3(model, adj) + self.self_SE3(model)

        model3 = Fadj.transpose(-1, -2).matmul(model) + self.self_AE2(model_AE)
        model3 = self.bn_AE2(model3.transpose(-1, -2)).transpose(-1, -2)
        model3 = F.relu(model3)
        model3 = F.dropout(model3, self.dropout, training=training)
        model3 = self.attribute_encoder3(model3, Fadj)


        model= torch.cat([model1,model2],2)
        model=torch.cat([model,model3],2)
        model=self.cnd3(model)

        # model = self.bn1(model.transpose(-1, -2)).transpose(-1, -2)
        # model = F.relu(model)
        # model = F.dropout(model, self.dropout, training=training)


        # return model1,model2,model3,model#torch.sigmoid(model) ,model # F.log_softmax(x, dim=1)
        return model11,model22,model33,model_,model

class QD_GCN_alloutput_NoG(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(QD_GCN_alloutput_NoG, self).__init__()

        self.graph_encoder1 = GraphConvolution(nfeat, nhid)
        self.structure_encoder1 = GraphConvolution(2, nhid)
        self.attribute_encoder1 = GraphConvolution(1, nhid)
        self.self_GE1= SelfLoop(nfeat, nhid)
        self.self_SE1 = SelfLoop(2, nhid)
        self.self_AE1 = SelfLoop(1, nhid)
        # self.sl1 = SelfLoop(nhid, nhid)
        # self.res1 = SelfLoop(nfeat, nhid)
        self.bn_GE1 = nn.BatchNorm1d(nhid)
        self.bn_SE1 = nn.BatchNorm1d(nhid)
        self.bn_AE1 = nn.BatchNorm1d(nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.cnd1 = SelfLoop(2 * nhid, nhid)

        self.graph_encoder2 = GraphConvolution(nhid, nhid)
        self.structure_encoder2 = GraphConvolution(nhid, nhid)
        self.attribute_encoder2 = GraphConvolution(nhid, nhid)
        self.self_GE2 = SelfLoop(nhid, nhid)
        self.self_SE2 = SelfLoop(nhid, nhid)
        self.self_AE2 = SelfLoop(nhid, nhid)
        # self.res2 = SelfLoop(nhid, nhid)
        # self.sl2fa = SelfLoop(nhid, nhid)
        self.bn_GE2 = nn.BatchNorm1d(nhid)
        self.bn_SE2 = nn.BatchNorm1d(nhid)
        self.bn_AE2 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.cnd2 = SelfLoop(2 * nhid, nhid)

        self.graph_encoder3 = GraphConvolution(nhid, nclass)
        self.structure_encoder3 = GraphConvolution(nhid, nclass)
        self.attribute_encoder3 = GraphConvolution(nhid, nclass)
        self.self_GE3 = SelfLoop(nhid, nclass)
        self.self_SE3 = SelfLoop(nhid, nclass)
        self.self_AE3 = SelfLoop(nhid, nclass)
        # self.sl3fa = SelfLoop(nhid, nclass)
        # self.res3 = SelfLoop(nhid, nclass)
        # self.bn3 = nn.BatchNorm1d(nclass)
        self.cnd3 = SelfLoop(2 * nclass, nclass)


        self.dropout = dropout

      

    def forward(self, node_input, att_input, adj, Fadj, feat, training=True):
        # y1 = x
        model11=node_input
        # model1=self.graph_encoder1(feat,adj)+ self.self_GE1(feat)
        model2 = self.structure_encoder1(node_input, adj) + self.self_SE1(node_input)
        model3 = self.attribute_encoder1(att_input, Fadj)
        
        # x = x1 + x2 +x3 # +self.sl1(x2)
        # model= torch.cat([model1,model2],2)
        model=torch.cat([model2,model3],2)
        model=self.cnd1(model)

        model = self.bn1(model.transpose(-1, -2)).transpose(-1, -2)
        model = F.relu(model)
        model = F.dropout(model, self.dropout, training=training)

        ##############################################################################################################

        # model1=self.bn_GE1(model1.transpose(-1, -2)).transpose(-1, -2)
        # model1 = F.relu(model1)
        # model1 = F.dropout(model1, self.dropout, training=training)
        # model1=self.graph_encoder2(model1,adj)+ self.self_GE2(model1)
        # model11 = model1
        # model_SE = self.bn_SE1(model2.transpose(-1, -2)).transpose(-1, -2)
        # model_SE = F.relu(model_SE)
        # model_SE = F.dropout(model_SE, self.dropout, training=training)
        # model2 = self.structure_encoder2(model, adj) + self.self_SE2(model_SE)
        model2 = self.structure_encoder2(model, adj) + self.self_SE2(model)
        model22 = model2
        model3 = Fadj.transpose(-1, -2).matmul(model) + self.self_AE1(att_input)
        model_AE = self.bn_AE1(model3.transpose(-1, -2)).transpose(-1, -2)
        model_AE = F.relu(model_AE)
        model_AE = F.dropout(model_AE, self.dropout, training=training)
        # model3 = self.attribute_encoder2(model3, Fadj)
        model3 = self.attribute_encoder2(model_AE, Fadj)
        model33 = model3

        # model= torch.cat([model1,model2],2)
        model=torch.cat([model2,model3],2)
        model=self.cnd2(model)
        model_ = model

        model = self.bn2(model.transpose(-1, -2)).transpose(-1, -2)
        model = F.relu(model)
        model = F.dropout(model, self.dropout, training=training)

        ################################################################################################################

        # model1=self.bn_GE2(model1.transpose(-1, -2)).transpose(-1, -2)
        # model1 = F.relu(model1)
        # model1 = F.dropout(model1, self.dropout, training=training)
        # model1=self.graph_encoder3(model1,adj)+ self.self_GE3(model1)

        # model_SE = self.bn_SE2(model2.transpose(-1, -2)).transpose(-1, -2)
        # model_SE = F.relu(model_SE)
        # model_SE = F.dropout(model_SE, self.dropout, training=training)
        # model2 = self.structure_encoder3(model, adj) + self.self_SE3(model_SE)
        model2 = self.structure_encoder3(model, adj) + self.self_SE3(model)

        model3 = Fadj.transpose(-1, -2).matmul(model) + self.self_AE2(model_AE)
        model3 = self.bn_AE2(model3.transpose(-1, -2)).transpose(-1, -2)
        model3 = F.relu(model3)
        model3 = F.dropout(model3, self.dropout, training=training)
        model3 = self.attribute_encoder3(model3, Fadj)


        # model= torch.cat([model1,model2],2)
        model=torch.cat([model2,model3],2)
        model=self.cnd3(model)

        # model = self.bn1(model.transpose(-1, -2)).transpose(-1, -2)
        # model = F.relu(model)
        # model = F.dropout(model, self.dropout, training=training)


        # return model1,model2,model3,model#torch.sigmoid(model) ,model # F.log_softmax(x, dim=1)
        return model11,model22,model33,model_,model

class ResGCN_BN_Update(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(ResGCN_BN_Update, self).__init__()

         # self.gc1 = GraphConvolution(nfeat, nclass)
        self.gc1 = GraphConvolution(2, nhid)
        self.fd1 = GraphConvolution(nfeat, nhid)
        self.sl1fa = SelfLoop(nfeat, nhid)
        self.fc1 = GraphConvolution(1, nhid)
        self.sl1s = SelfLoop(2, nhid)
        self.sl1f = SelfLoop(1, nhid)
        self.sl1 = SelfLoop(nhid, nhid)
        self.res1 = SelfLoop(nfeat, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn11 = nn.BatchNorm1d(nhid)
        self.bn12 = nn.BatchNorm1d(nhid)
        self.bn13 = nn.BatchNorm1d(nhid)
        self.cnd1 = SelfLoop(3 * nhid, nhid)

        self.gc2 = GraphConvolution(nhid, nhid)
        self.fc2 = GraphConvolution(nhid, nhid)
        self.sl2s = SelfLoop(nhid, nhid)
        self.sl2f = SelfLoop(nhid, nhid)
        self.sl2 = SelfLoop(nhid, nhid)
        self.res2 = SelfLoop(nhid, nhid)
        self.fd2 = GraphConvolution(nhid, nhid)
        self.sl2fa = SelfLoop(nhid, nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.bn21 = nn.BatchNorm1d(nhid)
        self.bn22 = nn.BatchNorm1d(nhid)
        self.bn23 = nn.BatchNorm1d(nhid)
        self.cnd2 = SelfLoop(3 * nhid, nhid)

        self.gc3 = GraphConvolution(nhid, nclass)
        self.fc3 = GraphConvolution(nhid, nclass)
        self.sl3s = SelfLoop(nhid, nclass)
        self.sl3f = SelfLoop(nhid, nclass)
        self.sl3 = SelfLoop(nhid, nclass)
        self.fd3 = GraphConvolution(nhid, nclass)
        self.sl3fa = SelfLoop(nhid, nclass)
        self.res3 = SelfLoop(nhid, nclass)
        self.bn3 = nn.BatchNorm1d(nclass)
        self.cnd3 = SelfLoop(3 * nclass, nclass)


        self.dropout = dropout
        self.dropout = dropout

    def forward(self, x, att, adj, feat, training=True):
        y1 = x
        x1 = self.gc1(x, adj) + self.sl1s(x)
        x2 = self.fc1(att, feat)
        x3=self.fd1(feat,adj)+ self.sl1fa(feat)
        # x = x1 + x2 +x3 # +self.sl1(x2)
        x= torch.cat([x1,x3],2)
        # x=torch.cat([x,x3],2)
        x=self.cnd1(x)

        x = self.bn1(x.transpose(-1, -2)).transpose(-1, -2)
        # x2 = self.bn1(x2.transpose(-1, -2)).transpose(-1, -2)
        x = F.relu(x)
        # x = F.dropout(x, self.dropout, training=training)
        y = x
        x2 = feat.transpose(-1, -2).matmul(x) + self.sl1f(att)
        x2 = self.bn12(x2.transpose(-1, -2)).transpose(-1, -2)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, self.dropout, training=training)

        x = F.dropout(x, self.dropout, training=training)

        # x1=x+self.sl1s(y1)
        x1=x
        # x1 = self.bn11(x1.transpose(-1, -2)).transpose(-1, -2)
        # x1 = F.relu(x1)
        # x1 = F.dropout(x1, self.dropout, training=training)

        x3=self.bn13(x3.transpose(-1, -2)).transpose(-1, -2)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, self.dropout, training=training)
        # x3=x


        y1=x1
        x1 = self.gc2(x1, adj) + self.sl2s(x1)
        y2 = x2
        x2 = self.fc2(x2, feat)
        x3 = self.fd2(x3, adj) + self.sl2fa(x3)
        # x = x + x2  # +self.sl2(x2)
        x = torch.cat([x1, x3], 2)
        # x = torch.cat([x, x3], 2)
        x = self.cnd2(x)

        x = self.bn2(x.transpose(-1, -2)).transpose(-1, -2)
        x = F.relu(x)
        # x = F.dropout(x, self.dropout, training=training)
        y = x
        x2 = feat.transpose(-1, -2).matmul(x) + self.sl2f(y2)
        x2 = self.bn21(x2.transpose(-1, -2)).transpose(-1, -2)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, self.dropout, training=training)

        x = F.dropout(x, self.dropout, training=training)

        x1=x
        # x1=x+ self.sl2s(y1)
        # x1 = self.bn21(x1.transpose(-1, -2)).transpose(-1, -2)
        # x1 = F.relu(x1)
        # x1 = F.dropout(x1, self.dropout, training=training)

        x3 = self.bn23(x3.transpose(-1, -2)).transpose(-1, -2)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, self.dropout, training=training)
        # x3 = x



        x1 = self.gc3(x1, adj) + self.sl3s(x1)  # +self.sl3(y)
        # x2 = self.sl3(x2)
        x2 = self.fc3(x2, feat)
        x3 = self.fd3(x3, adj) + self.sl3fa(x3)
        # x = x + x2#self.sl3(x2)
        x = torch.cat([x1, x3], 2)
        # x = torch.cat([x, x3], 2)
        x = self.cnd3(x)
        # x = self.bn3(x.transpose(-1, -2)).transpose(-1, -2)
        # x1 = F.relu(x1)  # +self.res1(y)
        # x2 = F.relu(x2)

        # x = F.dropout(x, self.dropout, training=training)

        return F.sigmoid(x) ,x # F.log_softmax(x, dim=1)

class ResGCN_BN_ADD(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(ResGCN_BN_ADD, self).__init__()

        # self.gc1 = GraphConvolution(nfeat, nclass)
        self.gc1 = GraphConvolution(2, nhid)
        self.fd1 = GraphConvolution(nfeat, nhid)
        self.sl1fa = SelfLoop(nfeat, nhid)
        self.fc1 = GraphConvolution(1, nhid)
        self.sl1s = SelfLoop(2, nhid)
        self.sl1f = SelfLoop(1, nhid)
        self.sl1 = SelfLoop(nhid, nhid)
        self.res1 = SelfLoop(nfeat, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn11 = nn.BatchNorm1d(nhid)
        self.bn12 = nn.BatchNorm1d(nhid)
        self.bn13 = nn.BatchNorm1d(nhid)
        self.cnd1 = SelfLoop(3 * nhid, nhid)

        self.gc2 = GraphConvolution(nhid, nhid)
        self.fc2 = GraphConvolution(nhid, nhid)
        self.sl2s = SelfLoop(nhid, nhid)
        self.sl2f = SelfLoop(nhid, nhid)
        self.sl2 = SelfLoop(nhid, nhid)
        self.res2 = SelfLoop(nhid, nhid)
        self.fd2 = GraphConvolution(nhid, nhid)
        self.sl2fa = SelfLoop(nhid, nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.bn21 = nn.BatchNorm1d(nhid)
        self.bn22 = nn.BatchNorm1d(nhid)
        self.bn23 = nn.BatchNorm1d(nhid)
        self.cnd2 = SelfLoop(3 * nhid, nhid)

        self.gc3 = GraphConvolution(nhid, nclass)
        self.fc3 = GraphConvolution(nhid, nclass)
        self.sl3s = SelfLoop(nhid, nclass)
        self.sl3f = SelfLoop(nhid, nclass)
        self.sl3 = SelfLoop(nhid, nclass)
        self.fd3 = GraphConvolution(nhid, nclass)
        self.sl3fa = SelfLoop(nhid, nclass)
        self.res3 = SelfLoop(nhid, nclass)
        self.bn3 = nn.BatchNorm1d(nclass)
        self.cnd3 = SelfLoop(3 * nclass, nclass)

        self.dropout = dropout
        self.dropout = dropout

    def forward(self, x, att, adj, feat, training=True):
        y1 = x
        x1 = self.gc1(x, adj) + self.sl1s(x)
        x2 = self.fc1(att, feat)
        x3 = self.fd1(feat, adj) + self.sl1fa(feat)
        x = x1 + x2 +x3 # +self.sl1(x2)

        # x = torch.cat([x1.unsqueeze(-1), x2.unsqueeze(-1)], -1)
        # x = torch.cat([x, x3.unsqueeze(-1)], -1)
        # x=torch.min(x,-1,keepdim=False)[0]
        # x = torch.mean(x, -1, keepdim=False)

        # x = torch.cat([x1, x2], 2)
        # x = torch.cat([x, x3], 2)
        # x = self.cnd1(x)

        x = self.bn1(x.transpose(-1, -2)).transpose(-1, -2)
        # x2 = self.bn1(x2.transpose(-1, -2)).transpose(-1, -2)
        x = F.relu(x)
        # x = F.dropout(x, self.dropout, training=training)
        y = x
        x2 = feat.transpose(-1, -2).matmul(x) + self.sl1f(att)
        x2 = self.bn12(x2.transpose(-1, -2)).transpose(-1, -2)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, self.dropout, training=training)

        x = F.dropout(x, self.dropout, training=training)

        # x1=x+self.sl1s(y1)
        x1 = x
        # x1 = self.bn11(x1.transpose(-1, -2)).transpose(-1, -2)
        # x1 = F.relu(x1)
        # x1 = F.dropout(x1, self.dropout, training=training)

        x3 = self.bn13(x3.transpose(-1, -2)).transpose(-1, -2)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, self.dropout, training=training)

        y1 = x1
        x1 = self.gc2(x1, adj) + self.sl2s(x1)
        y2 = x2
        x2 = self.fc2(x2, feat)
        x3 = self.fd2(x3, adj) + self.sl2fa(x3)
        x = x1 + x2 + x3  # +self.sl2(x2)

        # x = torch.cat([x1.unsqueeze(-1), x2.unsqueeze(-1)], -1)
        # x = torch.cat([x, x3.unsqueeze(-1)], -1)
        # x = torch.min(x, -1, keepdim=False)[0]
        # x = torch.mean(x, -1, keepdim=False)

        # x = torch.cat([x1, x2], 2)
        # x = torch.cat([x, x3], 2)
        # x = self.cnd2(x)

        x = self.bn2(x.transpose(-1, -2)).transpose(-1, -2)
        x = F.relu(x)
        # x = F.dropout(x, self.dropout, training=training)
        y = x
        x2 = feat.transpose(-1, -2).matmul(x) + self.sl2f(y2)
        x2 = self.bn21(x2.transpose(-1, -2)).transpose(-1, -2)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, self.dropout, training=training)

        x = F.dropout(x, self.dropout, training=training)

        x1 = x
        # x1=x+ self.sl2s(y1)
        # x1 = self.bn21(x1.transpose(-1, -2)).transpose(-1, -2)
        # x1 = F.relu(x1)
        # x1 = F.dropout(x1, self.dropout, training=training)
        x3 = self.bn23(x3.transpose(-1, -2)).transpose(-1, -2)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, self.dropout, training=training)

        x1 = self.gc3(x1, adj) + self.sl3s(x1)  # +self.sl3(y)
        # x2 = self.sl3(x2)
        x2 = self.fc3(x2, feat)
        x3 = self.fd3(x3, adj) + self.sl3fa(x3)
        x = x1 + x2 + x3#self.sl3(x2)

        # x = torch.cat([x1.unsqueeze(-1), x2.unsqueeze(-1)], -1)
        # x = torch.cat([x, x3.unsqueeze(-1)], -1)
        # x = torch.min(x, -1, keepdim=False)[0]
        # x = torch.mean(x, -1, keepdim=False)

        # x = torch.cat([x1, x2], 2)
        # x = torch.cat([x, x3], 2)
        # x = self.cnd3(x)
        # x = self.bn3(x.transpose(-1, -2)).transpose(-1, -2)
        # x1 = F.relu(x1)  # +self.res1(y)
        # x2 = F.relu(x2)

        # x = F.dropout(x, self.dropout, training=training)

        return F.sigmoid(x)  # F.log_softmax(x, dim=1)

class Self_ResGCN_BN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(Self_ResGCN_BN, self).__init__()

        # self.gc1 = GraphConvolution(nfeat, nclass)
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.fc1 = GraphConvolution(1, nhid)
        self.sl1s = SelfLoop(nfeat, nhid)
        self.sl1f = SelfLoop(1, nhid)
        self.sl1 = SelfLoop(nhid, nhid)
        self.res1 = SelfLoop(nfeat, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn11 = nn.BatchNorm1d(nhid)

        self.gc2 = GraphConvolution(nhid, nhid)
        self.fc2 = GraphConvolution(nhid, nhid)
        self.sl2s = SelfLoop(nhid, nhid)
        self.sl2f = SelfLoop(nhid, nhid)
        self.sl2 = SelfLoop(nhid, nhid)
        self.res2 = SelfLoop(nhid, nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.bn21 = nn.BatchNorm1d(nhid)

        self.gc3 = GraphConvolution(nhid, nclass)
        self.fc3 = GraphConvolution(nhid, nhid)
        self.sl3s = SelfLoop(nhid, nclass)
        self.sl3f = SelfLoop(nhid, nclass)
        self.sl3 = SelfLoop(nhid, nclass)
        self.res3 = SelfLoop(nhid, nclass)
        self.bn3 = nn.BatchNorm1d(nclass)
        self.dropout = dropout
        self.dropout = dropout

    def forward(self, x,att, adj,feat, training=True):
        y=x
        x1=self.gc1(x, adj)+self.sl1s(x)
        x2=self.fc1(att,feat)
        x = x1+x2#+self.sl1(x2)
        x = self.bn1(x.transpose(-1,-2)).transpose(-1,-2)
        # x2 = self.bn1(x2.transpose(-1, -2)).transpose(-1, -2)
        x = F.relu(x)
        y = x
        x2=feat.transpose(-1, -2).matmul(x)+self.sl1f(att)
        x2 = self.bn11(x2.transpose(-1, -2)).transpose(-1, -2)
        x2 = F.relu(x2)
        # x2 = F.dropout(x2, self.dropout, training=training)
        x = F.dropout(x, self.dropout, training=training)


        x = self.gc2(x, adj)+self.sl2s(x)
        y2=x2
        x2 = self.fc2(x2, feat)
        x = x +x2#+self.sl2(x2)
        x = self.bn2(x.transpose(-1, -2)).transpose(-1, -2)
        x = F.relu(x)
        y = x
        x2 = feat.transpose(-1, -2).matmul(x)+self.sl2f(y2)
        x2 = self.bn21(x2.transpose(-1, -2)).transpose(-1, -2)
        x2 = F.relu(x2)
        # x2 = F.dropout(x2, self.dropout, training=training)
        x = F.dropout(x, self.dropout, training=training)

        x = self.gc3(x, adj)+self.sl3s(x)#+self.sl3(y)
        # x2 = self.sl3(x2)
        # x2 = self.fc3(x2, feat)
        # x = x + self.sl3(x2)
        # x = self.bn3(x.transpose(-1, -2)).transpose(-1, -2)
        # x1 = F.relu(x1)  # +self.res1(y)
        # x2 = F.relu(x2)

        # x = F.dropout(x, self.dropout, training=training)

        return F.sigmoid(x) #F.log_softmax(x, dim=1)



class SPResGCN_BN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(SPResGCN_BN, self).__init__()

        # self.gc1 = GraphConvolution(nfeat, nclass)
        self.gc1 = SPGraphConvolution(nfeat, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.sl1 = SelfLoop(nfeat, nhid)
        self.gc2 = SPGraphConvolution(nhid, nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.sl2 = SelfLoop(nhid, nhid)
        self.gc3 = SPGraphConvolution(nhid, nclass)
        self.sl3 = SelfLoop(nhid, nclass)
        self.dropout = dropout
        self.dropout = dropout

    def sub_gc(self, x, adj, gc_fn,sl):
        y = []
        #print("x ")
        for x_sub in x:
        #    print(" x_sub ")
            y_sub = gc_fn(x_sub, adj)+sl(x_sub)
            y.append(y_sub.unsqueeze(0))
        y = torch.cat(y, 0)
        return y

    def forward(self, x, adj, training=True):
        x = self.sub_gc(x, adj, self.gc1,self.sl1)

        x = self.bn1(x.transpose(-1,-2)).transpose(-1,-2)
        x = F.relu(x)

        x = F.dropout(x, self.dropout, training=training)
        y = x
        # x = self.gc2(x, adj)
        # x = x + y
        # x = self.bn2(x.transpose(-1,-2)).transpose(-1,-2)
        # x = F.relu(x)
        # x = self.gc2(x, adj)
        x = self.sub_gc(x, adj, self.gc2,self.sl2)
        x = self.bn2(x.transpose(-1, -2)).transpose(-1, -2)
        x = F.relu(x)
        x = x+y
        x = F.dropout(x, self.dropout, training=training)
        # x = self.gc3(x, adj)
        x = self.sub_gc(x, adj, self.gc3,self.sl3)
        # print("weight max,min,mean ",self.gc3.weight.data.max(), self.gc3.weight.data.min(), self.gc3.weight.data.mean())
        # print("grad max,min,mean ", self.gc3.weight.grad_fn.max(), self.gc3.weight.grad_fn.min(),
        #       self.gc3.weight.grad_fn.mean())

        return F.sigmoid(x) #F.log_softmax(x, dim=1)

class ResDeepGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(ResDeepGCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.gc5 = GraphConvolution(nhid, nhid)
        self.gc6 = GraphConvolution(nhid, nhid)
        self.gc7 = GraphConvolution(nhid, nhid)
        self.gc8 = GraphConvolution(nhid, nhid)
        self.gc9 = GraphConvolution(nhid, nhid)
        self.gc10 = GraphConvolution(nhid, nhid)
        self.gc11 = GraphConvolution(nhid, nhid)
        self.gc12 = GraphConvolution(nhid, nhid)
        self.gc13 = GraphConvolution(nhid, nhid)
        self.gc14 = GraphConvolution(nhid, nhid)
        self.gc15 = GraphConvolution(nhid, nhid)
        self.gc16 = GraphConvolution(nhid, nhid)
        self.gcend = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, training=True):
        #print("[1]")
        #print("x size",x.shape)
        x1 = F.relu(self.gc1(x, adj))
        #x = F.dropout(x, self.dropout, training=self.training)
        #print("[2]")
        x = F.relu(self.gc2(x1, adj)+x1)
        #x = F.dropout(x, self.dropout, training=self.training)
        #print("[3]")
        # y=x1+x
        x = F.relu(self.gc3(x, adj)+x)
        #x = F.dropout(x, self.dropout, training=self.training)
        # #print("[4]")
        # y=y+x
        x = F.relu(self.gc4(x, adj)+x)
        # #x = F.dropout(x, self.dropout, training=self.training)
        # y = y + x
        x = F.relu(self.gc5(x, adj)+x)
        # # x = F.dropout(x, self.dropout, training=self.training)
        # y = y + x
        x = F.relu(self.gc6(x, adj)+x)
        # # x = F.dropout(x, self.dropout, training=self.training)
        # y = y + x
        x = F.relu(self.gc7(x, adj)+x)
        # # x = F.dropout(x, self.dropout, training=self.training)
        # y = y + x
        x = F.relu(self.gc8(x, adj)+x)
        # # x = F.dropout(x, self.dropout, training=self.training)
        # y = y + x
        x= F.relu(self.gc9(x, adj)+x)
        # # x = F.dropout(x, self.dropout, training=self.training)
        # #print("[5]")
        x = F.relu(self.gc10(x, adj) + x)
        # #x = F.dropout(x, self.dropout, training=self.training)
        # y = y + x
        x = F.relu(self.gc11(x, adj) + x)
        # # x = F.dropout(x, self.dropout, training=self.training)
        # y = y + x
        x = F.relu(self.gc12(x, adj) + x)
        # # x = F.dropout(x, self.dropout, training=self.training)
        # y = y + x
        x = F.relu(self.gc13(x, adj) + x)
        # # x = F.dropout(x, self.dropout, training=self.training)
        # y = y + x
        x = F.relu(self.gc14(x, adj) + x)
        # # x = F.dropout(x, self.dropout, training=self.training)
        # y = y + x
        x = F.relu(self.gc15(x, adj) + x)
        # # x = F.dropout(x, self.dropout, training=self.training)
        # y = y + x
        x = F.relu(self.gc16(x, adj) + x)
        # # x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcend(x, adj)
        # return F.log_softmax(x, dim = 1)
        #print('gcn5 ', self.gc5.weight.data.mean())
        return F.sigmoid(x) #F.log_softmax(x, dim=1)

class ResDeepGCN_BN8(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(ResDeepGCN_BN8, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.bn3 = nn.BatchNorm1d(nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.bn4 = nn.BatchNorm1d(nhid)
        self.gc5 = GraphConvolution(nhid, nhid)
        self.bn5 = nn.BatchNorm1d(nhid)
        self.gc6 = GraphConvolution(nhid, nhid)
        self.bn6 = nn.BatchNorm1d(nhid)
        self.gc7 = GraphConvolution(nhid, nhid)
        self.bn7 = nn.BatchNorm1d(nhid)
        self.gcend = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, training=True):
        x1 = self.gc1(x, adj)
        x1 = self.bn1(x1.transpose(-1, -2)).transpose(-1, -2)
        x1 = F.relu(x1)

        x2 = self.gc2(x1, adj)
        x2 = self.bn2(x2.transpose(-1, -2)).transpose(-1, -2)
        x2 = F.relu(x2)+x1

        x3 = self.gc3(x2, adj)
        x3 = self.bn3(x3.transpose(-1, -2)).transpose(-1, -2)
        x3 = F.relu(x3)+x2

        x4 = self.gc4(x3, adj)
        x4 = self.bn4(x4.transpose(-1, -2)).transpose(-1, -2)
        x4 = F.relu(x4)+x3

        x5 = self.gc5(x4, adj)
        x5 = self.bn5(x5.transpose(-1, -2)).transpose(-1, -2)
        x5 = F.relu(x5)+x4

        x6 = self.gc6(x5, adj)
        x6 = self.bn6(x6.transpose(-1, -2)).transpose(-1, -2)
        x6 = F.relu(x6)+x5

        x7 = self.gc7(x6, adj)
        x7 = self.bn7(x7.transpose(-1, -2)).transpose(-1, -2)
        x7 = F.relu(x7)+x6

        y = self.gcend(x7, adj)
        # return F.log_softmax(x, dim = 1)
        #print('gcn5 ', self.gc5.weight.data.mean())
        return F.sigmoid(y) #F.log_softmax(x, dim=1)

class ResBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super(ResBlock, self).__init__()
        self.gc = GraphConvolution(n_in, n_out)
        self.bn = nn.BatchNorm1d(n_out)
    def forward(self, x, adj):
        y = self.gc(x, adj)
        y = self.bn(y.transpose(-1, -2)).transpose(-1, -2)
        y = F.relu(y)
        if y.shape == x.shape:
            y = y + x
        return y

class ResDeepGCN_BN16(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(ResDeepGCN_BN16, self).__init__()

        self.res1 = ResBlock(nfeat, nhid)
        self.res2 = ResBlock(nhid, nhid)
        self.res3 = ResBlock(nhid, nhid)
        self.res4 = ResBlock(nhid, nhid)
        self.res5 = ResBlock(nhid, nhid)
        self.res6 = ResBlock(nhid, nhid)
        self.res7 = ResBlock(nhid, nhid)
        self.res8 = ResBlock(nhid, nhid)
        self.res9 = ResBlock(nhid, nhid)
        self.res10 = ResBlock(nhid, nhid)
        self.res11 = ResBlock(nhid, nhid)
        self.res12 = ResBlock(nhid, nhid)
        self.res13 = ResBlock(nhid, nhid)
        self.res14 = ResBlock(nhid, nhid)
        self.res15 = ResBlock(nhid, nhid)

        self.gcend = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, training=True):
        # lab = x[:,:,-2]
        x1 = self.res1(x,adj)
        x2 = self.res2(x1,adj)
        x3 = self.res3(x2,adj)
        x4 = self.res4(x3,adj)
        x5 = self.res5(x4,adj)
        x6 = self.res6(x5,adj)
        x7 = self.res7(x6,adj)
        x8 = self.res8(x7,adj)
        x9 = self.res9(x8,adj)
        x10 = self.res10(x9,adj)
        x11 = self.res11(x10,adj)
        x12 = self.res12(x11,adj)
        x13 = self.res13(x12,adj)
        x14 = self.res14(x13,adj)
        x15 = self.res15(x14,adj)
        # z = []
        # for i in range(lab.shape[0]):
        #     z.append(x15[i][lab[i]==1])
        # assert 1<0, (len(z), z[0].shape, x15.shape, lab.shape)
        # x16 = x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15
        # x15 += x1
        y = self.gcend(x15, adj)
        # return F.log_softmax(x, dim = 1)
        #print('gcn5 ', self.gc5.weight.data.mean())
        return F.sigmoid(y) #F.log_softmax(x, dim=1)

class ResDeepGCN_BN16_ATN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(ResDeepGCN_BN16_ATN, self).__init__()

        self.res1 = ResBlock(nfeat, nhid)
        self.res2 = ResBlock(nhid, nhid)
        self.res3 = ResBlock(nhid, nhid)
        self.res4 = ResBlock(nhid, nhid)
        self.res5 = ResBlock(nhid, nhid)
        self.res6 = ResBlock(nhid, nhid)
        self.res7 = ResBlock(nhid, nhid)
        self.res8 = ResBlock(nhid, nhid)
        self.res9 = ResBlock(nhid, nhid)
        self.res10 = ResBlock(nhid, nhid)
        self.res11 = ResBlock(nhid, nhid)
        self.res12 = ResBlock(nhid, nhid)
        self.res13 = ResBlock(nhid, nhid)
        self.res14 = ResBlock(nhid, nhid)
        self.res15 = ResBlock(nhid, nhid)

        # self.gcend = GraphConvolution(nhid, nclass)
        self.gcend = nn.Conv1d(nhid+3, nclass, kernel_size=1)
        self.dropout = dropout

    def forward(self, x, adj, training=True):
        lab = x[:,:,-2]
        x1 = self.res1(x,adj)
        x2 = self.res2(x1,adj)
        x3 = self.res3(x2,adj)
        x4 = self.res4(x3,adj)
        x5 = self.res5(x4,adj)
        x6 = self.res6(x5,adj)
        x7 = self.res7(x6,adj)
        x8 = self.res8(x7,adj)
        x9 = self.res9(x8,adj)
        x10 = self.res10(x9,adj)
        x11 = self.res11(x10,adj)
        x12 = self.res12(x11,adj)
        x13 = self.res13(x12,adj)
        x14 = self.res14(x13,adj)
        x15 = self.res15(x14,adj)
        z = []
        for i in range(lab.shape[0]):
            z.append(x15[i][lab[i]>0.5].unsqueeze(0))
        z = torch.cat(z, dim=0)
        # assert 1<0, (x15.shape, z.shape) # BxNxF, Bx3xF
        atn = x15.matmul(z.transpose(-1,-2)) # BxNx3
        x15 = torch.cat((x15, atn), dim=2)
        y = self.gcend(x15.transpose(-1,-2)).transpose(-1,-2)
        # return F.log_softmax(x, dim = 1)
        #print('gcn5 ', self.gc5.weight.data.mean())
        return F.sigmoid(y) #F.log_softmax(x, dim=1)

class DenseDeepGCN_BN16(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(DenseDeepGCN_BN16, self).__init__()

        self.res1 = ResBlock(nfeat, nhid)
        self.res2 = ResBlock(nhid, nhid)
        self.res3 = ResBlock(nhid, nhid)
        self.res4 = ResBlock(nhid, nhid)
        self.res5 = ResBlock(nhid, nhid)
        self.res6 = ResBlock(nhid, nhid)
        self.res7 = ResBlock(nhid, nhid)
        self.res8 = ResBlock(nhid, nhid)
        self.res9 = ResBlock(nhid, nhid)
        self.res10 = ResBlock(nhid, nhid)
        self.res11 = ResBlock(nhid, nhid)
        self.res12 = ResBlock(nhid, nhid)
        self.res13 = ResBlock(nhid, nhid)
        self.res14 = ResBlock(nhid, nhid)
        self.res15 = ResBlock(nhid, nhid)

        self.gcend = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, training=True):
        x1 = self.res1(x,adj)
        x2 = self.res2(x1,adj)
        x3 = self.res3(x2,adj)
        x3 += x1
        x4 = self.res4(x3,adj)
        x4 += x1+x2
        x5 = self.res5(x4,adj)
        x5 += x1+x2+x3
        x6 = self.res6(x5,adj)
        x6 += x1+x2+x3+x4
        x7 = self.res7(x6,adj)
        x7 += x1+x2+x3+x4+x5
        x8 = self.res8(x7,adj)
        x8 += x1 + x2 + x3 + x4 + x5 + x6
        x9 = self.res9(x8,adj)
        x9 += x1 + x2 + x3 + x4 + x5 + x6 + x7
        x10 = self.res10(x9,adj)
        x10 += x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8
        x11 = self.res11(x10,adj)
        x11 += x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9
        x12 = self.res12(x11,adj)
        x12 += x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10
        x13 = self.res13(x12,adj)
        x13 += x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11
        x14 = self.res14(x13,adj)
        x14 += x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12
        x15 = self.res15(x14,adj)
        x15 += x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13
        # x16 = x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15

        y = self.gcend(x15, adj)
        # return F.log_softmax(x, dim = 1)
        #print('gcn5 ', self.gc5.weight.data.mean())
        return F.sigmoid(y) #F.log_softmax(x, dim=1)


class DenseDeepGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(DenseDeepGCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.gc5 = GraphConvolution(nhid, nhid)
        self.gc6 = GraphConvolution(nhid, nhid)
        self.gc7 = GraphConvolution(nhid, nhid)
        self.gc8 = GraphConvolution(nhid, nhid)
        self.gc9 = GraphConvolution(nhid, nhid)
        self.gcend = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, training=True):
        #print("[1]")
        #print("x size",x.shape)
        x1 = F.relu(self.gc1(x, adj))
        #x = F.dropout(x, self.dropout, training=self.training)
        #print("[2]")
        x = F.relu(self.gc2(x1, adj)+x1)
        #x = F.dropout(x, self.dropout, training=self.training)
        #print("[3]")
        y=x1+x
        x = F.relu(self.gc3(x, adj)+y)
        #x = F.dropout(x, self.dropout, training=self.training)
        # #print("[4]")
        y=y+x
        x = F.relu(self.gc4(x, adj)+y)
        # #x = F.dropout(x, self.dropout, training=self.training)
        y = y + x
        x = F.relu(self.gc5(x, adj)+y)
        # # x = F.dropout(x, self.dropout, training=self.training)
        y = y + x
        x = F.relu(self.gc6(x, adj)+y)
        # # x = F.dropout(x, self.dropout, training=self.training)
        y = y + x
        x = F.relu(self.gc7(x, adj)+y)
        # # x = F.dropout(x, self.dropout, training=self.training)
        y = y + x
        x = F.relu(self.gc8(x, adj)+y)
        # # x = F.dropout(x, self.dropout, training=self.training)
        y = y + x
        x= F.relu(self.gc9(x, adj)+y)
        # # x = F.dropout(x, self.dropout, training=self.training)
        # #print("[5]")
        x = self.gcend(x, adj)
        # return F.log_softmax(x, dim = 1)
        #print('gcn5 ', self.gc5.weight.data.mean())
        return F.sigmoid(x) #F.log_softmax(x, dim=1)


"""
class GCNAtt(nn.Module):
    def __init__(self, nfeat, nhid, nclass, final_class, dropout):
        super(GCNAtt, self).__init__()

        self.nhid = nhid
        self.nclass = nclass
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.att_layer = Attention(n_expert = nhid, n_hidden = nhid, v_hidden = nclass)
        self.fc = FC(nhid * nclass, final_class)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        att = self.att_layer(x)
        x = att.matmul(x)
        x = x.reshape(x.shape[0], self.nhid * self.nclass) # x = x.squeeze()
        x = self.fc(x)
        return F.log_softmax(x)
        #return F.sigmoid(x) #F.log_softmax(x, dim=1)
"""

class GCNAtt(nn.Module):
    def __init__(self, nfeat, n_hid1, n_hid2, n_expert, att_hid, final_class, dropout):
        super(GCNAtt, self).__init__()

        self.n_expert = n_expert
        self.n_hid2 = n_hid2
        self.gc1 = GraphConvolution(nfeat, n_hid1)
        self.gc2 = GraphConvolution(n_hid1, n_hid2)
        self.att_layer = Attention(n_expert = n_expert, n_hidden = att_hid, v_hidden = n_hid2)
        #self.fc = FC(n_expert * n_hid2, final_class)
        self.fc1 = FC(n_expert * n_hid2, att_hid)
        self.fc2 = FC(att_hid, final_class)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        att = self.att_layer(x)
        assert 1 < 0, (att.shape, x.shape)
        x = att.matmul(x)
        assert 1<0, x.shape
        x = x.reshape(x.shape[0], self.n_expert * self.n_hid2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # assert 1<0, x.shape

        return F.log_softmax(x, dim = 1)
        #return F.sigmoid(x) #F.log_softmax(x, dim=1)


class GCNAtt_my(nn.Module):
    def __init__(self, nfeat, n_hid1, n_hid2, final_class, dropout):
        super(GCNAtt_my, self).__init__()

        # self.conv1 = nn.Conv1d(nfeat, n_hid1, 1)
        # self.conv2 = nn.Conv1d(n_hid1, n_hid2, 1)
        # self.atn_k = nn.Conv1d(n_hid2, n_hid2//8, 1)
        # self.atn_v = nn.Conv1d(n_hid2, n_hid2//2, 1)
        # self.conv3 = nn.Conv1d(n_hid2, n_hid2, 1)
        # self.pred = nn.Conv1d(n_hid2, final_class, 1)
        # self.sigmoid = nn.Sigmoid()
        # self.dropout = dropout
        self.gc1 = GraphConvolution(nfeat, n_hid1)
        self.gc2 = GraphConvolution(n_hid1, n_hid2)
        self.gc3 = GraphConvolution(n_hid2, final_class)
        self.atn_k = GraphConvolution(n_hid2, n_hid2 // 8)
        self.atn_v = GraphConvolution(n_hid2, n_hid2 // 2)
        self.atn_sk = GraphConvolution(n_hid2, n_hid2 // 8)
        self.atn_sv = GraphConvolution(n_hid2, n_hid2 // 2)
        # self.atn_k = nn.Conv1d(n_hid2, n_hid2//8, 1)
        # self.atn_v = nn.Conv1d(n_hid2, n_hid2//2, 1)
        self.gc1.weight.data.normal_(0, 0.001)
        self.gc1.bias.data.zero_()
        self.gc2.weight.data.normal_(0, 0.001)
        self.gc2.bias.data.zero_()
        self.gc3.weight.data.normal_(0, 0.001)
        self.gc3.bias.data.zero_()

        self.atn_k.weight.data.normal_(0, 0.001)
        self.atn_k.bias.data.zero_()
        self.atn_v.weight.data.normal_(0, 0.001)
        self.atn_v.bias.data.zero_()

        self.atn_sk.weight.data.normal_(0, 0.001)
        self.atn_sk.bias.data.zero_()
        self.atn_sv.weight.data.normal_(0, 0.001)
        self.atn_sv.bias.data.zero_()

        self.dropout = dropout


    def forward(self, x, adj,training=True):
        # x = self.conv1(x)
        # x = torch.bmm(adj, x.transpose(1,2)).transpose(1,2)
        # x = F.relu(x)
        # x = self.conv2(x)
        # x = torch.bmm(adj, x.transpose(1,2)).transpose(1,2)
        # x = F.relu(x)
        # atn_k = self.atn_k(x)
        # atn_v = self.atn_v(x)
        # atn_mat = torch.bmm(atn_k.transpose(1,2), atn_k)
        # atn_mat = F.softmax(atn_mat, 2)
        # agg_v = torch.bmm(atn_mat, atn_v.transpose(1,2)).transpose(1,2)
        # x = torch.cat((agg_v, atn_v), dim=1)
        # # x = self.conv3(x)
        # # x = torch.bmm(adj, x.transpose(1,2)).transpose(1,2)
        # # x = F.relu(x)
        # x = self.pred(x)
        # x = self.sigmoid(x)
        #
        # # print('conv1', self.conv1.weight.data.mean())
        #
        # return x
        x = F.relu(self.gc1(x, adj) )
        x = F.dropout(x, self.dropout, training=training)
        x = F.relu(self.gc2(x, adj) )
        x = F.dropout(x, self.dropout, training=training)

        self_adj=torch.zeros((adj.shape)).float()
        for j in range(adj.shape[0]):
            for i in range(adj.shape[1]):
                self_adj[j, i, i] = 1
        atn_k = self.atn_k(x,adj)#+self.atn_sk(x,self_adj)
        atn_v = self.atn_v(x,adj)#+self.atn_sv(x,self_adj)
        atn_k = atn_k.transpose(1, 2)
        atn_v = atn_v.transpose(1, 2)
        atn_mat = torch.bmm(atn_k.transpose(1,2), atn_k)
        atn_mat = F.softmax(atn_mat, 2)
        agg_v = torch.bmm(atn_mat, atn_v.transpose(1,2)).transpose(1,2)
        x = torch.cat((agg_v, atn_v), dim=1)
        x = x.transpose(1, 2)
        x = F.dropout(x, self.dropout, training=training)


        x = self.gc3(x, adj)
        return F.sigmoid(x)


class SPGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(SPGCN, self).__init__()

        # self.gc1 = GraphConvolution(nfeat, nclass)
        self.gc1 = SPGraphConvolution(nfeat, nhid)
        self.gc2 = SPGraphConvolution(nhid, nhid)
        self.gc3 = SPGraphConvolution(nhid, nclass)
        self.dropout = dropout
        # self.gc1.weight.data.normal_(0, 0.001)
        # self.gc1.bias.data.zero_()
        # self.gc2.weight.data.normal_(0, 0.001)
        # self.gc2.bias.data.zero_()
        # self.gc3.weight.data.normal_(0, 0.001)
        # self.gc3.bias.data.zero_()
        # self.dropout = dropout

    def forward(self, x, adj, training=True):
        x_list = []
        # print('1', x.shape)
        for i in range(x.shape[0]):
            x_list.append( F.relu(self.gc1(x[i], adj)+x[i]).unsqueeze(0) )
        # print('x_list', x_list[0].shape)
        x = torch.cat(x_list, 0)

        x = F.dropout(x, self.dropout, training=training)

        x_list = []
        # print('2', x.shape)
        for i in range(x.shape[0]):
            x_list.append(F.relu(self.gc2(x[i], adj) + x[i]).unsqueeze(0))
        x = torch.cat(x_list, 0)

        x_list = []
        for i in range(x.shape[0]):
            x_list.append(F.relu(self.gc3(x[i], adj)).unsqueeze(0))
        x = torch.cat(x_list, 0)
        # print('x', x.shape)
        return F.sigmoid(x)


    def forward_no_batch_size(self, x, adj, training=True):
        # print("SPGCN")
        # print('input', x.shape, adj.shape)

        x = F.relu(self.gc1(x, adj) + x)
        # print("SPGCN1")
        x = F.dropout(x, self.dropout, training=training)
        # print("SPGCN2")

        x = F.relu(self.gc2(x, adj)  + x)
        # print("SPGCN3")
        #
        # x = F.dropout(x, self.dropout, training=training)
        # # print("SPGCN4")

        x = self.gc3(x, adj)
        # x = F.relu(self.gc1(x, adj) )
        # x = F.dropout(x, self.dropout, training=training)
        # x = F.relu(self.gc2(x, adj) )
        # x = F.dropout(x, self.dropout, training=training)
        # x = self.gc3(x, adj)
        return F.sigmoid(x) #F.log_softmax(x, dim=1)


class SPGCN_Batch(SPGCN):
    def forward(self, x, adj, training=True):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=training)

        x = F.relu(self.gc2(x, adj))
        # x = F.dropout(x, self.dropout, training=training)

        x = self.gc3(x, adj)
        # x = F.relu(self.gc1(x, adj) )
        # x = F.dropout(x, self.dropout, training=training)
        # x = F.relu(self.gc2(x, adj) )
        # x = F.dropout(x, self.dropout, training=training)
        # x = self.gc3(x, adj)
        return F.sigmoid(x) #F.log_softmax(x, dim=1)
