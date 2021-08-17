import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, FC, Attention, SPGraphConvolution,SelfLoop


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

        # return model,torch.sigmoid(model)
        return torch.sigmoid(model) ,model # F.log_softmax(x, dim=1)



class CS_GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(CS_GCN, self).__init__()

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
        model1=self.graph_encoder1(feat,adj)+ self.self_GE1(feat)
        model2 = self.structure_encoder1(node_input, adj) + self.self_SE1(node_input)
        # model3 = self.attribute_encoder1(att_input, Fadj)
        
        # x = x1 + x2 +x3 # +self.sl1(x2)
        model= torch.cat([model1,model2],2)
        # model=torch.cat([model,model3],2)
        model=self.cnd1(model)

        model = self.bn1(model.transpose(-1, -2)).transpose(-1, -2)
        model = F.relu(model)
        model = F.dropout(model, self.dropout, training=training)

        ##############################################################################################################

        model1=self.bn_GE1(model1.transpose(-1, -2)).transpose(-1, -2)
        model1 = F.relu(model1)
        model1 = F.dropout(model1, self.dropout, training=training)
        model1=self.graph_encoder2(model1,adj)+ self.self_GE2(model1)

        model_SE = self.bn_SE1(model2.transpose(-1, -2)).transpose(-1, -2)
        model_SE = F.relu(model_SE)
        model_SE = F.dropout(model_SE, self.dropout, training=training)
        model2 = self.structure_encoder2(model, adj) + self.self_SE2(model_SE)
        # model2 = self.structure_encoder2(model, adj) + self.self_SE2(model)

        # model3 = Fadj.transpose(-1, -2).matmul(model) + self.self_AE1(att_input)
        # model_AE = self.bn_AE1(model3.transpose(-1, -2)).transpose(-1, -2)
        # model_AE = F.relu(model_AE)
        # model_AE = F.dropout(model_AE, self.dropout, training=training)
        # model3 = self.attribute_encoder2(model3, Fadj)
        # model3 = self.attribute_encoder2(model_AE, Fadj)


        model= torch.cat([model1,model2],2)
        # model=torch.cat([model,model3],2)
        model=self.cnd2(model)

        model = self.bn2(model.transpose(-1, -2)).transpose(-1, -2)
        model = F.relu(model)
        model = F.dropout(model, self.dropout, training=training)

        ################################################################################################################

        model1=self.bn_GE2(model1.transpose(-1, -2)).transpose(-1, -2)
        model1 = F.relu(model1)
        model1 = F.dropout(model1, self.dropout, training=training)
        model1=self.graph_encoder3(model1,adj)+ self.self_GE3(model1)

        model_SE = self.bn_SE2(model2.transpose(-1, -2)).transpose(-1, -2)
        model_SE = F.relu(model_SE)
        model_SE = F.dropout(model_SE, self.dropout, training=training)
        model2 = self.structure_encoder3(model, adj) + self.self_SE3(model_SE)
        # model2 = self.structure_encoder3(model, adj) + self.self_SE3(model)

        # model3 = Fadj.transpose(-1, -2).matmul(model) + self.self_AE2(model_AE)
        # model3 = self.bn_AE2(model3.transpose(-1, -2)).transpose(-1, -2)
        # model3 = F.relu(model3)
        # model3 = F.dropout(model3, self.dropout, training=training)
        # model3 = self.attribute_encoder3(model3, Fadj)


        model= torch.cat([model1,model2],2)
        # model=torch.cat([model,model3],2)
        model=self.cnd3(model)

        # model = self.bn1(model.transpose(-1, -2)).transpose(-1, -2)
        # model = F.relu(model)
        # model = F.dropout(model, self.dropout, training=training)

        # return model,torch.sigmoid(model)
        return torch.sigmoid(model) ,model # F.log_softmax(x, dim=1)


class CS_GCN_NoF(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(CS_GCN_NoF, self).__init__()

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
        model1=self.graph_encoder1(feat,adj)+ self.self_GE1(feat)
        model2 = self.structure_encoder1(node_input, adj) + self.self_SE1(node_input)
        # model3 = self.attribute_encoder1(att_input, Fadj)
        
        # x = x1 + x2 +x3 # +self.sl1(x2)
        # model= torch.cat([model1,model2],2)
        # model=torch.cat([model,model3],2)
        # model=self.cnd1(model)

        model2 = self.bn1(model2.transpose(-1, -2)).transpose(-1, -2)
        model2 = F.relu(model2)
        model2 = F.dropout(model2, self.dropout, training=training)

        ##############################################################################################################

        model1=self.bn_GE1(model1.transpose(-1, -2)).transpose(-1, -2)
        model1 = F.relu(model1)
        model1 = F.dropout(model1, self.dropout, training=training)
        model1=self.graph_encoder2(model1,adj)+ self.self_GE2(model1)

        # model_SE = self.bn_SE1(model2.transpose(-1, -2)).transpose(-1, -2)
        # model_SE = F.relu(model_SE)
        # model_SE = F.dropout(model_SE, self.dropout, training=training)
        # model2 = self.structure_encoder2(model, adj) + self.self_SE2(model_SE)
        model2 = self.structure_encoder2(model2, adj) + self.self_SE2(model2)

        # model3 = Fadj.transpose(-1, -2).matmul(model) + self.self_AE1(att_input)
        # model_AE = self.bn_AE1(model3.transpose(-1, -2)).transpose(-1, -2)
        # model_AE = F.relu(model_AE)
        # model_AE = F.dropout(model_AE, self.dropout, training=training)
        # model3 = self.attribute_encoder2(model3, Fadj)
        # model3 = self.attribute_encoder2(model_AE, Fadj)


        # model= torch.cat([model1,model2],2)
        # model=torch.cat([model,model3],2)
        # model=self.cnd2(model)

        model2 = self.bn2(model2.transpose(-1, -2)).transpose(-1, -2)
        model2 = F.relu(model2)
        model2 = F.dropout(model2, self.dropout, training=training)

        ################################################################################################################

        model1=self.bn_GE2(model1.transpose(-1, -2)).transpose(-1, -2)
        model1 = F.relu(model1)
        model1 = F.dropout(model1, self.dropout, training=training)
        model1=self.graph_encoder3(model1,adj)+ self.self_GE3(model1)

        # model_SE = self.bn_SE2(model2.transpose(-1, -2)).transpose(-1, -2)
        # model_SE = F.relu(model_SE)
        # model_SE = F.dropout(model_SE, self.dropout, training=training)
        # model2 = self.structure_encoder3(model, adj) + self.self_SE3(model_SE)
        model2 = self.structure_encoder3(model2, adj) + self.self_SE3(model2)

        # model3 = Fadj.transpose(-1, -2).matmul(model) + self.self_AE2(model_AE)
        # model3 = self.bn_AE2(model3.transpose(-1, -2)).transpose(-1, -2)
        # model3 = F.relu(model3)
        # model3 = F.dropout(model3, self.dropout, training=training)
        # model3 = self.attribute_encoder3(model3, Fadj)


        model= torch.cat([model1,model2],2)
        # model=torch.cat([model,model3],2)
        model=self.cnd3(model)

        # model = self.bn1(model.transpose(-1, -2)).transpose(-1, -2)
        # model = F.relu(model)
        # model = F.dropout(model, self.dropout, training=training)

        # return model,torch.sigmoid(model)
        return torch.sigmoid(model) ,model # F.log_softmax(x, dim=1)



class SimpleCS_GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(SimpleCS_GCN, self).__init__()

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
        # model1=self.graph_encoder1(feat,adj)+ self.self_GE1(feat)
        model2 = self.structure_encoder1(node_input, adj) + self.self_SE1(node_input)

        ##############################################################################################################

        model2=self.bn_GE1(model2.transpose(-1, -2)).transpose(-1, -2)
        model2 = F.relu(model2)
        model2 = F.dropout(model2, self.dropout, training=training)
        model2 = self.structure_encoder2(model2, adj) + self.self_SE2(model2)

        ################################################################################################################

        model2=self.bn_GE2(model2.transpose(-1, -2)).transpose(-1, -2)
        model2 = F.relu(model2)
        model2 = F.dropout(model2, self.dropout, training=training)
        model2 = self.structure_encoder3(model2, adj) + self.self_SE3(model2)


        return torch.sigmoid(model2) ,model2 # F.log_softmax(x, dim=1)

