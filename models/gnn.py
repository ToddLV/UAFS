import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math
from .models import register
import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(nn.Module):


    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bn = nn.BatchNorm2d(out_features)
        self.relu = nn.LeakyReLU(0.1)
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.mm(input, self.weight)
        output = output.view(1, -1,1,5)
#         output = output.view(1, -1,1,64)
        output = self.bn(output)
        output = torch.squeeze(output)
        output = output.t()
        output = self.relu(output)
        
        output = torch.mm(output, self.weight)
        output = output.view(1, -1,1,5)
#         output = output.view(1, -1,1,64)
        output = self.bn(output)
        output = torch.squeeze(output)
        output = output.t()
        output = self.relu(output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Adjacency(nn.Module):
    def __init__(self, input_dim, hidden_dim):

        super(Adjacency, self).__init__()
#         self.weight = nn.Linear(input_dim, hidden_dim)
        self.weight = Parameter(torch.FloatTensor(input_dim, hidden_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, x):
#         x_i = self.weight(x)
#         x_j = self.weight(x).t()
        x_i = torch.mm(x, self.weight)
        x_j = torch.mm(x, self.weight).t()
        A = torch.mm(x_i,x_j)

        A = F.softmax(A, dim=1)  # normalize
        return A  # (b, N, N)



class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(GCN, self).__init__()
        
        self.relu = nn.LeakyReLU(0.1)
        self.gc = GraphConvolution(nfeat, nhid)
        self.bn = nn.BatchNorm2d(nhid)
        self.weight1 = nn.Linear(nfeat, nhid)
        self.weight2 = nn.Linear(nhid, nclass)
        
    def forward(self, x, adj):
        output = self.weight1(x)
        output = torch.mm(adj, output)
       
        output = self.gc(output)
        output = x + output
        
        output = self.weight1(output)
        output = output.view(1, -1,1,5)
#         output = output.view(1, -1,1,64)
        output = self.bn(output)
        output = torch.squeeze(output)
        output = output.t()
        
        output = self.weight2(output)
        output = self.relu(output)

#         return F.softmax(x, dim=1)
        return output


class Normal(nn.Module):
    def __init__(self, x_shot, x_query):
        super(Normal, self).__init__()
        self.x_shot = x_shot
        self.x_query = x_query
        self.T = 10
    def forward(self, x_shot, x_query, logits):
        x_shot = x_shot.mean(dim=-2)
        x_shot = F.normalize(x_shot, dim=-1)
#         print('x_shot.shape',x_shot.shape)
        x_query = F.normalize(x_query, dim=-1)
#         print('x_query.shape',x_query.shape)
        mean = torch.bmm(x_query, x_shot.permute(0, 2, 1))
        mean = mean * 10
        logits_list = []
        std = logits.permute(1,0,2)
        axis = np.random.normal(loc=0, scale=1.0, size=self.T)
        for a in axis:
            logits = mean + (std * a)
            logits_list.append(logits)


        return logits_list

class Normalclassifier(nn.Module):
    def __init__(self, x_shot, x_query):
        super(Normalclassifier, self).__init__()
        self.x_shot = x_shot
        self.x_query = x_query
        self.T = 10
    def forward(self, x_shot, x_query, logits):
        x_shot = F.normalize(x_shot, dim=-1)
        x_query = F.normalize(x_query, dim=-1)
        mean = torch.mm(x_query, x_shot.t())
        mean = mean * 10
        logits_list = []
        std = logits
        axis = np.random.normal(loc=0, scale=1.0, size=self.T)
        for a in axis:
            logits = mean + (std * a)
            logits_list.append(logits)


        return logits_list



