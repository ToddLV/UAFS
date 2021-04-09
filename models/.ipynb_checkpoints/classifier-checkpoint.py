import math

import torch
import torch.nn as nn

import models
import utils
from .models import register


@register('classifier')
class Classifier(nn.Module):
    
    def __init__(self, encoder, encoder_args,
                 classifier, classifier_args):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        classifier_args['in_dim'] = self.encoder.out_dim
        self.classifier = models.make(classifier, **classifier_args)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


@register('linear-classifier')
class LinearClassifier(nn.Module):

    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.linear(x)

@register('uafs-classifier')
class UAFSClassifier(nn.Module):

    def __init__(self, in_dim, n_classes, metric='cos', temp=None):
        super().__init__()
        self.proto = nn.Parameter(torch.empty(n_classes, in_dim))
        nn.init.kaiming_uniform_(self.proto, a=math.sqrt(5))
        if temp is None:
            if metric == 'cos':
                temp = nn.Parameter(torch.tensor(10.))
            else:
                temp = 1.0
        self.metric = metric
        self.temp = temp

    def forward(self, x):
        x_shot = self.proto.chunk(32, dim=-1)
        x_query = x.chunk(32, dim=-1)
        out_logits = []
        for idx, ci in enumerate(x_shot):
            zi = x_query[idx]
            logits = utils.compute_logits(zi, ci, self.metric, self.temp)
            out_logits.append(logits)
        out_logits=torch.stack(out_logits)
        out_logits = torch.squeeze(out_logits)
        out_logits = out_logits.permute(1,2,0)
        out_logits = out_logits.chunk(out_logits.shape[0], dim=0)
        return [out_logits, self.proto, x]
#         return utils.compute_logits(x, self.proto, self.metric, self.temp)
    
# @register('nn-classifier')
# class NNClassifier(nn.Module):

#     def __init__(self, in_dim, n_classes, metric='cos', temp=None):
#         super().__init__()
#         self.proto = nn.Parameter(torch.empty(n_classes, in_dim))
#         nn.init.kaiming_uniform_(self.proto, a=math.sqrt(5))
#         if temp is None:
#             if metric == 'cos':
#                 temp = nn.Parameter(torch.tensor(10.))
#             else:
#                 temp = 1.0
#         self.metric = metric
#         self.temp = temp

#     def forward(self, x):
#         return utils.compute_logits(x, self.proto, self.metric, self.temp)

