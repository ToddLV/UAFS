import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchsnooper
import models
import utils
from .models import register


@register('meta-baseline')
class MetaBaseline(nn.Module):

    def __init__(self, encoder, encoder_args={}, method='cos',
                 temp=1., temp_learnable=True):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.method = method

        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp
#     @torchsnooper.snoop()
    def forward(self, x_shot, x_query):
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        x_shot = x_shot.view(*shot_shape, -1)
        x_query = x_query.view(*query_shape, -1)
        x_shotL = x_shot.chunk(32, dim=-1)
        x_queryL = x_query.chunk(32, dim=-1)
        out_logits = []
        for idx, ci in enumerate(x_shotL):
            zi = x_queryL[idx]
            if self.method == 'cos':
                ci = ci.mean(dim=-2)
                ci = F.normalize(ci, dim=-1)
                zi = F.normalize(zi, dim=-1)
                metric = 'dot'
            elif self.method == 'sqr':
                x_shot = ci.mean(dim=-2)
                metric = 'sqr'
            logits = utils.compute_logits(zi, ci, metric=metric, temp=self.temp)
            out_logits.append(logits)
        out_logits=torch.stack(out_logits)
        out_logits = torch.squeeze(out_logits)
        out_logits = out_logits.permute(1,2,0)
        out_logits = out_logits.chunk(out_logits.shape[0], dim=0)
        return [out_logits, x_shot, x_query]
#         for idx, ci in enumerate(x_shotL):
#             zi = x_queryL[idx]
#             zi = zi.chunk(75,dim=0)
#             logits_list = []
#             for i in zi:
#                 logits = utils.get_att_dis(i,ci)
#                 logits_list.append(logits)
#             logits_list=torch.stack(logits_list)
#             out_logits.append(logits_list)
#         out_logits = torch.stack(out_logits)
#         out_logits = out_logits.permute(1,2,0)
#         out_logits = out_logits.chunk(out_logits.shape[0], dim=0)
#         print('logits',out_logits)
#         return [out_logits, x_shot, x_query]

#         if self.method == 'cos':
#             x_shot = x_shot.mean(dim=-2)
#             x_shot = F.normalize(x_shot, dim=-1)
#             x_query = F.normalize(x_query, dim=-1)
#             metric = 'dot'
#         elif self.method == 'sqr':
#             x_shot = x_shot.mean(dim=-2)
#             metric = 'sqr'

#         logits = utils.compute_logits(
#             x_query, x_shot, metric=metric, temp=self.temp)

#         return logits

