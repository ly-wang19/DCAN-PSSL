import torch as th
from torch import nn

import dgl
import dgl.ops as F


class PAttentionReadout(nn.Module):
    def __init__(self, embedding_dim, batch_norm=False, feat_drop=0.0, activation=None):
        super().__init__()
        if batch_norm:
            self.batch_norm = nn.ModuleDict({
                'user': nn.BatchNorm1d(embedding_dim),
                'item': nn.BatchNorm1d(embedding_dim)
            })
        else:
            self.batch_norm = None
        self.feat_drop = nn.Dropout(feat_drop) if feat_drop > 0 else None
        self.fc_user = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.fc_key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.fc_last = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.fc_e = nn.Linear(embedding_dim, 1, bias=False)
        self.activation = activation

    def forward(self, g, feat_i, feat_u):
        if self.batch_norm is not None:
            feat_i = self.batch_norm['item'](feat_i)
            feat_u = self.batch_norm['user'](feat_u)
        if self.feat_drop is not None:
            feat_i = self.feat_drop(feat_i)
            feat_u = self.feat_drop(feat_u)
        feat_val = feat_i
        feat_key = self.fc_key(feat_i)
        feat_u = self.fc_user(feat_u)
        feat_qry = dgl.broadcast_nodes(g, feat_u)

        e = self.fc_e(th.sigmoid(feat_qry + feat_key))  # (num_nodes, 1)
        alpha = F.segment.segment_softmax(g.batch_num_nodes(), e)
        rst = F.segment.segment_reduce(g.batch_num_nodes(), alpha * feat_val, 'sum')
        if self.activation is not None:
            rst = self.activation(rst)
        return rst


class GatLayer(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_steps=1,
        batch_norm=True,
        feat_drop=0.0,
        relu=False,
    ):
        super().__init__()
        self.fc_i = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.fc_u = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.readout = PAttentionReadout(
            embedding_dim,
            batch_norm=batch_norm,
            feat_drop=feat_drop,
            activation=nn.ReLU() if relu else nn.PReLU(embedding_dim),
        )

    def forward(self, g, feat, feat_u):

        last_nodes = g.filter_nodes(lambda nodes: nodes.data['last'] == 1)
        ct_l = feat[last_nodes]
        ct_g = self.readout(g, feat, feat_u)
        # ct_g1 = self.readout(g, feat, ct_l)
        # print(ct_g1.shape)
        # print(ct_g.shape)
        # sr = th.cat((ct_l, ct_g), dim=1)
        return ct_l, ct_g
