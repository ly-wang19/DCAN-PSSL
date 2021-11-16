import torch as th
from torch import nn

from srs.layers.seframe import SEFrame
from srs.layers.gat import GatLayer
from srs.utils.data.collate import CollateFnRNNCNN,CollateFnGNN1
from srs.utils.data.load import BatchSampler
from srs.utils.Dict import Dict
from srs.utils.prepare_batch import prepare_batch_factory_recursive
from srs.utils.data.transform import seq_to_weighted_graph
import torch.nn.init as init
import torch.nn.functional as F
import copy

class Attention(nn.Module):
    def __init__(self, hidden_dim, session_len):
        super(Attention, self).__init__()
        self.attn_w0 = nn.Parameter(th.Tensor(session_len, hidden_dim))
        self.attn_w1 = nn.Parameter(th.Tensor(hidden_dim, hidden_dim))
        self.attn_w2 = nn.Parameter(th.Tensor(hidden_dim, hidden_dim))
        self.attn_bias = nn.Parameter(th.Tensor(hidden_dim))
        self.initial_()

    def initial_(self):
        init.normal_(self.attn_w0, 0, 0.05)
        init.normal_(self.attn_w1, 0, 0.05)
        init.normal_(self.attn_w2, 0, 0.05)
        init.constant_(self.attn_bias, 0)

    def forward(self, q, k, v, mask=None, dropout=None):
        alpha = th.matmul(
            th.relu(k.matmul(self.attn_w1) + q.matmul(self.attn_w2) + self.attn_bias),
            self.attn_w0.t(),
        )  # (B,seq,1)

        if mask is not None:
            alpha = alpha.masked_fill(mask == 0, -1e9)
        alpha = F.softmax(alpha, dim=-2)
        if dropout is not None:
            alpha = dropout(alpha)
        re = th.matmul(alpha.transpose(-1, -2), v)  # (B, 1, dim)
        return re


class MultiHeadedAttention(nn.Module):
    def __init__(self, hidden_dim, session_len, num_heads=1, dropout=0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert hidden_dim % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = hidden_dim // num_heads
        self.num_heads = num_heads

        self.attn = Attention(self.d_k, session_len)
        self.linears = self.clones(nn.Linear(hidden_dim, hidden_dim), 4)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def clones(self, module, N):
        "Produce N identical layers."
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    def forward(self, query, key, value, mask=None, linear=False):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        if linear is True:
            query, key, value = [lin(x).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2) for lin, x in zip(self.linears, (query, key, value))]
        else:
            query, key, value = [x.view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2) for x in (query, key, value)]

        # 2) Apply attention on all the projected vectors in batch.
        x = self.attn(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)
        return self.linears[-1](x) if linear is True else x



class SNARM(SEFrame):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim,
        knowledge_graph,
        num_layers,
        relu=False,
        batch_norm=True,
        feat_drop=0.0,
        **kwargs
    ):
        super().__init__(
            num_users,
            num_items,
            embedding_dim,
            knowledge_graph,
            num_layers,
            batch_norm=batch_norm,
            feat_drop=feat_drop,
            **kwargs,
        )
        self.pad_embedding = nn.Embedding(1, embedding_dim, max_norm=1)
        self.pad_indices = nn.Parameter(th.arange(1, dtype=th.long), requires_grad=False)
        self.pos_embedding = nn.Embedding(50, embedding_dim, max_norm=1)

        self.fc_i = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.fc_u = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.PSE_layer = SERecLayer(
            embedding_dim,
            num_steps=1,
            batch_norm=batch_norm,
            feat_drop=feat_drop,
            relu=relu,
        )
        self.fc_sr = nn.Linear(5 * embedding_dim, embedding_dim, bias=False)
        self.fc_sr1 = nn.Linear(7 * embedding_dim, embedding_dim, bias=False)

        self.mul = MultiHeadedAttention(2*embedding_dim, 50, 4)
        self.mul1 = MultiHeadedAttention(2*embedding_dim, 1, 4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs, extra_inputs=None):
        KG_embeddings = super().forward(extra_inputs)
        # print(self.pad_indices)
        KG_embeddings["item"] = th.cat([KG_embeddings["item"], self.pad_embedding(self.pad_indices)], dim=0)
        # print(KG_embeddings["item"].size())
        # pad_indice = KG_embeddings["item"].size(0) - 1
        # print(pad_indice)
        
        uids, padded_seqs, pos, g = inputs

        # pad_mask = (padded_seqs != -1) # B,seq
        # padded_seqs.masked_fill_(pad_mask == 0, pad_indice)
        # print(padded_seqs)
        # # print(padded_seqs.shape)
        # feat_u = KG_embeddings["user"][uids]
        # print(padded_seqs.shape)
        # print(pos.shape)

        emb_seqs = KG_embeddings["item"][padded_seqs]
        pos_emb = self.pos_embedding(pos)
        # emb_seqs = emb_seqs + feat_u.unsqueeze(1)
        feat = th.cat(
            [emb_seqs, pos_emb.unsqueeze(0).expand(emb_seqs.shape)], dim=-1
        )
        query = feat
        if self.dropout is not None:
            query = self.dropout(query)
        attn_output= self.mul(query, feat, feat)
        self.dropout(attn_output)
        # print(attn_output.shape)
        query1 = attn_output[:,-1,:]
        # if self.dropout is not None:
        #     query1 = self.dropout(query1)

        attn_output1 = self.mul1(query1, feat[:,:-1,:],feat[:,:-1,:])
        # print(attn_output1.shape)
        # self.dropout(attn_output1)
        # sr = th.cat([feat[:,-1,:], feat_u, attn_output[:,-1,:]], dim=1)
        
        # print(sr.shape)
        # logits = self.fc_sr(sr) @ self.item_embedding(self.item_indices).t()

        # graph 
        iid = g.ndata['iid']  # (num_nodes,)
        feat_i = KG_embeddings['item'][iid]
        feat_u = KG_embeddings['user'][uids]

        ct_l, ct_g = self.PSE_layer(g, feat_i, feat_u)
        # sr = th.cat([ct_l.unsqueeze(1), ct_g.unsqueeze(1),feat_u.unsqueeze(1)], dim=1)
        # sr = self.block(sr).reshape(sr.shape[0],sr.shape[1] * sr.shape[2])
        sr = th.cat([ct_l, ct_g, feat_u, attn_output1[:,-1,:],attn_output[:,-1,:]], dim=1)
        # print(sr.shape)
        logits = self.fc_sr1(sr) @ self.item_embedding(self.item_indices).t()
        return logits

seq_to_graph_fns = [seq_to_weighted_graph]
config = Dict({
    'Model': SNARM,
    'CollateFn': CollateFnGNN1,
    'seq_to_graph_fns': seq_to_graph_fns,
    'BatchSampler': BatchSampler,
    'prepare_batch_factory': prepare_batch_factory_recursive,
})


