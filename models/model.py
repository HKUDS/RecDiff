import dgl, math, torch
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
import math
from torch.nn.init import xavier_normal_, constant_, xavier_uniform_




class UUGCNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 weight=False,
                 bias=False,
                 activation=None):
        super(UUGCNLayer, self).__init__()
        self.bias = bias
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight = weight
        if self.weight:
            self.u_w = nn.Parameter(torch.Tensor(in_feats, out_feats))
            xavier_uniform_(self.u_w)
        self._activation = activation

    # def forward(self, graph, feat):
    def forward(self, graph, u_f):
        with graph.local_scope():
            if self.weight:
                u_f = torch.mm(u_f, self.u_w)
            node_f = u_f
            # D^-1/2
            # degs = graph.out_degrees().to(feat.device).float().clamp(min=1)
            degs = graph.out_degrees().to(u_f.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)
            # norm = norm.view(-1,1)
            # shp = norm.shape + (1,) * (feat.dim() - 1)
            # norm = t.reshape(norm, shp)

            node_f = node_f * norm

            graph.ndata['n_f'] = node_f
            # graph.edata['e_f'] = e_f
            graph.update_all(fn.copy_u(u='n_f', out='m'), reduce_func=fn.sum(msg='m', out='n_f'))

            rst = graph.ndata['n_f']

            degs = graph.in_degrees().to(u_f.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)
            # shp = norm.shape + (1,) * (feat.dim() - 1)
            # norm = t.reshape(norm, shp)
            rst = rst * norm

            if self._activation is not None:
                rst = self._activation(rst)

            return rst
class GCNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 weight=False,
                 bias=False,
                 activation=None):
        super(GCNLayer, self).__init__()
        self.bias = bias
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight = weight
        if self.weight:
            self.u_w = nn.Parameter(torch.Tensor(in_feats, out_feats))
            self.v_w = nn.Parameter(torch.Tensor(in_feats, out_feats))
            # self.e_w = nn.Parameter(t.Tensor(in_feats, out_feats))
            xavier_uniform_(self.u_w)
            xavier_uniform_(self.v_w)
            # init.xavier_uniform_(self.e_w)
        self._activation = activation

    # def forward(self, graph, feat):
    def forward(self, graph, u_f, v_f):
        with graph.local_scope():
            if self.weight:
                u_f = torch.mm(u_f, self.u_w)
                v_f = torch.mm(v_f, self.v_w)
                # e_f = t.mm(e_f, self.e_w)
            node_f = torch.cat([u_f, v_f], dim=0)
            # D^-1/2
            # degs = graph.out_degrees().to(feat.device).float().clamp(min=1)
            degs = graph.out_degrees().to(u_f.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)
            # norm = norm.view(-1,1)
            # shp = norm.shape + (1,) * (feat.dim() - 1)
            # norm = t.reshape(norm, shp)

            node_f = node_f * norm

            graph.ndata['n_f'] = node_f
            # graph.edata['e_f'] = e_f
            graph.update_all(fn.copy_u(u='n_f', out='m'), reduce_func=fn.sum(msg='m', out='n_f'))

            rst = graph.ndata['n_f']

            degs = graph.in_degrees().to(u_f.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)
            # shp = norm.shape + (1,) * (feat.dim() - 1)
            # norm = t.reshape(norm, shp)
            rst = rst * norm

            if self._activation is not None:
                rst = self._activation(rst)

            return rst

class GCNModel(nn.Module):
    def __init__(self,args, n_user,n_item):
        super(GCNModel, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_hid = args.n_hid
        self.n_layers = args.n_layers
        self.s_layers = args.s_layers
        self.embedding_dict = self.init_weight(n_user, n_item, self.n_hid)
        self.act = nn.LeakyReLU(0.5, inplace=True)
        self.layers = nn.ModuleList()
        self.uu_Layers = nn.ModuleList()
        self.weight = args.weight
        for i in range(0, self.n_layers):
            self.layers.append(GCNLayer(self.n_hid, self.n_hid, weight=self.weight, bias=False, activation=self.act))
        for i in range(0, self.s_layers):
            self.uu_Layers.append(UUGCNLayer(self.n_hid,self.n_hid,weight=self.weight, bias=False, activation=self.act))
    def init_weight(self, userNum, itemNum, hide_dim):
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(userNum, hide_dim))),
            'item_emb': nn.Parameter(initializer(torch.empty(itemNum, hide_dim))),
        })
        return embedding_dict
    def forward(self, uigraph, uugraph, isTrain=True):

        init_embedding = torch.concat([self.embedding_dict['user_emb'],self.embedding_dict['item_emb']],axis=0)
        init_user_embedding = self.embedding_dict['user_emb']
        all_embeddings = [init_embedding]
        all_uu_embeddings = [init_user_embedding]

        for i, layer in enumerate(self.layers):
            if i == 0:
                embeddings = layer(uigraph, self.embedding_dict['user_emb'], self.embedding_dict['item_emb'])
            else:
                embeddings = layer(uigraph, embeddings[:self.n_user], embeddings[self.n_user:])

            norm_embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]
        ui_embeddings = sum(all_embeddings)

        for i, layer in enumerate(self.uu_Layers):
            if i == 0:
                embeddings = layer(uugraph, self.embedding_dict['user_emb'])
            else:
                embeddings = layer(uugraph, embeddings)
            norm_embeddings = F.normalize(embeddings, p=2, dim=1)
            all_uu_embeddings +=[norm_embeddings]
        uu_embeddings = sum(all_uu_embeddings)

        return ui_embeddings,uu_embeddings




#Social hidden fully-connected architecture
class SDNet(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """

    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(SDNet, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims

        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
                                        for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
                                         for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)

        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)

        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)

        return h


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

