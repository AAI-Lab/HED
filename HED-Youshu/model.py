import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from scipy.sparse import csr_matrix
from models.hypersagnn import HyperSAGNN
from models.my_gcn_conv import GCNConv
from models.transformer_model import TransformerModel


def Split_HyperGraph_to_device(H, device, split_num=16):
    H_list = []
    length = H.shape[0] // split_num
    for i in range(split_num):
        if i == split_num - 1:
            H_list.append(H[length * i : H.shape[0]])
        else:
            H_list.append(H[length * i : length * (i + 1)])
    H_split = [SparseTensor.from_scipy(H_i).to(device) for H_i in H_list]
    return H_split


def normalize_Hyper(H):
    D_v = sp.diags(1 / (np.sqrt(H.sum(axis=1).A.ravel()) + 1e-8))
    D_e = sp.diags(1 / (np.sqrt(H.sum(axis=0).A.ravel()) + 1e-8))
    H_nomalized = D_v @ H @ D_e @ H.T @ D_v
    return H_nomalized


def mix_hypergraph(raw_graph, threshold=10):
    ui_graph, bi_graph, ub_graph = raw_graph

    #ii_graph = np.eye(32770)
    ii_graph = np.zeros((32770, 32770), dtype=np.int32)

    uu_graph = ub_graph @ ub_graph.T
    for i in range(ub_graph.shape[0]):
        for r in range(uu_graph.indptr[i], uu_graph.indptr[i + 1]):
            uu_graph.data[r] = 1 if uu_graph.data[r] > threshold else 0

    bb_graph = ub_graph.T @ ub_graph
    for i in range(ub_graph.shape[1]):
        for r in range(bb_graph.indptr[i], bb_graph.indptr[i + 1]):
            bb_graph.data[r] = 1 if bb_graph.data[r] > threshold else 0
    uu = np.eye(8039)
    uu_graph = uu_graph + uu
    bb = np.eye(4771)
    bb_graph = bb_graph + bb


    H1 = sp.vstack((uu_graph,ui_graph.T))
    H1 = sp.vstack((H1,ub_graph.T))
    H2 = sp.vstack((ui_graph,ii_graph))
    H2 = sp.vstack((H2,bi_graph))
    H3 = sp.vstack((ub_graph,bi_graph.T))
    H3 = sp.vstack((H3,bb_graph))
    H = sp.hstack((H1,H2))
    H = sp.hstack((H,H3))
    print("finsh mix hypergraph")

    return H

class UHBR(nn.Module):
    def __init__(self, raw_graph, device, dp, l2_norm, emb_size=32):
        super().__init__()

        ui_graph, bi_graph, ub_graph = raw_graph
        self.num_users, self.num_bundles, self.num_items = (
            ub_graph.shape[0],
            ub_graph.shape[1],
            ui_graph.shape[1],
        )
        
        H = mix_hypergraph(raw_graph)

        self.atom_graph = Split_HyperGraph_to_device(normalize_Hyper(H), device)
        print("finish generating hypergraph")

        # embeddings
        self.items_feature = nn.Parameter(
            torch.FloatTensor(self.num_items, emb_size).normal_(0, 0.5 / emb_size)
        )       
        self.users_feature = nn.Parameter(
            torch.FloatTensor(self.num_users, emb_size).normal_(0, 0.5 / emb_size)
        )
        self.bundles_feature = nn.Parameter(
            torch.FloatTensor(self.num_bundles, emb_size).normal_(0, 0.5 / emb_size)
        )
        self.user_bound = nn.Parameter(torch.FloatTensor(emb_size, 1).normal_(0, 0.5 / emb_size)).detach().numpy()
        self.drop = nn.Dropout(dp)
        self.embed_L2_norm = l2_norm
        print("finish generating embedding")

        #self.trans_model                     = TransformerModel(ntoken=self.num_items, ninp=16, nhead=4,
                                                                    #nhid=16, nlayers=3, dropout=0.2)
        #self.layer_norm                      = nn.LayerNorm(2)

        # hgnn ==================================
        self.hypersagnn_model                = HyperSAGNN(n_head=8, d_model=16, d_k=16, d_v=16,
                                                 node_embedding=32,
                                                 diag_mask=True, bottle_neck=8,dropout=0,
                                                 #emb_user=self.users_feature,emb_item=self.items_feature,emb_bundle=self.bundles_feature
                                                 ).to(device)
        print("finish hgnn")



    def propagate(self):
        embed_0 = torch.cat([self.users_feature,self.items_feature,self.bundles_feature], dim=0)
        embed_1 = torch.cat([G @ embed_0 for G in self.atom_graph], dim=0)
        all_embeds = embed_0 / 2 + self.drop(embed_1) / 3
        users_feature, items_feature , bundles_feature = torch.split(
            all_embeds, [self.num_users, self.num_items , self.num_bundles], dim=0
        )
        return users_feature, items_feature, bundles_feature

    def predict(self, users_feature, bundles_feature):
        pred = torch.sum(users_feature * bundles_feature, 2)
        return pred

    def regularize(self, users_feature, items_feature,  bundles_feature):
        loss = self.embed_L2_norm * (
            (users_feature ** 2).sum() + (bundles_feature ** 2).sum() + (items_feature ** 2).sum() 
        )
        return loss

    def forward(self, users, bundles):       
        users_feature, items_feature , bundles_feature= self.propagate()     
        users_embedding = users_feature[users].expand(-1, bundles.shape[1], -1)
        bundles_embedding = bundles_feature[bundles]
        pred = self.predict(users_embedding, bundles_embedding)
        loss = self.regularize(users_feature, items_feature, bundles_feature)        
        user_score_bound = users_feature[users].detach().cpu().numpy() @ self.user_bound
        return pred, user_score_bound, loss

    def evaluate(self, propagate_result, users):
        users_feature, items_feature, bundles_feature = propagate_result
        users_feature = users_feature[users]
        scores = users_feature @ (bundles_feature.T)
        return scores
