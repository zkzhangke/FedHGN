import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLHeteroGraph
from models.layers import RelGraphConvLayer
from utils import med_config
import tqdm


class RelGraphFeat(nn.Module):
    def __init__(self,
                 pub_node_feats,
                 pub_node_num,
                 device,
                 embed_size,
                 dropout=0.0):
        super(RelGraphFeat, self).__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)

        # create weight embeddings for each node for each relation
        self.pub_node_embeds = nn.ModuleDict()
        self.pub_ntypes=pub_node_feats.keys()
        for ntype in self.pub_ntypes:
            emb = th.nn.Embedding(num_embeddings=pub_node_num[ntype], embedding_dim=embed_size)
            emb.weight = th.nn.Parameter(pub_node_feats[ntype].to(device))
            self.pub_node_embeds[ntype] = emb


    def forward(self, all_nids,all_feat:dict):

        all_feat={k:all_feat[k].to(self.device) for k in all_feat.keys()}
        embeds = all_feat

        for ntype in self.pub_ntypes:
            embeds[ntype] = self.pub_node_embeds[ntype](all_nids[ntype]).to(self.device)
        return embeds

    def inference(self,all_nids,all_feat):
        all_feat={k:all_feat[k].to(self.device) for k in all_feat.keys()}
        embeds = all_feat
        for ntype in self.pub_ntypes:
            embeds[ntype]= self.pub_node_embeds[ntype](all_nids[ntype].to(self.device))

        return embeds

class TGCN(nn.Module):
    def __init__(self,
                 ntypes,etypes,feat_size,
                 node_feat_dim,
                 h_dim, num_classes,
                 num_bases,device,
                 num_conv_layers=1,
                 dropout=0,
                 use_self_loop=False):
        super(TGCN, self).__init__()
        self.device=device
        self.ntypes=ntypes
        self.feat_size=feat_size
        self.h_dim = h_dim
        self.out_dim = num_classes
        self.rel_names = list(set(etypes))
        self.rel_names.sort()
        if num_bases < 0 or num_bases > len(self.rel_names):
            self.num_bases = len(self.rel_names)
        else:
            self.num_bases = num_bases
        self.num_conv_layers = num_conv_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop


        self.inp_layers = nn.ModuleDict(
            {ntype:nn.Linear(in_features=node_feat_dim[ntype],out_features=feat_size).to(device) for ntype in ntypes})
        # i2h
        self.layers = nn.ModuleList()
        self.layers.append(RelGraphConvLayer(
            self.feat_size, self.h_dim, self.rel_names,
            num_bases=self.num_bases, activation=F.leaky_relu, self_loop=self.use_self_loop,
            dropout=self.dropout).to(device))
        # h2h
        for i in range(self.num_conv_layers-1):
            self.layers.append(RelGraphConvLayer(
                self.h_dim, self.h_dim, self.rel_names,
                num_bases=self.num_bases, activation=F.leaky_relu, self_loop=self.use_self_loop,
                dropout=self.dropout).to(device))
        self.linear_layer1=nn.ModuleDict(
            {ntype:nn.ModuleList([nn.Linear(in_features=self.h_dim,out_features=self.h_dim//2).to(device),
                                  nn.Dropout(self.dropout).to(device),nn.LeakyReLU().to(device)]) for ntype in self.ntypes})
        self.out_layer=nn.ModuleDict(
            {ntype:nn.ModuleList([nn.Linear(in_features=self.h_dim//2,out_features=self.out_dim).to(device),
                                  nn.Dropout(self.dropout).to(device),nn.Softmax().to(device)])for ntype in self.ntypes})

    def forward(self,blocks,feat):

        h={k:feat[k].to(self.device) for k in feat.keys()}
        blocks=[blk.to(self.device) for blk in blocks]

        for ntype in feat.keys():
            h[ntype]=self.inp_layers[ntype](h[ntype])

        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)

        for ntype in h.keys():
            for layer in self.linear_layer1[ntype]:
                h[ntype]=layer(h[ntype])

        for ntype in h.keys():
            for layer in self.out_layer[ntype]:
                h[ntype]=layer(h[ntype])

        return h


    def inference(self,test_loader,feat,labels,downstream,device):

        total_acc=0.0
        count=0
        labels=labels.to(device)
        for input_nodes, output_nodes, blocks in tqdm.tqdm(test_loader):
            if blocks[-1].number_of_dst_nodes(downstream)!=med_config.batch_size:
                continue
            h={k:feat[k][th.tensor(input_nodes[k].data,dtype=th.long)].to(device) for k in input_nodes.keys()}
            blocks=[blk.to(device) for blk in blocks]


            for ntype in input_nodes.keys():
                h[ntype]=self.inp_layers[ntype](h[ntype])
            for layer, block in zip(self.layers, blocks):
                h = layer(block.to(device), h)

            for ntype in h.keys():
                for layer in self.linear_layer1[ntype]:
                    h[ntype]=layer(h[ntype])

            for ntype in h.keys():
                for layer in self.out_layer[ntype]:
                    h[ntype]=layer(h[ntype])
            logits=h[downstream].argmax(dim=1)
            out_ids=th.tensor(output_nodes[downstream].data,dtype=th.long,device=device)
            acc = th.sum(logits == labels[out_ids]).item()
            total_acc += acc
            count += len(output_nodes[downstream])

        return total_acc/(count+1e-10)

    def inference_on_glb(self,demo_graph:DGLHeteroGraph,test_loader,labels,downstream,device):

        total_acc=0.0
        count=0
        node_feats={}
        labels=labels.to(device)
        for ntype in demo_graph.ntypes:
            node_feats[ntype]=demo_graph.nodes[ntype].data['feat']
        for input_nodes, output_nodes, blocks in tqdm.tqdm(test_loader):
            if blocks[-1].number_of_dst_nodes(downstream)!=med_config.batch_size:
                continue
            h={k:node_feats[k][th.tensor(input_nodes[k].data,dtype=th.long)].to(device) for k in input_nodes.keys()}

            for ntype in input_nodes.keys():
                h[ntype]=self.inp_layers[ntype](h[ntype])

            for layer, block in zip(self.layers, blocks):
                h = layer(block.to(device), h)

            for ntype in h.keys():
                for layer in self.linear_layer1[ntype]:
                    h[ntype]=layer(h[ntype])

            for ntype in h.keys():
                for layer in self.out_layer[ntype]:
                    h[ntype]=layer(h[ntype])
            logits=h[downstream].argmax(dim=1)
            out_ids=th.tensor(output_nodes[downstream].data,dtype=th.long,device=device)
            acc = th.sum(logits == labels[out_ids]).item()
            total_acc += acc
            count += len(output_nodes[downstream])

        return total_acc/(count+1e-10)


    def inference_on_glb_curve(self,demo_graph:DGLHeteroGraph,test_loader,labels,downstream,device):
        total_acc=0.0
        count=0
        loss=0.0
        node_feats={}
        labels=labels.to(device)
        for ntype in demo_graph.ntypes:
            node_feats[ntype]=demo_graph.nodes[ntype].data['feat']
        for input_nodes, output_nodes, blocks in tqdm.tqdm(test_loader):
            if blocks[-1].number_of_dst_nodes(downstream)!=med_config.batch_size:
                continue
            h={k:node_feats[k][th.tensor(input_nodes[k].data,dtype=th.long)].to(device) for k in input_nodes.keys()}

            for ntype in input_nodes.keys():
                h[ntype]=self.inp_layers[ntype](h[ntype])

            for layer, block in zip(self.layers, blocks):
                h = layer(block.to(device), h)

            for ntype in h.keys():
                for layer in self.linear_layer1[ntype]:
                    h[ntype]=layer(h[ntype])

            for ntype in h.keys():
                for layer in self.out_layer[ntype]:
                    h[ntype]=layer(h[ntype])
            logits=h[downstream].argmax(dim=1)

            out_ids=th.tensor(output_nodes[downstream].data,dtype=th.long,device=device)
            loss += F.cross_entropy(th.clip(h[downstream],1e-9,1.0-1e-9), labels[out_ids].long()).data*len(out_ids)
            acc = th.sum(logits == labels[out_ids]).item()
            total_acc += acc
            count += len(output_nodes[downstream])

        return loss/(count+1e-10),total_acc/(count+1e-10)