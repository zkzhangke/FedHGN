import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
import tqdm
from models.layers import RelGraphConvLayer
from torch.nn.modules import Linear,LeakyReLU,Dropout,ReLU,Tanh
from utils import dblp_config
from train.mb_eval_rgcn import set_loaders_light
from models.hete_models import TGCN


class TNGen(nn.Module):
    def __init__(self,ntypes,
                 etypes,feat_len,num_pred,
                 h_dim,
                 num_bases,
                 device,
                 num_hidden_layers=1,
                 dropout=0,
                 pred_in_n_rev_etypes=dblp_config.pred_in_n_rev_etypes,
                 use_self_loop=False):
        super(TNGen, self).__init__()
        self.ntypes=ntypes
        self.in_dim=feat_len
        self.h_dim = h_dim
        self.num_pred=num_pred
        self.rel_names = list(set(etypes))
        self.rel_names.sort()
        self.pred_in_n_rev_etypes=pred_in_n_rev_etypes
        self.device=device
        if num_bases < 0 or num_bases > len(self.rel_names):
            self.num_bases = len(self.rel_names)
        else:
            self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop

        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(RelGraphConvLayer(
            self.in_dim, self.h_dim, self.rel_names,
            num_bases=self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
            dropout=self.dropout))
        # h2h
        for i in range(self.num_hidden_layers):
            self.layers.append(RelGraphConvLayer(
                self.h_dim, self.h_dim, self.rel_names,
                num_bases=self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout))

        self.linear_layer=nn.ModuleDict(
            {ntype:nn.ModuleList([nn.Linear(in_features=self.h_dim,out_features=self.h_dim),
                                  nn.Dropout(self.dropout),
                                  nn.LeakyReLU()]) for ntype in self.ntypes})

        self.rand=Sampler()

        self.dGen=ReldGen(
            in_dim=self.h_dim,device=device,
            rel_names=list(pred_in_n_rev_etypes.keys()),
            dropout=dropout, activation=F.relu)

        self.fGen = RelfGen(
            in_feat=self.h_dim,
            feat_len=feat_len,
            num_pred=num_pred,device=device,
            rel_names=list(pred_in_n_rev_etypes.keys()),
            dropout=dropout, activation=F.tanh)



    def forward(self,seeds, h,blocks=None):
        i=0
        seeds={k:seeds[k].to(self.device) for k in seeds.keys()}
        h={k:h[k].to(self.device) for k in h.keys()}
        blocks=[blk.to(self.device) for blk in blocks]
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
            i+=1

        for ntype in h.keys():
            for layer in self.linear_layer[ntype]:
                h[ntype]=layer(h[ntype])

        for key_i in h.keys():
            h[key_i] = self.rand(h[key_i])

        pred_missing=self.dGen(h,seeds)
        pred_feat = self.fGen(h,seeds)

        return pred_missing,pred_feat

    def inference(self, g, batch_size, x,device):
        self.eval()
        y = {
            k: th.zeros(
                g.number_of_nodes(k),
                self.h_dim)
            for k in g.ntypes}

        sampler = dgl.dataloading.MultiLayerNeighborSampler([{('author', 'coauthor', 'author'): dblp_config.fanout,
                                                              ('author', 'write', 'paper'): dblp_config.fanout,
                                                              ('paper', 'belongto', 'author'): dblp_config.fanout,
                                                              ('paper', 'cite', 'paper'): dblp_config.fanout,
                                                              ('paper', 'citedby', 'paper'): dblp_config.fanout,
                                                              ('paper', 'in', 'venue'): dblp_config.fanout,
                                                              ('venue', 'have', 'paper'): dblp_config.fanout,
                                                              ('paper', 'with', 'keyword'): dblp_config.fanout,
                                                              ('keyword', 'mentionedby', 'paper'): dblp_config.fanout,
                                                              ('paper', 'study', 'mag_fos'): dblp_config.fanout,
                                                              ('mag_fos', 'include', 'paper'): dblp_config.fanout
                                                              }] * dblp_config.n_layers)

        sampled_ids={k: th.arange(g.number_of_nodes(k),dtype=th.int32) for k in dblp_config.private_types}

        dataloader  = dgl.dataloading.NodeDataLoader(
            g, sampled_ids, sampler,
            batch_size=batch_size, shuffle=True, num_workers=0)

        for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
            h = {k: x[k][th.tensor(input_nodes[k].data,dtype=th.long)].to(device) for k in input_nodes.keys()}
            blocks=[blk.to(device) for blk in blocks]
            for  layer,block in zip(self.layers,blocks):
                h = layer.to(device)(block, h)

            for ntype in h.keys():
                for layer in self.linear_layer[ntype]:
                    h[ntype]=layer.to(device)(h[ntype])

            for key_i in h.keys():
                h[key_i] = self.rand(h[key_i])

            for k in h.keys():
                y[k][th.tensor(output_nodes[k].data,dtype=th.long)] = h[k].cpu()
            # x = y


        pred_missing = self.dGen(y,{k[-1]: th.arange(g.number_of_nodes(k[-1]),dtype=th.int32) for k in self.pred_in_n_rev_etypes.keys()})
        pred_feat = self.fGen(y,{k[-1]: th.arange(g.number_of_nodes(k[-1]),dtype=th.int32) for k in self.pred_in_n_rev_etypes.keys()})
        # print("pred_missing")
        # print(pred_missing)
        return pred_missing,pred_feat

class HLocSagePlus(nn.Module):
    def __init__(self,etypes,ntypes,feat_len,num_pred,gen_portion,
                 gen_h_dim,cls_h_dim,num_classes,pub_node_num,pub_node_feats,
                 num_bases,batch_size,device,pred_in_n_rev_etypes,
                 gen_hidden_layers=1,cls_hidden_layers=1,num_workers=1,
                 dropout=0,
                 tngen_self_loop=False,
                 classifier_self_loop=False):
        super(HLocSagePlus,self).__init__()
        self.batch_size=batch_size
        self.device=device
        self.num_workers=num_workers
        self.tngen=TNGen(
            ntypes=ntypes,
            etypes=etypes,feat_len=feat_len,num_pred=num_pred,
            h_dim=gen_h_dim, num_bases=num_bases,
            pred_in_n_rev_etypes=pred_in_n_rev_etypes,
            num_hidden_layers=gen_hidden_layers,
            dropout=dropout,
            use_self_loop=tngen_self_loop,
            device=device)
        self.mend_graph=MendGraph(gen_portion=gen_portion,device=device)
        self.tgcn=TGCN(etypes=etypes,ntypes=ntypes,
                                              h_dim=cls_h_dim,feat_size=feat_len,
                                              pub_node_num=pub_node_num,pub_node_feats=pub_node_feats,
                                              num_classes=num_classes,
                                              num_bases=dblp_config.n_bases,
                                              num_conv_layers=cls_hidden_layers,
                                              dropout=dropout,
                                              device=device,
                                              use_self_loop=classifier_self_loop)


    def forward(self,g,cur_nodes,pub_node_ids,downstream,h,blocks=None):
        pred_missing,pred_feat=self.tngen(seeds=cur_nodes,h=h,blocks=blocks)
        mended_graph=self.mend_graph(graph=g,cur_node=cur_nodes,pred_num=pred_missing,pred_feat=pred_feat)
        node_feats={}
        for ntype in mended_graph.ntypes:
            node_feats[ntype]=mended_graph.nodes[ntype].data['feat']

        if len(cur_nodes[downstream])>0:
            train_loader=set_loaders_light(mended_graph,downstream,train_idx=cur_nodes[downstream],batch_size=len(cur_nodes[downstream]))
            input_nodes, output_nodes, blocks=next(iter(train_loader))
            h={k:node_feats[k][th.tensor(input_nodes[k].data,dtype=th.long).to(self.device)].to(self.device) for k in input_nodes.keys()}
            for ntype in dblp_config.public_types:
                input_nodes[ntype]=pub_node_ids[ntype][th.tensor(input_nodes[ntype].data,dtype=th.long)]
            input_nodes={k:input_nodes[k].to(self.device) for k in input_nodes.keys()}
            output_nodes={k:output_nodes[k].to(self.device) for k in output_nodes.keys()}
            blocks=[blk.to(self.device) for blk in blocks]
            pred=self.tgcn(input_nodes,blocks,h)
            return pred_missing,pred_feat,pred,output_nodes
        else:
            cur_nodes={k:cur_nodes[k].to(self.device) for k in cur_nodes.keys()}
            return pred_missing,pred_feat,None,cur_nodes
    def inference(self, g, batch_size, x):

        pred_missing,pred_feat = self.tngen.inference(g=g, batch_size=batch_size,
                                                          x=x)
        cur_node={ntype:g.nodes(ntype) for ntype in dblp_config.private_types}
        mended_graph = self.mend_graph.mend_graph(graph=g,cur_node=cur_node,
                                                         pred_num=pred_missing,pred_feat=pred_feat)
        return mended_graph


class Sampler(nn.Module):
    def __init__(self):
        super(Sampler, self).__init__()

    def forward(self, inputs):
        rand = th.normal(0, 1, size=inputs.shape,device=inputs.device)
        return inputs + rand

class MendGraph(nn.Module):
    def __init__(self,gen_portion,device):
        super(MendGraph, self).__init__()
        self.gen_portion=gen_portion
        self.device=device

    def mend_graph(self,graph,cur_node,pred_num:dict,pred_feat:dict):
        mended_g=graph.to("cpu")
        for nty in list(pred_num.keys()):
            length=len(cur_node[nty])
            nodes=cur_node[nty][np.random.choice(np.arange(length),int(self.gen_portion*length),replace=False)]
            for s_i in nodes:
                for e_i in dblp_config.pred_in_n_rev_etypes.keys():
                    if e_i[-1]==nty:
                        pred_missing=pred_num[nty][s_i.item()][e_i].float()
                        if pred_missing.device=='cpu':
                            num_links_t=th.round(pred_missing).view(-1).detach().numpy().item()
                        else:
                            num_links_t=th.round(pred_missing).view(-1).cpu().detach().numpy().item()
                        num_links=np.clip(num_links_t,0,dblp_config.num_pred).astype(dtype=np.int32)

                        if num_links>0:
                            total_exist=mended_g.num_nodes(e_i[0])
                            pred_f = pred_feat[nty][s_i.item()][e_i].view(dblp_config.num_pred, -1)
                            if pred_f.device!='cpu':
                                pred_f=pred_f.cpu()
                            feat_extend=th.vstack((mended_g.nodes[e_i[0]].data['feat'],pred_f[:num_links])).view(-1,dblp_config.feat_len)
                            mended_g = dgl.add_edges(g=mended_g,
                                                     u=th.arange(start=total_exist,
                                                                 end=total_exist+num_links,dtype=th.int32),
                                                     v=[s_i.item()]*num_links,
                                                     etype=e_i)
                            mended_g = dgl.add_edges(g=mended_g, u=[s_i.item()]*num_links,
                                                     v=th.arange(start=total_exist,
                                                                 end=total_exist+num_links,dtype=th.int32),
                                                     etype=dblp_config.pred_in_n_rev_etypes[e_i])

                            mended_g.nodes[e_i[0]].data['feat']=feat_extend
        return mended_g



    def forward(self,graph,cur_node,pred_num,pred_feat):
        mended_graph=self.mend_graph(graph=graph,cur_node=cur_node,
                                         pred_num=pred_num,pred_feat=pred_feat)
        return mended_graph


    def inference(self,graph,cur_node,pred_num,pred_feat):
        self.eval()
        mended_graph=self.mend_graph(graph=graph,cur_node=cur_node,
                                         pred_num=pred_num,pred_feat=pred_feat)
        return mended_graph



class ReldGen(nn.Module):
    def __init__(self,
                 in_dim,
                 rel_names,
                 device,
                 activation=F.leaky_relu,
                 dropout=0.0):
        super(ReldGen, self).__init__()
        self.in_dim = in_dim
        self.rel_names = rel_names
        self.activation = activation
        self.device=device


        self.rel_layer = {
                rel : nn.ModuleList([Linear(in_dim, 256).to(device),
                                     LeakyReLU().to(device),
                                     Linear(256, 32).to(device),
                                     LeakyReLU().to(device),
                                     Dropout(dropout).to(device),
                                     Linear(32, 1).to(device),ReLU().to(device)
                                     # Linear(32, dblp_config.num_pred+1),Softmax()
                                    ])
                for rel in rel_names
            }

        # weight for self loop


    def forward(self,inputs,seeds):
        seeds={k:seeds[k].to(self.device) for k in seeds.keys()}
        inputs={k:inputs[k].to(self.device) for k in inputs.keys()}
        def _apply(rel,h):
            for layer in self.rel_layer[rel]:
                h = layer(h)
            return h
        return {ntype:{s_i.item():{rel : _apply(rel, inputs[ntype][ind])
                            for rel in self.rel_names}
                       for ind,s_i in enumerate(seeds[ntype])}
                for ntype in list(seeds.keys())}


class RelfGen(nn.Module):
    def __init__(self,
                 in_feat,
                 feat_len,
                 rel_names,
                 num_pred,
                 device,
                 activation=F.relu,
                 dropout=0.0):
        super(RelfGen, self).__init__()
        self.in_feat = in_feat
        self.out_feat = feat_len*num_pred
        self.num_pred=num_pred
        self.rel_names = rel_names
        self.activation = activation
        self.device=device

        self.rel_layer = {
                rel : nn.ModuleList([Linear(in_feat,256).to(device),
                                     ReLU().to(device),
                                     Linear(256,2048).to(device),
                                     ReLU().to(device),
                                     Dropout(dropout).to(device),
                                     Linear(2048, self.out_feat).to(device),
                                     Tanh().to(device)])
                for rel in rel_names
            }



    def forward(self,inputs,seeds):
        seeds={k:seeds[k].to(self.device) for k in seeds.keys()}
        inputs={k:inputs[k].to(self.device) for k in inputs.keys()}
        def _apply(rel,h):
            for layer in self.rel_layer[rel]:
                h = layer(h)
            hs = h.view(self.num_pred,-1)
            return hs
        return {ntype:{s_i.item():{rel : _apply(rel, inputs[ntype][ind])
                            for rel in self.rel_names}
                       for ind, s_i in enumerate(seeds[ntype])}
                for ntype in list(seeds.keys())}

