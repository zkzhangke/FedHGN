from utils import med_config
from train.radam import RAdam
import torch as th
from dgl import DGLHeteroGraph
import torch.nn.functional as F
from models.med_hete_models import TGCN

def fedTGCN(num_owners,n_epochs,subgraphs:list,num_classes:int,etypes,ntypes,
            device,fed_lr,node_feat_dim,
            loaders:list,downstream:str):
    model_list=[]
    model_fvg = TGCN(etypes=etypes,ntypes=ntypes,
                             feat_size=med_config.feat_len,
                             h_dim=med_config.hidden, num_classes=num_classes,num_bases=med_config.n_bases,
                             num_conv_layers=med_config.n_layers,device=device,node_feat_dim=node_feat_dim,
                             dropout=med_config.dropout,use_self_loop=med_config.use_self_loop)

    cur_model_weights=model_fvg.state_dict()
    for owner_i in range(num_owners):
        model_i = TGCN(etypes=etypes,ntypes=ntypes,
                                 feat_size=med_config.feat_len,device=device,
                                 h_dim=med_config.hidden, num_classes=num_classes,
                                 num_bases=med_config.n_bases,
                                 num_conv_layers=med_config.n_layers,node_feat_dim=node_feat_dim,
                                 dropout=med_config.dropout,use_self_loop=med_config.use_self_loop)

        model_i.load_state_dict(cur_model_weights)
        model_list.append(model_i)

    for epoch in range(n_epochs):
        for owner_i in range(num_owners):
            subgraph=subgraphs[owner_i]
            train_loader = loaders[owner_i]
            labels_i = subgraph.nodes[downstream].data['label']
            node_feats={}
            for ntype in subgraph.ntypes:
                node_feats[ntype]=subgraph.nodes[ntype].data['feat']
            model_list[owner_i].load_state_dict(model_fvg.state_dict())
            optimizer = RAdam(model_list[owner_i].parameters(), lr=fed_lr,
                              weight_decay=med_config.weight_decay)
            model_list[owner_i].train()
            optimizer.zero_grad()
            for epoch_l in range(med_config.local_epochs):
                for i, (input_nodes, seeds, blocks) in enumerate(train_loader):
                    seeds=th.tensor(seeds[downstream].numpy(),dtype=th.long)
                    lbl=labels_i[seeds].to(device)
                    h={k:node_feats[k][th.tensor(input_nodes[k].data,dtype=th.long)] for k in input_nodes.keys()}

                    logits =  model_list[owner_i](blocks,h)[downstream]
                    loss_i = F.cross_entropy(th.clip(logits,1e-9,1.0-1e-9), lbl.long())
                    loss_i.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    train_acc = th.sum(logits.argmax(dim=1) == lbl).item() / len(seeds)
                    print("Epoch {:05d}|Owner {:05d} |local_epoch {:05d} | Train Acc: {:.4f} | Train Loss: {:.4f}".
                          format(epoch,owner_i,epoch_l, train_acc,loss_i.item()))
            model_list[owner_i].eval()
        for i in cur_model_weights.keys():
            cur_model_weights[i]=th.mean(th.stack([model_i.state_dict()[i] for model_i in model_list]),dim=0)
        model_fvg.load_state_dict(cur_model_weights)
    return model_fvg

def fed_test_model(demo_graph:DGLHeteroGraph,model:TGCN,downstream, g_test_loader,device, labels,pref=""):
    model.eval()
    test_acc = model.inference_on_glb(demo_graph=demo_graph,
                                      labels=labels,downstream=downstream,test_loader=g_test_loader,device=device)
    print(pref+" Global Test Acc: {:.4f}".format(test_acc))
    print()


