from dgl import DGLHeteroGraph
import dgl
import torch
from utils.functions import rm_duplicate_in_list
from train.med_mb_eval_rgcn import data_prep,set_loaders_light,map_nid_to_id
from utils import med_config
import numpy as np

def get_prime_graph(hidden_portion:float,graph:DGLHeteroGraph,star_ntype='admission'):
    graph=graph.to('cpu')
    if graph.device!='cpu':
        all_nodes=graph.nodes(star_ntype).cpu().numpy().reshape(-1)

    else:
        all_nodes=graph.nodes(star_ntype).numpy().reshape(-1)
    hide_nodes=np.random.choice(all_nodes,int(graph.num_nodes(star_ntype)*hidden_portion),replace=False)

    possible_nodes={}

    for e_i in graph.canonical_etypes:
        if e_i[0]==star_ntype and e_i[-1]!=star_ntype:
            if e_i[-1] not in list(possible_nodes.keys()):
                possible_nodes[e_i[-1]]=[]
            for i in graph.out_edges(hide_nodes,etype=e_i)[-1]:
                possible_nodes[e_i[-1]].append(i.item() )

        elif e_i[-1]==star_ntype and e_i[0]!=star_ntype:
            if e_i[0] not in list(possible_nodes.keys()):
                possible_nodes[e_i[0]]=[]
            for i in graph.in_edges(hide_nodes,etype=e_i)[0]:
                possible_nodes[e_i[0]].append(i.item())
    graph.remove_nodes(hide_nodes,ntype=star_ntype)
    for ntype in list(possible_nodes.keys()):
        print("org len(possible_nodes["+str(ntype)+"]) = "+str(len(possible_nodes[ntype])))
        possible_nodes[ntype]=rm_duplicate_in_list(possible_nodes[ntype])
        print("cleaned len(possible_nodes["+str(ntype)+"]) = "+str(len(possible_nodes[ntype])))

    degree={ntype:{n_i:0 for n_i in possible_nodes[ntype]} for ntype in list(possible_nodes.keys())}

    for ntype in list(possible_nodes.keys()):
        for n_i in possible_nodes[ntype]:
            for e_i in graph.canonical_etypes:
                if e_i[-1]==ntype:
                    degree[ntype][n_i]+=graph.in_degrees(n_i,etype=e_i)
                if e_i[0]==ntype:
                    degree[ntype][n_i]+=graph.out_degrees(n_i,etype=e_i)
    rmv={ntype:[] for ntype in list(degree.keys())}
    for ntype in list(degree.keys()):
        for n_i in list(degree[ntype].keys()):
            if degree[ntype][n_i]==0:
                rmv[ntype].append(n_i)
    for ntype in list(rmv.keys()):
        if len(rmv[ntype])>0:
            graph.remove_nodes(rmv[ntype],ntype=ntype)

    return graph

def get_train_idx(num_owners,subgraphs,demo_g,downstream):
    global_train_idx = []
    global_test_idx = []
    global_val_idx = []
    train_idx = {}
    test_idx = {}
    val_idx = {}
    labels = {}

    for i in range(num_owners):
        train_idx[i], test_idx[i], val_idx[i], labels[i], gtrain_idx, gtest_idx, gval_idx = \
            data_prep(subgraphs[i], downstream)
        global_train_idx += gtrain_idx
        global_test_idx += gtest_idx
        global_val_idx += gval_idx

    global_train_idx = list(set(global_train_idx))
    global_test_idx = list(set(global_test_idx)-set(global_train_idx))
    global_val_idx = list(set(global_val_idx) - set(global_train_idx))
    global_val_idx = list(set(global_val_idx) - set(global_test_idx))

    fed_loaders = {"train": [], 'valid': []}
    fed_labels = {i: [] for i in range(num_owners)}
    train_loaders={i:None for i in range(num_owners)}
    val_loaders = {i: None for i in range(num_owners)}
    for i in range(num_owners):
        train_loaders[i],val_loaders[i]=set_loaders_light(subgraphs[i], downstream, train_idx[i], val_idx[i])
        fed_loaders['train'].append(train_loaders[i])
        fed_loaders['valid'].append(val_loaders[i])
        fed_labels[i]=labels[i]

    g_train_loader_idx=map_nid_to_id(demo_g, torch.tensor(global_train_idx,dtype=torch.int32), downstream)
    g_val_loader_idx=map_nid_to_id(demo_g, torch.tensor(global_val_idx,dtype=torch.int32), downstream)

    return g_train_loader_idx, g_val_loader_idx,global_test_idx,train_loaders,val_loaders,labels,fed_loaders,fed_labels,\
           train_idx,test_idx,val_idx




def map_sub_to_g(cur_subg:DGLHeteroGraph,whole_g:DGLHeteroGraph,downstream:str):
    if cur_subg.device!='cpu':
        sub_NID = cur_subg.nodes[downstream].data[dgl.NID].cpu().numpy().reshape(-1)
        whole_NID = whole_g.nodes[downstream].data[dgl.NID].cpu().numpy().reshape(-1)
    else:
        sub_NID = cur_subg.nodes[downstream].data[dgl.NID].numpy().reshape(-1)
        whole_NID = whole_g.nodes[downstream].data[dgl.NID].numpy().reshape(-1)
    sub_in_whole_ids=[]
    for i,n in enumerate(sub_NID):
        ind =np.argwhere(whole_NID == n)[0][0]
        sub_in_whole_ids.append(ind)
    in_whole_ids=torch.tensor(sub_in_whole_ids,dtype=torch.int32).view(-1)
    return in_whole_ids



def get_tngen_labels(whole_g:DGLHeteroGraph,cur_subg:DGLHeteroGraph,downstream:str):
    nodes=cur_subg.nodes(downstream).numpy().reshape(-1)
    etypes=list(med_config.pred_in_n_rev_etypes.keys())
    node_labels={i:{'num':{e_i:0 for e_i in etypes},'feat':{e_i:None for e_i in etypes}} for i in nodes}
    nodes_gID=map_sub_to_g(cur_subg, whole_g, downstream)
    for i,gid in zip(nodes,nodes_gID):
        true_missing_num={etype_i:0 for etype_i in etypes}
        true_missing_feat={etype_i:None for etype_i in etypes}
        for e_i in etypes:
            if e_i in cur_subg.canonical_etypes:
                if e_i[-1]==downstream:
                    miss=len(whole_g.in_edges(gid,etype=e_i)[0])-len(cur_subg.in_edges(i,etype=e_i)[0])
                    if miss>0:
                        in_ids=whole_g.in_edges(gid,etype=e_i)[0].numpy().reshape(-1)
                        miss_gids_np=list(set(in_ids))
                        miss_gids=set([i.item() for i in miss_gids_np])
                        cur_ntyp_gID=map_sub_to_g(cur_subg, whole_g, e_i[0])
                        cur_in_ids=cur_subg.in_edges(i,etype=e_i)[0].numpy().reshape(-1)
                        has_gids=list(cur_ntyp_gID[cur_in_ids].numpy().reshape(-1))
                        diff=torch.tensor(list(miss_gids.difference(has_gids)),dtype=torch.int32)

                        feats = []
                        true_missing_num[e_i]+=len(diff)
                        for d_i in diff[:min(len(diff),med_config.num_pred)]:
                            feats.append(whole_g.nodes[e_i[0]].data['feat'][d_i])
                        true_missing_feat[e_i] = torch.stack(feats)
            else:
                if e_i[-1]==downstream:
                    miss=len(whole_g.in_edges(gid,etype=e_i)[0])
                    if miss>0:
                        miss_gids=whole_g.in_edges(gid,etype=e_i)[0].numpy().reshape(-1)
                        true_missing_num[e_i] += len(miss_gids)
                        feats = []
                        for n_i in miss_gids[:min(len(miss_gids),med_config.num_pred)]:
                            feats.append(whole_g.nodes[e_i[0]].data['feat'][n_i])

                        true_missing_feat[e_i] = torch.stack(feats)
        for e_i in etypes:
            node_labels[i]['num'][e_i]+=true_missing_num[e_i]
            node_labels[i]['feat'][e_i]=true_missing_feat[e_i]
    return node_labels




def rdm_get_prime_graph(hidden_portion:float,graph:DGLHeteroGraph,node_type,device='cpu'):
    graph=graph.to(device)
    all_nodes=graph.nodes(node_type).numpy().reshape(-1)
    hide_nodes=np.random.choice(all_nodes,int(graph.num_nodes(node_type)*hidden_portion),replace=False)

    possible_nodes={}

    for e_i in list(med_config.pred_in_n_rev_etypes.keys()):
        if e_i[-1]==node_type and e_i[0]!=node_type:
            if e_i[0] not in list(possible_nodes.keys()):
                possible_nodes[e_i[0]]=[]
            for i in graph.in_edges(hide_nodes,etype=e_i)[0]:
                possible_nodes[e_i[0]].append(i.item())

    graph.remove_nodes(hide_nodes,ntype=node_type)
    for ntype in list(possible_nodes.keys()):
        possible_nodes[ntype]=rm_duplicate_in_list(possible_nodes[ntype])

    degree={ntype:{n_i:0 for n_i in possible_nodes[ntype]} for ntype in list(possible_nodes.keys())}

    for ntype in list(possible_nodes.keys()):
        for n_i in possible_nodes[ntype]:
            for e_i in graph.canonical_etypes:
                if e_i[-1]==ntype:
                    degree[ntype][n_i]+=graph.in_degrees(n_i,etype=e_i)
                if e_i[0]==ntype:
                    degree[ntype][n_i]+=graph.out_degrees(n_i,etype=e_i)
    degrees=torch.zeros(graph.num_nodes(node_type),dtype=torch.int32,device=device)
    for e_i in graph.canonical_etypes:
        if e_i[-1]==node_type:
            degrees+=graph.in_degrees(graph.nodes(node_type),etype=e_i)
        if e_i[0]==node_type:
            degrees+=graph.out_degrees(graph.nodes(node_type),etype=e_i)

    rmv={ntype:[] for ntype in list(degree.keys())}
    for ntype in list(degree.keys()):
        for n_i in list(degree[ntype].keys()):
            if degree[ntype][n_i]==0:
                rmv[ntype].append(n_i)
    for ntype in list(rmv.keys()):
        if len(rmv[ntype])>0:
            graph.remove_nodes(rmv[ntype],ntype=ntype)
    ntype_rmv=[]
    for i,node_degree in enumerate(degrees):
        if node_degree.item()==0:
            ntype_rmv.append(i)
    if len(ntype_rmv)>0:
        graph.remove_nodes(ntype_rmv,ntype=node_type)

    return graph