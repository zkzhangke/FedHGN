from dgl import DGLHeteroGraph
import torch as th
import dgl
import numpy as np

def rm_duplicate_in_list(l:list):
    if len(l)==0:
        return l
    final_l=[]
    for x in l:
        if x not in final_l:
            final_l.append(x)
    return final_l

def extract_embed(node_embed:dict, input_nodes:dict):
    emb = {}
    for ntype,feat in node_embed.items():
        if ntype in input_nodes.keys() and len(input_nodes[ntype])>0:
            emb_n=[]
            for n in input_nodes[ntype]:
                emb_n.append(feat[n.item()].view(-1))
            emb[ntype] = th.stack(emb_n)
    return emb


def write_graph_info(graph:DGLHeteroGraph,path:str,prefix=''):
    total=0
    with open(path,'a+') as f:
        f.write(prefix+'\n')
    for n in graph.ntypes:
        with open(path,'a+') as f:
            f.write(n+'\t')
            total+=graph.number_of_nodes(n)
            f.close()
    with open(path, 'a+') as f:
        f.write('total\n')
    for n in graph.ntypes:
        with open(path, 'a+') as f:
            f.write(str(graph.number_of_nodes(n)) + "\t")
    with open(path, 'a+') as f:
        f.write(str(total) + "\n\n")
        f.close()

def map_sub_to_g(cur_subg:DGLHeteroGraph,whole_g:DGLHeteroGraph,downstream:str):
    if whole_g.device!='cpu':
        sub_NID = cur_subg.nodes[downstream].data[dgl.NID].cpu().numpy().reshape(-1)
        whole_NID = whole_g.nodes[downstream].data[dgl.NID].cpu().numpy().reshape(-1)
    else:
        sub_NID = cur_subg.nodes[downstream].data[dgl.NID].numpy().reshape(-1)
        whole_NID = whole_g.nodes[downstream].data[dgl.NID].numpy().reshape(-1)
    sub_in_whole_ids=[]
    for i,n in enumerate(sub_NID):
        ind =np.argwhere(whole_NID == n)[0][0]
        sub_in_whole_ids.append(ind)
    in_whole_ids=th.tensor(sub_in_whole_ids,dtype=th.int32).view(-1)
    return in_whole_ids

def map_g_to_sub(whole_g:DGLHeteroGraph,cur_subg:DGLHeteroGraph,downstream:str):
    if whole_g.device!='cpu':
        sub_NID = cur_subg.nodes[downstream].data[dgl.NID].cpu().numpy().reshape(-1)
        whole_NID = whole_g.nodes[downstream].data[dgl.NID].cpu().numpy().reshape(-1)
    else:
        sub_NID = cur_subg.nodes[downstream].data[dgl.NID].numpy().reshape(-1)
        whole_NID = whole_g.nodes[downstream].data[dgl.NID].numpy().reshape(-1)
    sub_in_whole_ids=[]
    for i,n in enumerate(sub_NID):
        ind =np.argwhere(whole_NID == n)
        if len(ind)==0:
            sub_in_whole_ids.append(-1)
        else:
            ind =ind[0][0]
            sub_in_whole_ids.append(ind)
    in_whole_ids=th.tensor(sub_in_whole_ids,dtype=th.int32).view(-1)
    return in_whole_ids