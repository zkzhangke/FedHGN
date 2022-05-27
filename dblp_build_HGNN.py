import sys
sys.path.append("/path/to/FedHGN/")
from .utils import dblp_config
import torch as th
import dgl
from utils.dblp_read_from_file import construct_graph
from dgl import DGLHeteroGraph
from utils.functions import rm_duplicate_in_list
from utils.get_gen_labels import get_train_idx
from collections import defaultdict
import argparse
import numpy as np
import h5py



def main(args):
    num_owners = args.num_owners
    num_classes=args.num_classes
    local_subgraphs_path = dblp_config.get_path("dblp_demo", num_owners)

    # load graph data
    demo_graph_file = dblp_config.root_path + "data/DBLP/dblp_demo_"+str(num_owners)+".bin"
    whole_graph_json = dblp_config.root_path + "data/DBLP/dblp.json"
    whole_graph_file = dblp_config.root_path + "data/DBLP/dblp.bin"
    whole_graph=construct_graph(whole_graph_json,whole_graph_file)
    rm_pub_nodes={k:[] for k in dblp_config.public_types}
    for k in rm_pub_nodes.keys():
        for i in whole_graph.nodes(k):
            if th.sum(whole_graph.nodes[k].data['feat'][i.item()]).item()==0:
                rm_pub_nodes[k].append(i.item())
        whole_graph.remove_nodes(nids=rm_pub_nodes[k],ntype=k)
    dgl.save_graphs(whole_graph_file,[whole_graph])
    indx_file = dblp_config.root_path + "data/DBLP/indx_owners_" + str(num_owners) + "train_" + str(
        dblp_config.train_portion) + "test_" + str(dblp_config.test_portion) + ".h5"

    category = 'venue'
    downstream = 'author'
    def get_demo_graph(offset:int,num_owners:int,graph:DGLHeteroGraph,target_node_type='venue'):

        venue_list=graph.nodes(target_node_type)
        topn={}
        for v_i in venue_list:
            topn[v_i]=graph.in_degree(v_i,('paper','in','venue')).item()
        topn_sort=[k.item() for k, v in sorted(topn.items(), key=lambda item: item[1],reverse=True)]
        venue_use = topn_sort[offset:num_owners+offset]
        print("venue_use list = "+str(venue_use))
        if num_owners+offset==1:
            venue_use=[venue_use]
        all_edge_types = graph.canonical_etypes
        all_node_types = graph.ntypes
        context_node_types = list.copy(all_node_types)
        context_node_types.remove(target_node_type)


        target_edge_types = []
        for etype_i in all_edge_types:
            if etype_i[-1] == target_node_type:
                target_edge_types.append(etype_i)
        do_nodes = {}
        for i in range(num_owners):
            do_nodes[i] = {}
            for node_type in context_node_types:
                do_nodes[i][node_type] = []
            do_nodes[i][target_node_type] = [venue_use[i]]

        for etype in target_edge_types:
            context_edges = graph.edges(etype=etype)
            for i in range(num_owners):
                target_ids = list.copy(do_nodes[i][target_node_type])
                context_type = etype[0]
                for ind in range(len(context_edges[-1])):
                    target_id_in_edge = context_edges[-1][ind].item()
                    if target_id_in_edge in target_ids:
                        context_id = context_edges[0][ind].item()
                        do_nodes[i][context_type].append(context_id)
        paper_context_etypes=[('paper','with','keyword'),('paper','study','mag_fos')]
        subgraphs=[]

        for owner_i in range(num_owners):
            for etype in paper_context_etypes:
                for i in graph.out_edges(do_nodes[owner_i]['paper'], etype=etype)[-1]:
                    do_nodes[owner_i][etype[-1]].append(i.item())
            for ntype in all_node_types:
                do_nodes[owner_i][ntype]=list.copy(rm_duplicate_in_list(do_nodes[owner_i][ntype]))


        demo_nodes = defaultdict(list)
        for ntype in all_node_types:
            demo_nodes[ntype]=[]
            for owner_i in range(num_owners):
                list.extend(demo_nodes[ntype],do_nodes[owner_i][ntype])
            demo_nodes[ntype] = list.copy(rm_duplicate_in_list(demo_nodes[ntype]))
        demo_graph=dgl.node_subgraph(graph,demo_nodes)
        for owner_i in range(num_owners):
            subgraph_i=dgl.node_subgraph(graph,do_nodes[owner_i])
            subgraphs.append(subgraph_i)

        return subgraphs,demo_graph

    print("get demo_graph")
    subgraphs,demo_graph=get_demo_graph(offset=1,num_owners=num_owners,
                   graph=whole_graph,
                   target_node_type=category)

    def get_author_labels(demo_graph:DGLHeteroGraph,num_classes:int):
        author_nodes=demo_graph.nodes('author')
        n_citations={i.item():th.tensor(0,dtype=th.int32) for i in author_nodes}
        for a_i in author_nodes:
            a_i=a_i.item()
            papers=demo_graph.out_edges(a_i,etype=('author','write','paper'))[-1]
            out_degrees=demo_graph.out_degrees(papers,etype=('paper','citedby','paper'))
            n_citations[a_i]=th.add(n_citations[a_i],th.sum(out_degrees))

        sorted_cite_authors={k: v for k, v in sorted(n_citations.items(), key=lambda item: item[1])}

        total_num=len(sorted_cite_authors.keys())
        avg_len=int(total_num/num_classes)

        sorted_inds=list(sorted_cite_authors.keys())

        inds_labels={i:sorted_inds[i*avg_len:(i+1)*avg_len] for i in range(num_classes-1)}
        inds_labels[num_classes-1]=sorted_inds[(num_classes-1)*avg_len:]
        label_dict={}

        for i in inds_labels.keys():
            for inds in inds_labels[i]:
                label_dict[inds]=i
        label_list=[label_dict[k.item()] for k in author_nodes]
        label_tensor=th.tensor(label_list,dtype=th.long).view(-1)
        return label_tensor
    def get_sublabels(demo_graph,cur_graph):
        sub_NID = cur_graph.nodes['author'].data[dgl.NID].numpy().reshape(-1)
        demo_NID = demo_graph.nodes['author'].data[dgl.NID].numpy().reshape(-1)
        sub_in_demo_ids=[]
        for i,n in enumerate(sub_NID):
            ind =np.argwhere(demo_NID == n)[0][0]
            sub_in_demo_ids.append(ind)
        in_whole_ids=th.tensor(sub_in_demo_ids,dtype=th.long).view(-1)
        cur_graph.nodes['author'].data['label']=demo_graph.nodes['author'].data['label'][in_whole_ids]
        return


    print("get label for demo_graph")
    demo_graph.nodes['author'].data['label']=get_author_labels(demo_graph,num_classes)

    def get_local_feat(graph:DGLHeteroGraph):
        nodes={ntype:np.arange(graph.num_nodes(ntype)) for ntype in dblp_config.private_types}
        etypes=graph.canonical_etypes
        pub_in_etypes=[
            ('venue', 'have', 'paper'),
            ('keyword', 'mentionedby', 'paper'),
            ('mag_fos', 'include', 'paper')
        ]
        node_feat={ntype:th.zeros((graph.num_nodes(ntype),dblp_config.feat_len)) for ntype in dblp_config.private_types}
        for ntype in nodes.keys():
            if ntype=='paper':
                for n_i in nodes[ntype]:
                    feat=th.zeros(dblp_config.feat_len)
                    length=0
                    for etype in pub_in_etypes:
                        pub_neighbors=graph.in_edges(n_i,etype=etype)[0]
                        length+=len(pub_neighbors)
                        feat+=th.sum(graph.nodes[etype[0]].data['feat'][th.tensor(pub_neighbors.numpy(),dtype=th.long)],dim=0)
                    feat=feat/length
                    node_feat[ntype][n_i]=feat
            elif ntype=='author':
                for n_i in nodes[ntype]:
                    length=0
                    feat=th.zeros(dblp_config.feat_len)
                    p_etypes=[]
                    for etype in etypes:
                        if etype[-1]==ntype and etype[0]=='paper':
                            p_etypes.append(etype)
                    for petype in p_etypes:
                        paper_neighbors=graph.in_edges(n_i,etype=petype)[0]
                        length+=len(paper_neighbors)
                        feat+=th.sum(graph.nodes['paper'].data['feat'][th.tensor(paper_neighbors.numpy(),dtype=th.long)],dim=0)
                    feat=feat/length
                    node_feat[ntype][n_i]=feat
            graph.nodes[ntype].data['feat']=node_feat[ntype]
        return graph

    for o_i in range(num_owners):
        print("start getting local "+str(o_i)+"'s local feat")
        subgraphs[o_i]=get_local_feat(subgraphs[o_i])
        print("finish getting local "+str(o_i)+"'s local feat")

    dgl.save_graphs(local_subgraphs_path, subgraphs)

    print("start getting demo's local feat")
    demo_graph=get_local_feat(demo_graph)
    print("finish getting demo's local feat")

    dgl.save_graphs(demo_graph_file, [demo_graph])
    # del demo_graph
    for i in range(num_owners):
        print("get label for subgraph "+str(i))
        get_sublabels(demo_graph,subgraphs[i])
    dgl.save_graphs(local_subgraphs_path, subgraphs)


    print("get train valid test idx")
    g_train_loader_idx, g_val_loader_idx, global_test_idx, _, _, _, _, _, \
    train_idx, test_idx, val_idx=get_train_idx(num_owners,subgraphs,demo_graph,downstream)

    hf = h5py.File(indx_file, 'w')
    hf.close()
    hf = h5py.File(indx_file, 'a')
    for i in range(num_owners):
        hf.create_dataset(name="train"+str(i),data=np.asarray(train_idx[i]))
        hf.create_dataset(name="val" + str(i), data=np.asarray(val_idx[i]))
        hf.create_dataset(name="test" + str(i), data=np.asarray(test_idx[i]))
    hf.create_dataset(name="global_test_idx", data=np.asarray(global_test_idx))
    hf.create_dataset(name="g_train_loader_idx", data=g_train_loader_idx.numpy())
    hf.create_dataset(name="g_val_loader_idx", data=g_val_loader_idx.numpy())
    hf.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FedHG')
    parser.add_argument("--num_owners", type=int, default=3,
                        help="number of data owner")
    parser.add_argument("--num_classes", type=int, default=5,
                        help="number of classes")
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    main(args)
