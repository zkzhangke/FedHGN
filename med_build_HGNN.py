import sys
sys.path.append("/path/to/kdd2022_fedhg/")
from utils import med_config
import torch as th
import dgl
from utils.med_read_from_file import construct_graph
from dgl import DGLHeteroGraph
from utils.functions import rm_duplicate_in_list
from utils.med_get_gen_labels import get_train_idx
from collections import defaultdict
import argparse
import numpy as np
import h5py



def main(args):
    num_owners = args.num_owners
    local_subgraphs_path = med_config.get_path("med_demo", num_owners)

    # load graph data
    demo_graph_file = med_config.root_path + "data/MIMICIII/demo_"+str(num_owners)+".bin"
    whole_graph_file = med_config.root_path + "data/MIMICIII/whole_graph.bin"
    whole_graph=construct_graph(whole_graph_file)
    rm_pub_nodes={k:[] for k in med_config.public_types}
    for k in rm_pub_nodes.keys():
        for i in whole_graph.nodes(k):
            if th.sum(whole_graph.nodes[k].data['feat'][i.item()]).item()==0:
                rm_pub_nodes[k].append(i.item())
        whole_graph.remove_nodes(nids=rm_pub_nodes[k],ntype=k)
    dgl.save_graphs(whole_graph_file,[whole_graph])
    indx_file = med_config.root_path + "data/MIMICIII/indx_owners_" + str(num_owners) + "train_" + str(
        med_config.train_portion) + "test_" + str(med_config.test_portion) + ".h5"

    # category = 'adm_type'
    category = 'insurance'
    downstream = 'patient'

    def get_demo_graph(offset:int,num_owners:int,graph:DGLHeteroGraph,target_node_type='insurance'):

        insurance_list=graph.nodes(target_node_type)
        topn={}
        for v_i in insurance_list:
            topn[v_i]=graph.in_degree(v_i,('admission','with','insurance')).item()
        topn_sort=[k.item() for k, v in sorted(topn.items(), key=lambda item: item[1],reverse=True)]
        insurance_use = topn_sort[offset:num_owners+offset]
        print("insurance_use list = "+str(insurance_use))
        # print("topn_sort list = "+str(topn_sort))
        if num_owners+offset==1:
            insurance_use=[insurance_use]
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
            do_nodes[i][target_node_type] = [insurance_use[i]]

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
        admission_context_etypes=[('admission','use','medicine'),
                                  ('admission','get','procedure'),
                                  ('admission', 'belongto', 'patient'),
                                  ('admission', 'in', 'adm_type')]
        subgraphs=[]

        for owner_i in range(num_owners):
            for etype in admission_context_etypes:
                for i in graph.out_edges(do_nodes[owner_i]['admission'], etype=etype)[-1]:
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
    subgraphs,demo_graph=get_demo_graph(offset=0,num_owners=num_owners,
                   graph=whole_graph,
                   target_node_type=category)



    def get_patient_labels(demo_graph:DGLHeteroGraph):
        patient_nodes=demo_graph.nodes('patient')
        n_icus={i.item():0.0 for i in patient_nodes}
        for a_i in patient_nodes:
            a_i=a_i.item()
            admissions=demo_graph.out_edges(a_i,etype=('patient','of','admission'))[-1]
            print("admissions = "+str(admissions))
            for adm_id in admissions:
                n_icus[a_i]+=demo_graph.nodes['admission'].data['n_icu'][adm_id].data
            n_icus[a_i]=n_icus[a_i]/len(admissions)
        print("n_icus = "+str(n_icus))
        label_tensor=th.zeros(demo_graph.num_nodes('patient'),dtype=th.int32)
        for i,v in zip(range(len(label_tensor)),n_icus.keys()):
            if n_icus[v].data==0:
                label_tensor[i]+=0
            elif n_icus[v].data<=1:
                label_tensor[i]+=1
            elif n_icus[v].data<=2:
                label_tensor[i]+=2
            elif n_icus[v].data<=3:
                label_tensor[i]+=3
            elif n_icus[v].data>3:
                label_tensor[i]+=4
        return label_tensor
    def get_sublabels(demo_graph,cur_graph):
        sub_NID = cur_graph.nodes['patient'].data[dgl.NID].numpy().reshape(-1)
        demo_NID = demo_graph.nodes['patient'].data[dgl.NID].numpy().reshape(-1)
        sub_in_demo_ids=[]
        for i,n in enumerate(sub_NID):
            ind =np.argwhere(demo_NID == n)[0][0]
            sub_in_demo_ids.append(ind)
        in_whole_ids=th.tensor(sub_in_demo_ids,dtype=th.long).view(-1)
        cur_graph.nodes['patient'].data['label']=demo_graph.nodes['patient'].data['label'][in_whole_ids]
        return


    print("get label for demo_graph")
    demo_graph.nodes['patient'].data['label']=get_patient_labels(demo_graph)
    dgl.save_graphs(demo_graph_file, [demo_graph])
    # del demo_graph
    for i in range(num_owners):
        print("get label for subgraph "+str(i))
        get_sublabels(demo_graph,subgraphs[i])
    dgl.save_graphs(local_subgraphs_path, subgraphs)
    for i in range(num_owners):
        print(subgraphs[i])

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
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    main(args)