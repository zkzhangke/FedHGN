import sys
sys.path.append("/path/to/FedHGN/")
from utils import dblp_config
import torch as th
import dgl
from train.fedavg import fedTGCN,fed_test_model
from utils.functions import map_sub_to_g
import h5py
import numpy as np
import argparse
from utils.get_gen_labels import set_loaders_light,map_nid_to_id



def main(args):
    num_owners = args.num_owners
    num_classes=args.num_classes
    fed_lr=args.fed_lr
    n_epochs = args.n_epochs
    device = 'cpu'

    local_subgraphs_path = dblp_config.get_path("dblp_demo", num_owners)
    if dblp_config.cuda:
        th.cuda.set_device(dblp_config.gpu)
        device = 'cuda:%d' % dblp_config.gpu
    # load graph data
    downstream = 'author'
    demo_graph_file = dblp_config.root_path + "data/DBLP/dblp_demo_" + str(num_owners) + ".bin"


    demo_graphs = dgl.load_graphs(demo_graph_file)
    demo_graph = demo_graphs[0][0].to('cpu')
    subgraphs = dgl.load_graphs(local_subgraphs_path)[0][:num_owners]
    indx_file = dblp_config.root_path + "data/DBLP/indx_owners_" + str(num_owners) + "train_" + str(
        dblp_config.train_portion) + "test_" + str(dblp_config.test_portion) + ".h5"


    train_idx = {}
    val_idx = {}

    hf = h5py.File(indx_file, 'r')
    for i in range(num_owners):
        train_idx[i]=np.asarray(hf.get("train" + str(i)))
        val_idx[i]=np.asarray(hf.get("val" + str(i)))
    hf = h5py.File(indx_file, 'r')
    global_test_idx=np.asarray(hf.get('g_test_loader_idx')).reshape(-1)
    hf.close()
    g_test_loader_idx=map_nid_to_id(demo_graph, th.tensor(global_test_idx,dtype=th.int32), downstream)


    fed_loaders = []
    train_loaders = {i: None for i in range(num_owners)}
    for i in range(num_owners):
        train_loaders[i], _ = set_loaders_light(subgraphs[i], downstream, train_idx[i], val_idx[i])
        fed_loaders.append(train_loaders[i])


    #fedavg
    pub_node_num={}
    pub_node_ids={}
    pub_node_feats={}
    for ntype in dblp_config.public_types:
        pub_node_num[ntype]=demo_graph.num_nodes(ntype)
        pub_node_feats[ntype]=demo_graph.nodes[ntype].data['feat']
        pub_node_ids[ntype]=demo_graph.nodes(ntype)

    pub_node_ids_dict={}
    for owner_i in range(num_owners):
        pub_node_ids_dict[owner_i]={ntype:map_sub_to_g(subgraphs[owner_i],demo_graph,ntype) for ntype in dblp_config.public_types}


    fed_model=fedTGCN(num_owners=num_owners,n_epochs=n_epochs,num_classes=num_classes,subgraphs=subgraphs,
                      pub_node_feats=pub_node_feats,pub_node_num=pub_node_num,device=device,
                      fed_lr=fed_lr,
                      loaders=fed_loaders,etypes=demo_graph.etypes,ntypes=demo_graph.ntypes,
                      pub_node_ids_dict=pub_node_ids_dict,
                      downstream=downstream)

    labels = demo_graph.nodes[downstream].data['label'].to(device)
    sampler = dgl.dataloading.MultiLayerNeighborSampler([{
        ('author', 'coauthor', 'author'): dblp_config.fanout,
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

    g_test_loader = dgl.dataloading.NodeDataLoader(
        demo_graph, {downstream: g_test_loader_idx}, sampler,
        batch_size=dblp_config.batch_size, shuffle=True, num_workers=0)
    fed_test_model(model=fed_model, downstream=downstream,
                   g_test_loader=g_test_loader, device=device,demo_graph=demo_graph,
                   labels=labels,pref="fedrgcn")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FedHG')
    parser.add_argument("--fed_lr", type=float, default=0.0001,
            help="fed rgcn learning rate")
    parser.add_argument("--num_classes", type=int, default=5,
            help="number of classes")
    parser.add_argument( "--n_epochs", type=int, default=50,
            help="number of training epoch")
    parser.add_argument("--flag", type=bool, default=False,
                        help="write dblp_txt_file")
    parser.add_argument("--num_owners", type=int, default=5,
                        help="number of data owner")
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    main(args)
