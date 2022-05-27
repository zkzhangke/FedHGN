import sys
sys.path.append("/path/to/kdd2022_fedhg/")
from utils import med_config
import torch as th
import dgl
from train.med_fedavg import fedTGCN,fed_test_model
import h5py
import numpy as np
import argparse
from utils.med_get_gen_labels import set_loaders_light,map_nid_to_id



def main(args):
    num_owners = args.num_owners
    num_classes=args.num_classes
    fed_lr=args.fed_lr
    n_epochs = args.n_epochs
    device = 'cpu'
    # load graph data
    downstream = 'patient'

    local_subgraphs_path = med_config.get_path("med_demo", num_owners)
    if med_config.cuda:
        th.cuda.set_device(med_config.gpu)
        device = 'cuda:%d' % med_config.gpu
    # load graph data
    demo_graph_file = med_config.root_path + "data/MIMICIII/demo_"+str(num_owners)+".bin"

    demo_graphs = dgl.load_graphs(demo_graph_file)
    demo_graph = demo_graphs[0][0].to('cpu')
    subgraphs = dgl.load_graphs(local_subgraphs_path)[0][:num_owners]
    indx_file = med_config.root_path + "data/MIMICIII/indx_owners_" + str(num_owners) + "train_" + str(
        med_config.train_portion) + "test_" + str(med_config.test_portion) + ".h5"



    train_idx = {}
    val_idx = {}

    hf = h5py.File(indx_file, 'r')
    for i in range(num_owners):
        train_idx[i]=np.asarray(hf.get("train" + str(i)))
        val_idx[i]=np.asarray(hf.get("val" + str(i)))
    hf = h5py.File(indx_file, 'r')
    global_test_idx=np.asarray(hf.get('global_test_idx')).reshape(-1)
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
    for ntype in med_config.public_types:
        pub_node_num[ntype]=demo_graph.num_nodes(ntype)
        pub_node_feats[ntype]=demo_graph.nodes[ntype].data['feat']
        pub_node_ids[ntype]=demo_graph.nodes(ntype)
    node_feat_dim={}
    for ntype in demo_graph.ntypes:
        node_feat_dim[ntype]=len(demo_graph.nodes[ntype].data['feat'][0])


    fed_model=fedTGCN(num_owners=num_owners,n_epochs=n_epochs,num_classes=num_classes,subgraphs=subgraphs,
                      device=device,node_feat_dim=node_feat_dim,
                      fed_lr=fed_lr,
                      loaders=fed_loaders,etypes=demo_graph.etypes,ntypes=demo_graph.ntypes,
                      downstream=downstream)


    labels = demo_graph.nodes[downstream].data['label'].to(device)
    sampler = dgl.dataloading.MultiLayerNeighborSampler([{('admission', 'seq', 'admission'): med_config.fanout,
                                                          ('admission', 'in', 'adm_type'): med_config.fanout,
                                                          ('adm_type', 'contain', 'admission'): med_config.fanout,
                                                          ('admission', 'belongto', 'patient'): med_config.fanout,
                                                          ('patient', 'of', 'admission'):  med_config.fanout,
                                                          ('patient', 'codiagnose', 'patient'):  med_config.fanout,
                                                          ('admission', 'with', 'insurance'): med_config.fanout,
                                                          ('insurance', 'choosenby', 'admission'): med_config.fanout,
                                                          ('admission', 'use', 'medicine'): med_config.fanout,
                                                          ('medicine', 'usedon', 'admission'): med_config.fanout,
                                                          ('admission', 'get', 'procedure'): med_config.fanout,
                                                          ('procedure', 'givento', 'admission'): med_config.fanout}] * med_config.n_layers)

    g_test_loader = dgl.dataloading.NodeDataLoader(
        demo_graph, {downstream: g_test_loader_idx}, sampler,
        batch_size=med_config.batch_size, shuffle=True, num_workers=0)
    fed_test_model(model=fed_model, downstream=downstream,
                   g_test_loader=g_test_loader, device=device,demo_graph=demo_graph,
                   labels=labels,pref="FedHG")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FedHG')
    parser.add_argument("--fed_lr", type=float, default=0.0001,
            help="fed rgcn learning rate")
    parser.add_argument("--num_classes", type=int, default=5,
            help="number of classes")
    parser.add_argument( "--n_epochs", type=int, default=50,
            help="number of training epoch")
    parser.add_argument("--num_owners", type=int, default=5,
                        help="number of data owner")
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    main(args)