import sys
sys.path.append("/path/to/kdd2022_fedhg/")
from utils import dblp_config
import torch as th
import dgl
from train.mb_eval_rgcn import train_model
from utils.test_classifier import test_clsfier
from utils.functions import map_sub_to_g
import numpy as np
from models.hete_models import TGCN
import argparse
import h5py



def main(args):
    num_owners = args.num_owners
    cur_owner=args.cur_owner
    num_classes=args.num_classes
    lr=args.lr
    n_epochs = args.n_epochs
    device = 'cpu'
    local_subgraphs_path = dblp_config.get_path("dblp_demo", num_owners)
    if dblp_config.cuda:
        th.cuda.set_device(dblp_config.gpu)
        device = 'cuda:%d' % dblp_config.gpu
    # load graph data
    downstream = 'author'
    demo_graph_file = dblp_config.root_path + "data/DBLP/dblp_demo_"+str(num_owners)+".bin"
    demo_graph=dgl.load_graphs(demo_graph_file)[0][0]

    indx_file = dblp_config.root_path + "data/DBLP/indx_owners_" + str(num_owners) + "train_" + str(
        dblp_config.train_portion) + "test_" + str(dblp_config.test_portion) + ".h5"

    subgraph = dgl.load_graphs(local_subgraphs_path)[0][cur_owner]


    hf = h5py.File(indx_file, 'r')
    train_idx = np.asarray(hf.get('train' + str(cur_owner))).reshape(-1)
    val_idx = np.asarray(hf.get('val' + str(cur_owner))).reshape(-1)
    hf.close()

    labels = subgraph.nodes[downstream].data['label']

    sampled_ids = {downstream:train_idx}
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


    train_loader = dgl.dataloading.NodeDataLoader(
        subgraph, sampled_ids, sampler,
        batch_size=dblp_config.batch_size,shuffle=True, num_workers=0)

    sampled_ids[downstream] = val_idx
    val_loader = dgl.dataloading.NodeDataLoader(
        subgraph, sampled_ids, sampler,
        batch_size=dblp_config.batch_size,shuffle=True, num_workers=0)


    pub_node_num={}
    pub_node_ids={}
    pub_node_feats={}
    for ntype in dblp_config.public_types:
        pub_node_num[ntype]=demo_graph.num_nodes(ntype)
        pub_node_feats[ntype]=demo_graph.nodes[ntype].data['feat']
        pub_node_ids[ntype]=map_sub_to_g(subgraph,demo_graph,ntype)

    all_node_num={}
    all_node_feats={}
    for ntype in subgraph.ntypes:
        all_node_num[ntype]=subgraph.num_nodes(ntype)
        all_node_feats[ntype]=subgraph.nodes[ntype].data['feat']

    model_i = TGCN(ntypes=subgraph.ntypes,etypes=subgraph.etypes,
                             feat_size=dblp_config.feat_len,pub_node_num=pub_node_num,pub_node_feats=pub_node_feats,
                             h_dim=dblp_config.hidden, num_classes=num_classes,num_bases=dblp_config.n_bases,
                             num_conv_layers=dblp_config.n_layers,device=device,
                             dropout=dblp_config.dropout,use_self_loop=dblp_config.use_self_loop)
    model_i=train_model(n_epochs=n_epochs,model=model_i,train_loader=train_loader,val_loader=val_loader,
                        labels_all=labels,downstream=downstream,feat=all_node_feats,pub_node_ids=pub_node_ids,
                        device=device,lr=lr)

    print("start test")
    test_clsfier(num_owners=num_owners,classfier=model_i,demo_graph=demo_graph,
                 downstream=downstream,device=device,pref='owner '+str(cur_owner))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FedHG')
    parser.add_argument("--num_classes", type=int, default=5,
            help="number of classes")
    parser.add_argument( "--n_epochs", type=int, default=50,
            help="number of training epoch")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate")
    parser.add_argument("--num_owners", type=int, default=5,
                        help="number of data owener")
    parser.add_argument("--cur_owner", type=int, default=0,
                        help="cur data owner for use")
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    main(args)