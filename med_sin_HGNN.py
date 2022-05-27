import sys
sys.path.append("/path/to/FedHGN/")
from utils import med_config
import torch as th
import dgl
from train.med_mb_eval_rgcn import train_model
from utils.med_test_classifier import test_clsfier
import numpy as np
from models.med_hete_models import TGCN
import argparse
import h5py


def main(args):
    cur_owner=args.cur_owner
    num_classes=args.num_classes

    num_owners = args.num_owners
    local_subgraphs_path = med_config.get_path("med_demo", num_owners)
    device='cpu'
    if med_config.cuda:
        th.cuda.set_device(med_config.gpu)
        device = 'cuda:%d' % med_config.gpu
    # load graph data
    demo_graph_file = med_config.root_path + "data/MIMICIII/demo_"+str(num_owners)+".bin"

    indx_file = med_config.root_path + "data/MIMICIII/indx_owners_" + str(num_owners) + "train_" + str(
        med_config.train_portion) + "test_" + str(med_config.test_portion) + ".h5"

    downstream = 'patient'

    lr=args.lr
    n_epochs = args.n_epochs
    demo_graph=dgl.load_graphs(demo_graph_file)[0][0]

    subgraph = dgl.load_graphs(local_subgraphs_path)[0][cur_owner]


    hf = h5py.File(indx_file, 'r')
    train_idx = np.asarray(hf.get('train' + str(cur_owner))).reshape(-1)
    val_idx = np.asarray(hf.get('val' + str(cur_owner))).reshape(-1)
    hf.close()

    labels = subgraph.nodes[downstream].data['label']

    sampled_ids = {downstream:train_idx}
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

    train_loader = dgl.dataloading.NodeDataLoader(
        subgraph, sampled_ids, sampler,
        batch_size=med_config.batch_size,shuffle=True, num_workers=0)

    sampled_ids[downstream] = val_idx
    val_loader = dgl.dataloading.NodeDataLoader(
        subgraph, sampled_ids, sampler,
        batch_size=med_config.batch_size,shuffle=True, num_workers=0)



    all_node_feats={}
    node_feat_dim={}
    for ntype in subgraph.ntypes:

        all_node_feats[ntype]=subgraph.nodes[ntype].data['feat']
        node_feat_dim[ntype]=len(subgraph.nodes[ntype].data['feat'][0])

    model_i = TGCN(ntypes=subgraph.ntypes,etypes=subgraph.etypes,
                             feat_size=med_config.feat_len,
                             h_dim=med_config.hidden, num_classes=num_classes,num_bases=med_config.n_bases,
                             num_conv_layers=med_config.n_layers,device=device,node_feat_dim=node_feat_dim,
                             dropout=med_config.dropout,use_self_loop=med_config.use_self_loop)


    model_i=train_model(n_epochs=n_epochs,model=model_i,train_loader=train_loader,val_loader=val_loader,
                        labels_all=labels,downstream=downstream,feat=all_node_feats,
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
