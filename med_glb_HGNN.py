import sys
sys.path.append("/path/to/kdd2022_fedhg/")
from utils import med_config
import torch as th
import dgl
from train.med_mb_eval_rgcn import train_model
from models.med_hete_models import TGCN
import argparse
import numpy as np
import h5py
from utils.med_test_classifier import test_clsfier



def main(args):
    num_owners = args.num_owners
    num_classes=args.num_classes
    lr = args.lr
    n_epochs = args.n_epochs
    device = 'cpu'
    if med_config.cuda:
        th.cuda.set_device(med_config.gpu)
        device = 'cuda:%d' % med_config.gpu

    demo_graph_file = med_config.root_path + "data/MIMICIII/demo_"+str(num_owners)+".bin"
    indx_file = med_config.root_path + "data/MIMICIII/indx_owners_" + str(num_owners) + "train_" + str(
        med_config.train_portion) + "test_" + str(med_config.test_portion) + ".h5"


    # load graph data
    downstream = 'patient'

    demo_graphs = dgl.load_graphs(demo_graph_file)
    demo_graph = demo_graphs[0][0]


    hf = h5py.File(indx_file, 'r')
    g_train_loader_idx = np.asarray(hf.get('g_train_loader_idx')).reshape(-1)
    g_val_loader_idx = np.asarray(hf.get('g_val_loader_idx')).reshape(-1)
    hf.close()


    labels = demo_graph.nodes[downstream].data['label']


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


    sampled_ids = {downstream:g_train_loader_idx}
    train_loader = dgl.dataloading.NodeDataLoader(
        demo_graph, sampled_ids, sampler,
        batch_size=med_config.batch_size, shuffle=True, num_workers=0)

    sampled_ids[downstream] = g_val_loader_idx
    val_loader = dgl.dataloading.NodeDataLoader(
        demo_graph, sampled_ids, sampler,
        batch_size=med_config.batch_size, shuffle=True, num_workers=0)


    all_node_feats={}
    node_feat_dim={}
    for ntype in demo_graph.ntypes:
        all_node_feats[ntype]=demo_graph.nodes[ntype].data['feat']
        node_feat_dim[ntype]=len(demo_graph.nodes[ntype].data['feat'][0])

    model = TGCN(ntypes=demo_graph.ntypes,etypes=demo_graph.etypes,
                           feat_size=med_config.feat_len,
                           h_dim=med_config.hidden, num_classes=num_classes,
                           num_bases=med_config.n_bases,device=device,
                           num_conv_layers=med_config.n_layers,
                           node_feat_dim=node_feat_dim,
                           dropout=med_config.dropout,use_self_loop=med_config.use_self_loop)

    model=train_model(n_epochs=n_epochs,model=model,train_loader=train_loader,val_loader=val_loader,
                        labels_all=labels, downstream=downstream,feat=all_node_feats,
                      device=device,lr=lr)


    test_clsfier(num_owners=num_owners,classfier=model,demo_graph=demo_graph,
                 downstream=downstream,device=device,pref='glb')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FedHG')
    parser.add_argument("--num_classes", type=int, default=5,
            help="number of classes")
    parser.add_argument( "--n_epochs", type=int, default=100,
            help="number of training epoch")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate")
    parser.add_argument("--num_owners", type=int, default=5,
                        help="number of data owener")
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    main(args)