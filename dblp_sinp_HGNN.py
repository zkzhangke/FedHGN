import sys
sys.path.append("/path/to/FedHGN/")
from utils import dblp_config
import torch as th
import dgl
from train.mb_eval_rgcn import set_gen_loaders
from utils.functions import map_sub_to_g
from utils.test_classifier import test_clsfier
from train.train_tngen import test_gen_model,train_locsageplus
from utils.get_gen_labels import get_tngen_labels,rdm_get_prime_graph
import numpy as np
import argparse
from models.gen_model import HLocSagePlus
import h5py


def main(args):
    num_owners = args.num_owners
    num_classes=args.num_classes
    cur_owner=args.cur_owner
    gen_portion=args.gen_portion
    gen_lr=args.gen_lr
    gencls_lr=args.gencls_lr

    hidden_portion=args.hidden_portion
    cls_epochs = args.cls_epochs
    gen_epochs = args.gen_epochs
    device = 'cpu'
    local_subgraphs_path = dblp_config.get_path("dblp_demo", num_owners)

    if dblp_config.cuda:
        th.cuda.set_device(dblp_config.gpu)
        device = 'cuda:%d' % dblp_config.gpu
    # load graph data
    downstream = 'author'

    demo_graph_file = dblp_config.root_path + "data/DBLP/dblp_demo_" + str(num_owners) + ".bin"
    indx_file = dblp_config.root_path + "data/DBLP/indx_owners_" + str(num_owners) + "train_" + str(
        dblp_config.train_portion) + "test_" + str(dblp_config.test_portion) + ".h5"

    demo_graphs = dgl.load_graphs(demo_graph_file)
    demo_graph = demo_graphs[0][0].to("cpu")
    subgraph = dgl.load_graphs(local_subgraphs_path)[0][cur_owner]

    hf = h5py.File(indx_file, 'r')
    train_idx = np.asarray(hf.get('train'+str(cur_owner))).reshape(-1)
    val_idx = np.asarray(hf.get('val' + str(cur_owner))).reshape(-1)
    test_idx = np.asarray(hf.get('test' + str(cur_owner))).reshape(-1)
    hf.close()


    node_gen_labels={ntype:{} for ntype in dblp_config.private_types}
    subgraph_prime = dgl.load_graphs(local_subgraphs_path)[0][cur_owner]
    for ntype in dblp_config.private_types:
        print("remove node "+str(ntype))
        subgraph_prime=rdm_get_prime_graph(graph=subgraph_prime,node_type=ntype,device='cpu', hidden_portion=hidden_portion)
    for ntype in dblp_config.private_types:
        print("get label for "+str(ntype))
        node_gen_labels[ntype]=get_tngen_labels(subgraph,subgraph_prime,ntype)




    labels_prime=subgraph_prime.nodes[downstream].data['label']

    etypes=demo_graph.etypes
    ntypes=demo_graph.ntypes


    gen_train_loader, gen_val_loader = set_gen_loaders(subgraph_prime)
    pub_node_num={}
    pub_node_ids={}
    pub_node_feats={}
    for ntype in dblp_config.public_types:
        pub_node_num[ntype]=demo_graph.num_nodes(ntype)
        pub_node_feats[ntype]=demo_graph.nodes[ntype].data['feat']
        pub_node_ids[ntype]=map_sub_to_g(subgraph,demo_graph,ntype)


    tngen_i = HLocSagePlus(ntypes=ntypes,etypes=etypes,feat_len=dblp_config.feat_len,num_pred=dblp_config.num_pred,
                              pred_in_n_rev_etypes=dblp_config.pred_in_n_rev_etypes,
                              num_classes=num_classes,
                              pub_node_num=pub_node_num,
                              batch_size=dblp_config.batch_size,
                              pub_node_feats=pub_node_feats,
                              gen_h_dim=dblp_config.latent_dim,
                              cls_h_dim=dblp_config.hidden,
                              num_bases=dblp_config.n_bases,
                              device=device,
                              gen_hidden_layers=dblp_config.n_layers - 1,
                              cls_hidden_layers=dblp_config.n_layers-1,
                              classifier_self_loop=dblp_config.use_self_loop,
                              gen_portion=gen_portion,
                              dropout=dblp_config.dropout,
                              tngen_self_loop=dblp_config.gen_self_loop)
    feat_i = {k: subgraph_prime.nodes[k].data['feat'] for k in subgraph_prime.ntypes}

    tngen_i = train_locsageplus(n_epochs=gen_epochs,model=tngen_i,gen_lr=gen_lr,
                                       gen_train_loader=gen_train_loader,pub_node_ids=pub_node_ids,
                                       node_gen_labels=node_gen_labels,feat=feat_i,
                                       graph=subgraph_prime,downstream=downstream,device=device,
                                       class_labels=labels_prime)

    labels_sub=subgraph.nodes[downstream].data['label']
    tngen_i.eval()
    model_test=test_gen_model(num_owners=num_owners,gen_model=tngen_i.tngen,graph=subgraph,downstream=downstream,
                          pub_node_num=pub_node_num,pub_node_feats=pub_node_feats,labels_train=labels_sub,
                              gen_portion=gen_portion,gencls_lr=gencls_lr,
                          n_epochs=cls_epochs,train_idx=train_idx,val_idx=val_idx,num_classes=num_classes,device='cpu',
                          global_graph=demo_graph,pub_node_ids=pub_node_ids,
                              pref='Single+ classifier_'+str(cur_owner)+"_hidden_"+str(hidden_portion))


    all_node_num={}
    all_node_feats={}
    for ntype in subgraph.ntypes:
        all_node_num[ntype]=subgraph.num_nodes(ntype)
        all_node_feats[ntype]=subgraph.nodes[ntype].data['feat']




    test_clsfier(num_owners=num_owners,classfier=model_test,demo_graph=demo_graph,downstream=downstream,
                 device=device,pref='single+_'+str(cur_owner))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FedHG')
    parser.add_argument("--hidden_portion", type=float, default=0.8,
            help="hidden portion for training tngen")
    parser.add_argument("--gen_lr", type=float, default=0.0001,
            help="generator learning rate")
    parser.add_argument("--gencls_lr", type=float, default=0.0001,
                        help="mended classifier learning rate")
    parser.add_argument("--num_classes", type=int, default=5,
            help="number of classes")
    parser.add_argument( "--gen_epochs", type=int, default=50,
            help="number of gen training epoch")
    parser.add_argument( "--cls_epochs", type=int, default=50,
                         help="number of cls training epoch")

    parser.add_argument("--batch_size", type=int, default=256,
                        help="batch size")
    parser.add_argument("--num_owners", type=int, default=5,
                        help="number of data owner")

    parser.add_argument("--gen_portion", type=float, default=0.2,
                        help="gen portion for tngen")
    parser.add_argument("--cur_owner", type=int, default=0,
                        help="cur data owner for use")
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    main(args)
