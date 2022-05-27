import sys
sys.path.append("/path/to/FedHGN/")
from utils import med_config
import torch as th
import dgl
from train.med_mb_eval_rgcn import set_gen_loaders
from utils.med_test_classifier import test_clsfier
from train.med_train_tngen import test_gen_model,train_locsageplus
from utils.med_get_gen_labels import get_tngen_labels,rdm_get_prime_graph
import numpy as np
import argparse
from models.med_gen_model import HLocSagePlus
import h5py


def main(args):
    cur_owner=args.cur_owner
    gen_portion=args.gen_portion
    gen_lr=args.gen_lr
    gencls_lr=args.gencls_lr
    num_classes=args.num_classes
    hidden_portion=args.hidden_portion
    cls_epochs = args.cls_epochs
    gen_epochs = args.gen_epochs

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


    demo_graphs = dgl.load_graphs(demo_graph_file)
    demo_graph = demo_graphs[0][0].to("cpu")
    subgraph = dgl.load_graphs(local_subgraphs_path)[0][cur_owner]

    hf = h5py.File(indx_file, 'r')
    train_idx = np.asarray(hf.get('train'+str(cur_owner))).reshape(-1)
    val_idx = np.asarray(hf.get('val' + str(cur_owner))).reshape(-1)
    hf.close()


    node_gen_labels={ntype:{} for ntype in med_config.private_types}
    subgraph_prime = dgl.load_graphs(local_subgraphs_path)[0][cur_owner]
    for ntype in med_config.private_types:
        print("remove node "+str(ntype))
        subgraph_prime=rdm_get_prime_graph(graph=subgraph_prime,node_type=ntype,device='cpu', hidden_portion=hidden_portion)
    for ntype in med_config.private_types:
        print("get label for "+str(ntype))
        node_gen_labels[ntype]=get_tngen_labels(subgraph,subgraph_prime,ntype)

    labels_prime=subgraph_prime.nodes[downstream].data['label']

    etypes=demo_graph.etypes
    ntypes=demo_graph.ntypes

    gen_train_loader, gen_val_loader = set_gen_loaders(subgraph_prime)

    node_feat_dim={}
    for ntype in subgraph.ntypes:
        node_feat_dim[ntype]=len(subgraph.nodes[ntype].data['feat'][0])
    tngen_i = HLocSagePlus(ntypes=ntypes,etypes=etypes,feat_len=med_config.feat_len,num_pred=med_config.num_pred,
                              pred_in_n_rev_etypes=med_config.pred_in_n_rev_etypes,
                              num_classes=num_classes,
                              batch_size=med_config.batch_size,
                              gen_h_dim=med_config.latent_dim,
                              cls_h_dim=med_config.hidden,
                              num_bases=med_config.n_bases,
                              device=device,
                              node_feat_dim=node_feat_dim,
                              gen_hidden_layers=med_config.n_layers - 1,
                              cls_hidden_layers=med_config.n_layers-1,
                              classifier_self_loop=med_config.use_self_loop,
                              gen_portion=gen_portion,
                              dropout=med_config.dropout,
                                tngen_self_loop=med_config.gen_self_loop)
    feat_i = {k: subgraph_prime.nodes[k].data['feat'] for k in subgraph_prime.ntypes}



    tngen_i = train_locsageplus(n_epochs=gen_epochs,model=tngen_i,gen_lr=gen_lr,
                                  gen_train_loader=gen_train_loader,node_feat_dim=node_feat_dim,
                                  node_gen_labels=node_gen_labels,feat=feat_i,
                                  graph=subgraph_prime,downstream=downstream,device=device,
                                  class_labels=labels_prime)


    labels_sub=subgraph.nodes[downstream].data['label']
    tngen_i.eval()
    model_test=test_gen_model(num_owners=num_owners,gen_model=tngen_i.tngen,graph=subgraph,downstream=downstream,
                          labels_train=labels_sub,
                              gen_portion=gen_portion,gencls_lr=gencls_lr,
                              node_feat_dim=node_feat_dim,
                          n_epochs=cls_epochs,train_idx=train_idx,val_idx=val_idx,
                              num_classes=num_classes,device='cpu',
                          global_graph=demo_graph,
                              pref='lsp classifier_'+str(cur_owner)+"_hidden_"+str(hidden_portion))


    all_node_feats={}
    for ntype in subgraph.ntypes:
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
