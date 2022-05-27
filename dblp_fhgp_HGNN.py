import sys
sys.path.append("/path/to/kdd2022_fedhg/")
from utils import dblp_config
import torch as th
import dgl
from train.mb_eval_rgcn import set_gen_loaders,set_loaders_light
from utils.functions import map_sub_to_g
from train.train_tngen import get_mended_graph
from train.fed_tngen import fed_train_tngen
from utils.get_gen_labels import map_nid_to_id,rdm_get_prime_graph,get_tngen_labels
from train.fedavg import fedTGCN,fed_test_model
import numpy as np
import argparse
from models.gen_model import HLocSagePlus
import h5py


def main(args):
    num_owners = args.num_owners
    num_classes=args.num_classes
    gen_portion=args.gen_portion
    fed_lr=args.fed_lr
    fed_gen_lr=args.fed_gen_lr
    n_epochs = args.n_epochs
    hidden_portion=args.hidden_portion
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
    demo_graph = demo_graphs[0][0].to('cpu')
    subgraphs = dgl.load_graphs(local_subgraphs_path)[0][:num_owners]


    train_idx = {}
    val_idx = {}

    hf = h5py.File(indx_file, 'r')
    for i in range(num_owners):
        train_idx[i]=np.asarray(hf.get("train" + str(i)))
        val_idx[i]=np.asarray(hf.get("val" + str(i)))
    global_test_idx=np.asarray(hf.get('g_test_loader_idx')).reshape(-1)
    hf.close()
    g_test_loader_idx=map_nid_to_id(demo_graph, th.tensor(global_test_idx,dtype=th.int32), downstream)



    fed_loaders = []
    train_loaders = {i: None for i in range(num_owners)}
    for i in range(num_owners):
        train_loaders[i], _ = set_loaders_light(subgraphs[i], downstream, train_idx[i], val_idx[i])
        fed_loaders.append(train_loaders[i])
    # check cuda

    etypes=demo_graph.etypes
    ntypes=demo_graph.ntypes

    # locsage+
    pub_node_ids_dict={}
    for owner_i in range(num_owners):
        pub_node_ids_dict[owner_i]={ntype:map_sub_to_g(subgraphs[owner_i],demo_graph,ntype) for ntype in dblp_config.public_types}
    pub_node_num={}
    pub_node_feats={}
    for ntype in dblp_config.public_types:
        pub_node_num[ntype]=demo_graph.num_nodes(ntype)
        pub_node_feats[ntype]=demo_graph.nodes[ntype].data['feat']


    tngen_models=[]
    subgraph_prime_all=[]
    fed_gen_labels={}
    fed_feat={}
    gen_train_loader_all=[]
    gen_val_loader_all=[]
    for owner_i in range(num_owners):
        print("build T-NGen for owner "+str(owner_i))
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
        tngen_models.append(tngen_i)

        print("get \\bar{G_i} for owner "+str(owner_i))

        node_gen_labels={ntype:{} for ntype in dblp_config.private_types}
        subgraph_prime = dgl.load_graphs(local_subgraphs_path)[0][owner_i]
        for ntype in dblp_config.private_types:
            print("remove node "+str(ntype))
            subgraph_prime=rdm_get_prime_graph(graph=subgraph_prime,node_type=ntype,device='cpu', hidden_portion=hidden_portion)
        for ntype in dblp_config.private_types:
            print("get label for "+str(ntype))
            node_gen_labels[ntype]=get_tngen_labels(subgraphs[owner_i],subgraph_prime,ntype)
        subgraph_prime_all.append(subgraph_prime)
        fed_feat[owner_i]={k: subgraph_prime.nodes[k].data['feat'] for k in subgraph_prime.ntypes}



        gen_train_loader, gen_val_loader = set_gen_loaders(subgraph_prime)
        gen_train_loader_all.append(gen_train_loader)
        gen_val_loader_all.append(gen_val_loader)
        print("get tngen labels for owner "+str(owner_i))
        fed_gen_labels[owner_i]=node_gen_labels
    print("start fed tngen")
    tngen_models=fed_train_tngen(num_owners=num_owners,models=tngen_models,
                                       local_epochs=dblp_config.fed_local_gen_epochs,
                                       subgraph_prime_all=subgraph_prime_all,
                                       fed_gen_lr=fed_gen_lr,
                                       pub_node_ids_all=pub_node_ids_dict,
                                       device=device,fed_feat=fed_feat,
                                       gen_train_loader_all=gen_train_loader_all,
                                       n_epochs=dblp_config.fed_global_gen_epochs,
                                       gen_labels=fed_gen_labels,downstream=downstream,
                                       )


    mended_graphs=[]
    for owner_i in range(num_owners):
        mended_graphs.append(get_mended_graph(gen_model=tngen_models[owner_i].tngen,device='cpu',
                                              graph=subgraphs[owner_i],gen_portion=gen_portion))

    pub_node_num={}
    pub_node_feats={}
    for ntype in dblp_config.public_types:
        pub_node_num[ntype]=demo_graph.num_nodes(ntype)
        pub_node_feats[ntype]=demo_graph.nodes[ntype].data['feat']

    pub_node_ids_dict={}
    fed_loaders = []#, 'valid': []}
    train_loaders = {i: None for i in range(num_owners)}
    for owner_i in range(num_owners):
        pub_node_ids_dict[owner_i]={ntype:map_sub_to_g(mended_graphs[owner_i],demo_graph,ntype) for ntype in dblp_config.public_types}
        train_loaders[owner_i], _ = set_loaders_light(mended_graphs[owner_i], downstream, train_idx[owner_i], val_idx[owner_i])
        fed_loaders.append(train_loaders[owner_i])


    fed_model=fedTGCN(num_owners=num_owners,n_epochs=n_epochs,num_classes=num_classes,subgraphs=mended_graphs,
                      pub_node_feats=pub_node_feats,pub_node_num=pub_node_num,device=device,fed_lr=fed_lr,
                      loaders=fed_loaders,etypes=demo_graph.etypes,ntypes=demo_graph.ntypes,pub_node_ids_dict=pub_node_ids_dict,
                      downstream=downstream)


    labels = demo_graph.nodes[downstream].data['label']
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
                   labels=labels,pref="FedHG+ h = "+str(hidden_portion)+"\t pred = "+str(gen_portion))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FedHG')
    parser.add_argument("--hidden_portion", type=float, default=0.8,
                        help="hidden portion for training tngen")
    parser.add_argument("--fed_lr", type=float, default=0.0001,
            help="learning rate")
    parser.add_argument("--fed_gen_lr", type=float, default=0.0001,
                        help="learning rate")
    parser.add_argument("--num_classes", type=int, default=5,
                        help="number of classes")
    parser.add_argument( "--n_epochs", type=int, default=50,
                         help="number of training epoch")
    parser.add_argument("--gen_portion", type=float, default=0.2,
                        help="gen portion for tngen")
    parser.add_argument("--num_owners", type=int, default=5,
                        help="number of data owener")
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    main(args)