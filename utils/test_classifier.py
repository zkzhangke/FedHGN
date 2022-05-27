from utils import dblp_config
import h5py
import numpy as np
from utils.get_gen_labels import map_nid_to_id
import torch as th
import dgl
from models.hete_models import TGCN


def test_clsfier(num_owners,classfier:TGCN,demo_graph,downstream,device,pref):
    indx_file = dblp_config.root_path + "data/DBLP/indx_owners_" + str(num_owners) + "train_" + str(
        dblp_config.train_portion) + "test_" + str(dblp_config.test_portion) + ".h5"
    hf = h5py.File(indx_file, 'r')
    global_test_idx=np.asarray(hf.get('g_test_loader_idx')).reshape(-1)
    hf.close()
    g_test_loader_idx=map_nid_to_id(demo_graph, th.tensor(global_test_idx,dtype=th.int32), downstream)
    sampler = dgl.dataloading.MultiLayerNeighborSampler([{('author', 'coauthor', 'author'): dblp_config.fanout,
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
        batch_size=dblp_config.batch_size,shuffle=True, num_workers=0)
    labels = demo_graph.nodes[downstream].data['label']

    classfier=classfier.to(device)
    test_acc=classfier.inference_on_glb(demo_graph=demo_graph,test_loader=g_test_loader,labels=labels,
                        downstream=downstream,device=device)
    print(str(pref)+" Global Test Acc: {:.4f}".format(test_acc))
    print()
