from utils import med_config
import h5py
import numpy as np
from utils.med_get_gen_labels import map_nid_to_id
import torch as th
import dgl
from models.med_hete_models import TGCN


def test_clsfier(num_owners,classfier:TGCN,demo_graph,downstream,device,pref):
    indx_file = med_config.root_path + "data/MIMICIII/indx_owners_" + str(num_owners) + "train_" + str(
        med_config.train_portion) + "test_" + str(med_config.test_portion) + ".h5"

    hf = h5py.File(indx_file, 'r')
    global_test_idx=np.asarray(hf.get('global_test_idx')).reshape(-1)
    hf.close()
    g_test_loader_idx=map_nid_to_id(demo_graph, th.tensor(global_test_idx,dtype=th.int32), downstream)
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
        batch_size=med_config.batch_size,shuffle=True, num_workers=0)
    labels = demo_graph.nodes[downstream].data['label']

    classfier=classfier.to(device)
    test_acc=classfier.inference_on_glb(demo_graph=demo_graph,test_loader=g_test_loader,labels=labels,
                        downstream=downstream,device=device)
    print(str(pref)+" Global Test Acc: {:.4f}".format(test_acc))
    print()
