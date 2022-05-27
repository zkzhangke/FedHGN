import dgl
import torch as th
from utils import dblp_config
from train.radam import RAdam
import torch.nn.functional as F
import numpy as np
from dgl import DGLHeteroGraph


def data_prep(graph:DGLHeteroGraph,downstream):

    local_len=graph.num_nodes(downstream)
    labeled_local_ids=graph.nodes(downstream)
    glb_NID= graph.nodes[downstream].data['ind']

    train_idx = labeled_local_ids[:int(local_len * dblp_config.train_portion)]
    val_idx = labeled_local_ids[int(local_len * dblp_config.train_portion):
                        int(local_len * (dblp_config.val_portion + dblp_config.train_portion))]
    test_idx = labeled_local_ids[int(local_len * (dblp_config.val_portion + dblp_config.train_portion)):]
    gtrain_idx =glb_NID[:int(local_len * dblp_config.train_portion)]
    gval_idx = glb_NID[int(local_len * dblp_config.train_portion):
                                int(local_len * (dblp_config.val_portion + dblp_config.train_portion))]
    gtest_idx = glb_NID[int(local_len * (dblp_config.val_portion + dblp_config.train_portion)):]

    labels = graph.nodes[downstream].data['label']

    return train_idx,test_idx,val_idx,labels,gtrain_idx,gtest_idx,gval_idx



def set_gen_loaders(graph:DGLHeteroGraph,val=None):
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

    sampled_ids={ntype:graph.nodes(ntype)[:int(dblp_config.gen_train_portion*graph.num_nodes(ntype))] for ntype in dblp_config.private_types}

    gen_train_loader = dgl.dataloading.NodeDataLoader(
        graph, sampled_ids, sampler,
        batch_size=dblp_config.batch_size, shuffle=True, num_workers=0)


    if val is None:
        return gen_train_loader
    val_sampled_ids = {ntype: graph.nodes(ntype)[int(dblp_config.gen_train_portion * graph.num_nodes(ntype)):
                                             int((dblp_config.gen_train_portion+dblp_config.gen_val_portion)
                                                 * graph.num_nodes(ntype))] for ntype in graph.ntypes}

    gen_val_loader = dgl.dataloading.NodeDataLoader(
        graph, val_sampled_ids, sampler,
        batch_size=dblp_config.batch_size, shuffle=True, num_workers=0)

    return gen_train_loader,gen_val_loader

def set_loaders_light(graph:DGLHeteroGraph,downstream,train_idx,val_idx=None,batch_size=dblp_config.batch_size):
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

    train_loader = dgl.dataloading.NodeDataLoader(
        graph, {downstream:train_idx}, sampler,
        batch_size=batch_size, shuffle=True, num_workers=0)


    if val_idx is None:
        return train_loader
    else:
        val_loader = dgl.dataloading.NodeDataLoader(
            graph, {downstream:val_idx}, sampler,
            batch_size=dblp_config.batch_size, shuffle=True, num_workers=0)

        return train_loader,val_loader



def train_model(n_epochs,feat,model,train_loader,val_loader,labels_all,pub_node_ids,
                downstream,device='cpu',lr=None):

    print("start training...")
    if device!='cpu':
        model=model.cuda()
    optimizer = RAdam(model.parameters(), lr=lr, weight_decay=dblp_config.l2norm)
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        for i, (input_nodes, seeds, blocks) in enumerate(train_loader):
            if blocks[-1].number_of_dst_nodes(downstream)!=dblp_config.batch_size:
                continue
            seeds=th.tensor(seeds[downstream].numpy(),dtype=th.long).to(device)
            h={k:feat[k][th.tensor(input_nodes[k].data,dtype=th.long)].to(device) for k in input_nodes.keys()}

            blocks = [blk.to(device)  for blk in blocks]
            labels=labels_all[seeds].to(device)

            for ntype in dblp_config.public_types:
                input_nodes[ntype]=pub_node_ids[ntype][th.tensor(input_nodes[ntype].data,dtype=th.long)]
            input_nodes={k:input_nodes[k].to(device) for k in input_nodes.keys()}
            out = model(input_nodes,blocks,h)
            logits=out[downstream]
            # print(seeds.cpu().numpy().reshape(-1))
            loss = F.cross_entropy(th.clip(logits,1e-9,1.0-1e-9), labels.long())
            train_acc = th.sum(logits.argmax(dim=1) == labels).item() / (len(seeds)+1e-10)
            print("Epoch {:05d} | Batch {:03d} | Train Acc: {:.4f} | Train Loss: {:.4f} ".
                  format(epoch, i, train_acc, loss.item()))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


        print("start evaluate ....")

        val_acc = evaluate(model=model, loader=val_loader,labels=labels_all,feat=feat,pub_node_ids=pub_node_ids,
                                     downstream=downstream,device=device)
        print("Epoch {:05d} | Valid Acc: {:.4f}".
              format(epoch, val_acc))
    print()

    return model


def evaluate(model,loader:dgl.dataloading.NodeDataLoader,pub_node_ids,feat,labels, downstream,device):
    model=model.to(device)
    model.eval()
    acc = model.inference(test_loader=loader,feat=feat,device=device,
                          labels=labels,pub_node_ids=pub_node_ids,downstream=downstream)
    return acc


def map_nid_to_id(graph:DGLHeteroGraph,nids:th.Tensor,downstream:str):
    if graph.device!='cpu':
        graph_NID = graph.nodes[downstream].data['ind'].cpu().numpy().reshape(-1)
    else:
        graph_NID = graph.nodes[downstream].data['ind'].numpy().reshape(-1)
    ids=[]
    for i,n in enumerate(nids):
        ind =np.argwhere(graph_NID==n.item()).reshape(-1)[0]
        ids.append(ind)
    ids_tensor=th.tensor(ids,dtype=th.int32).view(-1)
    return ids_tensor


