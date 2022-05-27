import torch as th
from utils import dblp_config
import torch.nn.functional as F
from models.gen_model import TNGen,MendGraph,HLocSagePlus
from train.loss_functions import greedy_loss
from models.hete_models import TGCN
from train.mb_eval_rgcn import set_loaders_light,train_model
from utils.test_classifier import test_clsfier
from train.radam import RAdam


def train_locsageplus(n_epochs,model:HLocSagePlus,feat:dict,gen_train_loader,pub_node_ids,gen_lr,
                      device,graph,downstream,class_labels,node_gen_labels):
    if dblp_config.cuda:
        model=model.cuda()
    optimizer = RAdam(model.parameters(), lr=gen_lr, weight_decay=dblp_config.l2norm)
    class_labels=class_labels.to(device)
    print("start training tngen...")

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        for i, (input_nodes, seeds, blocks) in enumerate(gen_train_loader):
            emb={k:feat[k][th.tensor(input_nodes[k].data,dtype=th.long).to(device)] for k in input_nodes.keys()}

            pred_missing,pred_feat,pred_class,out_nodes =model(g=graph,cur_nodes=seeds,
                                                               pub_node_ids=pub_node_ids,downstream=downstream,
                                                               h=emb,blocks=blocks)


            count=0
            pred_missing_all=[]
            true_num_all=[]
            pred_feat_all=[]
            true_feats_all=[]

            for nty in list(pred_missing.keys()):
                for s_i in seeds[nty]:
                    for e_i in dblp_config.pred_in_n_rev_etypes.keys():
                        if e_i[-1]==nty:
                            count+=1
                            true_num_all.append(node_gen_labels[nty][s_i.item()]['num'][e_i])
                            pred_missing_all.append(pred_missing[nty][s_i.item()][e_i])
                            pred_feat_all.append(pred_feat[nty][s_i.item()][e_i])
                            true_feats_all.append(node_gen_labels[nty][s_i.item()]['feat'][e_i])
            lossd= F.smooth_l1_loss(th.stack(pred_missing_all).view(-1),th.tensor(true_num_all,dtype=th.float,device=device))


            greedy = greedy_loss(pred_feats=pred_feat_all,
                                 true_feats=true_feats_all,
                                 pred_missing=pred_missing_all,
                                 true_missing=true_num_all,device=device)


            if pred_class is not None:
                if device=='cpu':
                    indx=th.tensor(out_nodes[downstream].numpy(),dtype=th.long,device=device)
                else:
                    indx=th.tensor(out_nodes[downstream].cpu().numpy(),dtype=th.long,device=device)
                loss_class = F.cross_entropy(th.clip(pred_class[downstream][:len(indx)],1e-9,1.0-1e-9), class_labels[indx].long())
                train_acc = th.sum(pred_class[downstream][:len(indx)].argmax(dim=1) == class_labels[indx]).item() / len(indx)
                print("Epoch {:05d} | Batch {:03d} | Lossd: {:.4f} | Greedy: {:.4f} | AccC: {:.4f}| LossC: {:.4f} ".
                      format(epoch,  i, lossd.mean().data,greedy.mean().item(), train_acc,loss_class.mean().item()))
                (dblp_config.a*lossd.mean()+dblp_config.b*greedy.mean()+dblp_config.c*loss_class).backward()
            else:
                print("Epoch {:05d} | Batch {:03d} | Lossd: {:.4f} | Greedy: {:.4f} ".
                      format(epoch,  i, lossd.mean().data,greedy.mean().item()))
                (dblp_config.a*lossd.mean()+dblp_config.b*greedy.mean()).backward()


            optimizer.step()
            optimizer.zero_grad()



    return model



def get_mended_graph(gen_model:TNGen,graph,gen_portion,device):
    gen_model.eval()
    feat={ntype:graph.nodes[ntype].data['feat'] for ntype in graph.ntypes}
    pred_missing,pred_feat = gen_model.inference(g=graph, batch_size=dblp_config.batch_size,x=feat,device=device)
    mend_model=MendGraph(gen_portion=gen_portion,device='cpu')
    cur_node={ntype:graph.nodes(ntype) for ntype in dblp_config.private_types}
    mended_graph=mend_model.inference(graph=graph,cur_node=cur_node,pred_num=pred_missing,pred_feat=pred_feat)
    mended_graph=mended_graph.to("cpu")
    for ntype in mended_graph.ntypes:
        mended_graph.nodes[ntype].data['feat']=th.tensor(mended_graph.nodes[ntype].data['feat'].data,device="cpu",requires_grad=False)
    return mended_graph

def test_gen_model(gen_model:TNGen,graph,downstream,pub_node_num,pub_node_feats,gen_portion,
                   n_epochs,train_idx,val_idx,labels_train,num_classes,num_owners,gencls_lr,
                   pub_node_ids,device,
                   global_graph=None,pref=''):
    gen_model.eval()
    gen_model.to(device)
    subg_feat={ntype:graph.nodes[ntype].data['feat'] for ntype in graph.ntypes}
    pred_missing,pred_feat = gen_model.inference(g=graph, batch_size=dblp_config.batch_size,x=subg_feat,device='cpu')
    mend_model=MendGraph(gen_portion=gen_portion,device='cpu')
    mend_model.eval()
    cur_node={ntype:graph.nodes(ntype) for ntype in dblp_config.private_types}
    mended_graph=mend_model.inference(graph=graph,cur_node=cur_node,pred_num=pred_missing,pred_feat=pred_feat)

    mendg_feat={ntype:th.tensor(mended_graph.nodes[ntype].data['feat'].data,requires_grad=False)
                for ntype in mended_graph.ntypes}
    train_loader,val_loader=set_loaders_light(mended_graph,downstream,train_idx,val_idx)
    etypes=mended_graph.etypes

    model_test = TGCN(etypes=etypes,ntypes=mended_graph.ntypes,
                             h_dim=dblp_config.hidden,feat_size=dblp_config.feat_len,
                             pub_node_num=pub_node_num,pub_node_feats=pub_node_feats,
                                num_classes=num_classes,
                                device=device,
                             num_bases=dblp_config.n_bases,
                            num_conv_layers=dblp_config.n_layers,
                             dropout=dblp_config.dropout,
                             use_self_loop=dblp_config.use_self_loop)

    print("start training classifier on mended graph...")
    model_test=train_model(n_epochs=n_epochs,feat=mendg_feat,model=model_test,train_loader=train_loader,
                val_loader=val_loader,labels_all=labels_train,pub_node_ids=pub_node_ids,
                downstream=downstream,device=device,lr=gencls_lr)

    print()
    if global_graph is not None:
        test_clsfier(num_owners=num_owners,classfier=model_test,demo_graph=global_graph,downstream=downstream,
                    device=device,pref=pref)
    return model_test