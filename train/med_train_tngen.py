import torch as th
from utils import med_config
import torch.nn.functional as F
from models.med_gen_model import TNGen,MendGraph,HLocSagePlus
from train.med_loss_functions import greedy_loss
from models.med_hete_models import TGCN
from train.med_mb_eval_rgcn import set_loaders_light,train_model
from utils.med_test_classifier import test_clsfier
from train.radam import RAdam


def train_locsageplus(n_epochs,model:HLocSagePlus,feat:dict,gen_train_loader,gen_lr,node_feat_dim,
                      device,graph,downstream,class_labels,node_gen_labels):
    if device!='cpu':
        model=model.cuda()
    optimizer = RAdam(model.parameters(), lr=gen_lr, weight_decay=med_config.l2norm)
    class_labels=class_labels.to(device)
    print("start training tngen...")

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        for i, (input_nodes, seeds, blocks) in enumerate(gen_train_loader):
            emb={k:feat[k][th.tensor(input_nodes[k].data,dtype=th.long)] for k in input_nodes.keys()}

            pred_missing,pred_feat,pred_class,out_nodes =model(g=graph,cur_nodes=seeds,
                                                               downstream=downstream,
                                                               h=emb,blocks=blocks)

            for nty in list(pred_missing.keys()):
                for s_i in seeds[nty]:
                    for e_i in med_config.pred_in_n_rev_etypes.keys():
                        if e_i[-1]==nty:
                            true_num_nty=node_gen_labels[nty][s_i.item()]['num'][e_i]
                            pred_missing_nty=pred_missing[nty][s_i.item()][e_i]
                            pred_feat_nty=pred_feat[nty][s_i.item()][e_i]
                            true_feats_nty=node_gen_labels[nty][s_i.item()]['feat'][e_i]
                            lossd= F.smooth_l1_loss(th.tensor([pred_missing_nty],device=device).view(-1),
                                                    th.tensor([true_num_nty],dtype=th.float,device=device))
                            greedy = greedy_loss(pred_feats=pred_feat_nty,
                                             true_feats=true_feats_nty,
                                             pred_missing=pred_missing_nty,
                                                 node_feat_dim=node_feat_dim[e_i[0]],
                                             true_missing=true_num_nty,device=device)
                            if greedy is not None:
                                print("Epoch {:05d} | Batch {:03d} | Lossd: {:.4f} | Greedy: {:.4f} ".
                                      format(epoch,  i, lossd.mean().data,greedy.mean().item()))
                                (med_config.a*lossd.mean()+med_config.b*greedy.mean()).backward(retain_graph=True)


            if pred_class is not None:
                if device=='cpu':
                    indx=th.tensor(out_nodes[downstream].numpy(),dtype=th.long,device=device)
                else:
                    indx=th.tensor(out_nodes[downstream].cpu().numpy(),dtype=th.long,device=device)
                loss_class = F.cross_entropy(th.clip(pred_class[downstream][:len(indx)],1e-9,1.0-1e-9), class_labels[indx].long())
                train_acc = th.sum(pred_class[downstream][:len(indx)].argmax(dim=1) == class_labels[indx]).item() / len(indx)
                print("Epoch {:05d} | Batch {:03d} | AccC: {:.4f}| LossC: {:.4f}".
                      format(epoch,  i,train_acc,loss_class.mean().item()))
                (med_config.c*loss_class).backward()


            optimizer.step()
            optimizer.zero_grad()


    return model





def get_mended_graph(gen_model:TNGen,graph,gen_portion,device):
    gen_model.eval()
    feat={ntype:graph.nodes[ntype].data['feat'] for ntype in graph.ntypes}
    pred_missing,pred_feat = gen_model.inference(g=graph, batch_size=med_config.batch_size,x=feat,device=device)
    mend_model=MendGraph(gen_portion=gen_portion,device='cpu')
    cur_node={ntype:graph.nodes(ntype) for ntype in med_config.private_types}
    mended_graph=mend_model.inference(graph=graph,cur_node=cur_node,pred_num=pred_missing,pred_feat=pred_feat)
    mended_graph=mended_graph.to("cpu")
    for ntype in mended_graph.ntypes:
        mended_graph.nodes[ntype].data['feat']=th.tensor(mended_graph.nodes[ntype].data['feat'].data,device="cpu",requires_grad=False)
    return mended_graph

def test_gen_model(gen_model:TNGen,graph,downstream,gen_portion,
                   n_epochs,train_idx,val_idx,labels_train,num_classes,num_owners,gencls_lr,
                   device,node_feat_dim,
                   global_graph=None,pref=''):
    gen_model.eval()
    gen_model.to(device)
    subg_feat={ntype:graph.nodes[ntype].data['feat'] for ntype in graph.ntypes}
    pred_missing,pred_feat = gen_model.inference(g=graph, batch_size=med_config.batch_size,x=subg_feat,device='cpu')
    mend_model=MendGraph(gen_portion=gen_portion,device='cpu')
    mend_model.eval()
    cur_node={ntype:graph.nodes(ntype) for ntype in med_config.private_types}
    mended_graph=mend_model.inference(graph=graph,cur_node=cur_node,pred_num=pred_missing,pred_feat=pred_feat)

    mendg_feat={ntype:th.tensor(mended_graph.nodes[ntype].data['feat'].data,requires_grad=False)
                for ntype in mended_graph.ntypes}
    train_loader,val_loader=set_loaders_light(mended_graph,downstream,train_idx,val_idx)
    etypes=mended_graph.etypes

    model_test = TGCN(etypes=etypes,ntypes=mended_graph.ntypes,
                             h_dim=med_config.hidden,feat_size=med_config.feat_len,
                                num_classes=num_classes,
                                device=device,
                                node_feat_dim=node_feat_dim,
                             num_bases=med_config.n_bases,
                            num_conv_layers=med_config.n_layers,
                             dropout=med_config.dropout,
                             use_self_loop=med_config.use_self_loop)

    print("start training classifier on mended graph...")
    model_test=train_model(n_epochs=n_epochs,feat=mendg_feat,model=model_test,train_loader=train_loader,
                val_loader=val_loader,labels_all=labels_train,
                downstream=downstream,device=device,lr=gencls_lr)

    print()

    if global_graph is not None:
        test_clsfier(num_owners=num_owners,classfier=model_test,demo_graph=global_graph,downstream=downstream,
                     device=device,pref=pref)
    return model_test