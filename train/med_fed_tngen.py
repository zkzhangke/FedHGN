from utils import med_config
from train.radam import RAdam
import torch as th
import torch.nn.functional as F
from train.med_loss_functions import greedy_loss

def fed_train_tngen(num_owners,models:list,n_epochs:int,fed_feat:dict,local_epochs:int,
                       subgraph_prime_all:list,device,fed_gen_lr,node_feat_dim,
                       gen_train_loader_all:list,gen_labels:dict,
                       downstream:str):
    print("start fed training tngen...")
    for epoch in range(n_epochs):
        for owner_i in range(num_owners):
            model = models[owner_i].to(device)
            model.requires_grad_(True)
            subgraph_prime_i=subgraph_prime_all[owner_i]
            class_labels = subgraph_prime_i.nodes[downstream].data['label'].to(device)
            feat_i=fed_feat[owner_i]
            gen_train_loader_i=gen_train_loader_all[owner_i]
            node_gen_labels=gen_labels[owner_i]
            optimizer = RAdam(model.parameters(), lr=fed_gen_lr,weight_decay=med_config.weight_decay)
            for epoch_l in range(local_epochs):
                model.train()
                optimizer.zero_grad()
                for i, (input_nodes, seeds, blocks) in enumerate(gen_train_loader_i):
                    emb={k:feat_i[k][th.tensor(input_nodes[k].data,dtype=th.long)] for k in input_nodes.keys()}
                    seeds={ntype:seeds[ntype] for ntype in med_config.private_types}
                    pred_missing,pred_feat,pred_class,out_nodes =model(g=subgraph_prime_i,cur_nodes=seeds,
                                                                       downstream=downstream,h=emb,blocks=blocks)

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
                                        print("Epoch {:05d} | Batch {:03d} | Lossd: {:.4f} | Greedy: {:.4f}".
                                              format(epoch,  i, lossd.mean().data,greedy.mean().item()))
                                        (med_config.a*lossd.mean()+med_config.b*greedy.mean()).backward(retain_graph=True)

                    if pred_class is not None and len(out_nodes[downstream])>0:
                        if out_nodes[downstream].device=='cpu':
                            indx=th.tensor(out_nodes[downstream].numpy(),dtype=th.long)
                        else:
                            indx=th.tensor(out_nodes[downstream].cpu().numpy(),dtype=th.long)
                        loss_class = F.cross_entropy(th.clip(pred_class[downstream][:len(indx)],1e-9,1.0-1e-9), class_labels[indx].long())
                        if loss_class.requires_grad:
                            train_acc = th.sum(pred_class[downstream][:len(indx)].argmax(dim=1) == class_labels[indx]).item() / len(indx)
                            print("Epoch {:05d} | Batch {:03d} | AccC: {:.4f}| LossC: {:.4f}".
                                  format(epoch,  i, train_acc,loss_class.mean().item()))
                            (med_config.c*loss_class).backward()

                    optimizer.step()
                    optimizer.zero_grad()
                w_list=[model.tngen.state_dict()]
                for j in range(num_owners):
                    if j != owner_i:
                        node_gen_labels_j=gen_labels[j]
                        feat_j=fed_feat[j]
                        gen_train_loader_j=gen_train_loader_all[j]
                        subgraph_prime_j=subgraph_prime_all[j]
                        class_labels_j = subgraph_prime_j.nodes[downstream].data['label'].to(device)
                        model_j=models[owner_i].to(device)
                        model_j.requires_grad_(False)
                        model_j.tngen.dGen.requires_grad_(True)
                        model_j.tngen.fGen.requires_grad_(True)
                        optimizer_j = RAdam(model_j.parameters(), lr=fed_gen_lr,weight_decay=med_config.weight_decay)
                        optimizer_j.zero_grad()
                        for i, (input_nodes_j, seeds_j, blocks_j) in enumerate(gen_train_loader_j):
                            emb={k:feat_j[k][th.tensor(input_nodes_j[k].data,dtype=th.long)] for k in input_nodes_j.keys()}
                            seeds_j={ntype:seeds_j[ntype] for ntype in med_config.private_types}
                            pred_missing_j,pred_feat_j,pred_class_j,out_nodes_j =model_j(g=subgraph_prime_j,cur_nodes=seeds_j,
                                                                               downstream=downstream,h=emb,blocks=blocks_j)
                            for nty in list(pred_missing_j.keys()):
                                for s_i in seeds_j[nty]:
                                    for e_i in med_config.pred_in_n_rev_etypes.keys():
                                        if e_i[-1]==nty:
                                            true_num_nty_j=node_gen_labels_j[nty][s_i.item()]['num'][e_i]
                                            pred_missing_nty_j=pred_missing_j[nty][s_i.item()][e_i]
                                            pred_feat_nty_j=pred_feat_j[nty][s_i.item()][e_i]
                                            true_feats_nty_j=node_gen_labels_j[nty][s_i.item()]['feat'][e_i]
                                            lossd_j= F.smooth_l1_loss(th.tensor([pred_missing_nty_j],device=device).view(-1),
                                                                    th.tensor([true_num_nty_j],dtype=th.float,device=device))
                                            greedy_j = greedy_loss(pred_feats=pred_feat_nty_j,
                                                                 true_feats=true_feats_nty_j,
                                                                 pred_missing=pred_missing_nty_j,
                                                                 node_feat_dim=node_feat_dim[e_i[0]],
                                                                 true_missing=true_num_nty_j,device=device)
                                            if greedy_j is not None:
                                                print("Owner {:02d}| From {:02d}| Epoch {:05d} |Fd: {:.4f} | Fg: {:.4f}".
                                                      format(owner_i,j,epoch_l, lossd_j.mean().data,greedy_j.mean().item()))
                                                (med_config.b_fl*med_config.a*lossd_j.mean()+med_config.b_fl*med_config.b*greedy_j.mean()).backward(retain_graph=True)

                            if pred_class_j is not None and len(out_nodes_j[downstream])>0:
                                if out_nodes_j[downstream].device=='cpu':
                                    indx=th.tensor(out_nodes_j[downstream].numpy(),dtype=th.long)
                                else:
                                    indx=th.tensor(out_nodes_j[downstream].cpu().numpy(),dtype=th.long)
                                loss_class_j = med_config.b_fl*F.cross_entropy(th.clip(pred_class_j[downstream][:len(indx)],1e-9,1.0-1e-9), class_labels_j[indx].long())
                                if loss_class_j.requires_grad:

                                    train_acc_j = th.sum(pred_class_j[downstream][:len(indx)].argmax(dim=1) == class_labels_j[indx]).item() / len(indx)
                                    print("Owner {:02d}| From {:02d}| Epoch {:05d} | FAcc: {:.4f}| Fc: {:.4f}".
                                          format(owner_i,j,epoch_l, train_acc_j,loss_class_j.mean().item()))
                                    (med_config.b_fl*med_config.c*loss_class_j).backward()

                            optimizer_j.step()
                            optimizer_j.zero_grad()
                        w_list.append(model_j.tngen.state_dict())
                weights={k:1.0/num_owners*w_list[0][k] for k in w_list[0].keys()}
                for w_j in w_list[1:]:
                    weights={k:weights[k]+1.0/num_owners*w_j[k] for k in weights.keys()}
                model.tngen.load_state_dict(weights)

            models[owner_i]=model

    return models