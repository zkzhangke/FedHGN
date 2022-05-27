from utils import dblp_config
from train.radam import RAdam
import torch as th
import torch.nn.functional as F
from train.loss_functions import greedy_loss




def fed_train_tngen(num_owners,models:list,n_epochs:int,fed_feat:dict,local_epochs:int,
                       subgraph_prime_all:list,pub_node_ids_all:dict,device,fed_gen_lr,
                       gen_train_loader_all:list,gen_labels:dict,downstream:str):
    print("start fed training tngen...")
    for epoch in range(n_epochs):
        for owner_i in range(num_owners):
            model = models[owner_i].to(device)
            pub_node_ids=pub_node_ids_all[owner_i]
            subgraph_prime_i=subgraph_prime_all[owner_i]
            class_labels = subgraph_prime_i.nodes[downstream].data['label'].to(device)
            feat_i=fed_feat[owner_i]
            gen_train_loader_i=gen_train_loader_all[owner_i]
            node_gen_labels=gen_labels[owner_i]
            optimizer = RAdam(model.parameters(), lr=fed_gen_lr,weight_decay=dblp_config.weight_decay)
            for epoch_l in range(local_epochs):
                model.train()
                optimizer.zero_grad()
                for i, (input_nodes, seeds, blocks) in enumerate(gen_train_loader_i):
                    emb={k:feat_i[k][th.tensor(input_nodes[k].data,dtype=th.long)] for k in input_nodes.keys()}
                    seeds={ntype:seeds[ntype] for ntype in dblp_config.private_types}
                    pred_missing,pred_feat,pred_class,out_nodes =model(g=subgraph_prime_i,cur_nodes=seeds,
                                                                       pub_node_ids=pub_node_ids,
                                                                       downstream=downstream,h=emb,blocks=blocks)

                    countd=0
                    countf=0
                    pred_missing_all=[]
                    true_num_all=[]
                    pred_feat_all=[]
                    true_feats_all=[]

                    for nty in list(pred_missing.keys()):
                        for s_i in seeds[nty]:
                            for e_i in dblp_config.pred_in_n_rev_etypes.keys():
                                if e_i[-1]==nty:
                                    countd+=1
                                    countf+=1
                                    true_num_all.append(node_gen_labels[nty][s_i.item()]['num'][e_i])
                                    pred_missing_all.append(pred_missing[nty][s_i.item()][e_i])
                                    pred_feat_all.append(pred_feat[nty][s_i.item()][e_i])
                                    true_feats_all.append(node_gen_labels[nty][s_i.item()]['feat'][e_i])
                    lossd= F.smooth_l1_loss(th.stack(pred_missing_all).view(-1),
                                            th.tensor(true_num_all,dtype=th.float,device=device))


                    greedy = greedy_loss(pred_feats=pred_feat_all,
                                         true_feats=true_feats_all,
                                         pred_missing=pred_missing_all,
                                         true_missing=true_num_all,
                                         device=device)

                    if pred_class is not None:
                        if out_nodes[downstream].device=='cpu':
                            indx=th.tensor(out_nodes[downstream].numpy(),dtype=th.long)
                        else:
                            indx=th.tensor(out_nodes[downstream].cpu().numpy(),dtype=th.long)
                        loss_class = F.cross_entropy(th.clip(pred_class[downstream][:len(indx)],1e-9,1.0-1e-9), class_labels[indx].long())
                        train_acc = th.sum(pred_class[downstream][:len(indx)].argmax(dim=1) == class_labels[indx]).item() / len(indx)
                        print("Epoch {:05d} | Batch {:03d} | Lossd: {:.4f} | Greedy: {:.4f} | AccC: {:.4f}| LossC: {:.4f}".
                              format(epoch,  i, lossd.mean().data,greedy.mean().item(), train_acc,loss_class.mean().item()))
                        (dblp_config.a*lossd.mean()+dblp_config.b*greedy.mean()+dblp_config.c*loss_class).backward()
                    else:
                        print("Epoch {:05d} | Batch {:03d} | Lossd: {:.4f} | Greedy: {:.4f} ".
                              format(epoch,  i, lossd.mean().data,greedy.mean().item()))
                        (dblp_config.a*lossd.mean()+dblp_config.b*greedy.mean()).backward()
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
                        pub_node_ids_j=pub_node_ids_all[j]
                        model_j=models[owner_i].to(device)
                        model_j.requires_grad_(False)
                        model_j.tngen.dGen.requires_grad_(True)
                        model_j.tngen.fGen.requires_grad_(True)
                        optimizer_j = RAdam(model_j.parameters(), lr=fed_gen_lr,weight_decay=dblp_config.weight_decay)
                        optimizer_j.zero_grad()
                        for i, (input_nodes_j, seeds_j, blocks_j) in enumerate(gen_train_loader_j):
                            emb={k:feat_j[k][th.tensor(input_nodes_j[k].data,dtype=th.long)] for k in input_nodes_j.keys()}
                            seeds_j={ntype:seeds_j[ntype] for ntype in dblp_config.private_types}
                            pred_missing_j,pred_feat_j,pred_class_j,out_nodes_j =model_j(g=subgraph_prime_j,cur_nodes=seeds_j,
                                                                               pub_node_ids=pub_node_ids_j,
                                                                               downstream=downstream,h=emb,blocks=blocks_j)


                            countd_j=0
                            countf_j=0
                            pred_missing_j_all=[]
                            true_num_j_all=[]
                            pred_feat_j_all=[]
                            true_feats_j_all=[]

                            for nty in list(pred_missing_j.keys()):
                                for s_i in seeds_j[nty]:
                                    for e_i in dblp_config.pred_in_n_rev_etypes.keys():
                                        if e_i[-1]==nty:
                                            countd_j+=1
                                            countf_j+=1
                                            true_num_j_all.append(node_gen_labels_j[nty][s_i.item()]['num'][e_i])
                                            pred_missing_j_all.append(pred_missing_j[nty][s_i.item()][e_i])
                                            pred_feat_j_all.append(pred_feat_j[nty][s_i.item()][e_i])
                                            true_feats_j_all.append(node_gen_labels_j[nty][s_i.item()]['feat'][e_i])
                            lossd_j= dblp_config.b_fl*F.smooth_l1_loss(th.stack(pred_missing_j_all).view(-1),
                                                                  th.tensor(true_num_j_all,dtype=th.float,device=device))


                            greedy_j= dblp_config.b_fl*greedy_loss(pred_feats=pred_feat_j_all,
                                                          pred_missing=pred_missing_j_all,
                                                          true_feats=true_feats_j_all,
                                                          true_missing=true_num_j_all,
                                                              device=device)

                            if pred_class_j is not None:
                                if out_nodes_j[downstream].device=='cpu':
                                    indx=th.tensor(out_nodes_j[downstream].numpy(),dtype=th.long)
                                else:
                                    indx=th.tensor(out_nodes_j[downstream].cpu().numpy(),dtype=th.long)
                                loss_class_j = dblp_config.b_fl*F.cross_entropy(th.clip(pred_class_j[downstream][:len(indx)],1e-9,1.0-1e-9), class_labels_j[indx].long())
                                train_acc_j = th.sum(pred_class_j[downstream][:len(indx)].argmax(dim=1) == class_labels_j[indx]).item() / len(indx)
                                print("Owner {:02d}| From {:02d}| Epoch {:05d} |Fd: {:.4f} | Fg: {:.4f} | FAcc: {:.4f}| Fc: {:.4f} ".
                                      format(owner_i,j,epoch_l, lossd_j.mean().data,greedy_j.mean().item(), train_acc_j,loss_class_j.mean().item()))
                                (dblp_config.a*lossd_j.mean()+dblp_config.b*greedy_j.mean()+dblp_config.c*loss_class_j).backward()
                            else:
                                print("Owner {:02d}| From {:02d}| Epoch {:05d} |Fd: {:.4f} | Fg: {:.4f} ".
                                      format(owner_i,j,epoch_l, lossd_j.mean().data,greedy_j.mean().item()))
                                (dblp_config.a*lossd_j.mean()+dblp_config.b*greedy_j.mean()).backward()

                            optimizer_j.step()
                            optimizer_j.zero_grad()
                        w_list.append(model_j.tngen.state_dict())
                weights={k:1.0/num_owners*w_list[0][k] for k in w_list[0].keys()}
                for w_j in w_list[1:]:
                    weights={k:weights[k]+1.0/num_owners*w_j[k] for k in weights.keys()}
                model.tngen.load_state_dict(weights)
            models[owner_i]=model
    return models

