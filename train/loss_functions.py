import torch as th
from utils import dblp_config
import numpy as np
import torch.nn.functional as F


def greedy_loss(pred_feats, true_feats,pred_missing,true_missing,device):
    pred_feats = th.stack(pred_feats).view(-1,dblp_config.num_pred, dblp_config.feat_len)

    true_missing_np = np.clip(a=np.stack(true_missing).reshape(-1),a_min=0,a_max=dblp_config.num_pred)
    if device!='cpu':
        pred_missing_np = th.round(th.stack(pred_missing).view(-1)).cpu().detach().numpy()
        pred_missing_np = np.clip(pred_missing_np,0,dblp_config.num_pred).astype(dtype=float)
    else:
        pred_missing_np = th.round(th.stack(pred_missing).view(-1)).detach().numpy()
        pred_missing_np = np.clip(pred_missing_np,0,dblp_config.num_pred).astype(dtype=float)

    loss=th.zeros(pred_feats.shape)
    if device!='cpu':
        loss=loss.cuda()
    for i,true_missing,pred_missing in zip(range(len(true_missing_np)),true_missing_np,pred_missing_np):
        true_missing=int(true_missing)
        pred_missing=int(pred_missing)
        if true_missing> 0 and pred_missing>0:
            for pred_j in range(pred_missing):
                if isinstance(true_feats[i][0], np.ndarray):
                    true_feats_tensor = th.tensor(true_feats[i][0])
                else:
                    true_feats_tensor = true_feats[i][0]
                if device!='cpu':
                    true_feats_tensor = true_feats_tensor.cuda()

                loss[pred_j]+= F.mse_loss(pred_feats[i][pred_j].unsqueeze(0).float(),
                                      true_feats_tensor.unsqueeze(0).float()).squeeze(0)
                for true_k in range(true_missing):

                    if isinstance(true_feats[i][true_k], np.ndarray):
                        true_feats_tensor = th.tensor(true_feats[i][true_k])

                    else:
                        true_feats_tensor = true_feats[i][true_k]
                    if device!='cpu':
                        true_feats_tensor = true_feats_tensor.cuda()

                    loss_jk = F.mse_loss(pred_feats[i][pred_j].unsqueeze(0).float(),
                                          true_feats_tensor.unsqueeze(0).float()).squeeze(0)
                    if th.sum(loss_jk) < th.sum(loss[pred_j].data):
                        loss[pred_j] = loss_jk
        elif true_missing == 0 and pred_missing==0:
            return loss
        elif true_missing== 0 :
            cur_pred_feat=pred_feats[i][:pred_missing].float().view(pred_missing,dblp_config.feat_len)
            loss[:pred_missing]+= F.mse_loss(cur_pred_feat,
                                 th.zeros(cur_pred_feat.shape,device=device).float()).squeeze(0)

        elif pred_missing==0:
            cur_pred_feat=pred_feats[i][:true_missing].view(true_missing,dblp_config.feat_len)
            if isinstance(true_feats[i][:true_missing], np.ndarray):
                true_feats_tensor = th.tensor(true_feats[i][:true_missing])
            else:
                true_feats_tensor = true_feats[i][:true_missing]
            if device!='cpu':
                true_feats_tensor = true_feats_tensor.cuda()
            loss[:true_missing]+= F.mse_loss(cur_pred_feat.float(),
                                             true_feats_tensor.float().view(true_missing,dblp_config.feat_len)).squeeze(0)

    return loss




