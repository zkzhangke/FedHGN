import torch as th
from utils import med_config
import numpy as np
import torch.nn.functional as F


def greedy_loss(pred_feats, true_feats,pred_missing,true_missing,node_feat_dim,device):
    pred_feats = pred_feats.view(med_config.num_pred,node_feat_dim)
    true_missing_np = np.clip(true_missing,a_min=0,a_max=med_config.num_pred)
    if device!='cpu':
        pred_missing_np = th.round(pred_missing).cpu().detach().numpy()
        pred_missing_np = np.clip(pred_missing_np,0,med_config.num_pred).astype(dtype=float)
    else:
        pred_missing_np = th.round(pred_missing).detach().numpy()
        pred_missing_np = np.clip(pred_missing_np,0,med_config.num_pred).astype(dtype=float)
    loss=th.zeros(pred_feats.shape)
    if device!='cpu':
        loss=loss.cuda()
    true_missing_np=int(true_missing_np)
    pred_missing_np=int(pred_missing_np)
    if true_missing_np> 0 and pred_missing_np>0:
        for pred_j in range(pred_missing_np):
            if isinstance(true_feats[0], np.ndarray):
                true_feats_tensor = th.tensor(true_feats[0])
            else:
                true_feats_tensor = true_feats[0]
            if device!='cpu':
                true_feats_tensor = true_feats_tensor.cuda()

            loss[pred_j]+= F.mse_loss(pred_feats[pred_j].unsqueeze(0).float(),
                                  true_feats_tensor.unsqueeze(0).float()).squeeze(0)
            for true_k in range(true_missing_np):

                if isinstance(true_feats[true_k], np.ndarray):
                    true_feats_tensor = th.tensor(true_feats[true_k])

                else:
                    true_feats_tensor = true_feats[true_k]
                if device!='cpu':
                    true_feats_tensor = true_feats_tensor.cuda()

                loss_jk = F.mse_loss(pred_feats[pred_j].unsqueeze(0).float(),
                                      true_feats_tensor.unsqueeze(0).float()).squeeze(0)
                if th.sum(loss_jk) < th.sum(loss[pred_j].data):
                    loss[pred_j] = loss_jk
    elif true_missing_np == 0 and pred_missing_np==0:
        return None
    elif true_missing_np== 0 :
        cur_pred_feat=pred_feats[:pred_missing_np].float().view(pred_missing_np,node_feat_dim)
        loss[:pred_missing_np]+= F.mse_loss(cur_pred_feat,
                             th.zeros(cur_pred_feat.shape,device=device).float()).squeeze(0)

    elif pred_missing_np==0:
        cur_pred_feat=pred_feats[:true_missing_np].view(true_missing_np,node_feat_dim)
        if isinstance(true_feats[:true_missing_np], np.ndarray):
            true_feats_tensor = th.tensor(true_feats[:true_missing_np])
        else:
            true_feats_tensor = true_feats[:true_missing_np]
        if device!='cpu':
            true_feats_tensor = true_feats_tensor.cuda()
        loss[:true_missing_np]+= F.mse_loss(cur_pred_feat.float(),
                                         true_feats_tensor.float().view(true_missing_np,node_feat_dim)).squeeze(0)

    return loss




