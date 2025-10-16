import torch
from torch import nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=0.5, flag=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.flag = flag

    def forward(self, anchor, anchor_phenotype, s, s_phenotype, t, t_phenotype):
        if self.flag:
            positive, positive_phenotype, negative, negative_phenotype = s, s_phenotype, t, t_phenotype
        else:
            positive, positive_phenotype, negative, negative_phenotype = self._set_pos_neg(anchor_phenotype, s, s_phenotype, t, t_phenotype)
        loss = self._loss(anchor, anchor_phenotype, positive, positive_phenotype, negative, negative_phenotype)
        return loss


    def _loss(self,anchor, anchor_phenotype, positive, positive_phenotype, negative, negative_phenotype):
        r = torch.abs(anchor_phenotype - negative_phenotype) / (torch.abs(anchor_phenotype - positive_phenotype) + 1e-2)
        margin = torch.clip(self.margin * r, 0.1, 0.9)
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)
        d_pos = F.pairwise_distance(anchor, positive, p=2)
        d_neg = F.pairwise_distance(anchor, negative, p=2)
        losses = F.softplus(d_pos - d_neg + margin)
        return losses.mean()

    def _set_pos_neg(self, anchor_phenotype, s, s_phenotype, t, t_phenotype):
        pos_mask = (torch.abs(anchor_phenotype - s_phenotype) <= torch.abs(anchor_phenotype - t_phenotype))
        positive, positive_phenotype = torch.where(pos_mask, s, t), torch.where(pos_mask, s_phenotype, t_phenotype)
        negative, negative_phenotype = torch.where(pos_mask, t, s), torch.where(pos_mask, t_phenotype, s_phenotype)
        return positive, positive_phenotype, negative, negative_phenotype


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, pred, true):
        return F.l1_loss(pred, true)