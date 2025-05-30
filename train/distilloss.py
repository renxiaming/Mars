import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import override # this could be removed since Python 3.12
from .loss import DetectionLoss


class CWDLoss(nn.Module):
    """Channel-wise Distillation Loss"""
    def __init__(self, device):
        super().__init__()
        self.device = device
        
    def forward(self, s_feats, t_feats):
        """
        s_feats: student features
        t_feats: teacher features
        """
        loss = 0
        for s_feat, t_feat in zip(s_feats, t_feats):
            # Normalize features
            s_feat = F.normalize(s_feat, dim=1)
            t_feat = F.normalize(t_feat, dim=1)
            # Compute channel-wise distillation loss
            loss += F.mse_loss(s_feat, t_feat)
        return loss / len(s_feats)


class ResponseLoss(nn.Module):
    """Response-based Distillation Loss"""
    def __init__(self, device, nc, teacher_class_indexes):
        super().__init__()
        self.device = device
        self.nc = nc
        self.teacher_class_indexes = teacher_class_indexes
        
    def forward(self, s_response, t_response):
        """
        s_response: student response (predictions)
        t_response: teacher response (predictions)
        """
        loss = 0
        for s_pred, t_pred in zip(s_response, t_response):
            # Extract class scores
            s_cls = s_pred[:, -self.nc:, :, :]  # student class scores
            t_cls = t_pred[:, -self.nc:, :, :]  # teacher class scores
            
            # Only compute loss for teacher classes (old classes)
            t_cls_selected = t_cls[:, self.teacher_class_indexes, :, :]
            s_cls_selected = s_cls[:, self.teacher_class_indexes, :, :]
            
            # Apply softmax and compute KL divergence
            t_soft = F.softmax(t_cls_selected / 4.0, dim=1)  # temperature=4
            s_log_soft = F.log_softmax(s_cls_selected / 4.0, dim=1)
            
            loss += F.kl_div(s_log_soft, t_soft, reduction='batchmean') * (4.0 ** 2)
            
        return loss / len(s_response)


class DistillationDetectionLoss(object):
    def __init__(self, mcfg, model):
        self.mcfg = mcfg
        self.histMode = False
        self.detectionLoss = DetectionLoss(mcfg, model)
        self.cwdLoss = CWDLoss(self.mcfg.device)
        self.respLoss = ResponseLoss(self.mcfg.device, self.mcfg.nc, self.mcfg.teacherClassIndexes)

    @override
    def __call__(self, rawPreds, batch):
        """
        rawPreds[0] & rawPreds[1] shape: (
            (B, regMax * 4 + nc, 80, 80),
            (B, regMax * 4 + nc, 40, 40),
            (B, regMax * 4 + nc, 20, 20),
            (B, 128 * w, 160, 160),
            (B, 256 * w, 80, 80),
            (B, 512 * w, 40, 40),
            (B, 512 * w * r, 20, 20),
            (B, 512 * w, 40, 40),
            (B, 256 * w, 80, 80),
            (B, 512 * w, 40, 40),
            (B, 512 * w * r, 20, 20),
        )
        """
        spreds = rawPreds[0]
        tpreds = rawPreds[1]

        sresponse, sfeats = spreds[:3], spreds[3:]
        tresponse, tfeats = tpreds[:3], tpreds[3:]

        loss = torch.zeros(3, device=self.mcfg.device)  # original, cwd distillation, response distillation
        loss[0] = self.detectionLoss(sresponse, batch) * self.mcfg.distilLossWeights[0]  # original
        loss[1] = self.cwdLoss(sfeats, tfeats) * self.mcfg.distilLossWeights[1]  # cwd distillation
        loss[2] = self.respLoss(sresponse, tresponse) * self.mcfg.distilLossWeights[2]  # response distillation

        return loss.sum()
