# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
import torch.nn as nn
from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import LOSSES_REGISTRY

@LOSSES_REGISTRY.register()
class COCO_CROWD_CrossEntropy(nn.Module):
    @configurable
    def __init__(self):
        super(COCO_CROWD_CrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    @classmethod
    def from_config(cls, cfg):
        return {}

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, outputs_dict):
        ret  = {}

        logits1 = outputs_dict['END_G_LOGITS']
        targets1 = outputs_dict['G_TARGET_IDS']
        logits1 = logits1.view(-1, logits1.shape[-1])
        targets1 = targets1.view(-1).long()
        if targets1.sum() == -len(targets1):
            loss1 = 0.0
        else:
            loss1 = self.criterion(logits1, targets1)

        logits2 = outputs_dict['MID_G_LOGITS']
        targets2 = outputs_dict['MID_G_TARGET_IDS']
        logits2 = logits2.view(-1, logits2.shape[-1])
        targets2 = targets2.view(-1).long()
        if targets2.sum() == -len(targets2):
            loss2=0.0
        else:
            loss2 = self.criterion(logits2, targets2)

        ret.update({'CrossEntropy Loss style 1': loss1, 'CrossEntropy Loss style 2': loss2})
            
        return ret

