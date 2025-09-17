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
class Style_CrossEntropy(nn.Module):
    @configurable
    def __init__(self):
        super(Style_CrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.BCE = nn.CrossEntropyLoss(ignore_index=-1)

    @classmethod
    def from_config(cls, cfg):
        return {}

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, outputs_dict):
        ret  = {}

        # logits1 = outputs_dict['END_G_LOGITS']
        # targets1 = outputs_dict['G_TARGET_IDS']
        # logits1 = logits1.view(-1, logits1.shape[-1])
        # targets1 = targets1.view(-1).long()
        # loss1 = self.criterion(logits1, targets1)
        # ret.update({'CrossEntropy Loss style 1': loss1})
        # Style_predict = outputs_dict['Style_predict']
        # Style_token= outputs_dict['STYLE_TOKEN'].view(-1)#.long()
        # loss1 = self.criterion(Style_predict, Style_token)
        # ret.update({'CrossEntropy Loss style 1': loss1 * 0.1}

        logits2 = outputs_dict['MID_G_LOGITS']
        targets2 = outputs_dict['MID_G_TARGET_IDS']
        logits2 = logits2.view(-1, logits2.shape[-1])
        targets2 = targets2.view(-1).long()
        loss2 = self.criterion(logits2, targets2)
        ret.update({'CrossEntropy Loss': loss2})

        if 'STYLE_LOGITS' in outputs_dict:
            logits3 = outputs_dict['STYLE_LOGITS']
            targets3 = outputs_dict['STYLE_TOKEN']
            logits3 = logits3.view(-1, logits3.shape[-1])
            targets3 = targets3.view(-1).long()
            loss3 = self.criterion(logits3, targets3)
            ret.update({'Style Loss': loss3})
        return ret

