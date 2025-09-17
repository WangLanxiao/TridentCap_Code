# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import PREDICTOR_REGISTRY

__all__ = ["WLXPredictor"]

@PREDICTOR_REGISTRY.register()
class WLXPredictor(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,   # include <BOS>/<EOS>
        dropout: float
    ):
        super(WLXPredictor, self).__init__()
        self.logits1 = nn.Linear(hidden_size, vocab_size)
        self.logits2 = nn.Linear(hidden_size, 5)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.DECODER_DIM,
            "vocab_size": cfg.MODEL.VOCAB_SIZE,
            "dropout": cfg.MODEL.PRED_DROPOUT
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs, mode='COCOe2d'):
       
        if mode == 'mid':
            hidden_states = batched_inputs['G_HIDDEN_STATES']
            if isinstance(hidden_states, list):
                hidden_states = hidden_states[-1]
            if self.dropout:
                hidden_states = self.dropout(hidden_states)
            logits = self.logits1(hidden_states)
        elif mode=='style':
            if self.dropout:
                batched_inputs = self.dropout(batched_inputs)
            logits = self.logits2(batched_inputs)

        return logits
