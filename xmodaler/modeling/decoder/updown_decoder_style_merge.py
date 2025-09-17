# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from ..layers.base_attention import BaseAttention
from .decoder import Decoder
from .build import DECODER_REGISTRY

__all__ = ["UpDownDecoderMerge"]

@DECODER_REGISTRY.register()
class UpDownDecoderMerge(Decoder):
    @configurable
    def __init__(
        self,
        *,
        hidden_size: int,
        token_embed_dim: int,
        visual_embed_dim: int,
        att_embed_size: int, 
        dropout1: float,
        dropout2: float,
        att_embed_dropout: float
    ):
        super(UpDownDecoderMerge, self).__init__()
        self.num_layers = 2
        self.hidden_size = hidden_size

        in_dim = hidden_size + token_embed_dim + visual_embed_dim
        self.lstm1 = nn.LSTMCell(in_dim, hidden_size)
        self.dropout1 = nn.Dropout(dropout1) if dropout1 > 0 else None

        in_dim = hidden_size + visual_embed_dim + visual_embed_dim
        self.lstm2 = nn.LSTMCell(in_dim, hidden_size)
        self.dropout2 = nn.Dropout(dropout2) if dropout2 > 0 else None

        self.att = BaseAttention(
            hidden_size = hidden_size,
            att_embed_size = att_embed_size,
            att_embed_dropout = att_embed_dropout
        )
        self.p_att_feats = nn.Linear(visual_embed_dim, att_embed_size)
        # self.p_att_feats = nn.Linear(visual_embed_dim*2, att_embed_size)

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.DECODER_DIM,
            "token_embed_dim": cfg.MODEL.S_E.WORD_EMBEDDING_SIZE,
            "visual_embed_dim": cfg.MODEL.VL_M.NUM_OUTPUT_CHANNELS,
            "att_embed_size": cfg.MODEL.UPDOWN.ATT_EMBED_SIZE,
            "dropout1": cfg.MODEL.UPDOWN.DROPOUT1,
            "dropout2": cfg.MODEL.UPDOWN.DROPOUT2,
            "att_embed_dropout": cfg.MODEL.UPDOWN.ATT_EMBED_DROPOUT
        }

    @classmethod
    def add_config(cls, cfg):
        cfg.MODEL.UPDOWN = CN()
        cfg.MODEL.UPDOWN.ATT_EMBED_SIZE = 512
        cfg.MODEL.UPDOWN.DROPOUT1 = 0.0
        cfg.MODEL.UPDOWN.DROPOUT2 = 0.0
        cfg.MODEL.UPDOWN.ATT_EMBED_DROPOUT = 0.0

    def preprocess(self, batched_inputs):
        att_feats = batched_inputs[kfg.ATT_FEATS]
        p_att_feats = self.p_att_feats(att_feats)
        init_states = self.init_states(att_feats.shape[0])
        batched_inputs.update(init_states)
        batched_inputs.update( { kfg.P_ATT_FEATS: p_att_feats } )
        return batched_inputs

    def forward(self, batched_inputs):
        wt = batched_inputs['MID_E_G_TOKEN_EMBED']       # b 512
        style_feats = batched_inputs['STYLE_FEATS'].squeeze(1)      # b 1 512
        att_feats = batched_inputs['MERGE_FEATS']        # b 37  512
        ext_att_masks = batched_inputs['EXT_ATT_MASKS']
        global_feats = batched_inputs['GLOBAL_FEATS']     # b 512
        p_att_feats = batched_inputs['P_ATT_FEATS']       # b 37  512
        hidden_states = batched_inputs['G_HIDDEN_STATES'] # [b 1024,b 1024]
        cell_states = batched_inputs['G_CELL_STATES']     # [b 1024,b 1024]

        # lstm1
        h2_tm1 = hidden_states[-1]
        input1 = torch.cat([h2_tm1, global_feats+style_feats, wt], 1)
        if self.dropout1 is not None:
            input1 = self.dropout1(input1)
        h1_t, c1_t = self.lstm1(input1, (hidden_states[0], cell_states[0]))
        att = self.att(h1_t, att_feats, p_att_feats, ext_att_masks)

        # lstm2
        # input2 = torch.cat([att,style_feats,h1_t], 1)
        input2 = torch.cat([att,style_feats,h1_t], 1)
        if self.dropout2 is not None:
            input2 = self.dropout2(input2)
        h2_t, c2_t = self.lstm2(input2, (hidden_states[1], cell_states[1]))

        hidden_states = [h1_t, h2_t]
        cell_states = [c1_t, c2_t]
        return { 
            'G_HIDDEN_STATES': hidden_states,
           'G_CELL_STATES': cell_states
        }

