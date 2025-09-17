# Copyright 2022 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import math
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.utils.initialization import trunc_normal_
from ..layers.create_act import get_act_layer
from .build import V_E_REGISTRY

__all__ = ["BaseVE"]

@V_E_REGISTRY.register()
class BaseVE(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        in_dim: int,
        out_dim: int,
        **kwargs
    ):
        super(BaseVE, self).__init__()
        self.embeddings = nn.Linear(in_dim, out_dim)
        self.g_embeddings = nn.Linear(in_dim, out_dim)
        self.embeddings_act = kwargs.pop("embeddings_act", None)
        self.embeddings_norm = kwargs.pop("embeddings_norm", None)
        self.embeddings_dropout = kwargs.pop("embeddings_dropout", None)
        self.embeddings_pos = kwargs.pop('embeddings_pos', None)

    @classmethod
    def from_config(cls, cfg):
        kwargs = {
            "in_dim": cfg.MODEL.V_E.IN_DIM,
            "out_dim": cfg.MODEL.V_E.OUT_DIM
        }

        activation_name = (cfg.MODEL.VISUAL_EMBED.ACTIVATION).lower()
        if activation_name != "none":
            activation = get_act_layer(activation_name)
            assert activation is not None

            act_kwargs = {}
            if activation_name in { "elu", "celu" }:
                act_kwargs["alpha"] = cfg.MODEL.VISUAL_EMBED.ELU_ALPHA
            embeddings_act = activation(**act_kwargs)
            kwargs['embeddings_act'] = embeddings_act

        if cfg.MODEL.VISUAL_EMBED.DROPOUT > 0:
            embeddings_dropout = nn.Dropout(cfg.MODEL.VISUAL_EMBED.DROPOUT)
            kwargs['embeddings_dropout'] = embeddings_dropout

        if cfg.MODEL.VISUAL_EMBED.USE_NORM:
            embeddings_norm = nn.LayerNorm(cfg.MODEL.VISUAL_EMBED.OUT_DIM)
            kwargs['embeddings_norm'] = embeddings_norm

        embeddings_pos = nn.Parameter(
            torch.zeros(1, cfg.DATALOADER.MAX_FEAT_NUM+1, cfg.MODEL.VISUAL_EMBED.OUT_DIM))
        trunc_normal_(embeddings_pos, std=.02)
        kwargs['embeddings_pos'] = embeddings_pos

        return kwargs

    def forward(self, batched_inputs):
        out={}
        if kfg.COCO_ATT_FEATS in batched_inputs:
            coco_feats = batched_inputs[kfg.COCO_ATT_FEATS]
            coco_embeddings = self.embeddings(coco_feats)
            coco_gfeats = self.g_embeddings(coco_feats.mean(-2, keepdim=True))
            coco_embeddings = torch.cat([coco_gfeats, coco_embeddings], dim=1)
            coco_att_masks = batched_inputs[kfg.COCO_ATT_MASKS]
            coco_vmasks = torch.cat([coco_att_masks, coco_att_masks[:, 0].unsqueeze(-1)], -1)
            coco_embeddings = coco_embeddings + self.embeddings_pos
            if self.embeddings_act is not None:
                coco_embeddings = self.embeddings_act(coco_embeddings)
            if self.embeddings_norm is not None:
                coco_embeddings = self.embeddings_norm(coco_embeddings)
            if self.embeddings_dropout is not None:
                coco_embeddings = self.embeddings_dropout(coco_embeddings)
            out.update({ kfg.COCO_ATT_FEATS: coco_embeddings, kfg.COCO_ATT_MASKS: coco_vmasks})
        if kfg.CROWD_ATT_FEATS in batched_inputs:
            crowd_feats = batched_inputs[kfg.CROWD_ATT_FEATS]
            crowd_embeddings = self.embeddings(crowd_feats)
            crowd_gfeats =  self.g_embeddings(crowd_feats.mean(-2,keepdim=True))
            crowd_embeddings = torch.cat([crowd_gfeats, crowd_embeddings], dim=1)
            crowd_att_masks = batched_inputs[kfg.CROWD_ATT_MASKS]
            crowd_vmasks = torch.cat([crowd_att_masks,crowd_att_masks[:,0].unsqueeze(-1)],-1)
            crowd_embeddings = crowd_embeddings + self.embeddings_pos
            if self.embeddings_act is not None:
                crowd_embeddings = self.embeddings_act(crowd_embeddings)
            if self.embeddings_norm is not None:
                crowd_embeddings = self.embeddings_norm(crowd_embeddings)
            if self.embeddings_dropout is not None:
                crowd_embeddings = self.embeddings_dropout(crowd_embeddings)
            out.update({kfg.CROWD_ATT_FEATS: crowd_embeddings,kfg.CROWD_ATT_MASKS: crowd_vmasks})
        return out