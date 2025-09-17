# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
from xmodaler.utils.registry import Registry

V_E_REGISTRY = Registry("V_E")
V_E_REGISTRY.__doc__ = """
Registry for backbone
"""

def build_visual_embed(cfg):
    V_E =  V_E_REGISTRY.get(cfg.MODEL.V_E.NAME)(cfg) if len(cfg.MODEL.V_E) > 0 else None
    return V_E

def add_visual_embed_config(cfg, tmp_cfg):
    if len(tmp_cfg.MODEL.V_E) > 0:
        V_E_REGISTRY.get(tmp_cfg.MODEL.V_E).add_config(cfg)