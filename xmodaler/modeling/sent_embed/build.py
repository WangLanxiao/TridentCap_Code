# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
from xmodaler.utils.registry import Registry

S_E_REGISTRY = Registry("S_E")
S_E_REGISTRY.__doc__ = """
Registry for backbone
"""

def build_sent_embed(cfg):
    S_E = S_E_REGISTRY.get(cfg.MODEL.S_E.NAME)(cfg) if len(cfg.MODEL.S_E) > 0 else None
    return S_E

def add_sent_config(cfg, tmp_cfg):
    if len(tmp_cfg.MODEL.S_E) > 0:
        S_E_REGISTRY.get(tmp_cfg.MODEL.S_E).add_config(cfg)