# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
from xmodaler.utils.registry import Registry

VL_Merge_REGISTRY = Registry("VL_Merge")
VL_Merge_REGISTRY.__doc__ = """
Registry for backbone
"""

def build_vl_merge(cfg):
    VL_M =  VL_Merge_REGISTRY.get(cfg.MODEL.VL_M.NAME)(cfg) if len(cfg.MODEL.VL_M) > 0 else None
    return VL_M

def add_vl_merge_config(cfg, tmp_cfg):
    if len(tmp_cfg.MODEL.VL_M) > 0:
        VL_Merge_REGISTRY.get(tmp_cfg.MODEL.VL_M.NAME).add_config(cfg)