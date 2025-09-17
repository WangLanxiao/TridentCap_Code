# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""


from .build import build_vl_merge, add_vl_merge_config
from .vl_concat_merge import VL_ConcatMerge

__all__ = list(globals().keys())
