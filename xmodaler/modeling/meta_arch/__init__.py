# -*- coding: utf-8 -*-
"""
From original at https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/meta_arch/__init__.py
Original copyright of Facebook code below, modifications by Yehao Li, Copyright 2021.	
"""
# Copyright (c) Facebook, Inc. and its affiliates.

from .build import META_ARCH_REGISTRY, build_model, add_config
from .rnn_att_enc_dec import RnnAttEncoderDecoder
from .wlx_att_enc_dec import WLXAttEncoderDecoder
from .transformer_enc_dec import TransformerEncoderDecoder
from .tden import TDENBiTransformer, TDENPretrain, TDENCaptioner
from .uniter import UniterPretrain, UniterForMMUnderstanding
from .tf_single import UTF_one_AttEncoderDecoder
from .tf_two import UTF_two_AttEncoderDecoder
from .tf_merge import TFMERGE
from .base_enc_dec_two_stage import BaseEncoderDecoderTWO
from .base_enc_dec_merge import BaseEncoderDecoderMerge


__all__ = list(globals().keys())