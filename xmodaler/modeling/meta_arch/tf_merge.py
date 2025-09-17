# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from ..decoder import build_decoder, add_decoder_config
from ..vl_merge import build_vl_merge, add_vl_merge_config
from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from .base_enc_dec_merge import BaseEncoderDecoderMerge
from .build import META_ARCH_REGISTRY
from ..sent_embed import build_sent_embed, add_sent_config
from xmodaler.functional import pad_tensor, dict_to_cuda
import random


__all__ = ["TFMERGE"]


@META_ARCH_REGISTRY.register()
class TFMERGE(BaseEncoderDecoderMerge):
    @configurable
    def __init__(
            self,
            *,
            vocab_size,
            max_seq_len,
            visual_embed,
            vl_merge,
            s_e,
            # decoder_2,
            decoder_1,
            predictor,
            greedy_decoder,
            beam_searcher,
            hidden_size
    ):
        super().__init__(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            visual_embed=visual_embed,
            predictor=predictor,
            greedy_decoder=greedy_decoder,
            beam_searcher=beam_searcher
        )

        self.vl_merge = vl_merge
        self.s_e = s_e
        self.decoder_1 = decoder_1
        # self.logits_style = nn.Linear(hidden_size, 5)
        # self.dropout_style = nn.Dropout(0.2)
        # self.decoder_2 = decoder_2

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret.update({"vl_merge": build_vl_merge(cfg),
                    "s_e": build_sent_embed(cfg),
                    # "decoder_2": build_decoder(cfg),
                    "decoder_1": build_decoder(cfg),
                    "hidden_size": cfg.MODEL.DECODER_DIM,
                    })
        return ret


    def get_extended_attention_mask(self, batched_inputs):
        att_masks = batched_inputs[kfg.ATT_MASKS]
        if att_masks is not None:
            att_masks = att_masks.to(dtype=next(self.parameters()).dtype)
            ext_att_masks = (1.0 - att_masks) * -10000.0
        else:
            ext_att_masks = None

        return {
            kfg.ATT_MASKS: att_masks,
            kfg.EXT_ATT_MASKS: ext_att_masks
        }

    def _forward(self, batched_inputs):
        inputs = batched_inputs
        vfeats, vmasks = pad_tensor(inputs[kfg.ATT_FEATS], padding_value=0, use_mask=True)
        inputs.update({kfg.ATT_MASKS: vmasks.cuda()})
        ###################   visual embed #################
        ve_out = self.visual_embed(batched_inputs)
        inputs.update(ve_out)
        ###################   language embed #################
        s_e_out = self.s_e(batched_inputs, mode='style')
        inputs.update(s_e_out)
        ###################   VL merge        #################
        vl_merge_out = self.vl_merge(batched_inputs, mode=1)
        inputs.update(vl_merge_out)
        masks = self.get_extended_attention_mask(batched_inputs)
        inputs.update(masks)
        #####################    initial  ######################
        inputs = self.decoder_1.preprocess(inputs)
        tokens_ids = batched_inputs['MID_G_TOKENS_IDS']
        batch_size, seq_len = tokens_ids.shape
        ss_prob = batched_inputs[kfg.SS_PROB]
        outputs = Variable(torch.zeros(batch_size, seq_len, self.vocab_size).cuda())
        Style_hidd=[]
        #####################    processing 1  ######################
        for t in range(seq_len):
            if t >= 1 and tokens_ids[:, t].max() == 0:
                break
            if self.training and t >= 1 and ss_prob > 0:
                prob = torch.empty(batch_size).cuda().uniform_(0, 1)
                mask = prob < ss_prob
                if mask.sum() == 0:
                    wt = tokens_ids[:, t].clone()
                else:
                    ind = mask.nonzero().view(-1)
                    wt = tokens_ids[:, t].data.clone()
                    prob_prev = torch.exp(outputs[:, t - 1].detach())
                    wt.index_copy_(0, ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, ind))
            else:
                wt = tokens_ids[:, t].clone()

            te_out = self.s_e(wt, mode='token')
            inputs.update({'MID_E_G_TOKEN_EMBED': te_out})
            decoder_out = self.decoder_1(inputs)
            inputs.update(decoder_out)
            Style_hidd.append(decoder_out['G_HIDDEN_STATES'][-1])
            logit = self.predictor(inputs, mode='mid')
            outputs[:, t] = logit
        inputs.update({'MID_G_LOGITS': outputs})
        Style_hidd=torch.stack(Style_hidd,1).mean(-2)
        style_class = self.predictor(Style_hidd, mode='style')
        inputs.update({'STYLE_LOGITS': style_class})
        # Style_predict=torch.stack(Style_predict, 1)
        # tokens_ids[torch.nonzero(tokens_ids)] = 1
        # Style_predict=Style_predict*tokens_ids[:,:Style_predict.size()[-2]].unsqueeze(-1)
        # Style_dp = self.dropout_style(Style_predict.mean(-2))
        # Style_out = self.logits_style(Style_dp)
        # inputs.update({'Style_predict': Style_out})
        # ###################   language embed #################
        # # mid_out = torch.exp(outputs.detach())
        # # coco_out = torch.multinomial(mid_out.reshape(-1, self.vocab_size), 1).reshape(batch_size, -1)
        # s_e_out = self.s_e(batched_inputs, mode='normal')
        # inputs.update(s_e_out)
        # ###################   VL merge        #################
        # vl_merge_out = self.vl_merge(inputs, mode=1)
        # inputs.update(vl_merge_out)
        # masks = self.get_extended_attention_mask(batched_inputs)
        # inputs.update(masks)
        # #####################    initial  ######################
        # inputs = self.decoder_2.preprocess(inputs)
        # tokens_ids = batched_inputs[kfg.G_TOKENS_IDS]
        # batch_size, seq_len = tokens_ids.shape
        # ss_prob = batched_inputs[kfg.SS_PROB]
        # outputs = Variable(torch.zeros(batch_size, seq_len, self.vocab_size).cuda())
        # #####################    processing 2  ######################
        # for t in range(seq_len):
        #     if t >= 1 and tokens_ids[:, t].max() == 0:
        #         break
        #     if self.training and t >= 1 and ss_prob > 0:
        #         prob = torch.empty(batch_size).cuda().uniform_(0, 1)
        #         mask = prob < ss_prob
        #         if mask.sum() == 0:
        #             wt = tokens_ids[:, t].clone()
        #         else:
        #             ind = mask.nonzero().view(-1)
        #             wt = tokens_ids[:, t].data.clone()
        #             prob_prev = torch.exp(outputs[:, t - 1].detach())
        #             wt.index_copy_(0, ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, ind))
        #     else:
        #         wt = tokens_ids[:, t].clone()
        #
        #     te_out = self.s_e(wt, mode='token')
        #     inputs.update({'MID_E_G_TOKEN_EMBED': te_out})
        #     decoder_out = self.decoder_2(inputs)
        #     inputs.update(decoder_out)
        #     logit = self.predictor(inputs, mode='mid')
        #     outputs[:, t] = logit
        # inputs.update({'END_G_LOGITS': outputs})
        return inputs