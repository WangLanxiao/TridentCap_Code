# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li, Jianjie Luo
@contact: yehaoli.sysu@gmail.com, jianjieluo.sysu@gmail.com
"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from xmodaler.config import configurable
from xmodaler.config import kfg
from .decode_strategy import DecodeStrategy
from .build import DECODE_STRATEGY_REGISTRY

@DECODE_STRATEGY_REGISTRY.register()
class GreedyDecoder(DecodeStrategy):
    def _forward(self, batched_inputs, model):
        is_sample = batched_inputs.get(kfg.DECODE_BY_SAMPLE, False)
        inputs = batched_inputs
        batch_size = batched_inputs[kfg.ATT_FEATS].size(0)
        ve_out = model.visual_embed(batched_inputs)
        inputs.update(ve_out)

        le_out = model.s_e(inputs, mode='style')
        inputs.update(le_out)

        vl_merge_out = model.vl_merge(inputs, mode=1)
        inputs.update(vl_merge_out)

        masks = model.get_extended_attention_mask(batched_inputs)
        inputs.update(masks)

        inputs = model.decoder_1.preprocess(inputs)

        ret = inputs
        sents = Variable(torch.zeros((batch_size, self.max_seq_len), dtype=torch.long).cuda()) + self.eos_token_id
        logprobs = Variable(torch.zeros(batch_size, self.max_seq_len).cuda())
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda()) + self.bos_token_id
        unfinished = wt.eq(wt)
        Style_hidd = []
        for t in range(self.max_seq_len):
            inputs.update({kfg.G_TOKENS_IDS: wt, kfg.TIME_STEP: t})
            te_out = model.s_e(wt, mode='token')
            inputs.update({'MID_E_G_TOKEN_EMBED': te_out})
            decoder_out = model.decoder_1(inputs)
            inputs.update(decoder_out)

            Style_hidd.append(decoder_out['G_HIDDEN_STATES'][-1])

            logit = model.predictor(inputs, mode='mid')
            logprobs_t = F.log_softmax(logit, dim=-1)
            if is_sample:
                probs_t = torch.exp(logprobs_t)
                wt = torch.multinomial(probs_t, 1)
                logP_t = logprobs_t.gather(1, wt)
            else:
                logP_t, wt = torch.max(logprobs_t, 1)
            wt = wt.view(-1).long()
            unfinished = unfinished * (wt != self.eos_token_id)
            wt = unfinished.type_as(wt) * wt + (1 - unfinished.type_as(wt)) * self.eos_token_id
            sents[:, t] = wt
            logprobs[:, t] = logP_t.view(-1)
            if unfinished.sum() == 0:
                break

        Style_hidd = torch.stack(Style_hidd, 1).mean(-2)
        style_class = model.predictor(Style_hidd, mode='style')
        _, style_class = torch.max(style_class, 1)

        ret.update({
            kfg.IDS: batched_inputs[kfg.IDS],
            kfg.G_SENTS_IDS: sents,
            kfg.G_LOGP: logprobs,
            'STYLE_CLASS': style_class
        })
        return ret