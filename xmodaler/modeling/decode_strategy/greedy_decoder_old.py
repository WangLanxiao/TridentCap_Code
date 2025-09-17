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
        if kfg.COCO_IMG_INPUT in batched_inputs:
            coco_backbone_out = model.backbone(inputs['COCO_IMG_INPUT'])
            inputs.update({kfg.COCO_ATT_FEATS: coco_backbone_out})
        if kfg.CROWD_IMG_INPUT in batched_inputs:
            crowd_backbone_out = model.backbone(inputs['CROWD_IMG_INPUT'])
            inputs.update({kfg.CROWD_ATT_FEATS: crowd_backbone_out})
        if self.mode == 'COCO':
            batch_size, _, _  = inputs[kfg.COCO_ATT_FEATS].shape
        elif self.mode == 'CROWD':
            batch_size, _, _ = inputs[kfg.CROWD_ATT_FEATS].shape

        ###################   visual embed #################
        v_e_out = model.v_e(batched_inputs)
        inputs.update(v_e_out)
        ###################   language embed #################
        s_e_out = model.s_e(batched_inputs, mode='embed')
        inputs.update(s_e_out)
        ###################   VL merge        #################
        vl_merge_out = model.vl_merge(batched_inputs)
        inputs.update(vl_merge_out)
        ###################   processing mask ################
        masks = model.get_extended_attention_mask(batched_inputs)
        inputs.update(masks)
        if self.mode == 'COCO':
            p_att_feats, g_hidden, g_cell = model.decoder_e2d.preprocess(inputs, mode='COCOe2d')
            inputs.update({kfg.COCO_E_P_ATT_FEATS: p_att_feats, kfg.COCO_E_G_HIDDEN_STATES: g_hidden,
                           kfg.COCO_E_G_CELL_STATES: g_cell})
            sents = Variable(torch.zeros((batch_size, self.crowd_max_seq_len), dtype=torch.long).cuda()) + self.eos_token_id
            logprobs = Variable(torch.zeros(batch_size, self.crowd_max_seq_len).cuda())
            wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda()) + self.bos_token_id
            unfinished = wt.eq(wt)
            for t in range(self.crowd_max_seq_len):
                te_out = model.s_e(wt, mode='token')
                inputs.update({kfg.COCO_E_G_TOKEN_EMBED: te_out, kfg.TIME_STEP: t })
                hidden_states, cell_states = model.decoder_e2d(inputs, mode='COCOe2d')
                inputs.update({kfg.COCO_E_G_HIDDEN_STATES: hidden_states, kfg.COCO_E_G_CELL_STATES: cell_states})
                logit = model.predictor(inputs, mode='COCOe2d')
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
            ret = inputs
            ret.update({
                kfg.COCO_IDS: batched_inputs[kfg.COCO_IDS],
                kfg.COCO_OUT_IDS: sents,
                kfg.COCO_D_G_LOGITS: logprobs
            })
            #####################################################################
            s_e_out = model.s_e(batched_inputs, mode='embed_based_output')
            inputs.update(s_e_out)
            ###################   VL merge        #################
            vl_merge_out = model.vl_merge(batched_inputs)
            inputs.update(vl_merge_out)
            #####################################################################
            p_att_feats, g_hidden, g_cell = model.decoder_d2e.preprocess(inputs, mode='COCOd2e')
            inputs.update({kfg.COCO_D_P_ATT_FEATS: p_att_feats, kfg.COCO_D_G_HIDDEN_STATES: g_hidden,kfg.COCO_D_G_CELL_STATES: g_cell})
            sents = Variable(torch.zeros((batch_size, self.coco_max_seq_len), dtype=torch.long).cuda()) + self.eos_token_id
            logprobs = Variable(torch.zeros(batch_size, self.coco_max_seq_len).cuda())
            wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda()) + self.bos_token_id
            unfinished = wt.eq(wt)
            for t in range(self.coco_max_seq_len):
                te_out = model.s_e(wt, mode='token')
                inputs.update({kfg.COCO_D_G_TOKEN_EMBED: te_out, kfg.TIME_STEP: t})
                hidden_states, cell_states = model.decoder_d2e(inputs, mode='COCOd2e')
                inputs.update({kfg.COCO_D_G_HIDDEN_STATES: hidden_states, kfg.COCO_D_G_CELL_STATES: cell_states})
                logit = model.predictor(inputs, mode='COCOd2e')
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
            ret.update({
                kfg.COCO_REOUT_IDS: sents,
                kfg.COCO_E_G_LOGITS: logprobs
            })
        elif self.mode == 'CROWD':
            p_att_feats, g_hidden, g_cell = model.decoder_d2e.preprocess(inputs, mode='CROWDd2e')
            inputs.update({kfg.CROWD_D_P_ATT_FEATS: p_att_feats, kfg.CROWD_D_G_HIDDEN_STATES: g_hidden,
                           kfg.CROWD_D_G_CELL_STATES: g_cell})
            sents = Variable(torch.zeros((batch_size, self.coco_max_seq_len), dtype=torch.long).cuda()) + self.eos_token_id
            logprobs = Variable(torch.zeros(batch_size, self.coco_max_seq_len).cuda())
            wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda()) + self.bos_token_id
            unfinished = wt.eq(wt)
            for t in range(self.coco_max_seq_len):
                te_out = model.s_e(wt, mode='token')
                inputs.update({kfg.CROWD_D_G_TOKEN_EMBED: te_out, kfg.TIME_STEP: t})
                hidden_states, cell_states = model.decoder_d2e(inputs, mode='CROWDd2e')
                inputs.update({kfg.CROWD_D_G_HIDDEN_STATES: hidden_states, kfg.CROWD_D_G_CELL_STATES: cell_states})
                logit = model.predictor(inputs, mode='CROWDd2e')
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
                sents[:,t] = wt
                logprobs[:,t] = logP_t.view(-1)
                if unfinished.sum() == 0:
                    break
            ret = inputs
            ret.update({
                kfg.CROWD_IDS: batched_inputs[kfg.CROWD_IDS],
                kfg.CROWD_OUT_IDS: sents,
                kfg.CROWD_E_G_LOGITS: logprobs
            })
            #####################################################################
            s_e_out = model.s_e(batched_inputs, mode='embed_based_output')
            inputs.update(s_e_out)
            ###################   VL merge        #################
            vl_merge_out = model.vl_merge(batched_inputs)
            inputs.update(vl_merge_out)
            #####################################################################
            p_att_feats, g_hidden, g_cell = model.decoder_e2d.preprocess(inputs, mode='CROWDe2d')
            inputs.update({kfg.CROWD_E_P_ATT_FEATS: p_att_feats, kfg.CROWD_E_G_HIDDEN_STATES: g_hidden,kfg.CROWD_E_G_CELL_STATES: g_cell})
            sents = Variable(torch.zeros((batch_size, self.crowd_max_seq_len), dtype=torch.long).cuda()) + self.eos_token_id
            logprobs = Variable(torch.zeros(batch_size, self.crowd_max_seq_len).cuda())
            wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda()) + self.bos_token_id
            unfinished = wt.eq(wt)
            for t in range(self.crowd_max_seq_len):
                te_out = model.s_e(wt, mode='token')
                inputs.update({kfg.CROWD_E_G_TOKEN_EMBED: te_out, kfg.TIME_STEP: t})
                hidden_states, cell_states = model.decoder_e2d(inputs, mode='CROWDe2d')
                inputs.update({kfg.CROWD_E_G_HIDDEN_STATES: hidden_states, kfg.CROWD_E_G_CELL_STATES: cell_states})
                logit = model.predictor(inputs, mode='CROWDe2d')
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
            ret.update({
                kfg.CROWD_REOUT_IDS: sents,
                kfg.CROWD_D_G_LOGITS: logprobs
            })
        return ret