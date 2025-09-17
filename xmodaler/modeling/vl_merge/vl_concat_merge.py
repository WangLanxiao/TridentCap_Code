import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.config import CfgNode as CN
from xmodaler.utils.initialization import trunc_normal_
from ..layers.create_act import get_act_layer
from .build import VL_Merge_REGISTRY
from xmodaler.modeling.layers import LowRankBilinearLayer
from torch.nn.parameter import Parameter
from ..layers.multihead_attention import MultiHeadAttention

__all__ = ["VL_ConcatMerge"]
@VL_Merge_REGISTRY.register()

class VL_ConcatMerge(nn.Module):
    @configurable
    def __init__(self,*,
                 hidden_size=512,
                 num_input_channels=256,
                 num_output_channels=256,
                 num_featmaps=1,
                 xlan_layer_num=3,
                 embed_dim: int,
                 att_heads: int,
                 att_mid_dim: int,
                 att_mid_drop: float,
                 dropout: float,
                 bifeat_emb_dropout: float,
                 emb_act_type: str,
                 act_type: str,
                 elu_alpha: float,
                 **kwargs):
        super(VL_ConcatMerge, self).__init__()
        self.hidden_size=hidden_size
        self.num_input_channels=num_input_channels
        self.num_output_channels=num_output_channels
        self.num_featmaps=num_featmaps

        # self.lang_fcs = nn.ModuleList([nn.Linear(self.hidden_size * 2, self.num_output_channels) for i in range(self.num_featmaps)])
        # self.visual_convs = nn.ModuleList([nn.Conv1d(self.num_input_channels, self.num_output_channels, 1, 1, 0) for i in range(self.num_featmaps)])
        # self.vl_convs = nn.ModuleList(
        #     [nn.Conv1d((self.num_output_channels * 3), self.num_output_channels, 1, 1, 0) for i in
        #      range(self.num_featmaps)])

        self.tf_pow = 2.0
        self.tf_scale = Parameter(torch.Tensor([1.0]))
        self.tf_sigma = Parameter(torch.Tensor([0.5]))


        # self.lang_fcs2 = nn.ModuleList(
        #     [nn.Linear(self.hidden_size * 2, self.num_output_channels) for i in range(self.num_featmaps)])
        # self.visual_convs2 = nn.ModuleList(
        #     [nn.Conv1d(self.num_input_channels, self.num_output_channels, 1, 1, 0) for i in range(self.num_featmaps)])
        # self.vl_convs2 = nn.ModuleList(
        #     [nn.Conv1d((self.num_output_channels * 2), self.num_output_channels, 1, 1, 0) for i in
        #      range(self.num_featmaps)])

        # self.layers = nn.ModuleList([])
        # self.bifeat_emb = nn.ModuleList([])
        # self.layer_norms = nn.ModuleList([])
        # for _ in range(xlan_layer_num):
        #     sublayer = LowRankBilinearLayer(
        #         embed_dim=embed_dim,
        #         att_heads=att_heads,
        #         att_mid_dim=att_mid_dim,
        #         att_mid_drop=att_mid_drop,
        #         dropout=dropout,
        #         act_type=act_type,
        #         elu_alpha=elu_alpha
        #     )
        #     self.layers.append(sublayer)
        #     self.bifeat_emb.append(nn.Sequential(
        #         nn.Linear(2 * embed_dim, embed_dim),
        #         get_act_layer(emb_act_type)(),
        #         nn.Dropout(bifeat_emb_dropout)
        #     ))
        #     self.layer_norms.append(torch.nn.LayerNorm(embed_dim))
        #
        # self.proj = nn.Linear(embed_dim * (xlan_layer_num + 1), embed_dim)
        self.layer_norm = torch.nn.LayerNorm(embed_dim)

        self.att_vis_new = MultiHeadAttention(
            d_model=embed_dim, \
            d_k=embed_dim, \
            d_v=embed_dim, \
            num_head=8, \
            dropout=0.2
        )
        self.att_style_new = MultiHeadAttention(
            d_model=embed_dim, \
            d_k=embed_dim, \
            d_v=embed_dim, \
            num_head=8, \
            dropout=0.2
        )

        # self.layers2 = nn.ModuleList([])
        # self.bifeat_emb2 = nn.ModuleList([])
        # self.layer_norms2 = nn.ModuleList([])
        # for _ in range(xlan_layer_num):
        #     sublayer = LowRankBilinearLayer(
        #         embed_dim=embed_dim,
        #         att_heads=att_heads,
        #         att_mid_dim=att_mid_dim,
        #         att_mid_drop=att_mid_drop,
        #         dropout=dropout,
        #         act_type=act_type,
        #         elu_alpha=elu_alpha
        #     )
        #     self.layers2.append(sublayer)
        #     self.bifeat_emb2.append(nn.Sequential(
        #         nn.Linear(2 * embed_dim, embed_dim),
        #         get_act_layer(emb_act_type)(),
        #         nn.Dropout(bifeat_emb_dropout)
        #     ))
        #     self.layer_norms2.append(torch.nn.LayerNorm(embed_dim))
        #
        # self.proj2 = nn.Linear(embed_dim * (xlan_layer_num + 1), embed_dim)
        # self.layer_norm2 = torch.nn.LayerNorm(embed_dim)


    @classmethod
    def from_config(cls, cfg):
        kwargs = {
            "hidden_size": cfg.MODEL.VL_M.HIDDEN_SIZE,
            "num_input_channels": cfg.MODEL.VL_M.NUM_INPUT_CHANNELS,
            "num_output_channels": cfg.MODEL.VL_M.NUM_OUTPUT_CHANNELS,
            "num_featmaps": cfg.MODEL.VL_M.NUM_FEATMAPS,
            "xlan_layer_num": cfg.MODEL.BILINEAR.ENCODE.LAYERS,
            "embed_dim": cfg.MODEL.BILINEAR.DIM,
            "att_heads": cfg.MODEL.BILINEAR.HEAD,
            "att_mid_dim": cfg.MODEL.BILINEAR.ENCODE.ATT_MID_DIM,
            "att_mid_drop": cfg.MODEL.BILINEAR.ENCODE.ATT_MID_DROPOUT,
            "dropout": cfg.MODEL.BILINEAR.ENCODE.DROPOUT,
            "bifeat_emb_dropout": cfg.MODEL.BILINEAR.ENCODE.BIFEAT_EMB_DROPOUT,
            "emb_act_type": cfg.MODEL.BILINEAR.BIFEAT_EMB_ACT,
            "act_type": cfg.MODEL.BILINEAR.ACT,
            "elu_alpha": cfg.MODEL.BILINEAR.ELU_ALPHA,
        }
        return kwargs

    @classmethod
    def add_config(cls, cfg):
        cfg.MODEL.BILINEAR = CN()
        cfg.MODEL.BILINEAR.DIM = 512
        cfg.MODEL.BILINEAR.HEAD = 8
        cfg.MODEL.BILINEAR.BIFEAT_EMB_ACT = "relu"
        cfg.MODEL.BILINEAR.ACT = "celu"
        cfg.MODEL.BILINEAR.ELU_ALPHA = 1.3

        cfg.MODEL.BILINEAR.ENCODE = CN()
        cfg.MODEL.BILINEAR.ENCODE.ATT_MID_DIM = [64, 32, 64]
        cfg.MODEL.BILINEAR.ENCODE.ATT_MID_DROPOUT = 0.1
        cfg.MODEL.BILINEAR.ENCODE.DROPOUT = 0.5
        cfg.MODEL.BILINEAR.ENCODE.BIFEAT_EMB_DROPOUT = 0.3
        cfg.MODEL.BILINEAR.ENCODE.LAYERS = 4

    def forward(self, batched_inputs, mode=1):
        out={}
        att_masks = batched_inputs[kfg.ATT_MASKS]
        style_feats = batched_inputs['STYLE_FEATS']  # b 1 512
        img_feats = batched_inputs[kfg.ATT_FEATS]
        batch, num, channel = img_feats.size()
        # style_feats = style_feats.expand(batch, num, self.hidden_size)
        lang_feat_emb = batched_inputs['SENT_FEATS'].expand(batch, num, self.hidden_size)

        verify_score = (F.normalize(img_feats, p=2, dim=-1) *
                        F.normalize(lang_feat_emb, p=2, dim=-1)).sum(dim=-1, keepdim=True)
        verify_score = self.tf_scale * \
                       torch.exp(- (1 - verify_score).pow(self.tf_pow) \
                                 / (2 * self.tf_sigma ** 2))

        fuse_img_feat = (self.layer_norm(img_feats) + self.layer_norm(lang_feat_emb)) * verify_score # + img_feats


        # feat_fuse = torch.cat((lang_feat_emb, style_feats, img_feats), dim=2)
        # feat_fuse_emb = self.vl_convs[0](feat_fuse.permute(0, 2, 1)).permute(0, 2, 1)


        style_feat = self.att_style_new(style_feats, fuse_img_feat, fuse_img_feat) + style_feats
        img_update = self.att_vis_new(img_feats, fuse_img_feat, fuse_img_feat) + fuse_img_feat
        gv_feat = img_update.mean(-2)

        # feat_arr = [gv_feat]
        # for i, layer in enumerate(self.layers):
        #     gv_feat = layer(gv_feat, feat_fuse_emb, att_masks, gv_feat, feat_fuse_emb)
        #     fuse_emb_cat = torch.cat([gv_feat.unsqueeze(1).expand_as(feat_fuse_emb), feat_fuse_emb], dim=-1)
        #     feat_fuse_emb = self.bifeat_emb[i](fuse_emb_cat) + feat_fuse_emb
        #     feat_fuse_emb = self.layer_norms[i](feat_fuse_emb)
        #     feat_arr.append(gv_feat)
        # gv_feat = torch.cat(feat_arr, dim=-1)
        # gv_feat = self.proj(gv_feat)
        gv_feat = self.layer_norm(gv_feat)
        style_feat = self.layer_norm(style_feat)
        out.update({'MERGE_FEATS': img_update, kfg.GLOBAL_FEATS: gv_feat, 'STYLE_FEATS': style_feat})

        # save_p='/data1/wlx/project/202303c2p_TransferLearning/vis/TNSE/'
        # exit_list=os.listdir(save_p)
        # new_sp=save_p+str(len(exit_list))+'.npy'
        # save={}
        # save['id']=np.stack(batched_inputs[kfg.IDS])
        # save['style_label']=np.stack(batched_inputs['STYLE_TOKEN'].cpu())[:,0]
        # save['img']=np.array(batched_inputs[kfg.ATT_FEATS].cpu())
        # save['lang']=np.array(batched_inputs['SENT_FEATS'].cpu())
        # save['dual']=np.array(fuse_img_feat.cpu())
        # save['style']=np.array(style_feat.cpu())
        # save['content']=np.array(img_update.cpu())
        # np.save(new_sp, save)
        return out
