# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li, Jianjie Luo
@contact: yehaoli.sysu@gmail.com, jianjieluo.sysu@gmail.com
"""
import os
import copy
import pickle
import random
from tqdm import tqdm
import numpy as np
from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import read_np, dict_as_tensor, boxes_to_locfeats
from ..build import DATASETS_REGISTRY
from itertools import combinations,product
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

__all__ = ["TF_MERGE"]

@DATASETS_REGISTRY.register()
class TF_MERGE:
    @configurable
    def __init__(
        self,
        stage: str,
        anno_file: list,
        ann_weight: list,
        seq_per_sample_net1: int,
        seq_per_sample_net2: int,
        max_feat_num: int,
        max_seq_len_net1: int,
        max_seq_len_net2: int,
        test_style: int,
        feats_folder: str,
        style: list,
    ):
        self.stage = stage
        self.style = style
        self.anno_file = anno_file
        self.ann_weight = ann_weight
        self.seq_per_sample_net1 = seq_per_sample_net1
        self.seq_per_sample_net2 = seq_per_sample_net2
        self.max_feat_num = max_feat_num
        self.feats_folder = [
                    feats_folder+'coco_btud36',
                    feats_folder+'senticap_btud36',
                    feats_folder+'senticap_btud36',
                    feats_folder+'flickr_btud36',
                    feats_folder+'flickr_btud36',
                             ]
        self.max_seq_len_net1 = max_seq_len_net1
        self.max_seq_len_net2 = max_seq_len_net2
        self.test_style = test_style

        
    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        ann_files = {
            "train": cfg.DATALOADER.ANNO_FOLDER_TRAIN,
            "val": cfg.DATALOADER.ANNO_FOLDER_VAL,
            "test": cfg.DATALOADER.ANNO_FOLDER_TEST
        }
        ann_weight = {
            "train": cfg.DATALOADER.ANNO_WEIGHT_TRAIN,
            "val": [1.0],
            "test": [1.0]
        }
        ret = {
            "stage": stage,
            "anno_file": ann_files[stage],
            "ann_weight": ann_weight[stage],
            "style": cfg.DATALOADER.TYPE,
            "seq_per_sample_net1": cfg.DATALOADER.SEQ_PER_SAMPLE_NET1,
            "seq_per_sample_net2": cfg.DATALOADER.SEQ_PER_SAMPLE_NET2,
            "max_feat_num": cfg.DATALOADER.MAX_FEAT_NUM,
            "feats_folder": cfg.DATALOADER.FEATS_FOLDER,
            "max_seq_len_net1": cfg.MODEL.MAX_SEQ_LEN_NET1,
            "max_seq_len_net2": cfg.MODEL.MAX_SEQ_LEN_NET2,
            "test_style": cfg.INFERENCE.STYLE
        }
        return ret

    def _preprocess_datalist(self, datalist,stage):
        t_style = self.style[self.test_style]
        update_list = {}
        for sub_data in datalist:
            if sub_data['filename'] not in update_list:
                update_list[sub_data['filename']] = {}
            update_list[sub_data['filename']][sub_data['style']] = sub_data

        new_datalist = []
        if stage=='train':
            for sub_data_name in update_list:
                for sub_style in update_list[sub_data_name]:
                    if sub_style != 'normal':
                        sub_data = copy.copy(update_list[sub_data_name]['normal'])
                        sub_data['mid_tokens_ids'] = update_list[sub_data_name][sub_style]['tokens_ids']
                        sub_data['mid_target_ids'] = update_list[sub_data_name][sub_style]['target_ids']
                        sub_data['image_id'] = update_list[sub_data_name][sub_style]['image_id']
                        sub_data['style'] = self.style.index(sub_style)
                        new_datalist.append(sub_data)
        else:
            for sub_data_name in update_list:
                if t_style not in update_list[sub_data_name]:
                    continue
                sub_data=copy.copy(update_list[sub_data_name]['normal'])
                sub_data['mid_tokens_ids']=update_list[sub_data_name][t_style]['tokens_ids']
                sub_data['mid_target_ids']=update_list[sub_data_name][t_style]['target_ids']
                sub_data['image_id'] = update_list[sub_data_name][t_style]['image_id']
                sub_data['style'] = self.test_style
                new_datalist.append(sub_data)
        return new_datalist

    def load_data(self, cfg):
        datalist=[]
        orginum = len(pickle.load(open(self.anno_file[0], 'rb'), encoding='bytes'))
        for idx,sub in enumerate(self.anno_file):
            orig_data=pickle.load(open(sub, 'rb'), encoding='bytes')
            if idx>1 and self.ann_weight[idx]>0.0:
                num_d=min(int(self.ann_weight[idx]*orginum),len(orig_data))
                sample_data = random.sample(orig_data, num_d)
                datalist = datalist + sample_data
            elif idx > 1 and self.ann_weight[idx] == 0.0:
                continue
            else:
                datalist = datalist + orig_data
        datalist = self._preprocess_datalist(datalist, self.stage)
        return datalist
        
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image_id = dataset_dict['image_id']
        style = dataset_dict['style']
        filename = dataset_dict['filename'][:-4]

        if len(self.feats_folder) > 0:
            if 'COCO' in filename:
                feat_path = os.path.join(self.feats_folder[0], filename + '.npz')
            else:
                feat_path = os.path.join(self.feats_folder[4], filename + '.npz')
            content = read_np(feat_path)
            att_feats = content['x'][0:self.max_feat_num].astype('float32')
            ret = { kfg.IDS: image_id, kfg.ATT_FEATS: att_feats}

            if "bbox" in content:
                boxes = content['bbox'][0:self.max_feat_num]
                image_h = content['image_h']
                image_w = content['image_w']
                image_locations = boxes_to_locfeats(boxes, image_w, image_h)

                g_image_feat = np.mean(att_feats, axis=0)
                att_feats = np.concatenate([np.expand_dims(g_image_feat, axis=0), att_feats], axis=0)
                g_image_location = np.array([0, 0, 1, 1, 1])
                image_locations = np.concatenate([np.expand_dims(g_image_location, axis=0), image_locations], axis=0)

                ret.update({
                    kfg.ATT_FEATS: att_feats,
                    kfg.GLOBAL_FEATS: g_image_feat,
                    kfg.ATT_FEATS_LOC: image_locations.astype('float32'),
                })
            
        if self.stage != 'train':
            g_tokens_type = np.ones((self.max_seq_len_net1,), dtype=np.int64)
            tokens_ids = [dataset_dict['tokens_ids'].astype(np.int64)[0]]
            target_ids = [dataset_dict['target_ids'].astype(np.int64)[0]]
            ret.update({ kfg.G_TOKENS_TYPE: g_tokens_type,
                         kfg.G_TOKENS_IDS: tokens_ids,
                         kfg.G_TARGET_IDS: target_ids,
                         'STYLE_TOKEN': style
                         })
            dict_as_tensor(ret)
            return ret
        sent_num = len(dataset_dict['tokens_ids'])
        sent_num2 = len(dataset_dict['mid_tokens_ids'])
        proposal = list(product(list(range(sent_num)), list(range(sent_num2))))
        proposal_num = len(proposal)
        if proposal_num >= self.seq_per_sample_net1:
            selects = random.sample(range(proposal_num), self.seq_per_sample_net1)
        else:
            selects = random.choices(range(proposal_num), k = (self.seq_per_sample_net1 - proposal_num))
            selects += list(range(proposal_num))

        selects1=[ proposal[i][0] for i in selects ]
        selects2=[ proposal[i][1] for i in selects ]

        tokens_ids = [ dataset_dict['tokens_ids'][i,:].astype(np.int64) for i in selects1 ]
        mid_tokens_ids = [ dataset_dict['mid_tokens_ids'][i,:].astype(np.int64) for i in selects2 ]
        target_ids = [ dataset_dict['target_ids'][i,:].astype(np.int64) for i in selects1 ]
        mid_target_ids = [ dataset_dict['mid_target_ids'][i,:].astype(np.int64) for i in selects2 ]
        g_tokens_type = [ np.ones((len(dataset_dict['tokens_ids'][i,:]), ), dtype=np.int64) for i in selects1 ]
        mid_g_tokens_type = [ np.ones((len(dataset_dict['mid_tokens_ids'][i,:]), ), dtype=np.int64) for i in selects2 ]
        ret.update({
            kfg.SEQ_PER_SAMPLE_NET1: self.seq_per_sample_net1,
            kfg.G_TOKENS_IDS: tokens_ids,
            kfg.G_TARGET_IDS: target_ids,
            kfg.G_TOKENS_TYPE: g_tokens_type,
            'MID_'+kfg.G_TOKENS_IDS: mid_tokens_ids,
            'MID_'+kfg.G_TARGET_IDS: mid_target_ids,
            'MID_'+kfg.G_TOKENS_TYPE: mid_g_tokens_type,
            'STYLE_TOKEN': style
        })
        dict_as_tensor(ret)
        # ret.update({kfg.STYLE: style})
        return ret
