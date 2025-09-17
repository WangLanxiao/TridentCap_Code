# 最终test cls ppc
import torch
import os
import csv
import pickle
import json
from tqdm import tqdm
from CLS.cls_lstm import Cls_Classifier

GT_TXT=[
        './tools/text_srilm_ro.txt',
        './tools/text_srilm_fu.txt',
        './tools/text_srilm_po.txt',
        './tools/text_srilm_ne.txt',
        ]
PPL_TRAIN=[
        './tools/text_srilm_ro.count',
        './tools/text_srilm_fu.count',
        './tools/text_srilm_po.count',
        './tools/text_srilm_ne.count',
        ]
PPL_LM=[
        './tools/text_srilm_ro.lm',
        './tools/text_srilm_fu.lm',
        './tools/text_srilm_po.lm',
        './tools/text_srilm_ne.lm',
        ]
for id in range(0,4):
    os.system('./tools/srilm-1.7.1/bin/i686-m64/ngram-count -text ' + GT_TXT[id] + ' -order 3 -write '+PPL_TRAIN[id])
    os.system('./tools/srilm-1.7.1/bin/i686-m64/ngram-count -read ' + PPL_TRAIN[id] + ' -order 3 -lm '+PPL_LM[id]+' -kndiscount1 -kndiscount2 -kndiscount3')


