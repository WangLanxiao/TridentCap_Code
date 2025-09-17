# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import os
import sys
import tempfile
import json
from json import encoder
from xmodaler.config import kfg
from xmodaler.config import configurable
from .build import EVALUATION_REGISTRY

sys.path.append(kfg.COCO_PATH)
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

@EVALUATION_REGISTRY.register()
class COCOEvaler(object):
    def __init__(self, cfg, annfile, output_dir):
        super(COCOEvaler, self).__init__()
        test_style=cfg.INFERENCE.STYLE
        self.test_stylename=cfg.DATALOADER.TYPE[test_style]
        ALL=[
         '../open_source_dataset/transfer_learning/',
         '../open_source_dataset/transfer_learning/stylecaption_merge/senticap_test_positive.json',
         '../open_source_dataset/transfer_learning/stylecaption_merge/senticap_test_negative.json',
         '../open_source_dataset/transfer_learning/stylecaption_merge/flickrstyle_test_funny.json',
         '../open_source_dataset/transfer_learning/stylecaption_merge/flickrstyle_test_romantic.json'
        ]
        annfile=ALL[test_style]
        self.coco = COCO(annfile)
        if not os.path.exists(kfg.TEMP_DIR):
            os.mkdir(kfg.TEMP_DIR)

        if output_dir is not None:
            self.output_dir = os.path.join(output_dir, 'results')
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir)
        else:
            self.output_dir = None

    def eval(self, results, epoch):
        cls=0
        if 'TYPE_predict' in results[0]:
            num=0
            for sub in results:
                if sub['TYPE_predict'] == sub['TYPE_gt']:
                    num += 1
            cls = num / len(results) * 100

        in_file = tempfile.NamedTemporaryFile(mode='w', delete=False, dir=kfg.TEMP_DIR)
        json.dump(results, in_file)
        in_file.close()

        cocoRes = self.coco.loadRes(in_file.name)
        cocoEval = COCOEvalCap(self.coco, cocoRes)
        cocoEval.evaluate()
        os.remove(in_file.name)

        results.append('the style accuracy is ' + str(cls)+'%')
        #################################################  save  results  #######################
        if self.output_dir is not None:
            json.dump(results,
                      open(os.path.join(self.output_dir, str(epoch) + '_' + self.test_stylename + '.json'), "w"))
        print('============================================================')
        print('the style accuracy is ', cls, '%')
        print('============================================================')
        return cocoEval.eval