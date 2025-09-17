"""
From original at https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/__init__.py
Original copyright of Facebook code below, modifications by Yehao Li, Copyright 2021.	
"""

# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
from collections import OrderedDict
import torch
import xmodaler.utils.comm as comm
from xmodaler.checkpoint import XmodalerCheckpointer
from xmodaler.config import get_cfg
from xmodaler.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch, build_engine
from xmodaler.modeling import add_config
from torch.utils.tensorboard import SummaryWriter
import json
torch.backends.cudnn.enabled = True

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    tmp_cfg = cfg.load_from_file_tmp(args.config_file)
    add_config(cfg, tmp_cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts) 
    
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = build_engine(cfg)
    trainer.resume_or_load(resume=args.resume)
    
    if args.eval_only:
        result_json_path = cfg.OUTPUT_DIR+'/test_'+cfg.DATALOADER.TYPE[cfg.INFERENCE.STYLE]+ '.json'
        writer = SummaryWriter(log_dir=cfg.OUTPUT_DIR+'/test_accuracy_'+cfg.DATALOADER.TYPE[cfg.INFERENCE.STYLE])
        res = None
        # if trainer.val_data_loader is not None:
        #     res = trainer.test(trainer.cfg, trainer.model, trainer.val_data_loader, trainer.val_evaluator, epoch=-1)
        # if comm.is_main_process():
        #     print(res)
        epoch=str(int(cfg.MODEL.WEIGHTS.split('Epoch_')[-1].split('_')[0]))+'_'+cfg.DATALOADER.MODE
        if trainer.test_data_loader is not None:
            res = trainer.test(trainer.cfg, trainer.model, trainer.test_data_loader, trainer.test_evaluator, epoch=epoch)
        if comm.is_main_process():
            print(res)
        for sub in res:
            writer.add_scalar(tag=sub,  # name
                              scalar_value=res[sub],  # zongzuobiao
                              global_step=int(cfg.MODEL.WEIGHTS.split('Iter_')[1].split('.pth')[0])  # hengzuobiao
                              )
        if len(res)>0:
            res['epoch']=epoch
            json.dump(res, open(result_json_path, "a"))
        return res
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )