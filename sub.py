import deeplabutils
from detectron2.config import CfgNode
import os
from detectron2.engine import launch
import torch
import importlib
import yaml
import json


import sys

def main(cfg,imgs,train_ratio):
    print("après")
    deeplabutils.register_train_test(imgs,train_ratio=train_ratio)

    trainer = deeplabutils.MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    print("Starting training job...")
    trainer.train()
    print("Training job finished.")
   
if __name__ == "__main__":
    _,cfg,imgs,split = sys.argv
    with open(cfg,'r') as f :
        cfg_data= f.read()
        cfg = CfgNode.load_cfg(cfg_data)
    with open(imgs,'r' ) as f:
        imgs = json.load(f)
    print("avabt_launch")
    launch(main, num_gpus_per_machine=4,dist_url="auto",args=(cfg,imgs,float(split),))
