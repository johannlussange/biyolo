#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, Metadata
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data import DatasetCatalog
import random
from detectron2.utils.visualizer import Visualizer
import cv2


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    os.system('export DETECTRON2_DATASETS=/home/jlussang/detectron2/datasets_generated')
    register_coco_instances("coco_macha_train", {},
                            "/home/jlussang/detectron2/datasets_generated/coco/annotations/instances_train2017.json",
                            "/home/jlussang/detectron2/datasets_generated/coco/train2017/")
    #coco_macha_metadata = MetadataCatalog.get("coco_macha_train")
    coco_macha_metadata = MetadataCatalog.get("coco_macha_train").set(thing_classes=["hah", "hbh", "hch", "hdh", "heh", "hfh", "hgh", "hhh", "hih", "hjh", "hkh", "hlh", "hmh", "hnh", "hoh", "hph", "hqh", "hrh", "hsh"])
    #coco_macha_metadata = MetadataCatalog.get("coco_macha_train").set(thing_classes=["0"])
    args = default_argument_parser().parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(
        #"../configs/COCO-InstanceSegmentation/mask_rcnn_regnety.yaml"
        "../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" # choiced
        #"../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
        #"../configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
        #"../configs/new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py"
    )
    cfg.DATASETS.TRAIN = ("coco_macha_train",)
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 1 #2

    #cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
    #cfg.MODEL.WEIGHTS = "../tools/output/model_0999999.pth"
    cfg.MODEL.WEIGHTS = "../tools/output/biyolo_230_random/model_6d.pth"
    cfg.SOLVER.IMS_PER_BATCH = 2 #1
    GPU_num = 4
    #cfg.SOLVER.BASE_LR = 0.002/3 #GPU_num*0.00125 # OLD (for 2D segmentation)
    cfg.SOLVER.BASE_LR = 0.00025 # 0.00025/10.0 (for 3D inference)

    # NEW https: // detectron2.readthedocs.io / en / latest / modules / config.html  # config-references
    #cfg.SOLVER.BASE_LR_END = 0.0
    #cfg.SOLVER.WARMUP_ITERS = 100
    #cfg.SOLVER.MOMENTUM = 0.9
    #cfg.SOLVER.NESTEROV = False
    #cfg.SOLVER.WEIGHT_DECAY = 0.0001
    #cfg.SOLVER.GAMMA = 0.1 # OLD
    #cfg.SOLVER.STEPS = (30000,) # OLD
    #cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    #cfg.SOLVER.WARMUP_ITERS = 1000
    #cfg.SOLVER.WARMUP_METHOD = "linear"
    #cfg.SOLVER.CHECKPOINT_PERIOD = 25000

    cfg.SOLVER.MAX_ITER = (
        1000000 # 300 (did 52 859 in 10h already and then failed)
    )  # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        128 # 128
    )  # number of anchors per image to sample for training (faster, and good enough for this toy dataset)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 19  # biyolo
    #cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 2dsegmentation
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print('cfg.OUTPUT_DIR=', cfg.OUTPUT_DIR)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    #trainer.resume_or_load(resume=True) #Using to last training
    trainer.train()

    # Saving the model
    model_save_name = 'coco_macha_train.pth'
    path = F"/home/jlussang/detectron2/datasets_generated/coco/{model_save_name}"
    torch.save(trainer.model.state_dict(), path)

