# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import random

import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data.detection_utils import read_image
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer, ColorMode
from predictor import VisualizationDemo


# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        # default="configs/quick_schedules/mask_rcnn_R_50_FPN_3x_inference_acc_test.yaml",
        default="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser



# Create config
mp.set_start_method("spawn", force=True)
args = get_parser().parse_args()
setup_logger(name="fvcore")
logger = setup_logger()
logger.info("Arguments: " + str(args))
cfg = setup_cfg(args)

os.system('export DETECTRON2_DATASETS=/home/jlussang/detectron2/datasets_generated')
register_coco_instances("coco_macha_train", {},
                        "/home/jlussang/detectron2/datasets_generated/coco/annotations/instances_train2017.json",
                        "/home/jlussang/detectron2/datasets_generated/coco/train2017/")
register_coco_instances("coco_macha_val", {},
                        "/home/jlussang/detectron2/datasets_generated/coco/annotations/instances_val2017.json",
                        "/home/jlussang/detectron2/datasets_generated/coco/val2017/")
register_coco_instances("coco_macha_test", {},
                        "/home/jlussang/detectron2/datasets_generated/coco/annotations/instances_test2017.json",
                        "/home/jlussang/detectron2/datasets_generated/coco/test2017/")
register_coco_instances("coco_macha_all", {},
                        "/home/jlussang/detectron2/datasets_generated/coco/annotations/instances_train2017.json",
                        "/home/jlussang/detectron2/datasets_generated/coco/all2017/")
coco_macha_metadata1 = MetadataCatalog.get("coco_macha_train")
coco_macha_metadata2 = MetadataCatalog.get("coco_macha_val")
coco_macha_metadata3 = MetadataCatalog.get("coco_macha_test")
coco_macha_metadata4 = MetadataCatalog.get("coco_macha_all")
# args = default_argument_parser().parse_args()
cfg = get_cfg()
cfg.merge_from_file(
    "../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)
cfg.DATASETS.TRAIN = ()
cfg.DATASETS.TEST = ("coco_macha_val",)  # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.WEIGHTS = "../tools/output/model_0459999.pth"
cfg.SOLVER.IMS_PER_BATCH = 2
GPU_num = 1
cfg.SOLVER.BASE_LR = GPU_num*0.00125
#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.999 # 2Dsegmentation
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05  # 3Dinference
cfg.SOLVER.MAX_ITER = (
    1000000  # 300 (converges already at 400k, which is 24h)
)  # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    128  # 128
)  # faster, and good enough for this toy dataset
#cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # 2Dsegmentation
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 19 # 3Dinference
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
predictor = DefaultPredictor(cfg)


# Loading 300x300 black background
data_background = '/home/jlussang/detectron2/demo/black.jpg'
imz = cv2.imread(data_background)


def masks_bw(filename):
    # Make prediction
    data_f = '/home/jlussang/detectron2/datasets_generated/coco/all2017/' + filename
    im = cv2.imread(data_f)
    outputs = predictor(im)
    print(filename)

    # Retrieve masks
    mask=outputs["instances"].get("pred_masks")
    mask=mask.to("cpu")
    mask=mask.numpy()
    mask=mask.astype(np.uint8) #XYZ

    # Generate black background of same dimensions (as instance_mode=ColorMode.IMAGE_BW does not work)
    #num, h, w = mask.shape
    #background = np.zeros((h, w))
    #cv2.imwrite(data_background, background)


    # Setting Visualizer() parameters
    v = Visualizer(imz[:, :, ::-1], metadata=coco_macha_metadata4, scale=1.0, instance_mode=ColorMode.IMAGE)

    # Drawing all binary masks stacked on the black background image
    cv2.imwrite('/home/jlussang/detectron2/results/' + filename, imz) # in case no masks
    k=0
    for m in mask:  # m is a pytorch Tensor
        v.draw_binary_mask(m, color='white', edge_color='white', alpha=1.0, area_threshold=-1000)
        #v = v.draw_binary_mask(mask[0], color=None, edge_color=None, text=None, alpha=0.5, area_threshold=0)
        #v.draw_binary_mask(m, edge_color='white', alpha=1.0, area_threshold=1.0)
        k+=1
    if k>=1:
        v = v.draw_binary_mask(mask[0], color='white', edge_color='white', alpha=0.0, area_threshold=-1000)
        #v = v.draw_binary_mask(mask[0], edge_color='white', alpha=0.0, area_threshold=1.0)
        # Outputing the results
        img = v.get_image()[:, :, ::-1]
        #cv2.imshow('', img)
        #cv2.waitKey()
        cv2.imwrite('/home/jlussang/detectron2/results/' + filename, img)
        print('Instance written... k=', k)

'''
            binary_mask (ndarray): numpy array of shape (H, W), where H is the image height and
                W is the image width. Each value in the array is either a 0 or 1 value of uint8
                type.
            color: color of the mask. Refer to `matplotlib.colors` for a full list of
                formats that are accepted. If None, will pick a random color.
            edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
                full list of formats that are accepted.
            text (str): if None, will be drawn in the object's center of mass.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            area_threshold (float): a connected component small than this will not be shown.
'''

def masks_bw_all(filepath):
    for file in sorted(os.listdir(filepath)):
        if file.endswith('.jpg'):
            #input = os.path.join(filepath, file)
            #file_name1 = os.path.splitext(os.path.basename(file))[0]
            #file_name2 = os.path.basename(file)
            #print('input=', input)
            #print('file_name1=', file_name1)
            #print('file_name2=', file_name2)
            masks_bw(os.path.basename(file))

masks_bw_all('/home/jlussang/detectron2/datasets_generated/coco/all2017/')
