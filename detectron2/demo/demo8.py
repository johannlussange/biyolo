# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data.detection_utils import read_image
from detectron2.engine import default_argument_parser, DefaultPredictor
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

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


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False



if __name__ == "__main__":
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
    #coco_macha_metadata = MetadataCatalog.get("coco_macha_train")
    # coco_macha_metadata2 = MetadataCatalog.get("coco_macha_val")
    #coco_macha_metadata = MetadataCatalog.get("coco_macha_train").set(thing_classes=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61"])
    #coco_macha_metadata = MetadataCatalog.get("coco_macha_train").set(thing_classes=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"])
    # args = default_argument_parser().parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(
        #"../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
        "../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        #"../configs/COCO-InstanceSegmentation/mask_rcnn_regnety.yaml"
    )
    # cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
    # cfg.DATASETS.TEST = ("coco_macha_test")  # no metrics implemented for this dataset
    # cfg.DATASETS.TRAIN = ("coco_macha_train",)
    cfg.DATASETS.TRAIN = ()
    cfg.DATASETS.TEST = ("coco_macha_all",)  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 1
    #cfg.MODEL.WEIGHTS = "../tools/output/2Dsegmentation_230_random/model_6d.pth"
    cfg.MODEL.WEIGHTS = "../tools/output/biyolo_230_random/model_6d.pth"
    cfg.SOLVER.IMS_PER_BATCH = 2 # 2 choiced
    GPU_num = 1
    # cfg.SOLVER.BASE_LR = GPU_num*0.00125 # Multiply this by the number of GPU !!!
    cfg.SOLVER.BASE_LR = GPU_num*0.00125 # 0.00025
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.999 # 2Dsegmentation
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05 # 3Dinference
    cfg.SOLVER.MAX_ITER = (
        1000000  # 300 (converges already at 400k, which is 24h)
    )  # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        128  # 128
    )  #  number of anchors per image to sample for training (faster, and good enough for this toy dataset)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 19  # 62 classes (bench) # choiced
    #cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 62 classes (bench)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    demo = VisualizationDemo(cfg)
    predictor = DefaultPredictor(cfg)

    def masks_bw(filename):
        # Make prediction
        data_f = '/home/jlussang/detectron2/datasets_generated/coco/all2017/' + filename
        image_raster = cv2.imread(data_f)
        outputs = predictor(image_raster)
        mask_array = outputs['instances'].pred_masks.to("cpu").numpy()
        num_instances = mask_array.shape[0]
        scores = outputs['instances'].scores.to("cpu").numpy()
        labels = outputs['instances'].pred_classes.to("cpu").numpy()
        bbox = outputs['instances'].pred_boxes.to("cpu").tensor.numpy()
        mask_array = np.moveaxis(mask_array, 0, -1)
        mask_array_instance = []
        #https://stackoverflow.com/questions/66599909/converting-detectron2-instance-segmentation-to-opencv-mat-array

        output = np.zeros_like(image_raster)  # black
        print(filename, ' : num_instances = ', num_instances)
        for i in range(num_instances):
            mask_array_instance.append(mask_array[:, :, i:(i + 1)])
            output = np.where(mask_array_instance[i] == True, 255, output)
        cv2.imwrite('/home/jlussang/detectron2/results/' + filename, output)

        # Checking numpy array to see if instances are herein recorded
        imax = output.shape[0]
        jmax = output.shape[1]
        for i in range(0, imax):
            for j in range(0, jmax):
                if (output[i][j][0] > 1): print('Found instance at ', i, j)

        '''
        image_black = np.zeros_like(image_raster) #black
        h = image_raster.shape[0]
        w = image_raster.shape[1]
        image_black_mask = np.zeros([h, w, 3], np.uint8)
        color = (200, 100, 255)
        print(filename, ' : num_instances = ', num_instances)
        for i in range(num_instances):
            image_black = np.zeros_like(image_raster)
            mask_array_instance.append(mask_array[:, :, i:(i + 1)])
            image_black = np.where(mask_array_instance[i] == True, 255, image_black)
            array_image_black = np.asarray(image_black)
            image_black_mask[np.where((array_image_black == [255, 255, 255]).all(axis=2))] = color
        image_black_mask = np.asarray(image_black_mask)
        #output = cv2.addWeighted(image_black, 0.5, image_black_mask, 0.5, 0)
        output = cv2.addWeighted(image_raster, 0.5, image_black_mask, 0.5, 0)
        cv2.imwrite('/home/jlussang/detectron2/results/' + filename, output)
        '''




    def masks_bw_all(filepath):
        for file in sorted(os.listdir(filepath)):
            if file.endswith('.jpg'):
                masks_bw(os.path.basename(file))

    masks_bw_all('/home/jlussang/detectron2/datasets_generated/coco/val2017/')

    '''
    images = glob.glob('/home/jlussang/detectron2/datasets_generated/coco/val2017/*')
    for idx, image in enumerate(images):
        img = cv2.imread(image)
        outputs = predictor(img)
        mask = outputs['instances'].get('pred_masks')
        mask = mask.to('cpu')
        num, h, w = mask.shape
        bin_mask = np.zeros((h, w))
        for m in mask:
            bin_mask += m
        filename = '/home/jlussang/detectron2/results/' + str(idx + 1) + '.png'
        cv2.imwrite(filename, bin_mask)
    '''
