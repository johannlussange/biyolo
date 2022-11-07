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
    #cfg.MODEL.WEIGHTS = "../tools/output/biyolo/biyolo_6d.pth"
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

    # Issue of not all instances being draw, for example on image 1008.jpg, the following marked with X are not drawn:
    # '8 99%', X'4 94%', X'6 90%', '7 88%', X'5 84%', '4 78%', X'5 45%', X'3 45%', '4 41%', X'5 29%', X'9 17%', '7 16%', '3 13%', X'13 11%', X'7 7%']

    # demo9.py calls "predictions, visualized_output = demo.run_on_image_uniques(img, img_black)"
    # This function within predictor.py returns predictions = self.predictor(image), vis_output = visualizer.draw_instance_predictions_uniques(predictions=instances)
    # This latter function within visualizerx.py returns self.output = VisImage(self.img, scale=scale)
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            img_black = np.zeros_like(img)  # black
            start_time = time.time()
            #predictions, visualized_output = demo.run_on_image(img)
            predictions, visualized_output = demo.run_on_imagebw(img, img_black)
            instances_nr = len(predictions["instances"])
            for i in range(0, instances_nr):
                predictions, visualized_output, label = demo.run_on_image_uniques(img, img_black, i)
                logger.info(
                    "{}: {} in {:.2f}s".format(
                        path,
                        "detected {} instances".format(len(predictions["instances"]))
                        if "instances" in predictions
                        else "finished",
                        time.time() - start_time,
                    )
                )

                if args.output:
                    if os.path.isdir(args.output):
                        assert os.path.isdir(args.output), args.output
                        out_filename = os.path.join(args.output, os.path.basename(path))
                        labelstr = str(label)
                        labelstr2 = (labelstr.split("['")[1]).split("']")[0]
                        print('label=', labelstr)
                        height = labelstr2.split(" ")[0]
                        score_pct = labelstr2.split(" ")[1]
                        score = score_pct.split("%")[0]
                        #print('split =', labelstr2)
                        #print('height =', height)
                        #print('score_pct =', score_pct)
                        #print('score =', score)
                        out_filename2 = '../results/' + os.path.splitext(os.path.basename(out_filename))[0] + '_' + str(i) + '_' + str(height) + '_' + str(score) + '.png'
                        print('out_filename2 = ', out_filename2)
                    else:
                        assert len(args.input) == 1, "Please specify a directory with args.output"
                        out_filename2 = args.output
                    visualized_output.save(out_filename2)
                else:
                    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                    cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                    if cv2.waitKey(0) == 27:
                        break  # esc to quit

