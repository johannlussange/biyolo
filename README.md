# biyolo
One-shot 3D reconstruction of roof planes from urban satellite images with LOD2

This code take as input one raster satellite image & one shapefile of the associated buildings' roof contours with their corners heights. 

It then creates randomized training/validation/testing sets and produce .json annotations in the COCO format via pycococreator: 

https://github.com/waspinator/pycococreator

This is used to then train and use a MaskR-CNN model (namely, mask_rcnn_R_50_FPN_3x) via the Detectron2 suite to perform a 2D segmentation of the roof planes' contours: 

https://github.com/facebookresearch/detectron2

Once this is done, it likewise uses the same model to perform a 3D reconstruction of the roof corners' heights, with the height being set as a class ("a" for 1 meter, "b" for 2 meters, "c" for 3 meters, etc.). 

Both of these results are then combined and post-processed in several functions of main.py to reconstruct all of the roof planes of the entire urban area of the raster image in 3D. 

The fully trained weights of these two MASKR-CNN models (which take 230 x 230 raster images as input) are available here:

For the 2D segmentation: https://drive.google.com/file/d/1g5h3rBpcm1K6v87sJNlbJieKhwp5u7Ja/view?usp=share_link

For the 3D reconstruction: https://drive.google.com/file/d/1MEw8ecUmwW_9HRUsoxPomLiYdhd7z02l/view?usp=share_link

Remark: A few files of both the detectron2 suite and pycococreator are changed for this code to work. The tweaked native files are given here, and to be placed in the directories of the detectron2 and pycococreator repositories as shown here in the arborescence. 

