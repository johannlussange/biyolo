import os
import os.path
import glob
import gdal
from osgeo import gdal
from osgeo import ogr
import json
import numpy as np
import random
import cv2
from itertools import zip_longest
import csv
import pandas as pd
import fiona
import matplotlib.pyplot as plt
from PIL import ImageChops
import PIL.ImageOps
from PIL import Image
from shapely.geometry import Point, Polygon
import rasterio
from rasterio.plot import show
import geopandas as gpd
from math import floor, ceil
from skimage.morphology import flood, flood_fill, label
import copy


RootPath = '/user/jlussang/home/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/0/'
#fShapefileInput_InputPath = RootPath + 'Mour.shp_bativec2.shp'
fShapefileInput_InputPath = RootPath + '20AUG13104849-res32cm_bativec.shp'
#fShapefileInput_InputPath = RootPath + '20AUG13104951-res38cm_bativec.shp'
fShapefileInput_OutputPath = RootPath + 'ShapefileInput/{}.shp'
fShapefileInput_InputPath1 = RootPath + 'RESIDENTIAL_2A_WV03-2020-08-13-10-48-49.shp'
#fShapefileInput_InputPath1 = RootPath + '20AUG13104849-res32cm_RESIDENTIAL.shp'
#fShapefileInput_InputPath1 = RootPath + '20AUG13104951-res38cm_RESIDENTIAL.shp'
fShapefileInput_InputPath2 = RootPath + 'COMMERCIAL_2A_WV03-2020-08-13-10-48-49.shp'
#fShapefileInput_InputPath2 = RootPath + '20AUG13104849-res32cm_COMMERCIAL.shp'
#fShapefileInput_InputPath2 = RootPath + '20AUG13104951-res38cm_COMMERCIAL.shp'
fShapefileInput_InputPath3 = RootPath + 'INDUSTRIAL_2A_WV03-2020-08-13-10-48-49.shp'
#fShapefileInput_InputPath3 = RootPath + '20AUG13104849-res32cm_INDUSTRIAL.shp'
#fShapefileInput_InputPath3 = RootPath + '20AUG13104951-res38cm_INDUSTRIAL.shp'
fRasterOutput_InputPath = RootPath + 'ShapefileInput/*.shp'
fRasterOutput_OutputPath1 = RootPath + 'WV03-2020-08-13-10-48-49_RGB_PSH.tif'
#fRasterOutput_OutputPath1 = RootPath + '20AUG13104849-res32cm_RGBIr_Byte_LxSatLab.tif'
#fRasterOutput_OutputPath1 = RootPath + '20AUG13104951-res38cm_RGBIr_Byte_LxSatLab.tif'
fTifToJpgConvert_InputPath = RootPath + 'RasterOutput/'
fBatchResize_Output = RootPath + 'RasterOutputResized/'
fRasterize_Output = RootPath + 'Rasterized/'
fRasterMosaics_Input2 = RootPath + 'MergedRasterOutput/'
fRasterMosaics_Input3 = RootPath + 'MergedRasterOutputResized/'



# Step 1: Split full vector file in individual vector polygons: https://gis.stackexchange.com/questions/294115/clipping-raster-with-multiple-polygons-and-naming-the-resulting-rasters/294354
def fShapefileInput(InputPath, OutputPath):
    with fiona.open(InputPath, 'r') as dst_in:
        k = 0
        for index, feature in enumerate(dst_in):
            with fiona.open(OutputPath.format(index), 'w', **dst_in.meta) as dst_out:
                dst_out.write(feature)
                k += 1
                print('Writing polygon ', k)

# Step 2: Use these individual vector polygons to clip the full raster image
def fRasterOutput(InputPath, OutputPath1, OutputPath2):
    polygons = glob.glob(InputPath)  # Retrieve all the .shp files
    for polygon in polygons:
        feat = fiona.open(polygon, 'r')
        OutputPath = 'gdalwarp -dstnodata -9999 -cutline {} -crop_to_cutline -of GTiff ' + OutputPath1 + ' ' + OutputPath2 + '{}.tif'
        os.system(OutputPath.format(polygon, feat.name))

# Step 3: Convert these .tif rasters to .jpg to avoid format conflicts
def fTifToJpgConvert(InputPath):
    #command = 'gdal_translate -of png -a_nodata 0 -b 1 -b 2 -b 3 {input} {output}'
    command = 'gdal_translate -of JPEG {input} {output}'
    for file in os.listdir(InputPath):
        if file.endswith('.tif'):
            input = os.path.join(InputPath, file)
            filename = os.path.splitext(os.path.basename(file))[0]
            output = os.path.join(InputPath, filename + '.jpg')
            os.system(command.format(input=input, output=output))

def fTifToPngConvert(InputPath):
    #command = 'gdal_translate -of png -a_nodata 0 -b 1 -b 2 -b 3 {input} {output}'
    command = 'gdal_translate -of png {input} {output}'
    for file in os.listdir(InputPath):
        if file.endswith('.tif'):
            input = os.path.join(InputPath, file)
            filename = os.path.splitext(os.path.basename(file))[0]
            output = os.path.join(InputPath, filename + '.png')
            os.system(command.format(input=input, output=output))

def fCrushTifXml(InputPath):
    command2 = 'rm ' + InputPath + '*.tif'
    os.system(command2)
    command3 = 'rm ' + InputPath + '*.xml'
    os.system(command3)

# Step 4: Backup and resize rasters
def fBatchResize(InputPath, OutputPath):
    #command = 'cp -r ' + InputPath + '*.* ' + OutputPath
    #os.system(command)
    os.listdir(OutputPath)
    for file in os.listdir(OutputPath):
        if file.endswith('.jpg'):
            f_img = OutputPath + file
            print(f_img)
            img = Image.open(f_img)
            img = img.resize((480, 320))
            img.save(f_img)

# Step 4: Backup and resize rasters
def fJpegResize(input, output, width, height):
    img = Image.open(input)
    img = img.resize((width, height))
    img.save(output)

# Step 5: Rasterize the full vector file of roof lines: https://stackoverflow.com/questions/2220749/rasterizing-a-gdal-layer
def fRasterize(InputRaster, Output):
    # Get coordinates of raster image
    src = gdal.Open(InputRaster)
    x_min, xres, xskew, y_max, yskew, yres = src.GetGeoTransform()
    y_min = x_min + (src.RasterXSize * xres)
    x_max = y_max + (src.RasterYSize * yres)
    raster_width = src.RasterXSize
    raster_height = src.RasterYSize
    del src
    # Merge vector files
    VectorFile1 = gpd.read_file(fShapefileInput_InputPath1)
    VectorFile2 = gpd.read_file(fShapefileInput_InputPath2)
    VectorFile3 = gpd.read_file(fShapefileInput_InputPath3)
    VectorFile = gpd.GeoDataFrame(pd.concat([VectorFile1, VectorFile2, VectorFile3]))
    VectorFile.to_file(RootPath + 'MergedVector.shp')
    # Polygon to polyline in QGIS: Vector -> Geometry tools -> Polygons to lines.
    # Then right click go to export -> save as feature -> select ESRI shape -> then under geometry select linestring -> save as MergedVectorLine.
    # Rasterize this file
    command = 'gdal_rasterize -l "MergedVectorLine" -burn 255 -burn 0 -burn 0 -ts '
    #command = 'gdal_rasterize -l "MergedVectorLine" -burn 0 -ot UInt16 -ts '
    #command = 'gdal_rasterize -l "MergedVectorLine" -3d -ts '
    command += str(raster_width) + ' ' + str(raster_height) + ' -a_nodata -9999 -te ' + str(x_min) + ' ' + str(x_max) + ' ' + str(y_min) + ' ' + str(y_max) + ' -of GTiff '
    command += RootPath + 'MergedVectorLine.shp '
    command += Output + 'Rasterized_z.tif'
    os.system(command)
    #Command = 'gdal_translate ' + Output + 'Rasterized.tif -scale 1 0 0 1 ' + Output + 'Rasterized.tif'
    #os.system(Command)

def fFormatClearBinarization(InputPath, Name):
    #command = 'gdal_translate -of png -a_nodata 0 -b 1 -b 2 -b 3 {input} {output}'
    # Name.tif to Name.png
    #command1 = 'gdal_translate -of png -a_nodata -9999 -b 1 '
    command1 = 'gdal_translate -of png '
    command1 += InputPath + Name + '.tif '
    command1 += InputPath + Name + '.png'
    os.system(command1)

    command1 = 'gdal_translate -of GTiff '
    command1 += InputPath + Name + '.png '
    command1 += InputPath + Name + '.tif'
    os.system(command1)
    '''
    # Name.png to NameBinarized.png
    Image = cv2.imread(InputPath + Name + '.png')
    Image_gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    ret, Image_bin = cv2.threshold(Image_gray, 127, 255, cv2.THRESH_BINARY)
    # NameBinarized.png inversion
    Image_bin = cv2.bitwise_not(Image_bin)
    cv2.imwrite(InputPath + Name + 'Binarized.png', Image_bin)
    # NameBinarized.png to NameBinarized.tif
    command1 = 'gdal_translate -of GTiff '
    command1 += InputPath + Name + 'Binarized.png '
    command1 += InputPath + Name + 'Binarized.tif'
    os.system(command1)
    '''

# Step 7: Merge the two rasters, i. e. the rasterized roof lines + raw satellite raster image, and then convert output to jpg in order to avoid format conflicts
def fRasterMerger(Output, Input):
    command = 'gdal_merge.py -o '
    command += Output + 'Merged.tif '
    command += Input + ' '
    command += Output + 'Rasterized.tif'
    os.system(command)
    #fTifToPngConvert(Output)

# Step 8: Generate individual rasters from individual vectors and resize them
def fRasterMosaics(Input1, Input2, Input3, Input4):
    '''
    polygons = glob.glob(Input1)  # Retrieve all the .shp files
    for polygon in polygons:
        feat = fiona.open(polygon, 'r')
        #Command1 = 'gdalwarp -s_srs EPSG:2154 -dstnodata -9999 -cutline {} -crop_to_cutline -of GTiff -to SRC_METHOD=NO_GEOTRANSFORM '
        Command1 = 'gdalwarp -cutline {} -crop_to_cutline '
        Command1 += Input2 + 'Rasterized.tif '
        Command1 += Input3 + '{}_label.tif'
        os.system(Command1.format(polygon, feat.name))
    '''
    # Copy results and re-size them
    Command2 = 'cp -r ' + Input3 + '*.tif ' + Input4
    os.system(Command2)
    os.listdir(Input4)
    for file in os.listdir(Input4):
        f_img = Input4 + file
        # Binarization
        Imagecv = cv2.imread(f_img)
        Image_graycv = cv2.cvtColor(Imagecv, cv2.COLOR_BGR2GRAY)
        ret, Image_bin = cv2.threshold(Image_graycv, 5, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f_img, Image_bin)
        # Resizing
        img = Image.open(f_img)
        img = img.resize((480, 320))
        img.save(f_img)
    # Convert these .tif rasters to .png
    fTifToPngConvert(Input4)

# Gives all sorts of metrics on a given image
def fMetrics(Input):
    ds = gdal.Open(Input)
    width = ds.RasterXSize - 1
    height = ds.RasterYSize - 1
    gt = ds.GetGeoTransform()
    minx = gt[0]
    miny = gt[3] + width * gt[4] + height * gt[5]
    maxx = gt[0] + width * gt[1] + height * gt[2]
    maxy = gt[3]
    Command = 'gdalinfo ' + Input  # To get coordinates of a raster
    os.system(Command)
    print('Width=', width, ', Height=', height, ', Min_x=', minx, ', Min_y=', miny, ', Max_x=', maxx, ', Max_y=', maxy)

# Extracts a file with the CRS--EPSG:32631 xyz-coordinates of all corner points of one given shp polygon
def fVectorCoordOutput(InputPath):
    # https://gis.stackexchange.com/questions/200384/how-to-read-geographic-coordinates-when-the-shapefile-has-a-projected-spatial-re
    CSVFile = open(InputPath + 'CoordinatesOutputUnit.txt', 'w')
    with CSVFile as f:
        FileOutput = csv.writer(f)
        infile = ogr.Open(InputPath)
        layer = infile.GetLayer()
        feature = layer.GetFeature(0)
        coordinates = json.loads(feature.ExportToJson())['geometry']['coordinates'][
            0]  # First point is upper left, and then clockwise
        coordinates.pop(4)  # Removes last point, which is the first
        # print(coordinates)
        # print(f)
        # f.write(f)
        # f.write('\n')
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=1)
        npcoordinates = np.array(coordinates)
        # FileOutput.writerows(coordinates) # (for the list)
        FileOutput.writerows(npcoordinates)  # (for the array)
        # print(npcoordinates[0][0])
        # Equation system
        # https://math.stackexchange.com/questions/2686606/equation-of-a-plane-passing-through-3-points
        A = npcoordinates[0]
        B = npcoordinates[1]
        C = npcoordinates[2]
        X1 = np.subtract(B, A)
        X2 = np.subtract(C, A)
        X = np.cross(X1, X2)
        d = -X[0] * A[0] - X[1] * A[1] - X[2] * A[2]
        Res = [X[0], X[1], X[2], d]
        print(Res)  # Equation of plane a, b, c, d !
        # print(X[0]*A[0] + X[1]*A[1] + X[2]*A[2] + d) # Check

# Extracts a file with the CRS--EPSG:32631 xyz-coordinates of all corner points of shp polygons in given directory
def fVectorCoordOutputAll(InputPath):
    CSVFile = open(InputPath + 'CoordinatesOutput.txt', 'w')
    with CSVFile as f:
        FileOutput = csv.writer(f)
        for file in os.listdir(InputPath):
            if file.endswith('.shp'):
                input = os.path.join(InputPath, file)
                infile = ogr.Open(input)
                layer = infile.GetLayer()
                feature = layer.GetFeature(0)
                coordinates = json.loads(feature.ExportToJson())['geometry']['coordinates'][
                    0]  # First point is upper left, and then clockwise
                np.set_printoptions(suppress=True)
                np.set_printoptions(precision=2)
                npcoordinates = np.array(coordinates)
                print(file)
                # print(npcoordinates)
                f.write(file)
                f.write('\n')
                # FileOutput.writerows(coordinates) # (for the list)
                FileOutput.writerows(npcoordinates)  # (for the array)
                A = npcoordinates[0]
                B = npcoordinates[1]
                C = npcoordinates[2]
                X1 = np.subtract(B, A)
                X2 = np.subtract(C, A)
                X = np.cross(X1, X2)
                d = -X[0] * A[0] - X[1] * A[1] - X[2] * A[2]
                Res = [[X[0], X[1], X[2], d]]
                # print(Res)
                FileOutput.writerows(Res)

# Extracts a np array with the CRS--EPSG:32631 xyz-coordinates of all corner points of shp polygons in given directory + the plane coefficients A, B, C, D determining the plane
def fPolygonsList(InputPath):
    L = []
    for file in os.listdir(InputPath): #InputPath is the directory of all shp polygons
        if file.endswith('.shp'):
            input = os.path.join(InputPath, file)
            infile = ogr.Open(input)
            layer = infile.GetLayer()
            feature = layer.GetFeature(0)
            coordinates = json.loads(feature.ExportToJson())['geometry']['coordinates'][0]  # First point is upper left, and then clockwise
            npcoordinates = np.array(coordinates)
            A = npcoordinates[0]
            B = npcoordinates[1]
            C = npcoordinates[2]
            X1 = np.subtract(B, A)
            X2 = np.subtract(C, A)
            X = np.cross(X1, X2)
            d = -X[0] * A[0] - X[1] * A[1] - X[2] * A[2]
            #Res = [[1000000 * X[0] / d, 1000000 * X[1] / d, 1000000 * X[2] / d]] # We assume d=1000000
            Res = [[X[0] / d, X[1] / d, X[2] / d]] # We assume d=1
            Res = np.concatenate((npcoordinates, Res))
            L.append(Res)
    #M = np.vstack(L)
    return L

# Extracts a np array with the CRS--EPSG:32631 xyz-coordinates of all corner points of shp polygons in given directory
def fPolygonsList2(InputPath):
    L=[]
    for file in os.listdir(InputPath):  # InputPath is the directory of all shp polygons
        if file.endswith('.shp'):
            input = os.path.join(InputPath, file)
            infile = ogr.Open(input)
            layer = infile.GetLayer()
            feature = layer.GetFeature(0)
            coordinates = json.loads(feature.ExportToJson())['geometry']['coordinates'][0]  # First point is upper left, and then clockwise
            npcoordinates = np.array(coordinates)
            #L.append(npcoordinates)
            L.append(npcoordinates[:-1])
    return L


'''
# Extracts a np array with the CRS--EPSG:32631 xyz-coordinates of all corner points of shp polygons in given directory
def fPolygonsList3(InputPath):
    L = []
    k = 0
    for file in os.listdir(InputPath):  # InputPath is the directory of all shp polygons
        if file.endswith('.shp'):
            input = os.path.join(InputPath, file)
            infile = ogr.Open(input)
            layer = infile.GetLayer()
            feature = layer.GetFeature(0)
            #coordinates = json.loads(feature.ExportToJson())['geometry']['coordinates'][0]  # First point is upper left, and then clockwise
            coordinates = json.loads(feature.ExportToJson())['geometry']['coordinates'][0]  # First point is upper left, and then clockwise
            print('k=', k, ' => ', coordinates)
            k += 1
            #exit()
            npcoordinates = np.array(coordinates)
            #L.append(npcoordinates)
            #L.append(npcoordinates[:-1])
    return L
'''

def fPolygonsList4(input):
    L = []
    for feat in fiona.open(input):
        #print('---------')
        #print(feat['geometry']['coordinates'][0][0], ' VS. ', feat['geometry']['coordinates'][0][len(feat['geometry']['coordinates'][0]) - 1])
        Lsub = feat['geometry']['coordinates'][0]
        #print(Lsub[0], ' VS. ', Lsub[len(Lsub) - 1])
        del Lsub[-1]
        #print(Lsub[0], ' VS. ', Lsub[len(Lsub) - 1])
        L.append(Lsub)
        #print(L[len(L)-1][0], ' VS. ', L[len(L)-1][len(L[len(L)-1]) - 1])
    return L

def fPlaneCoeffStats(InputPath):
    L = fPolygonsList(InputPath)
    A_L = []
    B_L = []
    C_L = []
    for p in range(0, len(L)):  # polygons
        A_L.append(L[p][len(L[p]) - 1][0]) # List of all plane coeff A/D
        B_L.append(L[p][len(L[p]) - 1][1]) # List of all plane coeff B/D
        C_L.append(L[p][len(L[p]) - 1][2]) # List of all plane coeff C/D
    d = [A_L, B_L, C_L]
    export_data = zip_longest(*d, fillvalue='')
    with open('/user/jlussang/home/Desktop/Grid/PlaneCoeffDistrib.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(("A/D", "B/D", "C/D"))
        wr.writerows(export_data)
    myfile.close()
    # Now normalizing to RGB
    A_RGB = []
    B_RGB = []
    C_RGB = []
    mini = min(min(A_L), min(B_L), min(C_L))
    maxi = max(max(A_L), max(B_L), max(C_L))
    delta = maxi - mini
    for k in range(0, len(A_L)):
        A_RGB.append(255 * (A_L[k]-mini) / delta)
        B_RGB.append(255 * (B_L[k] - mini) / delta)
        C_RGB.append(255 * (C_L[k] - mini) / delta)
    d = [A_RGB, B_RGB, C_RGB]
    export_data = zip_longest(*d, fillvalue='')
    with open('/user/jlussang/home/Desktop/Grid/PlaneCoeffDistrib_RGB.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(("A/D", "B/D", "C/D"))
        wr.writerows(export_data)
    myfile.close()
    # Same but sorted
    A_L.sort()
    B_L.sort()
    C_L.sort()
    d = [A_L, B_L, C_L]
    export_data = zip_longest(*d, fillvalue='')
    with open('/user/jlussang/home/Desktop/Grid/PlaneCoeffDistrib_sorted.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow((" ", " ", " "))
        wr.writerow((" ", " ", " "))
        wr.writerow(("Sorted A/D", "Sorted B/D", "Sorted C/D"))
        wr.writerows(export_data)
    myfile.close()




# Extracts a list of the names of the polygons in order
def fPolygonsNames(InputPath):
    L = []
    for file in os.listdir(InputPath): #InputPath is the directory of all shp polygons
        if file.endswith('.shp'):
            Res = file
            #Res.split('.')
            #L.append(Res[0])
            L.append(Res)
    return L


def fResizeRaster(input, output, width, height):
    #Command = 'gdalwarp -of GTiff -co COMPRESS=DEFLATE -ts ' + str(width+1) + ' ' + str(height+1) + ' -r cubic -overwrite ' + input + ' ' + output
    Command = 'gdalwarp -of GTiff -co COMPRESS=DEFLATE -ts ' + str(width) + ' ' + str(height) + ' -r bilinear -overwrite ' + input + ' ' + output
    os.system(Command)

def fClipRaster(input, output, xmin, ymin, xmax, ymax):
    Command = 'gdalwarp -te ' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax) + ' ' + input + ' ' + output
    #Command += ' SRC_METHOD=NO_GEOTRANSFORM'
    os.system(Command)

# Takes a detectron2 masked image and replace mask colors by dsm altitude values
def fDsming(treshold, input_dsm, input_mask, output):
    M_dsm = cv2.imread(input_dsm, cv2.IMREAD_UNCHANGED) # np array
    M_mask = cv2.imread(input_mask)  # np array
    width = M_dsm.shape[0]
    height = M_dsm.shape[1]
    M_result = np.zeros((width, height), dtype=np.float32)
    print(M_dsm.shape)
    print(M_mask.shape)
    print(M_result.shape)
    for i in range(0, width, 1):
        for j in range(0, height, 1):
            if (M_dsm[i][j]<0): M_dsm[i][j] = 100.0
            print('i=', i,'/', width, ' ; j=', j,'/', height)
            if (M_mask[i][j][0] >= treshold) or (M_mask[i][j][1] >= treshold) or (M_mask[i][j][2] >= treshold):
                M_result[i][j] = M_dsm[i][j]
                print('Found roof RGB: ', M_mask[i][j][0], M_mask[i][j][1], M_mask[i][j][2], ' => ', M_result[i][j])
    cv2.imwrite(output, M_result)

def fReproject(input, output):
    Command = 'gdalwarp -t_srs EPSG:4326 ' + input + ' ' + output
    os.system(Command)

def fRasterNumpyze(input):
    M = cv2.imread(input)
    # Other method for opencv issues with tiff files
    #g_input = gdal.Open(input)
    #M = np.array(g_input.GetRasterBand(1).ReadAsArray()) # This is just the first band though !
    return M

def fRasterCoordinize(input, width, height):
    im = cv2.imread(input)
    width = im.shape[0]
    height = im.shape[1]
    ds = gdal.Open(input)
    M = np.arange(float(width*height*2)).reshape(width, height, 2)
    M.fill(0)
    gt = ds.GetGeoTransform()
    for i in range(0, width): #row nb
        for j in range(0, height): #column nb
            #print(gt[0] + j*gt[1] + i*gt[2])
            M[i][j][0] = gt[0] + j*gt[1] + i*gt[2]
            M[i][j][1] = gt[3] + j*gt[4] + i*gt[5]
    #print(M[10][20])
    return M

# Outputs a matrix M_M as mask for a soup of planes defined by their 3 coefficients A/D, B/D, C/D (with D=1000000)
def fTrainingSetGenerator():
    # np arrays of the two stereo rasters
    Ax = fRasterNumpyze('/user/jlussang/home/Desktop/LuxCarta/Capsule/RawData/BasicAx.tif')
    Bx = fRasterNumpyze('/user/jlussang/home/Desktop/LuxCarta/Capsule/RawData/BasicBx.tif')
    # print(Ax)
    # print(Bx)

    # np array of the coordinates (lat, lon)
    M_Ax = fRasterCoordinize('/user/jlussang/home/Desktop/LuxCarta/Capsule/RawData/BasicAx.tif', 768, 768)
    # print(M_Ax[0])

    # Empty np array of the mask
    M_M = np.arange(float(Ax.shape[0] * Ax.shape[1] * 3)).reshape(Ax.shape[0], Ax.shape[1], 3)
    M_M.fill(0)
    for i in range(0, M_M.shape[0]):
        for j in range(0, M_M.shape[1]):
            M_M[i][j][2] = 1  # C/D is set by default at 1 for places with no building

    # List of nparrays, each consisting of the corner points and plane coeff of each shp polygon
    M_L = fPolygonsList('/user/jlussang/home/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/0/ShapefileInput/')
    # print(M_L)
    # print(len(M_L))

    CSVFile = open('/user/jlussang/home/Desktop/OutputLog.txt', 'w')
    for p in range(0, len(M_L)):  # polygons
        Polygon_coord = []
        Polygon_lats = []
        Polygon_lons = []
        for c in range(0, len(M_L[
                                  p]) - 1):  # number of entries of each polygon array (minus 2 because the last point is the first, and because of plane coefficients at the end)
            Polygon_coord.append((M_L[p][c][0], M_L[p][c][1]))
            Polygon_lats.append(M_L[p][c][0])
            Polygon_lons.append(M_L[p][c][1])
        minlat = min(Polygon_lats)
        minlon = min(Polygon_lons)
        maxlat = max(Polygon_lats)
        maxlon = max(Polygon_lons)
        if (maxlat < M_Ax[0][0][0]) or (minlat > M_Ax[0][M_Ax.shape[0] - 1][0]) or (
                maxlon < M_Ax[M_Ax.shape[0] - 1][0][1]) or (minlon > M_Ax[0][0][1]):
            # print('Polygon_coord=', Polygon_coord)
            # print('Polygon_lats=', Polygon_lats)
            # print('Polygon_lons=', Polygon_lons)
            # print('maxlat=', maxlat, '<', M_Ax[0][0][0], ' or minlat=', minlat, '>', M_Ax[0][M_Ax.shape[0]-1][0], ' or maxlon=', maxlon, '<', M_Ax[M_Ax.shape[0]-1][0][1], ' or minlon=', minlon, '>', M_Ax[0][0][1])
            # print('p=', p)
            continue
        # print('p=', p)
        imin = 0
        imax = M_Ax.shape[0]
        jmin = 0
        jmax = M_Ax.shape[1]
        for i in range(0, M_Ax.shape[0]):  # rows
            if (M_Ax[i][0][1] >= maxlon):
                imin = i
            if (M_Ax[i][0][1] > minlon):
                imax = i + 1
            for j in range(0, M_Ax.shape[1]):  # columns
                if (M_Ax[0][j][0] < minlat):
                    jmin = j
                if (M_Ax[0][j][0] <= maxlat):
                    jmax = j + 1
        imax = max(M_Ax.shape[0], imax)
        jmax = max(M_Ax.shape[1], jmax)
        print('imin=', imin, ' imax=', imax, ' jmin=', jmin, ' jmax=', jmax)
        Polygonus = Polygon(Polygon_coord)
        for i in range(imin, imax):  # rows
            for j in range(jmin, jmax):  # columns
                P = Point(M_Ax[i][j][0], M_Ax[i][j][1])
                Cond = P.within(Polygonus)
                print('i=', i, '/', M_Ax.shape[0], ' j=', j, '/', M_Ax.shape[1], ' p=', p, '/', len(M_L))
                if Cond == True:
                    M_M[i][j][0] = M_L[p][len(M_L[p]) - 1][0]  # plane coeff A/D
                    M_M[i][j][1] = M_L[p][len(M_L[p]) - 1][1]  # plane coeff B/D
                    M_M[i][j][2] = M_L[p][len(M_L[p]) - 1][2]  # plane coeff C/D
                    print('Point (', M_Ax[i][j][0], ', ', M_Ax[i][j][1], ') is in polygon !')
                    CSVFile.write('Point (' + str(M_Ax[i][j][0]) + ', ' + str(M_Ax[i][j][1]) + ') is in polygon !')
                    CSVFile.write('\n')
                    CSVFile.write(
                        '=> M_M[' + str(i) + '][' + str(j) + '][0] = A/D = ' + str(M_L[p][len(M_L[p]) - 1][0]))
                    CSVFile.write('\n')
                    CSVFile.write(
                        '   M_M[' + str(i) + '][' + str(j) + '][1] = B/D = ' + str(M_L[p][len(M_L[p]) - 1][1]))
                    CSVFile.write('\n')
                    CSVFile.write(
                        '   M_M[' + str(i) + '][' + str(j) + '][2] = C/D = ' + str(M_L[p][len(M_L[p]) - 1][2]))
                    CSVFile.write('\n')

    CSVFile = open('/user/jlussang/home/Desktop/OutputM_M.txt', 'w')
    with CSVFile:
        FileOutput = csv.writer(CSVFile)
        FileOutput.writerows(M_M)

# Slices a raster image into individual rasters of size tilesize x tilesize
def fGridify(input, output, tilesize):
    src = gdal.Open(input)
    width = src.RasterXSize
    height = src.RasterYSize
    count=1
    for i in range(0, width, tilesize):
        for j in range(0, height, tilesize):
            print('i=', i, '/', width, ' ; j=', j, '/', height, ' => image 00000000000' + str(count), '.tif')
            #gdaltranString = "gdal_translate -of GTiff -srcwin "+str(i)+", "+str(j)+", "+str(tilesize)+", " +str(tilesize)+" /user/jlussang/home/Desktop/Grid/MourmelonSmall2.tif /user/jlussang/home/Desktop/Grid/GridsRaster768/00000"+str(count)+".tif"
            gdaltranString = "gdal_translate -of GTiff -srcwin "+str(i)+", "+str(j)+", "+str(tilesize)+", " +str(tilesize)+" "+input+" "+output+str('00000000000')+str(count)+".tif"
            count += 1
            os.system(gdaltranString)
    fTifToJpgConvert(output)

def fGridifyPadding_general(input, output, tilesize, padding):
    src = gdal.Open(input)
    width = src.RasterXSize
    height = src.RasterYSize
    count=1
    for i in range(0, width, tilesize-padding):
        for j in range(0, height, tilesize-padding):
            #print('i=', i, '/', width, ' ; j=', j, '/', height, ' => image 00000000000' + str(count), '.tif')
            #gdaltranString = "gdal_translate -of GTiff -srcwin "+str(i)+", "+str(j)+", "+str(tilesize)+", " +str(tilesize)+" /user/jlussang/home/Desktop/Grid/MourmelonSmall2.tif /user/jlussang/home/Desktop/Grid/GridsRaster768/00000"+str(count)+".tif"
            #gdaltranString = "gdal_translate -of GTiff -srcwin "+str(i)+", "+str(j)+", "+str(tilesize)+", " +str(tilesize)+" "+input+" "+output+str('00000000000')+str(count)+".tif"
            # In order to prevent the contrast issue with gdal_translate, plot a histogram in GDAL of the whole raster (right click property, histogram) and note the min & max values of the three RGB bands
            # Then use the following argument as -scale min max -exponent 0.7 to change the brightness

            # FOR GENERAL
            gdaltranString = "gdal_translate -of GTiff -srcwin "+str(i)+", "+str(j)+", "+str(tilesize)+", " +str(tilesize)+" "+input+" "+output+str('00000000000')+str(count)+".tif"

            # FOR RASTER
            #gdaltranString = "gdal_translate -scale 0 950 0 255 -exponent 0.7 -ot Byte -srcwin "+str(i)+", "+str(j)+", "+str(tilesize)+", " +str(tilesize)+" "+input+" "+output+str('00000000000')+str(count)+".tif -of GTiff" #-scale 55 200

            # GOOD FOR DTM
            #gdaltranString = "gdal_translate -of GTiff -srcwin "+str(i)+", "+str(j)+", "+str(tilesize)+", " +str(tilesize)+" "+input+" "+output+str('00000000000')+str(count)+".tif"

            # BAD FOR DTM
            #gdaltranString = "gdal_translate -scale 105 136 0 255 -exponent 0.7 -ot Byte -srcwin "+str(i)+", "+str(j)+", "+str(tilesize)+", " +str(tilesize)+" "+input+" "+output+str('00000000000')+str(count)+".tif -of GTiff" #-scale 55 200

            count += 1
            os.system(gdaltranString)
    fTifToJpgConvert(output)

def fGridifyPadding_raster(input, output, tilesize, padding):
    src = gdal.Open(input)
    width = src.RasterXSize
    height = src.RasterYSize
    count=1
    for i in range(0, width, tilesize-padding):
        for j in range(0, height, tilesize-padding):
            #print('i=', i, '/', width, ' ; j=', j, '/', height, ' => image 00000000000' + str(count), '.tif')
            #gdaltranString = "gdal_translate -of GTiff -srcwin "+str(i)+", "+str(j)+", "+str(tilesize)+", " +str(tilesize)+" /user/jlussang/home/Desktop/Grid/MourmelonSmall2.tif /user/jlussang/home/Desktop/Grid/GridsRaster768/00000"+str(count)+".tif"
            #gdaltranString = "gdal_translate -of GTiff -srcwin "+str(i)+", "+str(j)+", "+str(tilesize)+", " +str(tilesize)+" "+input+" "+output+str('00000000000')+str(count)+".tif"
            # In order to prevent the contrast issue with gdal_translate, plot a histogram in GDAL of the whole raster (right click property, histogram) and note the min & max values of the three RGB bands
            # Then use the following argument as -scale min max -exponent 0.7 to change the brightness

            # FOR RASTER
            #gdaltranString = "gdal_translate -scale 0 950 0 255 -exponent 0.7 -ot Byte -srcwin "+str(i)+", "+str(j)+", "+str(tilesize)+", " +str(tilesize)+" "+input+" "+output+str('00000000000')+str(count)+".tif -of GTiff" #-scale 55 200

            # GOOD FOR DTM
            gdaltranString = "gdal_translate -of GTiff -srcwin "+str(i)+", "+str(j)+", "+str(tilesize)+", " +str(tilesize)+" "+input+" "+output+str('00000000000')+str(count)+".tif"

            # BAD FOR DTM
            #gdaltranString = "gdal_translate -scale 105 136 0 255 -exponent 0.7 -ot Byte -srcwin "+str(i)+", "+str(j)+", "+str(tilesize)+", " +str(tilesize)+" "+input+" "+output+str('00000000000')+str(count)+".tif -of GTiff" #-scale 55 200

            count += 1
            os.system(gdaltranString)
    fTifToPngConvert(output)

def fGridifyPadding_dtm(input, output, tilesize, padding):
    src = gdal.Open(input)
    width = src.RasterXSize
    height = src.RasterYSize
    count=1
    for i in range(0, width, tilesize-padding):
        for j in range(0, height, tilesize-padding):
            print('i=', i, '/', width, ' ; j=', j, '/', height, ' => image 00000000000' + str(count), '.tif')
            #gdaltranString = "gdal_translate -of GTiff -srcwin "+str(i)+", "+str(j)+", "+str(tilesize)+", " +str(tilesize)+" /user/jlussang/home/Desktop/Grid/MourmelonSmall2.tif /user/jlussang/home/Desktop/Grid/GridsRaster768/00000"+str(count)+".tif"
            #gdaltranString = "gdal_translate -of GTiff -srcwin "+str(i)+", "+str(j)+", "+str(tilesize)+", " +str(tilesize)+" "+input+" "+output+str('00000000000')+str(count)+".tif"
            # In order to prevent the contrast issue with gdal_translate, plot a histogram in GDAL of the whole raster (right click property, histogram) and note the min & max values of the three RGB bands
            # Then use the following argument as -scale min max -exponent 0.7 to change the brightness

            # FOR RASTER
            #gdaltranString = "gdal_translate -scale 0 950 0 255 -exponent 0.7 -ot Byte -srcwin "+str(i)+", "+str(j)+", "+str(tilesize)+", " +str(tilesize)+" "+input+" "+output+str('00000000000')+str(count)+".tif -of GTiff" #-scale 55 200

            # GOOD FOR DTM
            gdaltranString = "gdal_translate -of GTiff -srcwin "+str(i)+", "+str(j)+", "+str(tilesize)+", " +str(tilesize)+" "+input+" "+output+str('00000000000')+str(count)+".tif"

            # BAD FOR DTM
            #gdaltranString = "gdal_translate -scale 105 136 0 255 -exponent 0.7 -ot Byte -srcwin "+str(i)+", "+str(j)+", "+str(tilesize)+", " +str(tilesize)+" "+input+" "+output+str('00000000000')+str(count)+".tif -of GTiff" #-scale 55 200

            count += 1
            os.system(gdaltranString)
    fTifToJpgConvert(output)

def fUngridify(inputdir, output, tilesize, padding):
    tilesize_pad = tilesize - padding
    width = int((ceil(8508/tilesize_pad))*tilesize_pad)
    height = int((ceil(9725/tilesize_pad))*tilesize_pad)
    PatronBlack = np.zeros((height, width, 3), dtype=np.uint8)
    for file in sorted(os.listdir(inputdir)):
        if file.endswith('.png'):
            input = os.path.join(inputdir, file)
            file_name = os.path.splitext(os.path.basename(file))[0]
            file_nr = int(file_name)
            print('file_name=', file_name, ', file_nr=', file_nr)
            # Identifying i0 and j0 in PatronBlack that correspond to i=0, j=0 of the tile image file_name
            #height_corrected = max(height, (floor(height/tilesize)+1)*tilesize)
            i0 = floor((tilesize_pad*(file_nr-1)) % height)
            j0 = tilesize_pad * floor((tilesize_pad * (file_nr - 1)) / height)
            print(file_nr, ': i0=', i0, '/', height, ' ; j0=', j0, '/', width)
            tile = cv2.imread(input)  # np array
            for i in range(i0, min(i0+tilesize_pad, height), 1):
                for j in range(j0, min(j0+tilesize_pad, width), 1):
                    PatronBlack[i][j][0] = tile[i - i0][j - j0][0]
                    PatronBlack[i][j][1] = tile[i - i0][j - j0][1]
                    PatronBlack[i][j][2] = tile[i - i0][j - j0][2]
    cv2.imwrite(output, PatronBlack)

def fRasterize2(InputRaster, Output):
    # Get coordinates of raster image
    src = gdal.Open(InputRaster)
    x_min, xres, xskew, y_max, yskew, yres = src.GetGeoTransform()
    y_min = x_min + (src.RasterXSize * xres)
    x_max = y_max + (src.RasterYSize * yres)
    raster_width = src.RasterXSize
    raster_height = src.RasterYSize
    del src
    # Rasterize this file
    command = 'gdal_rasterize -l "8" -burn 255 -burn 0 -burn 0 -ts '
    #command = 'gdal_rasterize -l "MergedVectorLine" -burn 0 -ot UInt16 -ts '
    #command = 'gdal_rasterize -l "MergedVectorLine" -3d -ts '
    command += str(raster_width) + ' ' + str(raster_height) + ' -a_nodata -9999 -te ' + str(x_min) + ' ' + str(x_max) + ' ' + str(y_min) + ' ' + str(y_max) + ' -of GTiff '
    command += '/user/jlussang/home/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/0/ShapefileInput/8.shp '
    command += Output + 'RasterizedRoof.tif'
    os.system(command)
    #Command = 'gdal_translate ' + Output + 'Rasterized.tif -scale 1 0 0 1 ' + Output + 'Rasterized.tif'
    #os.system(Command)

# Outputs several matrix M_M as mask for a soup of planes defined by their 3 coefficients A/D, B/D, C/D (with D=1000000)
def fTrainingSetGeneratorCOCO(input, name):
    input2 = str(input+name+'.tif')
    src = gdal.Open(input2)
    width = src.RasterXSize
    height = src.RasterYSize
    # np arrays of the two stereo rasters
    Ax = fRasterNumpyze(input2)
    # np array of the coordinates (lat, lon)
    M_Ax = fRasterCoordinize(input2, width, height)

    # List of nparrays, each consisting of the corner points and plane coeff of each shp polygon
    M_L = fPolygonsList('/user/jlussang/home/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/0/ShapefileInput/')
    # List of the file names of each polygon
    M_L_names = fPolygonsNames('/user/jlussang/home/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/0/ShapefileInput/')
    # print(M_L)
    # print(len(M_L))
    #M_L[p][len(M_L[p]) - 1][0]
    CSVFile = open('/user/jlussang/home/Desktop/Grid/OutputLog.txt', 'w')
    #print(len(M_L))
    kID = 0
    for p in range(0, len(M_L)):  # polygons
        Polygon_coord = []
        Polygon_lats = []
        Polygon_lons = []
        # number of entries of each polygon array (minus 1 because of plane coefficients at the end)
        for c in range(0, len(M_L[p]) - 1):
            Polygon_coord.append((M_L[p][c][0], M_L[p][c][1]))
            Polygon_lats.append(M_L[p][c][0])
            Polygon_lons.append(M_L[p][c][1])
        minlat = min(Polygon_lats)
        minlon = min(Polygon_lons)
        maxlat = max(Polygon_lats)
        maxlon = max(Polygon_lons)
        imin = 0
        imax = M_Ax.shape[0]
        jmin = 0
        jmax = M_Ax.shape[1]
        if (maxlat < M_Ax[0][0][0]) or (minlat > M_Ax[0][M_Ax.shape[1] - 1][0]) or (
                maxlon < M_Ax[M_Ax.shape[0] - 1][0][1]) or (minlon > M_Ax[0][0][1]):
            # print('Polygon_coord=', Polygon_coord)
            # print('Polygon_lats=', Polygon_lats)
            # print('Polygon_lons=', Polygon_lons)
            # print('maxlat=', maxlat, '<', M_Ax[0][0][0], ' or minlat=', minlat, '>', M_Ax[0][M_Ax.shape[0]-1][0], ' or maxlon=', maxlon, '<', M_Ax[M_Ax.shape[0]-1][0][1], ' or minlon=', minlon, '>', M_Ax[0][0][1])
            # print('p=', p)
            continue
        print('Polygon ', M_L_names[p])
        for i in range(0, imax):  # rows
            if (M_Ax[i][0][1] >= maxlon):
                imin = i
            if (M_Ax[i][0][1] > minlon):
                imax = i + 1
            for j in range(0, jmax):  # columns
                if (M_Ax[0][j][0] < minlat):
                    jmin = j
                if (M_Ax[0][j][0] <= maxlat):
                    jmax = j + 1
        imax = max(M_Ax.shape[0], imax)
        jmax = max(M_Ax.shape[1], jmax)
        #print('imin=', imin, ' imax=', imax, ' jmin=', jmin, ' jmax=', jmax)
        Polygonus = Polygon(Polygon_coord)

        # Empty np array of the mask with coefficients
        M_M = np.arange(float(Ax.shape[0] * Ax.shape[1] * 3)).reshape(Ax.shape[0], Ax.shape[1], 3)
        M_M.fill(0)
        # Empty np array of the mask with b/w polygons
        # M_Mbw = np.arange(float(Ax.shape[0] * Ax.shape[1] * 3)).reshape(Ax.shape[0], Ax.shape[1], 3)
        M_Mbw = np.zeros((height, width, 3), dtype=np.uint8)
        M_Mbw.fill(0)  # black everywhere
        for i in range(0, M_M.shape[0]):
            for j in range(0, M_M.shape[1]):
                M_M[i][j][2] = 1  # C/D is set by default at 1 for places with no building
        CondInit = 0
        for i in range(imin, imax):  # rows
            for j in range(jmin, jmax):  # columns
                P = Point(M_Ax[i][j][0], M_Ax[i][j][1])
                Cond = P.within(Polygonus)
                #print('i=', i, '/', M_Ax.shape[0], ' j=', j, '/', M_Ax.shape[1], ' p=', p, '/', len(M_L))
                if Cond == True:
                    CondInit = 1
                    M_M[i][j][0] = M_L[p][len(M_L[p]) - 1][0]  # plane coeff A/D
                    M_M[i][j][1] = M_L[p][len(M_L[p]) - 1][1]  # plane coeff B/D
                    M_M[i][j][2] = M_L[p][len(M_L[p]) - 1][2]  # plane coeff C/D
                    M_Mbw[i][j] = [255, 255, 255]  # white
                    #print('Point (', M_Ax[i][j][0], ', ', M_Ax[i][j][1], ') is in polygon !')
                    CSVFile.write('Point (' + str(M_Ax[i][j][0]) + ', ' + str(M_Ax[i][j][1]) + ') is in polygon !')
                    CSVFile.write('\n')
                    CSVFile.write(
                        '=> M_M[' + str(i) + '][' + str(j) + '][0] = A/D = ' + str(M_L[p][len(M_L[p]) - 1][0]))
                    CSVFile.write('\n')
                    CSVFile.write(
                        '   M_M[' + str(i) + '][' + str(j) + '][1] = B/D = ' + str(M_L[p][len(M_L[p]) - 1][1]))
                    CSVFile.write('\n')
                    CSVFile.write(
                        '   M_M[' + str(i) + '][' + str(j) + '][2] = C/D = ' + str(M_L[p][len(M_L[p]) - 1][2]))
                    CSVFile.write('\n')
        img = Image.fromarray(M_Mbw, 'RGB')
        img.save('/user/jlussang/home/Desktop/Grid/detectron2_results/Masks/' + str(name) + '_' + str(M_L_names[p].split('.')[0]) + '_' + str(kID) + '.png')
        if CondInit == 1:
            kID += 1
        '''
        CSVFile2 = open('/user/jlussang/home/Desktop/Grid/OutputM_M.txt', 'w')
        with CSVFile2:
            FileOutput = csv.writer(CSVFile2)
            FileOutput.writerows(M_M)
        CSVFile3 = open('/user/jlussang/home/Desktop/Grid/OutputM_Mbw.txt', 'w')
        with CSVFile3:
            FileOutput = csv.writer(CSVFile3)
            FileOutput.writerows(M_Mbw)
        '''

def fTrainingSetGeneratorCOCO_blended(input, name):
    resolution = 1 # pixel resolution of the raster in meters (here 38 cm but for simplicity, we keep 1 m)
    input2 = str(input+name+'.tif')
    src = gdal.Open(input2)
    width = src.RasterXSize
    height = src.RasterYSize
    # np arrays of the two stereo rasters
    Ax = fRasterNumpyze(input2)
    # np array of the coordinates (lat, lon)
    M_Ax = fRasterCoordinize(input2, width, height)
    # List of nparrays, each consisting of the corner points
    #M_L = fPolygonsList('/user/jlussang/home/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/0/ShapefileInput/')
    # List of the file names of each polygon
    #M_L_names = fPolygonsNames('/user/jlussang/home/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/0/ShapefileInput/')
    # List of nparrays, each consisting of the corner points
    #M_L = fPolygonsList4('/user/jlussang/home/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/0/MergedVector.shp')
    M_La = fPolygonsList4('/user/jlussang/home/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/0/COMMERCIAL_2A_WV03-2020-08-13-10-48-49.shp')
    M_Lb = fPolygonsList4('/user/jlussang/home/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/0/INDUSTRIAL_2A_WV03-2020-08-13-10-48-49.shp')
    M_Lc = fPolygonsList4('/user/jlussang/home/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/0/RESIDENTIAL_2A_WV03-2020-08-13-10-48-49.shp')
    M_L = M_La + M_Lb + M_Lc
    CSVFile = open('/user/jlussang/home/Desktop/Grid/OutputLog.txt', 'w')
    #M_Msup = np.zeros((Ax.shape[0], Ax.shape[1], 3), dtype=np.uint8) # All point in one masked image
    kID = 0
    for p in range(0, len(M_L)):  # polygons
        print(p,'/', len(M_L))
        Polygon_coord = []
        Polygon_lats = []
        Polygon_lons = []
        # number of entries of each polygon array
        for c in range(0, len(M_L[p])):
            Polygon_coord.append((M_L[p][c][0], M_L[p][c][1], M_L[p][c][2])) # latitude, longitude, altitutde of each roof corner
            Polygon_lats.append(M_L[p][c][0])
            Polygon_lons.append(M_L[p][c][1])
        minlat = min(Polygon_lats)
        minlon = min(Polygon_lons)
        maxlat = max(Polygon_lats)
        maxlon = max(Polygon_lons)
        imin = 0
        imax = M_Ax.shape[0]
        jmin = 0
        jmax = M_Ax.shape[1]
        if (maxlat < M_Ax[0][0][0]) or (minlat > M_Ax[0][M_Ax.shape[1] - 1][0]) or (
                maxlon < M_Ax[M_Ax.shape[0] - 1][0][1]) or (minlon > M_Ax[0][0][1]):
            continue
        #print('Polygon ', M_L_names[p], ' (has ', len(Polygon_coord), ' corners): ', Polygon_coord)
        for i in range(0, imax):  # rows
            if (M_Ax[i][0][1] >= maxlon):
                imin = i
            if (M_Ax[i][0][1] > minlon):
                imax = i + 1
            for j in range(0, jmax):  # columns
                if (M_Ax[0][j][0] < minlat):
                    jmin = j
                if (M_Ax[0][j][0] <= maxlat):
                    jmax = j + 1
        imax = max(M_Ax.shape[0], imax)
        jmax = max(M_Ax.shape[1], jmax)
        # Finding for each corner of polygon p its i and j indices in M_Ax
        for k in range(0, len(Polygon_coord)): # corner points
            # Empty np array of the mask with corners
            M_M = np.zeros((Ax.shape[0], Ax.shape[1], 3), dtype=np.uint8)
            coord_i = imin
            coord_j = jmin
            for i in range(0, M_Ax.shape[0]): # rows
                if (M_Ax[i][j][1] >= Polygon_coord[k][1]): coord_i = i
                for j in range(0, M_Ax.shape[1]): # columns
                    if (M_Ax[i][j][0] <= Polygon_coord[k][0]): coord_j = j
            #print('     Corner k=', k, ' treated...: coord_i=', coord_i, ' (polygon imin, imax =', imin, imax, '); coord_j=', coord_j, '(jmin, jmax =', jmin, jmax, ')')
            if (coord_i >= 1) and (coord_i <= M_Ax.shape[0]-2) and (coord_j >= 1) and (coord_j <= M_Ax.shape[1]-2):
                M_M[coord_i][coord_j] = [255, 255, 255]
                M_M[coord_i-1][coord_j-1] = [255, 255, 255]
                M_M[coord_i-1][coord_j] = [255, 255, 255]
                M_M[coord_i-1][coord_j+1] = [255, 255, 255]
                M_M[coord_i][coord_j-1] = [255, 255, 255]
                M_M[coord_i][coord_j+1] = [255, 255, 255]
                M_M[coord_i+1][coord_j-1] = [255, 255, 255]
                M_M[coord_i+1][coord_j] = [255, 255, 255]
                M_M[coord_i+1][coord_j+1] = [255, 255, 255]
                '''             
                M_Msup[coord_i][coord_j] = [255, 255, 255]
                M_Msup[coord_i-1][coord_j-1] = [255, 255, 255]
                M_Msup[coord_i-1][coord_j] = [255, 255, 255]
                M_Msup[coord_i-1][coord_j+1] = [255, 255, 255]
                M_Msup[coord_i][coord_j-1] = [255, 255, 255]
                M_Msup[coord_i][coord_j+1] = [255, 255, 255]
                M_Msup[coord_i+1][coord_j-1] = [255, 255, 255]
                M_Msup[coord_i+1][coord_j] = [255, 255, 255]
                M_Msup[coord_i+1][coord_j+1] = [255, 255, 255]
                '''
                coord_z = int(Polygon_coord[k][2]/resolution) # class of the corner keypoint, in units of resolution
                #coord_z -= 131
                print('         Writing roof corner k=', k, ' at xyz=(', Polygon_coord[k][0], ', ', Polygon_coord[k][1], ', ', Polygon_coord[k][2], ' => (coord_i,coord_j, coord_z)=(', coord_i, ' ,', coord_j, ', ', coord_z, ') => Outputing ', str(name) + '_' + str(coord_z) + '_' + str(kID) + '.png')
                img = Image.fromarray(M_M, 'RGB')
                # Output name is "raster grid name/nb"_"shapefile polygon name/nb"_"roof corner point name/nb"_"altitude in resolution units".png
                #img.save('/user/jlussang/home/Desktop/Grid/detectron2_results/Masks/' + str(name) + '_' + str(M_L_names[p].split('.')[0]) + '_' + str(k) + '_' + str(coord_z) + '.png')

                img.save('/user/jlussang/home/Desktop/Grid/detectron2_results/Masks_blended_300/' + str(name) + '_h' + str(coord_z) + '_' + str(kID) + '.png')
                #img.save('/user/jlussang/home/Desktop/Grid/detectron2_results/Masks_blended_300/' + str(name) + '_roof_' + str(kID) + '.png')

                #print(input_path+file_name.split('_')[0]+'_'+file_name.split('_')[3]+'_'+file_name.split('_')[2]+'.png')
                kID += 1
    #imgsup = Image.fromarray(M_Msup, 'RGB')
    #imgsup.save('/user/jlussang/home/Desktop/Grid/detectron2_results/Masks/Sup/maskedsup_' + str(name) + '.png')



#fTrainingSetGeneratorCOCO_dtm('/user/jlussang/home/Desktop/Grid/detectron2_results/Gridify_finput2_150/', '/user/jlussang/home/Desktop/Grid/detectron2_results/Gridify_dtm_150/', filename)
def fTrainingSetGeneratorCOCO_dtm(inputraster, inputdtm, name):
    resolution = 1 # pixel resolution of the raster in meters (here 38 cm but for simplicity, we keep 1 m)
    inputraster = str(inputraster + name + '.tif')
    inputdtm = str(inputdtm + name + '.tif')
    src = gdal.Open(inputraster)
    width = src.RasterXSize
    height = src.RasterYSize
    # np arrays of the raster
    Ax = fRasterNumpyze(inputraster)
    # np array of the underlying dtm file
    ds = gdal.Open(inputdtm)
    DTMx = np.array(ds.GetRasterBand(1).ReadAsArray())
    # np array of the coordinates (lat, lon)
    M_Ax = fRasterCoordinize(inputraster, width, height)
    # List of nparrays, each consisting of the corner points
    M_La = fPolygonsList4('/user/jlussang/home/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/2/ShapefileInput_raw/20AUG13104951-res38cm_COMMERCIAL.shp')
    M_Lb = fPolygonsList4('/user/jlussang/home/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/2/ShapefileInput_raw/20AUG13104951-res38cm_INDUSTRIAL.shp')
    M_Lc = fPolygonsList4('/user/jlussang/home/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/2/ShapefileInput_raw/20AUG13104951-res38cm_RESIDENTIAL.shp')
    M_L = M_La + M_Lb + M_Lc
    L_pairingpoints = [] # list of unique (i, j) points of roofs in M_M
    #print(M_L[0])
    #print(M_L[0][1][0], M_L[0][1][1], M_L[0][1][2])
    #exit()
    #M_L = fPolygonsList('/user/jlussang/home/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/0/ShapefileInput/')
    # List of the file names of each polygon
    #M_L_names = fPolygonsNames('/user/jlussang/home/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/2/ShapefileInput/')
    #CSVFile = open('/user/jlussang/home/Desktop/OutputLog.txt', 'w')
    #M_Msup = np.zeros((Ax.shape[0], Ax.shape[1], 3), dtype=np.uint8) # All point in one masked image
    kID = 0
    for p in range(0, len(M_L)):  # polygons
        #print(p,'/', len(M_L))
        Polygon_coord = []
        Polygon_lats = []
        Polygon_lons = []
        # number of entries of each polygon array
        for c in range(0, len(M_L[p])):
            #print(M_L[p][c])
            Polygon_coord.append((M_L[p][c][0], M_L[p][c][1], M_L[p][c][2])) # latitude, longitude, altitutde of each roof corner
            Polygon_lats.append(M_L[p][c][0])
            Polygon_lons.append(M_L[p][c][1])
        minlat = min(Polygon_lats)
        minlon = min(Polygon_lons)
        maxlat = max(Polygon_lats)
        maxlon = max(Polygon_lons)
        imin = 0
        imax = M_Ax.shape[0]
        jmin = 0
        jmax = M_Ax.shape[1]
        if (maxlat < M_Ax[0][0][0]) or (minlat > M_Ax[0][M_Ax.shape[1] - 1][0]) or (
                maxlon < M_Ax[M_Ax.shape[0] - 1][0][1]) or (minlon > M_Ax[0][0][1]):
            continue
        #print('Polygon ', p, ' (has ', len(Polygon_coord), ' corners): ', Polygon_coord)
        for i in range(0, imax):  # rows
            if (M_Ax[i][0][1] >= maxlon):
                imin = i
            if (M_Ax[i][0][1] > minlon):
                imax = i + 1
            for j in range(0, jmax):  # columns
                if (M_Ax[0][j][0] < minlat):
                    jmin = j
                if (M_Ax[0][j][0] <= maxlat):
                    jmax = j + 1
        imax = max(M_Ax.shape[0], imax)
        jmax = max(M_Ax.shape[1], jmax)
        # Finding for each corner of polygon p its i and j indices in M_Ax
        for k in range(0, len(Polygon_coord)): # corner points
            # Empty np array of the mask with corners
            M_M = np.zeros((Ax.shape[0], Ax.shape[1], 3), dtype=np.uint8)
            coord_i = imin
            coord_j = jmin
            for i in range(0, M_Ax.shape[0]): # rows
                if (M_Ax[i][j][1] >= Polygon_coord[k][1]): coord_i = i
                for j in range(0, M_Ax.shape[1]): # columns
                    if (M_Ax[i][j][0] <= Polygon_coord[k][0]): coord_j = j
            #print('     Corner k=', k, ' treated...: coord_i=', coord_i, ' (polygon imin, imax =', imin, imax, '); coord_j=', coord_j, '(jmin, jmax =', jmin, jmax, ')')
            if (coord_i >= 1) and (coord_i <= M_Ax.shape[0]-2) and (coord_j >= 1) and (coord_j <= M_Ax.shape[1]-2):
                M_M[coord_i][coord_j] = [255, 255, 255]
                M_M[coord_i-1][coord_j-1] = [255, 255, 255]
                M_M[coord_i-1][coord_j] = [255, 255, 255]
                M_M[coord_i-1][coord_j+1] = [255, 255, 255]
                M_M[coord_i][coord_j-1] = [255, 255, 255]
                M_M[coord_i][coord_j+1] = [255, 255, 255]
                M_M[coord_i+1][coord_j-1] = [255, 255, 255]
                M_M[coord_i+1][coord_j] = [255, 255, 255]
                M_M[coord_i+1][coord_j+1] = [255, 255, 255]

                pairingpoint = (coord_i, coord_j)
                if pairingpoint not in L_pairingpoints:
                    coord_z = int(round((Polygon_coord[k][2]-DTMx[coord_i][coord_j])/resolution)) # class of the corner keypoint, in units of resolution
                    if (coord_z > 50): coord_z = 17 # prevents some crazy outliers
                    print('Writing roof corner k=', k, ' of polygon ', p, ' at xyz=(', Polygon_coord[k][0], ', ', Polygon_coord[k][1], ', ', Polygon_coord[k][2], ' => (coord_i,coord_j, coord_z)=(', coord_i, ' ,', coord_j, ', ', coord_z, ') => Outputing ', str(name) + '_' + str(coord_z) + '_' + str(kID) + '.png')
                    img = Image.fromarray(M_M, 'RGB')
                    # Output name is "raster grid name/nb"_"shapefile polygon name/nb"_"roof corner point name/nb"_"altitude in resolution units".png
                    #img.save('/user/jlussang/home/Desktop/Grid/detectron2_results/Masks/' + str(name) + '_' + str(M_L_names[p].split('.')[0]) + '_' + str(k) + '_' + str(coord_z) + '.png')


                    # NEW
                    print(coord_z, chr(97 + coord_z), coord_i, coord_j, DTMx[coord_i][coord_j])
                    img.save('/user/jlussang/home/Desktop/Grid/detectron2_results/Masks_biyolo_230_random/' + str(name) + '_h' + str(chr(97 + coord_z)) + 'h_' + str(kID) + '.png')
                    print(str(name) + '_h' + str(chr(97 + coord_z)) + 'h_' + str(kID) + '.png')

                    #img.save('/user/jlussang/home/Desktop/Grid/detectron2_results/Masks_directdisparity_300/' + str(name) + '_roof_' + str(kID) + '.png')
                    #print(input_path+file_name.split('_')[0]+'_'+file_name.split('_')[3]+'_'+file_name.split('_')[2]+'.png')
                    kID += 1
                    #CSVFile.write(str(pairingpoint))
                    #CSVFile.write('\n')
                L_pairingpoints.append(pairingpoint)
    #imgsup = Image.fromarray(M_Msup, 'RGB')
    #imgsup.save('/user/jlussang/home/Desktop/Grid/detectron2_results/Masks/Sup/maskedsup_' + str(name) + '.png')



# Outputs several matrix M_M as mask for a soup of planes defined by their 3 coefficients A/D, B/D, C/D (with D=1000000)
def fTrainingSetGeneratorCOCO_2Dseg(input, name):
    input2 = str(input+name+'.tif')
    src = gdal.Open(input2)
    width = src.RasterXSize
    height = src.RasterYSize
    # np arrays of the two stereo rasters
    Ax = fRasterNumpyze(input2)
    # np array of the coordinates (lat, lon)
    M_Ax = fRasterCoordinize(input2, width, height)

    # List of nparrays, each consisting of the corner points and plane coeff of each shp polygon
    #M_L = fPolygonsList('/user/jlussang/home/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/0/ShapefileInput/') # OLD
    # List of nparrays, each consisting of the corner points
    M_La = fPolygonsList4('/user/jlussang/home/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/2/ShapefileInput_raw/20AUG13104951-res38cm_COMMERCIAL.shp')
    M_Lb = fPolygonsList4('/user/jlussang/home/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/2/ShapefileInput_raw/20AUG13104951-res38cm_INDUSTRIAL.shp')
    M_Lc = fPolygonsList4('/user/jlussang/home/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/2/ShapefileInput_raw/20AUG13104951-res38cm_RESIDENTIAL.shp')
    M_L = M_La + M_Lb + M_Lc

    # List of the file names of each polygon
    #M_L_names = fPolygonsNames('/user/jlussang/home/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/0/ShapefileInput/') # OLD
    # print(M_L)
    # print(len(M_L))
    #M_L[p][len(M_L[p]) - 1][0]
    #CSVFile = open('/user/jlussang/home/Desktop/Grid/OutputLog.txt', 'w')
    #print(len(M_L))
    kID = 0
    for p in range(0, len(M_L)):  # polygons
        Polygon_coord = []
        Polygon_lats = []
        Polygon_lons = []
        # number of entries of each polygon array (minus 1 because of plane coefficients at the end)
        #for c in range(0, len(M_L[p]) - 1): # OLD
        for c in range(0, len(M_L[p])):
            #Polygon_coord.append((M_L[p][c][0], M_L[p][c][1])) # OLD
            Polygon_coord.append((M_L[p][c][0], M_L[p][c][1], M_L[p][c][2])) # latitude, longitude, altitutde of each roof corner
            Polygon_lats.append(M_L[p][c][0])
            Polygon_lons.append(M_L[p][c][1])
        minlat = min(Polygon_lats)
        minlon = min(Polygon_lons)
        maxlat = max(Polygon_lats)
        maxlon = max(Polygon_lons)
        imin = 0
        imax = M_Ax.shape[0]
        jmin = 0
        jmax = M_Ax.shape[1]
        if (maxlat < M_Ax[0][0][0]) or (minlat > M_Ax[0][M_Ax.shape[1] - 1][0]) or (
                maxlon < M_Ax[M_Ax.shape[0] - 1][0][1]) or (minlon > M_Ax[0][0][1]):
            # print('Polygon_coord=', Polygon_coord)
            # print('Polygon_lats=', Polygon_lats)
            # print('Polygon_lons=', Polygon_lons)
            # print('maxlat=', maxlat, '<', M_Ax[0][0][0], ' or minlat=', minlat, '>', M_Ax[0][M_Ax.shape[0]-1][0], ' or maxlon=', maxlon, '<', M_Ax[M_Ax.shape[0]-1][0][1], ' or minlon=', minlon, '>', M_Ax[0][0][1])
            # print('p=', p)
            continue
        #print('Polygon ', M_L_names[p])
        for i in range(0, imax):  # rows
            if (M_Ax[i][0][1] >= maxlon):
                imin = i
            if (M_Ax[i][0][1] > minlon):
                imax = i + 1
            for j in range(0, jmax):  # columns
                if (M_Ax[0][j][0] < minlat):
                    jmin = j
                if (M_Ax[0][j][0] <= maxlat):
                    jmax = j + 1
        imax = max(M_Ax.shape[0], imax)
        jmax = max(M_Ax.shape[1], jmax)
        #print('imin=', imin, ' imax=', imax, ' jmin=', jmin, ' jmax=', jmax)
        Polygonus = Polygon(Polygon_coord)

        # Empty np array of the mask with coefficients
        M_M = np.arange(float(Ax.shape[0] * Ax.shape[1] * 3)).reshape(Ax.shape[0], Ax.shape[1], 3)
        M_M.fill(0)
        # Empty np array of the mask with b/w polygons
        # M_Mbw = np.arange(float(Ax.shape[0] * Ax.shape[1] * 3)).reshape(Ax.shape[0], Ax.shape[1], 3)
        M_Mbw = np.zeros((height, width, 3), dtype=np.uint8)
        M_Mbw.fill(0)  # black everywhere
        for i in range(0, M_M.shape[0]):
            for j in range(0, M_M.shape[1]):
                M_M[i][j][2] = 1  # C/D is set by default at 1 for places with no building
        CondInit = 0
        for i in range(imin, imax):  # rows
            for j in range(jmin, jmax):  # columns
                #print('i=', i, '/', M_Ax.shape[0], ' j=', j, '/', M_Ax.shape[1], ' p=', p, '/', len(M_L), 'M_Ax[i][j]=', M_Ax[i][j])
                P = Point(M_Ax[i][j][0], M_Ax[i][j][1])
                Cond = P.within(Polygonus)
                if Cond == True:
                    CondInit = 1
                    M_M[i][j][0] = M_L[p][len(M_L[p]) - 1][0]  # plane coeff A/D
                    M_M[i][j][1] = M_L[p][len(M_L[p]) - 1][1]  # plane coeff B/D
                    M_M[i][j][2] = M_L[p][len(M_L[p]) - 1][2]  # plane coeff C/D
                    M_Mbw[i][j] = [255, 255, 255]  # white
                    #print('Point (', M_Ax[i][j][0], ', ', M_Ax[i][j][1], ') is in polygon !')
                    '''
                    CSVFile.write('Point (' + str(M_Ax[i][j][0]) + ', ' + str(M_Ax[i][j][1]) + ') is in polygon !')
                    CSVFile.write('\n')
                    CSVFile.write(
                        '=> M_M[' + str(i) + '][' + str(j) + '][0] = A/D = ' + str(M_L[p][len(M_L[p]) - 1][0]))
                    CSVFile.write('\n')
                    CSVFile.write(
                        '   M_M[' + str(i) + '][' + str(j) + '][1] = B/D = ' + str(M_L[p][len(M_L[p]) - 1][1]))
                    CSVFile.write('\n')
                    CSVFile.write(
                        '   M_M[' + str(i) + '][' + str(j) + '][2] = C/D = ' + str(M_L[p][len(M_L[p]) - 1][2]))
                    CSVFile.write('\n')
                    '''
        img = Image.fromarray(M_Mbw, 'RGB')
        #img.save('/user/jlussang/home/Desktop/Grid/detectron2_results/Masks/' + str(name) + '_' + str(M_L_names[p].split('.')[0]) + '_' + str(kID) + '.png') # OLD
        img.save('/user/jlussang/home/Desktop/Grid/detectron2_results/Masks_2Dsegmentation_230_random/' + str(name) + '_0_' + str(kID) + '.png')
        if CondInit == 1:
            kID += 1


def fTrainingSetGeneratorCOCOAll(input_path):
    for file in os.listdir(input_path):
    #for file in sorted(os.listdir(input_path)):
        if file.endswith('.tif'):
            input = os.path.join(input_path, file)
            filename = os.path.splitext(os.path.basename(file))[0]
            print(filename)
            #fTrainingSetGeneratorCOCO(input_path, filename)
            #fTrainingSetGeneratorCOCO_blended(input_path, filename)
            #fTrainingSetGeneratorCOCO_dtm(input_path, '/user/jlussang/home/Desktop/Grid/detectron2_results/Gridify_dtm_230_random/', filename)
            fTrainingSetGeneratorCOCO_2Dseg(input_path, filename)


#Removes the plain black image masks in the training set
def fCleanCocoDataset(input_path):
    for file in os.listdir(input_path):
        input = os.path.join(input_path, file)
        filename = os.path.splitext(os.path.basename(file))[0]
        print(filename)
        M = cv2.imread(input)
        Cond = 0
        for i in range(0, M.shape[0]):
            for j in range(0, M.shape[1]):
                if ((M[i][j][0] == 255) or (M[i][j][1] == 255) or (M[i][j][1] == 255)):
                    Cond = 1
        if (Cond==0):
            print('Removing ', filename)
            os.remove(input)

# Renames the training set according to pycococreator
# Our masks format is Mask_tifid_shpid.png, e. g. Mask_000000000007_2275.png
# Pycoco example has images as 1000.jpeg, 1001.jpeg, and masks as 1000_square_0.png,
# 1001_circle_0.png, 1001_square_1.png, 1001_square_2.png, 1001_circle_3.png
# So we need to have mask ID's as such: 000000000007_2275.png (hence just remove the first part 'Mask_')
def fRenameCocoDataset(input_path):
    for file in os.listdir(input_path):
        input = os.path.join(input_path, file)
        filename = os.path.splitext(os.path.basename(file))[0]
        filename2 = filename.split('_')[0] + '_h' + filename.split('_')[1] + 'h_' + filename.split('_')[2] + '.png'
        filename += '.png'
        print(filename)
        print(filename2)
        os.rename(input_path+filename, input_path+filename2)
        '''
        #print(filename2)
        filename3 = filename.replace("h", "")
        filename3 = filename3.replace("d", "")
        numb = (filename3.split('_')[1])
        #print(chr(97 + int(numb)))
        #print(filename3)
        #print(numb)
        filename4 = filename.split('_')[0] + '_' + chr(97 + int(numb)) + '_' + filename.split('_')[2]
        print(filename4)
        #for i in range(0, 20, 1): print(chr(97 + i))
        #if (int(numb)-10<0): classroof = numb[0]
        #else: classroof = numb[0] + 'd' + numb[1]
        #print(classroof)
        #filename2 = (filename.split('h')[0]) + 'h' + classroof + '_' + (filename.split('h')[1]).split('_')[1]
        '''

def fRenameCocoDataset2(input_path):
    for file in os.listdir(input_path):
        input = os.path.join(input_path, file)
        filename_backup = os.path.basename(file)
        filename = os.path.splitext(os.path.basename(file))[0]
        filename2 = str(int(filename.split('_')[1])%2)
        if int(filename2)==0: filename2 = 'circle'
        elif int(filename2) == 1: filename2 = 'square'
        else:  filename2 = 'triangle'
        filename = filename.split('_')[0] + '_' + filename2 + '_' + filename.split('_')[2] + '.png'
        filename_backup = str(filename_backup)
        filename = str(filename)
        print(filename_backup, filename)
        os.rename(input_path+filename_backup, input_path+filename)

def fCocoAnnotationGen(input_path):
    for file in os.listdir(input_path):
        input = os.path.join(input_path, file)
        filename = os.path.splitext(os.path.basename(file))[0]
        filename2 = filename.split('Mask_')[1]+'.png'
        filename += '.png'
        print(filename)
        print(filename2)
        os.rename(input_path+filename, input_path+filename2)

# This is for the coco annotation procedure
def fCocoAnnotationGen2(input_path):
    new_filename = 'xyz'
    annotation_id = -1
    for file in sorted(os.listdir(input_path)):
        input = os.path.join(input_path, file)
        filename = os.path.splitext(os.path.basename(file))[0]
        filename_backup = os.path.splitext(os.path.basename(file))[0] + '.png'
        annotation_id += 1
        if filename.split('_')[0] != new_filename:
            annotation_id = 0
            #print(filename.split('_')[0])
            #print(new_filename)
            new_filename = filename.split('_')[0]
        filename += '_' + str(annotation_id) + '.png'
        print(filename)
        os.rename(input_path + filename_backup, input_path + filename)

def fCocoAnnotation_CATEGORIES(input_path):
    CSVFile = open('/user/jlussang/home/Desktop/Log.txt', 'w')
    file_id = 0
    CSVFile.write('CATEGORIES = [')
    CSVFile.write('\n')
    for file in sorted(os.listdir(input_path)):
        if file.endswith('.shp'):
            input = os.path.join(input_path, file)
            file_name = os.path.splitext(os.path.basename(file))[0]
            #file_name = os.path.basename(file)
            #print('file_name', file_name)
            #print('file_id', file_id)
            CSVFile.write('\t')
            CSVFile.write('{')
            CSVFile.write('\n')

            CSVFile.write('\t')
            CSVFile.write('\t')
            CSVFile.write("'id': ")
            CSVFile.write(str(file_id+1))
            CSVFile.write(",")
            CSVFile.write('\n')

            CSVFile.write('\t')
            CSVFile.write('\t')
            CSVFile.write("'name': '")
            CSVFile.write(str(int(file_name)%14))
            CSVFile.write("',")
            CSVFile.write('\n')

            CSVFile.write('\t')
            CSVFile.write('\t')
            CSVFile.write("'supercategory': 'roof',")
            CSVFile.write('\n')

            CSVFile.write('\t')
            CSVFile.write('},')
            CSVFile.write('\n')
            file_id += 1
    CSVFile.write(']')
    CSVFile.write('\n')


def fCocoAnnotation_CATEGORIES2(mini, maxi):
    CSVFile = open('/user/jlussang/home/Desktop/Log.txt', 'w')
    CSVFile.write('CATEGORIES = [')
    CSVFile.write('\n')
    for i in range(1, maxi-mini+2):
        CSVFile.write('\t')
        CSVFile.write('{')
        CSVFile.write('\n')

        CSVFile.write('\t')
        CSVFile.write('\t')
        CSVFile.write("'id': ")
        CSVFile.write(str(i))
        CSVFile.write(",")
        CSVFile.write('\n')

        CSVFile.write('\t')
        CSVFile.write('\t')
        CSVFile.write("'name': '")
        CSVFile.write(str(mini+i-1)) # 200 units of resolution 38cm for instance, i.e. 76 m
        CSVFile.write("',")
        CSVFile.write('\n')

        CSVFile.write('\t')
        CSVFile.write('\t')
        CSVFile.write("'supercategory': 'roof',")
        CSVFile.write('\n')

        CSVFile.write('\t')
        CSVFile.write('},')
        CSVFile.write('\n')
    CSVFile.write(']')
    CSVFile.write('\n')

def fCocoAnnotation_ArgumentsMetadata(mini, maxi):
    CSVFile = open('/user/jlussang/home/Desktop/LogMetadata.txt', 'w')
    for i in range(1, maxi-mini+2):
        CSVFile.write('"')
        CSVFile.write(str(mini+i-1))
        CSVFile.write('", ')

def fCocoAnnotation_NameChange(input_path):
    kID = 0
    for file in sorted(os.listdir(input_path)):
        if file.endswith('.png'):
            input = os.path.join(input_path, file)
            file_name = os.path.splitext(os.path.basename(file))[0]
            print(input)
            print(input_path + file_name.split('_')[0] + '_0_' + file_name.split('_')[2] + '.png')
            #os.rename(input, input_path + file_name.split('_')[0] + '_0_' + file_name.split('_')[2] + '.png')
            #print(input_path+file_name.split('_')[0] + '_roof_' + str(kID) + '.png')
            #os.rename(input, input_path+file_name.split('_')[0] + '_roof_' + str(kID) + '.png')
            #kID += 1


def fCocoAnnotation_minimax_classes(input_path):
    mini = 9999
    maxi = -9999
    CSVFile = open('/user/jlussang/home/Desktop/class_distribution.csv', 'w')
    for file in sorted(os.listdir(input_path)):
        if file.endswith('.png'):
            input = os.path.join(input_path, file)
            file_name = os.path.splitext(os.path.basename(file))[0]
            class_id = int(file_name.split('_')[1])
            if (class_id < mini): mini = class_id
            if (class_id > maxi): maxi = class_id
            if (class_id > 60): print(file_name)
            CSVFile.write(str(class_id))
            CSVFile.write('\n')
    print('mini=', mini, ', maxi=', maxi, ' (i.e. ', mini, ' & ', maxi, ' meters)')


# Takes a detectron2 masked image and assign a unique id to each segmented roof
def fRoofLabeling(input, output):
    M = cv2.imread(input, cv2.IMREAD_UNCHANGED)
    height = M.shape[0]
    width = M.shape[1]
    #M2 = (np.add.reduce(M, 2))/1000.0
    M2 = M.copy()/1000.0 # XYZ new
    M2 = np.ceil(M2)  # M2 is np array of input image with 0 for no roof and 1 for roof
    roof_id = 10  # roof ID
    '''for i in range(0, height, 1):
        for j in range(0, width, 1):
            #if (M[i][j] >= 10):
                print(M2[i][j])
    exit()'''
    for i in range(0, height, 1):
        for j in range(0, width, 1):
            if (M[i][j] >= 10) and (M2[i][j] <= 9):  # Not black and not already filled
                M2 = flood_fill(M2, (i, j), roof_id)
                print('Flooded at i=', i, '/', height, ', j=', j, '/', width, ' with roof_id #', roof_id, ' => (M2[i][j]=', M2[i][j], ')')
                roof_id += 1
    cv2.imwrite(output, M2)

    # Now turning this into a list (of roofs) of lists of (x,y,z) points...
    print('Now turning this into a list (of roofs) of lists of (x,y,z) points')
    List_roofs = []
    for k in range(roof_id-10):
        Roof_points = []
        List_roofs.append(Roof_points)
    for i in range(0, height, 1):
        for j in range(0, width, 1):
            if (M2[i][j] >= 10):
                List_roofs[int(M2[i][j]-10)].append([i*0.30, j*0.30, M[i][j]])

    # Now outputing this List_roofs as csv file...
    print('Now outputing this List_roofs as csv file')
    CSVFile = open('/home/jlussang/Desktop/Grid/detectron2_results/List_roofs.csv', 'w')
    with CSVFile:
        FileOutput = csv.writer(CSVFile)
        FileOutput.writerows(List_roofs) # Each line k of this csv file corresponds to the k=roof_id list of points (i,j,z)

def fcgal_write():
    Command1 = 'make /home/jlussang/cgal/CSVInput'
    Command2 = '/home/jlussang/cgal/CSVInput'
    os.system(Command1)
    os.system(Command2)

def fcgal_read(input):
    List_roofs = []
    with open(input) as CSVFile:
        FileInput = csv.reader(CSVFile, delimiter=',')
        i=0
        for row in FileInput:
            print(i)
            i += 1
            List_roofs.append(row)  # (file converted to a list)
    #print(List_roofs[1][6])

def Black(height, width):
    background = np.zeros((height, width))
    cv2.imwrite('/home/jlussang/Desktop/black.jpg', background)

def Blender(input1, input2):
    finput1_pil = PIL.Image.open(input1)
    finput2_pil = PIL.Image.open(input2)
    finput1_pil_rgb = finput1_pil.convert("RGB")
    finput2_pil_rgb = finput2_pil.convert("RGB")
    height, width = finput1_pil.size
    Blended = np.zeros((width, height, 3), dtype=np.uint8)
    #print(Blended.shape)
    for i in range(0, width, 1):
        for j in range(0, height, 1):
            #Blended[i][j][0] = finput1_pil_rgb.getpixel((j, i))[0] # getpixel inverts indices !
            #Blended[i][j][1] = finput1_pil_rgb.getpixel((j, i))[1] # getpixel inverts indices !
            Blended[i][j][0] = 0  # getpixel inverts indices !
            Blended[i][j][1] = 0  # getpixel inverts indices !
            Blended[i][j][2] = finput2_pil_rgb.getpixel((j, i))[0] # getpixel inverts indices !
            if finput2_pil_rgb.getpixel((j, i))[0] < 50:
                Blended[i][j][0] = finput1_pil_rgb.getpixel((j, i))[0] # getpixel inverts indices !
                Blended[i][j][1] = finput1_pil_rgb.getpixel((j, i))[1] # getpixel inverts indices !
        #print('Wrote Blended [', i, '] [', j, '] =', Blended[i][j], ' => ', int(100.0*i/width), '%')
    res = Image.fromarray(Blended, 'RGB')
    return res
    #res.save('/home/jlussang/Desktop/Blended.tif')
    #res.save('/home/jlussang/Desktop/Blended.png')

def fBlender_all(input1, input2):
    for file in os.listdir(input1):
    #for file in sorted(os.listdir(input_path)):
        if file.endswith('.tif'):
            input = os.path.join(input1, file)
            filename = os.path.splitext(os.path.basename(file))[0] + '.jpg'
            print(filename)
            res = Blender(input1 + filename, input2 + filename)
            res.save('/home/jlussang/Desktop/Grid/detectron2_results/Gridify_biyolo_230_random/' + filename)
            #res = Blender(input1 + filename, '/home/jlussang/Desktop/black.jpg')
            #res.save('/home/jlussang/Desktop/Gridify_biyolo_300/' + filename)

#Outputs a list of unique random numbers from 1 to size
#print(fJrandomize(1755))
def fJrandomize(size):
    L = []
    V = []
    for i in range(1, size):
        L.append(i)
    for i in range(1, size):
        index = ceil(np.random.uniform()*len(L)) - 1
        elem = L.pop(index)
        V.append(elem)
    return V

def fJrandomize2(first, size):
    L = []
    V = []
    for i in range(first, size):
        L.append(i)
    for i in range(first, size):
        index = ceil(np.random.uniform()*len(L)) - 1
        elem = L.pop(index)
        V.append(elem)
    return V



# For maths in CP
#fCalcul('/user/jlussang/home/Desktop/Calculs.txt', 500)
def fCalcul(InputPath, size):
    CSVFile = open(InputPath, 'w')
    for i in range(1, size):
        x = ceil(abs(np.random.normal(10, 30, size=None)))
        y = ceil(abs(np.random.normal(10, 30, size=None)))
        pm = ceil(abs(np.random.normal(10, 30, size=None)))
        if (x >= 100): x = 90
        if (y >= 100): y = 90
        if ((x >= 10) and (y >= 10)):
            if ((pm % 4 == 0) and (x >= y)): CSVFile.write(str(x) + ' - ' + str(y) + '  = ')
            else: CSVFile.write(str(x) + ' + ' + str(y) + '  = ')
            CSVFile.write('\n')
            CSVFile.write('\n')
            '''
            if ((pm % 4 == 0) and (x >= y)): CSVFile.write(str(x) + ' - ' + str(y) + '  = _____________')
            else: CSVFile.write(str(x) + ' + ' + str(y) + '  = _____________')
            CSVFile.write('\n')
            CSVFile.write('\t' + '\t' + '   _____________')
            CSVFile.write('\n')
            CSVFile.write('\t' + '\t' + '   _____________')
            CSVFile.write('\n')
        else:
            if ((x <= 9) and (y <= 9)):
                if ((pm % 4 == 0) and (x >= y)): CSVFile.write(str(x) + ' - ' + str(y) + '    = _____________')
                else: CSVFile.write(str(x) + ' + ' + str(y) + '    = _____________')
                CSVFile.write('\n')
                continue
            CSVFile.write(str(x) + ' + ' + str(y) + '   = _____________')
            CSVFile.write('\n')
        '''

def fRandomizeCOCOdatasets(input_rasters, input_masks, first, size):
    output_rasters_train = '/user/jlussang/home/Desktop/pycococreator2/examples/shapes/train/roofs_train2021_train/'
    output_rasters_val = '/user/jlussang/home/Desktop/pycococreator2/examples/shapes/train/roofs_train2021_val/'
    output_rasters_test = '/user/jlussang/home/Desktop/pycococreator2/examples/shapes/train/roofs_train2021_test/'
    output_masks_train = '/user/jlussang/home/Desktop/pycococreator2/examples/shapes/train/annotationsgeo_train/'
    output_masks_val = '/user/jlussang/home/Desktop/pycococreator2/examples/shapes/train/annotationsgeo_val/'
    output_masks_test = '/user/jlussang/home/Desktop/pycococreator2/examples/shapes/train/annotationsgeo_test/'
    #J = fJrandomize(size+1)
    J = fJrandomize2(first, size + 1)
    train_min = 0
    train_max = int(len(J)*0.6)
    val_min = train_max + 1
    val_max = int(len(J)*0.8)
    test_min = val_max + 1
    test_max = int(len(J))
    # print(train_min, train_max, val_min, val_max, test_min, test_max) # 0 1053 1054 1404 1405 1755
    for i in range(train_min, train_max + 1):
        Command1 = 'cp ' + input_rasters + '00000000000' + str(J[i]) + '.jpg ' + output_rasters_train
        Command2 = 'cp ' + input_masks + '00000000000' + str(J[i]) + '_* ' + output_masks_train
        os.system(Command1)
        os.system(Command2)
    for i in range(val_min, val_max + 1):
        Command3 = 'cp ' + input_rasters + '00000000000' + str(J[i]) + '.jpg ' + output_rasters_val
        Command4 = 'cp ' + input_masks + '00000000000' + str(J[i]) + '_* ' + output_masks_val
        os.system(Command3)
        os.system(Command4)
    for i in range(test_min, test_max):
        Command5 = 'cp ' + input_rasters + '00000000000' + str(J[i]) + '.jpg ' + output_rasters_test
        Command6 = 'cp ' + input_masks + '00000000000' + str(J[i]) + '_* ' + output_masks_test
        os.system(Command5)
        os.system(Command6)

# To copy specific files from one dir to another, taking the names from a third dir
def fCopycat(matrix_xxx, input_all, goal_xxx):
    kID = 0
    for file in sorted(os.listdir(matrix_xxx)):
        if file.endswith('.jpg'):
            input = os.path.join(matrix_xxx, file)
            file_name = os.path.splitext(os.path.basename(file))[0]
            #print(input)
            #print(file_name + '.jpg')
            Command = 'cp ' + input_all + file_name + '_* ' + goal_xxx
            print(Command)
            os.system(Command)

# Function that colors 2Dsegmentation bw results as {0,0,value} according to the tile being in the train, val, or test set
def f2Dsegmentation_color(input_all, output_all, name_dir, value):
    for file in sorted(os.listdir(name_dir)):
        if file.endswith('.png'):
            file_name = os.path.splitext(os.path.basename(file))[0]
            print(input_all + file_name + '.png')
            finput_pil = PIL.Image.open(input_all + file_name + '.png')
            finput_pil_rgb = finput_pil.convert("RGB")
            height, width = finput_pil.size
            seg_color = np.zeros((width, height, 3), dtype=np.uint8)
            #Command = 'cp ' + input_all + file_name + '.jpg ' + goal_xxx
            #print(Command)
            #os.system(Command)
            for i in range(0, width, 1):
                for j in range(0, height, 1):
                    #dseg_color[i][j][0] = finput_pil_rgb.getpixel((j, i))[0] # getpixel inverts indices !
                    #dseg_color[i][j][1] = finput_pil_rgb.getpixel((j, i))[1] # getpixel inverts indices !
                    #dseg_color[i][j][2] = finput_pil_rgb.getpixel((j, i))[2] # getpixel inverts indices !
                    if finput_pil_rgb.getpixel((j, i))[0] > 50:
                        seg_color[i][j][0] = 0
                        seg_color[i][j][1] = 0
                        seg_color[i][j][2] = value  # getpixel inverts indices !
            res = Image.fromarray(seg_color, 'RGB')
            res.save(output_all + file_name + '.png')

# This function blends the color_coded 2Dsegmentation results with the 3Dreconstruction results
def fBlend_2D3D(input_2Dsegmentation, input_3Dreconstruction, output2D3D):
    for file2D in sorted(os.listdir(input_2Dsegmentation)):
        if file2D.endswith('.png'):
            print('file2D=', file2D)
            filename2D = os.path.splitext(os.path.basename(file2D))[0]
            print('filename2D=', filename2D)
            finput2D = PIL.Image.open(input_2Dsegmentation + file2D)
            finput2D_rgb = finput2D.convert("RGB")
            height, width = finput2D.size
            Blended = np.asarray(finput2D_rgb)
            for file3D in sorted(os.listdir(input_3Dreconstruction)):
                if file3D.endswith('.png'):
                    filename3D = os.path.splitext(os.path.basename(file3D))[0]
                    filename3D_trunk = filename3D.split('_')[0]
                    z = int(filename3D.split('_')[2])
                    if (filename3D_trunk == filename2D):
                        print('Match found between filename2D=', filename2D, ' and filename3D_trunk=', filename3D_trunk)
                        finput3D = PIL.Image.open(input_3Dreconstruction + file3D)
                        finput3D_rgb = finput3D.convert("RGB")
                        for i in range(0, width, 1):
                            for j in range(0, height, 1):
                                if (finput3D_rgb.getpixel((j, i))[0] + finput3D_rgb.getpixel((j, i))[1] + finput3D_rgb.getpixel((j, i))[2]) > 1:
                                    Blended[i][j][0] = z # red_z
                                    #Blended[i][j][1] = 0
                                    #Blended[i][j][2] = z + 230
                                    print('Wrote roof corner at Blended [', i, '] [', j, '] =', z)
            res = Image.fromarray(Blended, 'RGB')
            res.save(output2D3D + filename2D + '.png')

# This function blends the color_coded 2Dsegmentation results with the 3Dreconstruction results
def fBlend_2D3D_GT(input_2Dsegmentation, input_3Dreconstruction, output2D3D):
    for file2D in sorted(os.listdir(input_2Dsegmentation)):
        if file2D.endswith('.png'):
            print('file2D=', file2D)
            filename2D = os.path.splitext(os.path.basename(file2D))[0]
            print('filename2D=', filename2D)
            finput2D = PIL.Image.open(input_2Dsegmentation + file2D)
            finput2D_rgb = finput2D.convert("RGB")
            height, width = finput2D.size
            Blended = np.asarray(finput2D_rgb)
            #Blended = np.zeros((width, height, 3), dtype=np.uint8)
            for file3D in sorted(os.listdir(input_3Dreconstruction)):
                if file3D.endswith('.png'):
                    filename3D = os.path.splitext(os.path.basename(file3D))[0]
                    filename3D_trunk = filename3D.split('_')[0]
                    #z = int(filename3D.split('_')[2])
                    if (filename3D_trunk == filename2D):
                        print('Match found between filename2D=', filename2D, ' and filename3D_trunk=', filename3D_trunk)
                        finput3D = PIL.Image.open(input_3Dreconstruction + file3D)
                        finput3D_rgb = finput3D.convert("RGB")
                        for i in range(0, width, 1):
                            for j in range(0, height, 1):
                                if (finput3D_rgb.getpixel((j, i))[0]) > 1:
                                    Blended[i][j][0] = finput3D_rgb.getpixel((j, i))[0]
                                    #Blended[i][j][1] = finput2D_rgb.getpixel((j, i))[1]
                                    #Blended[i][j][2] = finput2D_rgb.getpixel((j, i))[2]
                                    #print('Wrote roof corner at Blended [', i, '] [', j, ']')
            res = Image.fromarray(Blended, 'RGB')
            res.save(output2D3D + filename2D + '.png')

# This function blends the tiles with individual roof planes so as to have tiles with multiple roof planes
def fBlend_2D2D(input_matrixdir, input_2Dsegmentation, width, height, output2D2D):
    for file in sorted(os.listdir(input_matrixdir)):
        if file.endswith('.jpg'):
            filename_matrix = os.path.splitext(os.path.basename(file))[0]
            # print('input_matrixfile=', input_matrixfile)
            # print('input_2Dsegmentation=', input_2Dsegmentation)
            # print('output2D2D=', output2D2D)
            M = np.zeros((width, height, 3), dtype=np.uint8)
            for file2D in sorted(os.listdir(input_2Dsegmentation)):
                filename2D = os.path.splitext(os.path.basename(file2D))[0]  # e.g. 000000000001000_0_0
                filename3D_trunk = filename2D.split('_')[0]  # e.g. 000000000001000
                if ((file2D.endswith('.png')) and (filename3D_trunk == filename_matrix)):
                    finput2D = PIL.Image.open(input_2Dsegmentation + file2D)
                    # print('file2D=', file2D)  # e.g. 000000000001000_0_0.png
                    # print('filename2D=', filename2D, ', filename3D_trunk=', filename3D_trunk)
                    # print('input_2Dsegmentation= ', input_2Dsegmentation)  # e.g. /user/jlussang/home/Desktop/Grid/detectron2_results/Masks_2Dsegmentation_230_random_colored/
                    finput2D_rgb = finput2D.convert("RGB")
                    Blended = np.asarray(finput2D_rgb)  # Blended is a np array of image 000000000001000_0_0.png
                    print('filename3D_trunk=', filename3D_trunk, ' == input_matrixfile=', filename_matrix,
                          ' => processing file2D=', file2D)
                    for i in range(0, width, 1):
                        for j in range(0, height, 1):
                            if (Blended[i][j][2] > 1):
                                #print('Blended[i][j][2]=', Blended[i][j][2])
                                M[i][j][2] = Blended[i][j][2]
                            if (Blended[i][j][1] > 1):
                                #print('Blended[i][j][1]=', Blended[i][j][1])
                                M[i][j][1] = Blended[i][j][1]
            res = Image.fromarray(M, 'RGB')
            res.save(output2D2D + filename_matrix + '.png')

# This function blends the tiles with individual roof plane corners so as to have tiles with multiple roof plane corners (and their elevation as color-coded)
def fBlend_3D3D(input_matrixdir, input_3Dsegmentation, width, height, output3D3D):
    height_scale = []
    for i in range(0, 25, 1):
        height_scale.append(chr(97 + i)) # list as alphabet-coded heights
    #print(height_scale)
    for file in sorted(os.listdir(input_matrixdir)):
        if file.endswith('.jpg'):
            filename_matrix = os.path.splitext(os.path.basename(file))[0]
            # print('input_matrixfile=', input_matrixfile)
            # print('input_2Dsegmentation=', input_2Dsegmentation)
            # print('output2D2D=', output2D2D)
            M = np.zeros((width, height, 3), dtype=np.uint8)
            for file2D in sorted(os.listdir(input_3Dsegmentation)):
                filename2D = os.path.splitext(os.path.basename(file2D))[0]  # e.g. 000000000001000_0_0
                filename3D_trunk = filename2D.split('_')[0]  # e.g. 000000000001000
                if ((file2D.endswith('.png')) and (filename3D_trunk == filename_matrix)):
                    finput2D = PIL.Image.open(input_3Dsegmentation + file2D)
                    z_char = file2D.split('_')[1].split('h')[1]
                    if (not z_char): z_char = 'h'
                    print('file2D=', file2D, ' z_char=', z_char)  # e.g. 000000000001000_0_0.png
                    # print('filename2D=', filename2D, ', filename3D_trunk=', filename3D_trunk)
                    # print('input_2Dsegmentation= ', input_2Dsegmentation)  # e.g. /user/jlussang/home/Desktop/Grid/detectron2_results/Masks_2Dsegmentation_230_random_colored/
                    z = height_scale.index(z_char) + 1
                    finput2D_rgb = finput2D.convert("RGB")
                    Blended = np.asarray(finput2D_rgb)  # Blended is a np array of image 000000000001000_0_0.png
                    print('filename3D_trunk=', filename3D_trunk, ' == input_matrixfile=', filename_matrix, ' => processing file2D=', file2D, ', z=', z)
                    for i in range(0, width, 1):
                        for j in range(0, height, 1):
                            if (Blended[i][j][0] > 1):
                                print('Blended[i][j][0]=', Blended[i][j][0])
                                M[i][j][0] = z + 200
                                break # for GT are of one pixel only
            res = Image.fromarray(M, 'RGB')
            res.save(output3D3D + filename_matrix + '.png')

# This function shrinks via a 4-connectivity the GT of 2D segmentation so that roof lines can be distinguished
def fShrink_2D2D(inputdir, outputdir):
    for file in sorted(os.listdir(inputdir)):
        if file.endswith('.png'):
            finput = PIL.Image.open(inputdir + file)
            finput_rgb = finput.convert("RGB")
            height, width = finput.size
            MShrank = np.asarray(finput_rgb, dtype=np.uint8)
            Shrank = np.zeros((width, height, 3), dtype=np.uint8)
            height, width = finput.size
            print('Shrinking planes of file ', file)
            for i in range(2, width-2, 1):
                for j in range(2, height-2, 1):
                    if (MShrank[i][j][2] > 1) and (MShrank[i+1][j][2] > 1) and (MShrank[i-1][j][2] > 1) and (MShrank[i][j+1][2] > 1) and (MShrank[i][j-1][2] > 1):
                        Shrank[i][j][2] = MShrank[i][j][2]
            res = Image.fromarray(Shrank, 'RGB')
            res.save(outputdir + file)

# This function deletes blue pixels that are horizontally on vertically in between two black pixels
def fLink_delete(inputdir, outputdir):
    finput = PIL.Image.open(inputdir)
    finput_rgb = finput.convert("RGB")
    height, width = finput.size
    finput_np = np.asarray(finput_rgb, dtype=np.uint8)
    height, width = finput.size
    for i in range(1, width-1, 1):
        if (i % 200 == 0): print('Link delete at i = ', i, '/', width)
        for j in range(1, height-1, 1):
            if (finput_np[i][j][2] > 1) and (finput_np[i-1][j][2] < 1) and (finput_np[i+1][j][2] < 1): # vertically
                finput_np[i][j][2] = 0
            if (finput_np[i][j][2] > 1) and (finput_np[i][j-1][2] < 1) and (finput_np[i][j+1][2] < 1): # horizontally
                finput_np[i][j][2] = 0
    res = Image.fromarray(finput_np, 'RGB')
    res.save(outputdir)

# This function fills the 2 pixel borders of each tile so as not to have black lines on the fully reconstructed raster image via Ungridify()
def fBorders_fill(input, output):
    for file in sorted(os.listdir(input)):
        if file.endswith('.png'):
            filename = os.path.splitext(os.path.basename(file))[0]
            print(filename)
            finput = PIL.Image.open(input + file)
            finput_rgb = finput.convert("RGB")
            height, width = finput.size
            Borders_fill = np.asarray(finput_rgb)
            for i in range(0, width, 1):
                if Borders_fill[i][2][2] >= 1:
                    Borders_fill[i][0][2] = Borders_fill[i][2][2] # Filling left border of the tile
                    Borders_fill[i][1][2] = Borders_fill[i][2][2] # Filling left border of the tile
                if Borders_fill[i][height-3][2] >= 1:
                    Borders_fill[i][height-1][2] = Borders_fill[i][height-3][2] # Filling left border of the tile
                    Borders_fill[i][height-2][2] = Borders_fill[i][height-3][2] # Filling left border of the tile
                if Borders_fill[i][2][1] >= 1:
                    Borders_fill[i][0][1] = Borders_fill[i][2][1] # Filling left border of the tile
                    Borders_fill[i][1][1] = Borders_fill[i][2][1] # Filling left border of the tile
                if Borders_fill[i][height-3][1] >= 1:
                    Borders_fill[i][height-1][1] = Borders_fill[i][height-3][1] # Filling left border of the tile
                    Borders_fill[i][height-2][1] = Borders_fill[i][height-3][1] # Filling left border of the tile
            for j in range(0, height, 1):
                if Borders_fill[2][j][2] >= 1:
                    Borders_fill[0][j][2] = Borders_fill[2][j][2] # Filling left border of the tile
                    Borders_fill[1][j][2] = Borders_fill[2][j][2] # Filling left border of the tile
                if Borders_fill[width-3][j][2] >= 1:
                    Borders_fill[width-1][j][2] = Borders_fill[width-3][j][2] # Filling left border of the tile
                    Borders_fill[width-2][j][2] = Borders_fill[width-3][j][2] # Filling left border of the tile
                if Borders_fill[2][j][1] >= 1:
                    Borders_fill[0][j][1] = Borders_fill[2][j][1] # Filling left border of the tile
                    Borders_fill[1][j][1] = Borders_fill[2][j][1] # Filling left border of the tile
                if Borders_fill[width-3][j][1] >= 1:
                    Borders_fill[width-1][j][1] = Borders_fill[width-3][j][1] # Filling left border of the tile
                    Borders_fill[width-2][j][1] = Borders_fill[width-3][j][1] # Filling left border of the tile
            res = Image.fromarray(Borders_fill, 'RGB')
            res.save(output + filename + '.png')

# This function expands the pixel roof corners as nxn squares of pixels
def fSquare_expand(inputdir, outputdir,n):
    finput = PIL.Image.open(inputdir)
    finput_rgb = finput.convert("RGB")
    height, width = finput.size
    MSquare = np.asarray(finput_rgb, dtype=np.uint8)
    Square = np.zeros((width, height, 3), dtype=np.uint8)
    height, width = finput.size
    for i in range(int(n/2), width-int(n/2), 1):
        for j in range(int(n/2)-2, height-int(n/2)-2, 1):
            if (MSquare[i][j][2] > 1):
                Square[i][j][2] = MSquare[i][j][2]
            if (MSquare[i][j][0] > 1):
                print('Expanding square at i,j=', i, ',', j)
                for i2 in range(-int(n/2), int(n/2), 1):
                    for j2 in range(-int(n/2)+2, int(n/2)+2, 1):
                        Square[i+i2][j+j2][0] = MSquare[i][j][0]
    res = Image.fromarray(Square, 'RGB')
    res.save(outputdir)

# This function replaces the corner squares (i.e. roof corners) in the tiles by black if over black
# Or by white ({255,255,255}) if over pixel {0,0,200} (train set), by red ({255,0,0}) if over pixel {0,0,210} (val set), by green ({0,255,0}) if over pixel {0,0,220} (test set)
def fCorners_colorcode(input, output):
    for file in sorted(os.listdir(input)):
        if file.endswith('.png'):
            filename = os.path.splitext(os.path.basename(file))[0]
            print(filename)
            finput = PIL.Image.open(input + file)
            finput_rgb = finput.convert("RGB")
            height, width = finput.size
            Corners_colorcode = np.asarray(finput_rgb)
            for i in range(0, width, 1):
                for j in range(0, height, 1):
                    '''
                    if Corners_colorcode[i][j][0] > 0 :
                        print('Corners_colorcode[', i, '][', j, '] = {', Corners_colorcode[i][j][0], Corners_colorcode[i][j][1], Corners_colorcode[i][j][2], '}')
                    '''
                    if Corners_colorcode[i][j][0] + Corners_colorcode[i][j][1] + Corners_colorcode[i][j][2] < 0.5:
                        #print('Corners_colorcode[', i, '][', j, '] = {', Corners_colorcode[i][j][0], Corners_colorcode[i][j][1], Corners_colorcode[i][j][2], '} passed over black pixel')
                        continue # i.e. if black pixel
                    if (Corners_colorcode[i][j][0] >= 1) and (Corners_colorcode[i][j][1] < 1) and (Corners_colorcode[i][j][2] < 1): # i.e. black corner over black background
                        #print('Corners_colorcode[', i, '][', j, '] = {', Corners_colorcode[i][j][0], Corners_colorcode[i][j][1], Corners_colorcode[i][j][2], '} filled black')
                        Corners_colorcode[i][j][0] = 0 # Filling it black
                        Corners_colorcode[i][j][1] = 0 # Filling it black
                        Corners_colorcode[i][j][2] = 0 # Filling it black
                        continue
                    if (Corners_colorcode[i][j][0] >= 1) and (Corners_colorcode[i][j][1] < 1) and (Corners_colorcode[i][j][2] == 200):  # i.e. red_z corner over 230-blue train set
                        print('Corners_colorcode[', i, '][', j, '] = {', Corners_colorcode[i][j][0], Corners_colorcode[i][j][1], Corners_colorcode[i][j][2], '} filled red')
                        Corners_colorcode[i][j][0] += 200  # Filling it red_z
                        continue
                    if (Corners_colorcode[i][j][0] >= 1) and (Corners_colorcode[i][j][1] < 1) and (Corners_colorcode[i][j][2] == 210):  # i.e. green_z corner over 230-blue val set
                        print('Corners_colorcode[', i, '][', j, '] = {', Corners_colorcode[i][j][0], Corners_colorcode[i][j][1], Corners_colorcode[i][j][2], '} filled green')
                        Corners_colorcode[i][j][1] = 210 + Corners_colorcode[i][j][0]  # Filling it green_z
                        Corners_colorcode[i][j][0] = 0  # Filling it green_z
                        continue
                    if (Corners_colorcode[i][j][0] >= 1) and (Corners_colorcode[i][j][1] < 1) and (Corners_colorcode[i][j][2] == 220):  # i.e. yellow_z corner over 230-blue test set
                        print('Corners_colorcode[', i, '][', j, '] = {', Corners_colorcode[i][j][0], Corners_colorcode[i][j][1], Corners_colorcode[i][j][2], '} filled yellow')
                        Corners_colorcode[i][j][0] += 220  # Filling it yellow_z (red + green)
                        Corners_colorcode[i][j][1] = Corners_colorcode[i][j][0]  # Filling it yellow_z (red + green)
                        continue
            res = Image.fromarray(Corners_colorcode, 'RGB')
            res.save(output + filename + '.png')

# This function limes and erases roof corner pixels that are not contiguous to the black background
def fRoofCornerShrinking(input, output):
    Raster = PIL.Image.open(input)
    Raster_rgb = Raster.convert("RGB")
    Raster_np = np.asarray(Raster_rgb)
    height, width = Raster.size
    for i in range(0, width, 1):
        for j in range(0, height, 1):
            #print('Screening image at i=', i, '/', height, ', j=', j, '/', width, '...')
            if (Raster_np[i][j][0] >= 1) or (Raster_np[i][j][1] >= 1): # if there is a roof corner
                cond1 = (Raster_np[i][max(0, j-1)][2] >= 1)  # no contiguous black pixel on the left
                cond2 = (Raster_np[i][min(height-1, j+1)][2] >= 1)  # no contiguous black pixel on the right
                cond3 = (Raster_np[max(0, i-1)][max(0, j-1)][2] >= 1)  # no contiguous black pixel on the top-left
                cond4 = (Raster_np[max(0, i-1)][j][2] >= 1)  # no contiguous black pixel on the top
                cond5 = (Raster_np[max(0, i-1)][min(height-1, j+1)][2] >= 1)  # no contiguous black pixel on the top-right
                cond6 = (Raster_np[min(width-1, i+1)][max(0, j-1)][2] >= 1)  # no contiguous black pixel on the bottom-left
                cond7 = (Raster_np[min(width-1, i+1)][j][2] >= 1)  # no contiguous black pixel on the bottom
                cond8 = (Raster_np[min(width-1, i+1)][min(height-1, j+1)][2] >= 1)  # no contiguous black pixel on the bottom-right
                if (cond1 and cond2 and cond3 and cond4 and cond5 and cond6 and cond7 and cond8):
                    Raster_np[i][j][0] = 0 # erase this pixel roof corner
                    Raster_np[i][j][1] = 0 # erase this pixel roof corner
                    print('Roof corner pixel erased at i=', i, '/', height, ', j=', j, '/', width)
    res = Image.fromarray(Raster_np, 'RGB')
    res.save(output)

# This takes a raster with invisible corners (on R of RGB) and blue 2D segmentation (B on RGB) and blackens diagonal pixels between roofs
def fRoofSeparator(input, output):
    Raster = PIL.Image.open(input)
    Raster_rgb = Raster.convert("RGB")
    Raster_np = np.asarray(Raster_rgb)
    height, width = Raster.size
    for i in range(1, width-1, 1):
        for j in range(1, height-1, 1):
            condAB = (Raster_np[i][j][2] >= 1) # if there is a roof pixel
            condA1 = (Raster_np[i - 1][j + 1][2] >= 1)  # roof pixel on top-right
            condA2 = (Raster_np[i - 1][j][2] < 1)  # black pixel on top
            condA3 = (Raster_np[i][j + 1][2] < 1)  # black pixel on the right
            condB1 = (Raster_np[i + 1][j - 1][2] >= 1)  # roof pixel on bottom-left
            condB2 = (Raster_np[i + 1][j][2] < 1)  # black pixel on bottom
            condB3 = (Raster_np[i][j - 1][2] < 1)  # black pixel on the left
            condC1 = (Raster_np[i - 1][j - 1][2] >= 1)  # roof pixel on top-left
            condC2 = (Raster_np[i - 1][j][2] < 1)  # black pixel on top
            condC3 = (Raster_np[i][j - 1][2] < 1)  # black pixel on the left
            condD1 = (Raster_np[i + 1][j + 1][2] >= 1)  # roof pixel on bottom-right
            condD2 = (Raster_np[i + 1][j][2] < 1)  # black pixel on bottom
            condD3 = (Raster_np[i][j + 1][2] < 1)  # black pixel on the right
            condA = condAB and condA1 and condA2 and condA3
            condB = condAB and condB1 and condB2 and condB3
            condC = condAB and condC1 and condC2 and condC3
            condD = condAB and condD1 and condD2 and condD3
            if condA or condB or condC or condD:
                Raster_np[i][j][0] = 0 # blackens this pixel
                Raster_np[i][j][1] = 0 # blackens this pixel
                Raster_np[i][j][2] = 0 # blackens this pixel
                print('Diagonal pixel erased at i=', i, '/', height, ', j=', j, '/', width)
    res = Image.fromarray(Raster_np, 'RGB')
    res.save(output)

def fRoofBlackening(input, output):
    Raster = PIL.Image.open(input)
    Raster_rgb = Raster.convert("RGB")
    Raster_np = np.asarray(Raster_rgb)
    height, width = Raster.size
    for i in range(1, width, 1):
        for j in range(1, height, 1):
            if ((Raster_np[i][j][0] >= 0.1) and (Raster_np[i][j][2] <= 0.1)):
                Raster_np[i][j][0] = 0
                Raster_np[i][j][1] = 0
                Raster_np[i][j][2] = 0
            print('Roof corner pixel blackening at i=', i, '/', height, ', j=', j, '/', width)
    res = Image.fromarray(Raster_np, 'RGB')
    res.save(output)

def fGreenBlackening(input, output):
    Raster = PIL.Image.open(input)
    Raster_rgb = Raster.convert("RGB")
    Raster_np = np.asarray(Raster_rgb)
    Result = np.asarray(Raster_rgb)
    height, width = Raster.size
    for i in range(1, width, 1):
        if (i%200==0): print('Green roof lines blackening at i = ', i, '/', width)
        for j in range(1, height, 1):
            if ((Raster_np[i][j][1] >= 1)): # Pure green
                Result[i][j][1] = 0
                Result[i][j][2] = 0
    res = Image.fromarray(Result, 'RGB')
    res.save(output)

def fResultComparison(input, output, opt):
    Raster = PIL.Image.open(input)
    Raster_rgb = Raster.convert("RGB")
    Raster_np = np.asarray(Raster_rgb)
    height, width = Raster.size
    for i in range(1, width, 1):
        if (i % 200 == 0): print('Data post-processing at i = ', i, '/', width)
        for j in range(1, height, 1):
            if ((Raster_np[i][j][0] == 0) and (Raster_np[i][j][1] == 0) and (Raster_np[i][j][2] == 0)):
                Raster_np[i][j][0] = 255
                Raster_np[i][j][1] = 255
                Raster_np[i][j][2] = 255
                continue
            if ((Raster_np[i][j][0] == 0) and (Raster_np[i][j][1] == 0) and (Raster_np[i][j][2] >= 1) and opt == 'green'):
                Raster_np[i][j][1] = Raster_np[i][j][2]
                Raster_np[i][j][2] = 0
    res = Image.fromarray(Raster_np, 'RGB')
    res.save(output)


# This function assigns via label() unique ids to each continuous set of roof corner pixels
# as a numpy array for the whole raster image, and then for each roof corner pixel, it records all associated pixels of
# same id in a 30x30 square around it, and finds the ones with most black pixel neighbors in 8-connectivity, and erases the others
def fRoofCornerExtremeShrinking(input, output):
    Raster = PIL.Image.open(input)
    Raster_rgb = Raster.convert("RGB")
    Raster_np = np.asarray(Raster_rgb)
    height, width = Raster.size
    M = np.zeros((width, height), dtype=np.uint32)
    M_connectivity_color = np.zeros((width, height, 3), dtype=np.uint8)
    # Assigning unique ids to each continuous set of roof corner pixels
    for i in range(0, width, 1):
        for j in range(0, height, 1):
            if (Raster_np[i][j][0] <= 0.1):
                continue
            M[i][j] = 200 # Filling M_connectivity with 200 when there is a roof corner pixel, 0 otherwise
            print('STEP1: Labeled roof corner at i=', i, '/', height, ', j=', j, '/', width, ' with ', M[i][j])
    # Out of this np array, now use scikit-image's function label() with 8-connectivity
    # Note: connectivity=1 is for the 4 vertical-horizontal pixels, while connectivity=2 is for the 8 vertical-horizontal-diagonal pixels
    M_connectivity = label(M, connectivity=2)
    # print(list(set(list((M_connectivity.flatten().tolist()))))) # Check unique assigned id's
    '''
    for i in range(2000, 3000, 1):
        for j in range(2000, 3000, 1):
            M_connectivity_color[i][j][0] = int(M_connectivity[i][j])
            M_connectivity_color[i][j][1] = int(0)
            M_connectivity_color[i][j][2] = int(0)
    res = Image.fromarray(M_connectivity_color, 'RGB')
    res.save('/user/jlussang/home/Desktop/Grid/detectron2_results/M_connectivity_color.png')
    exit()
    '''
    print('STEP2: scikit-image function label() activated...')
    # For each roof corner pixel, recording all pixels of same id in a 30x30 square around it,
    for i in range(0, width, 1):
        for j in range(0, height, 1):
            if (M_connectivity[i][j] <= 0.1):
                continue # Consider only roof corner pixels
            selected_i, selected_j = i, j
            blackneighbors_max = 0
            print('STEP3: Finding the most bordeline roof corner pixel in a 30x30 grid around pixel i=', i, '/', height, ', j=', j, '/', width)
            for i2 in range(max(1, i-15), min(i+15, width-1), 1):
                for j2 in range(max(1, j-15), min(j+15, height-1), 1):
                    if (M_connectivity[i][j] != M_connectivity[i2][j2]) or (Raster_np[i2][j2][0] <= 0.1):
                        continue # If the pixel in this 30x30 square if not of same id and not a roof corner, continue
                    # Computing the number of black pixels around this pixel with 8-connectivity
                    blackneighbors = 0
                    if (Raster_np[i2-1][j2-1][2] <= 0.1): blackneighbors += 1 # top-left
                    if (Raster_np[i2-1][j2][2] <= 0.1): blackneighbors += 1 # top
                    if (Raster_np[i2-1][j2+1][2] <= 0.1): blackneighbors += 1 # top-right
                    if (Raster_np[i2][j2-1][2] <= 0.1): blackneighbors += 1 # left
                    if (Raster_np[i2][j2+1][2] <= 0.1): blackneighbors += 1 # right
                    if (Raster_np[i2+1][j2-1][2] <= 0.1): blackneighbors += 1 # bottom-left
                    if (Raster_np[i2+1][j2][2] <= 0.1): blackneighbors += 1 # bottom
                    if (Raster_np[i2+1][j2+1][2] <= 0.1): blackneighbors += 1 # bottom-right
                    if (blackneighbors_max <= blackneighbors):
                        blackneighbors_max = blackneighbors
                        selected_i, selected_j = i2, j2
            for i2 in range(max(1, i - 15), min(i + 15, width - 1), 1):
                for j2 in range(max(1, j - 15), min(j + 15, height - 1), 1):
                    if (M_connectivity[i][j] != M_connectivity[i2][j2]) or (Raster_np[i2][j2][0] <= 0.1):
                        continue # If the pixel in this 30x30 square if not of same id and not a roof corner, continue
                    if ((i2 == selected_i) and (j2 == selected_j)):
                        #Raster_np[i2][j2] = [255, 255, 255] # out-comment this line only if wanting to produce a white visual for the selected pixel ZYX
                        continue
                    # Erasing all roof corner pixels of same id as pixel (i,j), except the one (selected_i, selected_j)
                    Raster_np[i2][j2][0] = 0
                    print('       Found it at i=', selected_i, ', j=', selected_j, ' with ', blackneighbors_max, ' black pixel neighbors...')
    res = Image.fromarray(Raster_np, 'RGB')
    res.save(output)

# Expands a square roof corner pixel with one same pixel in each 8-direction there is non-corner roof plane pixel, or a black pixel followed by the latter
def fRoofCornerEnlarge(input, output):
    Raster = PIL.Image.open(input)
    Raster_rgb = Raster.convert("RGB")
    Raster_np = np.asarray(Raster_rgb)
    height, width = Raster.size
    M = np.zeros((width, height), dtype=np.uint32)
    for i in range(2, width-2, 1):
        for j in range(2, height-2, 1):
            condAB = (Raster_np[i][j][0] >= 1) and (Raster_np[i][j][2] < 1)  # there is a roof corner pixel but no roof plane
            condA1 = (Raster_np[i - 1][j - 1][2] >= 1) and (Raster_np[i - 1][j - 1][0] < 1)  # non-corner roof pixel on top-left
            condA2 = (Raster_np[i - 1][j][2] >= 1) and (Raster_np[i - 1][j][0] < 1) # non-corner roof pixel on top
            condA3 = (Raster_np[i - 1][j + 1][2] >= 1) and (Raster_np[i - 1][j + 1][0] < 1) # non-corner roof pixel on top-right
            condA4 = (Raster_np[i][j - 1][2] >= 1) and (Raster_np[i][j - 1][0] < 1) # non-corner roof pixel on the left
            condA5 = (Raster_np[i][j + 1][2] >= 1) and (Raster_np[i][j + 1][0] < 1) # non-corner roof pixel on the right
            condA6 = (Raster_np[i + 1][j - 1][2] >= 1) and (Raster_np[i + 1][j - 1][0] < 1) # non-corner roof pixel on bottom-left
            condA7 = (Raster_np[i + 1][j][2] >= 1) and (Raster_np[i + 1][j][0] < 1) # non-corner roof pixel on bottom
            condA8 = (Raster_np[i + 1][j + 1][2] >= 1) and (Raster_np[i + 1][j + 1][0] < 1) # non-corner roof pixel on bottom-right
            condB1 = (Raster_np[i - 2][j - 2][2] >= 1) and (Raster_np[i - 2][j - 2][0] < 1) and (Raster_np[i - 1][j - 1][2] < 1) and (Raster_np[i - 1][j - 1][0] < 1) # non-corner roof pixel on 2-top-left after a black pixel on 1-top-left
            condB2 = (Raster_np[i - 2][j][2] >= 1) and (Raster_np[i - 2][j][0] < 1) and (Raster_np[i - 1][j][2] < 1) and (Raster_np[i - 1][j][0] < 1) # non-corner roof pixel on 2-top after a black pixel on 1-top
            condB3 = (Raster_np[i - 2][j + 2][2] >= 1) and (Raster_np[i - 2][j + 2][0] < 1) and (Raster_np[i - 1][j + 1][2] < 1) and (Raster_np[i - 1][j + 1][0] < 1) # non-corner roof pixel on 2-top-right after a black pixel on 1-top-right
            condB4 = (Raster_np[i][j - 2][2] >= 1) and (Raster_np[i][j - 2][0] < 1) and (Raster_np[i][j - 1][2] < 1) and (Raster_np[i][j - 1][0] < 1) # non-corner roof pixel on the 2-left after a black pixel on 1-left
            condB5 = (Raster_np[i][j + 2][2] >= 1) and (Raster_np[i][j + 2][0] < 1) and (Raster_np[i][j + 1][2] < 1) and (Raster_np[i][j + 1][0] < 1) # non-corner roof pixel on the 2-right after a black pixel on 1-right
            condB6 = (Raster_np[i + 2][j - 2][2] >= 1) and (Raster_np[i + 2][j - 2][0] < 1) and (Raster_np[i + 1][j - 1][2] < 1) and (Raster_np[i + 1][j - 1][0] < 1) # non-corner roof pixel on 2-bottom-left after a black pixel on 1-bottom-left
            condB7 = (Raster_np[i + 2][j][2] >= 1) and (Raster_np[i + 2][j][0] < 1) and (Raster_np[i + 1][j][2] < 1) and (Raster_np[i + 1][j][0] < 1) # non-corner roof pixel on 2-bottom after a black pixel on 1-bottom
            condB8 = (Raster_np[i + 2][j + 2][2] >= 1) and (Raster_np[i + 2][j + 2][0] < 1) and (Raster_np[i + 1][j + 1][2] < 1) and (Raster_np[i + 1][j + 1][0] < 1) # non-corner roof pixel on 2-bottom-right after a black pixel on 1-bottom-right
            if (condAB and condA1):
                Raster_np[i - 1][j - 1][0] = Raster_np[i][j][0]
                print('Enlarged roof corner pixel i=', i, ', j=', j, ' in top-left')
            if (condAB and condA2):
                Raster_np[i - 1][j][0] = Raster_np[i][j][0]
                print('Enlarged roof corner pixel i=', i, ', j=', j, ' in top')
            if (condAB and condA3):
                Raster_np[i - 1][j + 1][0] = Raster_np[i][j][0]
                print('Enlarged roof corner pixel i=', i, ', j=', j, ' in top-right')
            if (condAB and condA4):
                Raster_np[i][j - 1][0] = Raster_np[i][j][0]
                print('Enlarged roof corner pixel i=', i, ', j=', j, ' in left')
            if (condAB and condA5):
                Raster_np[i][j + 1][0] = Raster_np[i][j][0]
                print('Enlarged roof corner pixel i=', i, ', j=', j, ' in right')
            if (condAB and condA6):
                Raster_np[i + 1][j - 1][0] = Raster_np[i][j][0]
                print('Enlarged roof corner pixel i=', i, ', j=', j, ' in bottom-left')
            if (condAB and condA7):
                Raster_np[i + 1][j][0] = Raster_np[i][j][0]
                print('Enlarged roof corner pixel i=', i, ', j=', j, ' in bottom')
            if (condAB and condA8):
                Raster_np[i + 1][j + 1][0] = Raster_np[i][j][0]
                print('Enlarged roof corner pixel i=', i, ', j=', j, ' in bottom-right')
            if (condAB and condB1):
                Raster_np[i - 2][j - 2][0] = Raster_np[i][j][0]
                print('Enlarged roof corner pixel i=', i, ', j=', j, ' in top-left 2 pixels away')
            if (condAB and condB2):
                Raster_np[i - 2][j][0] = Raster_np[i][j][0]
                print('Enlarged roof corner pixel i=', i, ', j=', j, ' in top 2 pixels away')
            if (condAB and condB3):
                Raster_np[i - 2][j + 2][0] = Raster_np[i][j][0]
                print('Enlarged roof corner pixel i=', i, ', j=', j, ' in top-right 2 pixels away')
            if (condAB and condB4):
                Raster_np[i][j - 2][0] = Raster_np[i][j][0]
                print('Enlarged roof corner pixel i=', i, ', j=', j, ' in left 2 pixels away')
            if (condAB and condB5):
                Raster_np[i][j + 2][0] = Raster_np[i][j][0]
                print('Enlarged roof corner pixel i=', i, ', j=', j, ' in right 2 pixels away')
            if (condAB and condB6):
                Raster_np[i + 2][j - 2][0] = Raster_np[i][j][0]
                print('Enlarged roof corner pixel i=', i, ', j=', j, ' in bottom-left 2 pixels away')
            if (condAB and condB7):
                Raster_np[i + 2][j][0] = Raster_np[i][j][0]
                print('Enlarged roof corner pixel i=', i, ', j=', j, ' in bottom 2 pixels away')
            if (condAB and condB8):
                Raster_np[i + 2][j + 2][0] = Raster_np[i][j][0]
                print('Enlarged roof corner pixel i=', i, ', j=', j, ' in bottom-right 2 pixels away')
    res = Image.fromarray(Raster_np, 'RGB')
    res.save(output)

# Takes the whole masked image and assigns a unique id to each segmented roof and its corners
# Now our whole input is made of 2D roof shapes as {0,0,200} for train,{0,0,210} for train, {0,0,220} for test,
# and 3D small corners {z,0,200} over train roofs, {z,0,210} over val roofs, {z,0,220} over test roofs
# Output list_frag.csv by fRoofCornerLabeling(), where each line is the roofs id, as {x,y,z,roof_id,dataset_id} (z=0 for all non-corners and dataset_id=0,1,2 for train,val,test resp.)
def fRoofCornerLabeling(input):
    Resolution = 0.38 # working with raster data at such resolution
    default_height = 5
    Raster = PIL.Image.open(input)
    Raster_rgb = Raster.convert("RGB")
    Raster_np = np.asarray(Raster_rgb)
    height, width = Raster.size
    #print(height, width) # 8580 9900
    M = np.zeros((width, height))
    #M_connectivity = np.zeros((width, height), dtype=np.uint8)
    M_delaunay = np.zeros((width, height), dtype=np.uint8)
    diff = 200 # diff=200 when large, 0 when sample 0000000000031.png

    #Checking that Raster_np is well encoded
    mylistR = []
    mylistG = []
    mylistB = []
    for i in range(0, width, 1):
        if (i % 50 == 0): print('Step 1/5: Build a single np_array from raster: i = ', i, '/', width)
        for j in range(0, height, 1):
            mylistR.append(Raster_np[i][j][0])
            mylistG.append(Raster_np[i][j][1])
            mylistB.append(Raster_np[i][j][2])
    mylistR = sorted(list(set(mylistR)))
    mylistG = sorted(list(set(mylistG)))
    mylistB = sorted(list(set(mylistB)))
    print(mylistR)
    print(mylistG)
    print(mylistB)
    #exit()

    # STEP 1: First build a width x height single np array M encoding the 3D corners and 2D roof planes as follows
    for i in range(0, width, 1):
    # for i in range(140, 160, 1):
        if (i%200==0): print('STEP 1/7: Build a single np_array from raster: i = ', i, '/', width)
        for j in range(0, height, 1):
            z = 0
            if (Raster_np[i][j][0] >= 1):
                z = int(Raster_np[i][j][0]) - diff
            if (Raster_np[i][j][2] == 200):
                M[i][j] = 100 + z # M[i][j] = 100 + z being the np array for train set roofs
                #if (Raster_np[i][j][0] >= 1): print('train Raster_np[i][j][0]=', Raster_np[i][j][0], ', Raster_np[i][j][2]=', Raster_np[i][j][2], ' M[i][j]=', M[i][j])
            if (Raster_np[i][j][2] == 210):
                M[i][j] = 125 + z # M[i][j] = 125 + z being the np array for val set roofs
                #if (Raster_np[i][j][0] >= 1): print('val Raster_np[i][j][0]=', Raster_np[i][j][0], ', Raster_np[i][j][2]=', Raster_np[i][j][2], ' M[i][j]=', M[i][j])
            if (Raster_np[i][j][2] == 220):
                M[i][j] = 150 + z # M[i][j] = 150 + z being the np array for test set roofs
                #if (Raster_np[i][j][0] >= 1): print('test Raster_np[i][j][0]=', Raster_np[i][j][0], ', Raster_np[i][j][2]=', Raster_np[i][j][2], ' M[i][j]=', M[i][j])
    '''
    #Checking that M is well encoded
    for i in range(0, width, 1):
        for j in range(0, height, 1):
            if (M[i][j] > 999999) or (M[i][j] < 0):
                M[i][j] = 0
    for i in range(0, width, 1):
        for j in range(0, height, 1):
            if (M[i][j] > 999999) or (M[i][j] < 0):
                print('M[', i, '][', j, '] = ', M[i][j], ' is ill-posed')
    list_checkdigits = []
    for i in range(0, width, 1):
        #if (i % 50 == 0): print('Checking that M is well encoded: i = ', i, '/', width)
        for j in range(0, height, 1):
            list_checkdigits.append(M[i][j])
    list_checkdigits = sorted(list(set(list_checkdigits)))
    print(list_checkdigits)
    '''

    # STEP 2: Now flood_fill() this M np_array with tolerance 70 so on the whole picture a 2D segmentation with 100 can be joined to a contiguous one at 150+19=169, but not the black background
    roof_id = 500  # roof ID
    for i in range(0, width, 1):
    #for i in range(140, 160, 1):
        #if (i % 50 == 0): print('Step 2/5: Flood_fill() with roof_id = ', roof_id, ' this single np_array: i = ', i, '/', width)
        for j in range(0, height, 1):
            if (M[i][j] >= 100) and (M[i][j] <= 120):
                M = flood_fill(M, (i, j), roof_id, tolerance=70)
                print('STEP 2/7: Flooded train at i=', i, '/', height, ', j=', j, '/', width, ' with temp roof_id #', roof_id, ' => (M[i][j]=', M[i][j], ')')
                roof_id += 1
                continue
            if (M[i][j] >= 125) and (M[i][j] <= 145):
                M = flood_fill(M, (i, j), roof_id, tolerance=70)
                print('STEP 2/7: Flooded val at i=', i, '/', height, ', j=', j, '/', width, ' with temp roof_id #', roof_id, ' => (M[i][j]=', M[i][j], ')')
                roof_id += 1
                continue
            if (M[i][j] >= 150) and (M[i][j] <= 170):
                M = flood_fill(M, (i, j), roof_id, tolerance=70)
                print('STEP 2/7: Flooded test at i=', i, '/', height, ', j=', j, '/', width, ' with temp roof_id #', roof_id, ' => (M[i][j]=', M[i][j], ')')
                roof_id += 1
                continue
    '''
    # Checking that M is well encoded
    for i in range(0, width, 1):
        for j in range(0, height, 1):
            if (M[i][j] > 999999) or (M[i][j] < 0):
                M[i][j] = 0
    for i in range(0, width, 1):
        for j in range(0, height, 1):
            if (M[i][j] > 999999) or (M[i][j] < 0):
                print('M[', i, '][', j, '] = ', M[i][j], ' is ill-posed')
    
    list_checkdigits = []
    for i in range(0, width, 1):
        # if (i % 50 == 0): print('Checking that M is well encoded: i = ', i, '/', width)
        for j in range(0, height, 1):
            list_checkdigits.append(M[i][j])
    list_checkdigits = sorted(list(set(list_checkdigits)))
    print(list_checkdigits)
    '''
    # STEP 3: Now turning this into a list of k roofs, each consisting of a list of (x,y,z) points with z = 0 for all non-corner points
    cv2.imwrite('/home/jlussang/Desktop/Grid/detectron2_results/M_floodfill.png', M)
    #print('Now turning this into a list (of roofs) of lists of (x,y,z) points')
    list_frags = []
    for k in range(0, roof_id-500, 1):
        Roof_points = []
        list_frags.append(Roof_points)
    print('len(list_frags)=', len(list_frags))
    for i in range(0, width, 1):
    #for i in range(140, 160, 1):
        if (i % 200 == 0): print('STEP 3/7: Turn this single np_array into a list of ', roof_id - 500, ' lists of incomplete (x,y,z) points: i = ', i, '/', width)
        for j in range(0, height, 1):
            if (M[i][j] >= 500): # if there is a roof id
                id = int(M[i][j]) - 500 # fetch this roof id and renormalize it
                z = max(0, int(Raster_np[i][j][0])-diff)  # fetch altitude (always zero if point {i,j} is not on a roof corner)
                if (Raster_np[i][j][2] == 200): # if this {i,j} point is from the train train set
                    index_dataset = 0 # train
                if (Raster_np[i][j][2] == 210): # if this {i,j} point is from the val data set
                    index_dataset = 1 # val
                if (Raster_np[i][j][2] == 220): # if this {i,j} point is from the test data set
                    index_dataset = 2 # test
                #if (Raster_np[i][j][0] >= 1) or (Raster_np[i][j][1] >= 1):
                    #print('Raster_np[i][j]=', Raster_np[i][j][0], Raster_np[i][j][1], Raster_np[i][j][2], ' and z=', z)
                list_frags[id].append([i, j, z, int(id), int(index_dataset)]) # roof corner data as {i, j, z, int(id), int(index_dataset)}
                #print('          {i, j, z, id, index_dataset} = {', i, j, z, int(id), int(index_dataset), '} => len(list_frags[id]) = ', len(list_frags[id]))
    # Outputing this List_roofs as csv file...
    #print('Now outputing list_frag as .csv file, with ', roof_nr, ' roofs')
    CSVFile = open('/home/jlussang/Desktop/Grid/detectron2_results/list_frag.csv', 'w')
    # Each line k of the .csv file below corresponds to the k=roof_id list of points (i,j,z)
    with CSVFile:
        FileOutput = csv.writer(CSVFile)
        FileOutput.writerows(list_frags)

    # STEP 4: Finding for each roof plane list_frags2[k] its 3 most distant roof corners by the largest 2D triangle area formed by these points (Delaunay triangulation)
    #list_frags2 = list_frags.copy()
    list_frags2 = copy.deepcopy(list_frags)
    average_height_regression = 0
    average_height_regression_nr = 0
    List_roofs_coeffs = []
    total_nr_corners = 0
    for k in range(0, len(list_frags2), 1):
        Roof_planes = []
        List_roofs_coeffs.append(Roof_planes) # For each k roof plane, its associated A, B, C, D plane 3D coefficients, and three widest points' coordinates
        for p in range(0, len(list_frags2[k]), 1):
            if (list_frags2[k][p][2] > 0.1):
                total_nr_corners += 1 # We count the total roof corners detected by our model
                if (list_frags2[k][p][4] <= 0.1): # If a roof corner is in the train set
                    average_height_regression += list_frags2[k][p][2]
                    average_height_regression_nr += 1
    average_height_regression /= average_height_regression_nr # average height of roof corners of the train set, that will be used by linear regression as default_height
    default_height = average_height_regression
    print('2D segmentation stats: Out of 6,609 roof planes as GT, the 2D segmentation model found ', len(list_frags2), ' (i.e. ', int(len(list_frags2)*100.0/6609), '%)')
    print('3D reconstruction stats: Out of 23,024 roof corners as GT, the 3D reconstruction model found ', total_nr_corners,' (i.e. ', int(total_nr_corners * 100.0 / 23024), '%)')
    print('average_height_regression = ', average_height_regression, ' with average_height_regression_nr = ', average_height_regression_nr)
    for k in range(0, len(list_frags2), 1):
        nr_corners = 0
        corner_list = []
        for p in range(0, len(list_frags2[k]), 1):
            if (list_frags2[k][p][2] > 0.1):
                nr_corners += 1  # Work only on 3D roof corners, not 2D segmentations
                corner_list.append(list_frags2[k][p][2])
        corner_list_sorted = sorted(corner_list)  # ascending order
        #print('          Found roof plane k = ', k, ' having ', nr_corners, ' roof corners')
    for k in range(0, len(list_frags2), 1):
        print('STEP 4/7: Finding the Delaunay triangulation for the corners of each roof plane k = ', k, '/', len(list_frags2), ' this roof k having ', len(list_frags2[k]), ' points')
        largest_area = 0
        x1_wide, y1_wide, z1_wide, x2_wide, y2_wide, z2_wide, x3_wide, y3_wide, z3_wide = 0, 0, default_height, 100, 100, default_height, 200, 200, default_height
        for p1 in range(0, len(list_frags2[k])-2, 1):
            if (list_frags2[k][p1][2] < 0.1): continue # Work only on 3D roof corners, not 2D segmentations
            for p2 in range(p1 + 1, len(list_frags2[k])-1, 1):
                if (list_frags2[k][p2][2] < 0.1): continue  # Work only on 3D roof corners, not 2D segmentations
                for p3 in range(p1 + 2, len(list_frags2[k]), 1):
                    if (list_frags[k][p3][2] < 0.1): continue  # Work only on 3D roof corners, not 2D segmentations
                    x1, y1, z1 = list_frags2[k][p1][0], list_frags2[k][p1][1], list_frags2[k][p1][2]
                    x2, y2, z2 = list_frags2[k][p2][0], list_frags2[k][p2][1], list_frags2[k][p2][2]
                    x3, y3, z3 = list_frags2[k][p3][0], list_frags2[k][p3][1], list_frags2[k][p3][2]
                    area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
                    if (largest_area <= area):
                        largest_area = area
                        x1_wide, y1_wide, z1_wide, x2_wide, y2_wide, z2_wide, x3_wide, y3_wide, z3_wide = x1, y1, z1, x2, y2, z2, x3, y3, z3
        # Computing the whole number of corners for this roof k
        nr_corners = 0
        corner_list = []
        for p in range(0, len(list_frags2[k]), 1):
            if (list_frags2[k][p][2] > 0.1):
                nr_corners += 1  # Work only on 3D roof corners, not 2D segmentations
                corner_list.append(list_frags2[k][p][2])
        corner_list_sorted = sorted(corner_list) # ascending order
        print('          Found roof plane k = ', k, ' having ', nr_corners, ' roof corners and largest_area =', largest_area, ' within (', x1_wide, y1_wide, z1_wide, '), (', x2_wide, y2_wide, z2_wide, '), (', x3_wide, y3_wide, z3_wide, ')')
        # Failsafe: if the Delaunay triangulation area is below 20% of the total roof area, or above twice its area, or if the number of roof corners is 0 or 1
        cond_impossible_coeff = 0
        if (largest_area <= int(len(list_frags2[k]) / 5)) or (largest_area >= int(len(list_frags2[k]) * 2) or (nr_corners <= 2)): # Issue XYZ: previously (nr_corners <= 1)
            print('          Roof plane corners not relevant...')
            z1_wide, z2_wide, z3_wide = default_height, default_height, default_height
            corner_list_sorted = [z1_wide, z2_wide, z3_wide]
            cond_impossible_coeff = 1

        # STEP 5: Performing height-regularization of these three Delaunay roof corners, knowing 2 of them should have averaged same height
        roof_procedure = [z1_wide, z2_wide, z3_wide]
        print('STEP 5/7: Performing height-regularization of these corners, with z1, z2, z3 =', roof_procedure)
        #z1_wide, z2_wide, z3_wide = 7, 4, 5 # test
        roof_procedure_sorted = sorted(roof_procedure) # ascending order
        roof_procedure_sorted_copy = roof_procedure_sorted.copy()
        #print('          Roof_procedure_sorted =', roof_procedure_sorted)
        #print('          Roof_procedure_sorted_copy =', roof_procedure_sorted_copy)
        indexz1 = roof_procedure_sorted_copy.index(z1_wide)
        roof_procedure_sorted_copy[indexz1] = -999
        indexz2 = roof_procedure_sorted_copy.index(z2_wide)
        roof_procedure_sorted_copy[indexz2] = -999
        indexz3 = roof_procedure_sorted_copy.index(z3_wide)
        roof_procedure_sorted_copy[indexz3] = -999
        roof_procedure_indexed = [indexz1, indexz2, indexz3]
        roof_procedure_sorted2 = roof_procedure_sorted.copy()
        #print('          Roof_procedure_indexed =', roof_procedure_indexed)
        #print('          Roof_procedure_sorted2 =', roof_procedure_sorted2)
        if (abs(roof_procedure_sorted[1]-roof_procedure_sorted[0]) <= abs(roof_procedure_sorted[2]-roof_procedure_sorted[1])): # if z1 and z2 are the lowest heights of the roof plane
            roof_procedure_sorted2[0] = abs(roof_procedure_sorted[1] + roof_procedure_sorted[0]) / 2
            roof_procedure_sorted2[1] = abs(roof_procedure_sorted[1] + roof_procedure_sorted[0]) / 2
            roof_procedure_sorted2[2] = roof_procedure_sorted[2]
            print('          New roof_procedure_sorted low =', roof_procedure_sorted2)
            # With extreme values instead ZZZ (uncomment if not needed)
            roof_procedure_sorted2[0] = corner_list_sorted[0]
            roof_procedure_sorted2[1] = corner_list_sorted[0]
            roof_procedure_sorted2[2] = roof_procedure_sorted[2]
            print('          With extreme values instead =', roof_procedure_sorted2)
        if (abs(roof_procedure_sorted[1]-roof_procedure_sorted[0]) > abs(roof_procedure_sorted[2]-roof_procedure_sorted[1])): # if z2 and z3 are the highest heights of the roof plane
            roof_procedure_sorted2[0] = roof_procedure_sorted[0]
            roof_procedure_sorted2[1] = abs(roof_procedure_sorted[2] + roof_procedure_sorted[1]) / 2
            roof_procedure_sorted2[2] = abs(roof_procedure_sorted[2] + roof_procedure_sorted[1]) / 2
            print('          New roof_procedure_sorted high =', roof_procedure_sorted2)
            # With extreme values instead ZZZ (uncomment if not needed)
            roof_procedure_sorted2[0] = roof_procedure_sorted[0]
            roof_procedure_sorted2[1] = corner_list_sorted[int(len(corner_list_sorted)-1)]
            roof_procedure_sorted2[2] = corner_list_sorted[int(len(corner_list_sorted)-1)]
            print('          With extreme values instead =', roof_procedure_sorted2)
        z1_wide, z2_wide, z3_wide = roof_procedure_sorted2[indexz1], roof_procedure_sorted2[indexz2], roof_procedure_sorted2[indexz3]
        roof_procedure = [z1_wide, z2_wide, z3_wide]
        print('          z1, z2, z3 regularized as = ', roof_procedure)
        # For outputting the Delaunay triangulation result as a raster
        M_delaunay[x1_wide][y1_wide] = 255
        M_delaunay[x2_wide][y2_wide] = 255
        M_delaunay[x3_wide][y3_wide] = 255

        # STEP 6: Compute the 3D roof plane coefficients out of these three Delaunay corners
        A = np.array([x1_wide * Resolution, y1_wide * Resolution, z1_wide])
        B = np.array([x2_wide * Resolution, y2_wide * Resolution, z2_wide])
        C = np.array([x3_wide * Resolution, y3_wide * Resolution, z3_wide])
        X1 = np.subtract(B, A)
        X2 = np.subtract(C, A)
        X = np.cross(X1, X2)
        d = -X[0] * A[0] - X[1] * A[1] - X[2] * A[2]
        # Result is 3D plane coeff and Delaunay corners (x,y,z) coordinates
        List_roofs_coeffs[k].append([X[0], X[1], X[2], d, x1_wide, y1_wide, z1_wide, x2_wide, y2_wide, z2_wide, x3_wide, y3_wide, z3_wide])
        print('          STEP 6/7: Computing the 3D coeff. of roof plane k =', k, ' from its Delaunay corners (', x1_wide, y1_wide, z1_wide, '), (', x2_wide, y2_wide, z2_wide, '), (', x3_wide, y3_wide, z3_wide, '), as = {', int(X[0]), int(X[1]), int(X[2]), int(d), '}')
        '''
        # Equation check
        print(x1_wide * Resolution * X[0] + y1_wide * Resolution * X[1] + z1_wide * X[2] + d) # Should be zero
        print(x2_wide * Resolution * X[0] + y2_wide * Resolution * X[1] + z2_wide * X[2] + d) # Should be zero
        print(x3_wide * Resolution * X[0] + y3_wide * Resolution * X[1] + z3_wide * X[2] + d) # Should be zero
        # Delaunay triangulation check as green corners on the original raster
        Raster_np[x1_wide][y1_wide][0] = 0
        Raster_np[x1_wide][y1_wide][1] = 255
        Raster_np[x1_wide][y1_wide][2] = 0
        Raster_np[x2_wide][y2_wide][0] = 0
        Raster_np[x2_wide][y2_wide][1] = 255
        Raster_np[x2_wide][y2_wide][2] = 0
        Raster_np[x3_wide][y3_wide][0] = 0
        Raster_np[x3_wide][y3_wide][1] = 255
        Raster_np[x3_wide][y3_wide][2] = 0
        '''

        # STEP 7: Now derive z for any point with known (x, y) contained in such a 3D plane
        print('          STEP 7/7: Computing from these 3D roof plane coeff. the height of all the points of roof plane k = ', k, '/', len(list_frags2), '...')
        for p in range(0, len(list_frags2[k]), 1):
            if (list_frags2[k][p][2] < 0.1) and (cond_impossible_coeff == 0):  # Work only 2D segmentations points whose height is yet unknown
                #print(list_frags[k][p])
                #print(X[0], X[1], X[2], d, x1_wide, y1_wide, z1_wide, x2_wide, y2_wide, z2_wide, x3_wide, y3_wide, z3_wide)
                list_frags2[k][p][2] = ceil(- (list_frags2[k][p][0] * Resolution * X[0]) / X[2] - (list_frags2[k][p][1] * Resolution * X[1]) / X[2] - d / X[2])
            if (cond_impossible_coeff == 1):
                list_frags2[k][p][2] = default_height # Default height for all roof points

    # Now outputing this List_roofs_coeffs as csv file...
    print('Outputting list_frag2.csv where each line k corresponds to the k roof plane points P(i, j, z, roof_id, dataset_id)')
    CSVFile2 = open('/home/jlussang/Desktop/Grid/detectron2_results/List_roofs_coeffs.csv', 'w')
    # Each line k of the .csv file below corresponds to the k list roof planes of 3D coefficients (A,B,C,D)
    with CSVFile2:
        FileOutput2 = csv.writer(CSVFile2)
        FileOutput2.writerows(List_roofs_coeffs)
    # Now outputing this list_frags2 as csv file...
    #print('Now outputing list_frag2 as .csv file, with ', roof_nr, ' roofs')
    CSVFile3 = open('/home/jlussang/Desktop/Grid/detectron2_results/list_frag2.csv', 'w')
    # Each line k of the .csv file below corresponds to the k=roof_id list of points (i,j,z)
    with CSVFile3:
        FileOutput3 = csv.writer(CSVFile3)
        FileOutput3.writerows(list_frags2)
    cv2.imwrite('/home/jlussang/Desktop/Grid/detectron2_results/M_delaunay.png', M_delaunay)

def fNormalizingResults(input, output):
    Raster = PIL.Image.open(input)
    Raster_rgb = Raster.convert("RGB")
    Raster_np = np.asarray(Raster_rgb)
    height, width = Raster.size
    #M = Raster_np.copy()
    M = np.zeros((width, height, 3), dtype=np.uint8)
    for i in range(0, width, 1):
        if (i % 50 == 0): print('Redening raster: i = ', i, '/', width)
        for j in range(0, height, 1):
            if (Raster_np[i][j][0] >= 1):
                M[i][j][2] = int(Raster_np[i][j][0]) + 200
                M[i][j][0] = int(Raster_np[i][j][2])
            if (Raster_np[i][j][2] >= 1) and (Raster_np[i][j][0] < 1):
                M[i][j][0] = int(Raster_np[i][j][2])
    cv2.imwrite(output, M)

def f2Daccuracy(inputGT, inputMOD):
    RasterGT = PIL.Image.open(inputGT)
    RasterGT_rgb = RasterGT.convert("RGB")
    RasterGT_np = np.asarray(RasterGT_rgb)
    height, width = RasterGT.size
    RasterMOD = PIL.Image.open(inputMOD)
    RasterMOD_rgb = RasterMOD.convert("RGB")
    RasterMOD_np = np.asarray(RasterMOD_rgb)
    # IoU over all datasets
    totalpixelnr_GT_total = 0
    totalpixelnr_IoU_total = 0
    totalpixelnr_GT_train = 0
    totalpixelnr_IoU_train = 0
    totalpixelnr_GT_val = 0
    totalpixelnr_IoU_val = 0
    totalpixelnr_GT_test = 0
    totalpixelnr_IoU_test = 0
    for i in range(0, width, 1):
        if (i % 100 == 0): print('Computing 2D accuracy statistic: i = ', i, '/', width)
        for j in range(0, height, 1):
            if (RasterGT_np[i][j][2] >= 1):
                totalpixelnr_GT_total += 1
                if (RasterMOD_np[i][j][2] >= 1):
                    totalpixelnr_IoU_total += 1
            if (RasterGT_np[i][j][2] == 220):
                totalpixelnr_GT_train += 1
                if (RasterMOD_np[i][j][2] == 220):
                    totalpixelnr_IoU_train += 1
            if (RasterGT_np[i][j][2] == 210):
                totalpixelnr_GT_val += 1
                if (RasterMOD_np[i][j][2] == 210):
                    totalpixelnr_IoU_val += 1
            if (RasterGT_np[i][j][2] == 220):
                totalpixelnr_GT_test += 1
                if (RasterMOD_np[i][j][2] == 220):
                    totalpixelnr_IoU_test += 1
    result_total = totalpixelnr_IoU_total*100.0/totalpixelnr_GT_total
    result_train = totalpixelnr_IoU_train*100.0/totalpixelnr_GT_train
    result_val = totalpixelnr_IoU_val*100.0/totalpixelnr_GT_val
    result_test = totalpixelnr_IoU_test*100.0/totalpixelnr_GT_test
    print('2D statistic accuracy: result_total = ', result_total, '%, result_train = ', result_train, '%, result_val = ', result_val, '%, result_test =', result_test)
    with open('/home/jlussang/Desktop/Grid/detectron2_results/f2Daccuracy.txt', 'w') as f:
        f.write('result_total = ')
        f.write(str(result_total))
        f.write(' %')
        f.write('\n')
        f.write('result_train = ')
        f.write(str(result_train))
        f.write(' %')
        f.write('\n')
        f.write('result_val = ')
        f.write(str(result_val))
        f.write(' %')
        f.write('\n')
        f.write('result_test = ')
        f.write(str(result_test))
        f.write(' %')
        f.write('\n')

# Returns a list of roof planes made of a list of roof points, each point being a string tuple (x,y,z,id,dataset)
def fRoofCornerLabeling_CSVreader(inputcsv):
    print('STEP 1: Building string list')
    with open(inputcsv) as CSVFile:
        roof_list_string = []
        FileInput = csv.reader(CSVFile, delimiter=',')
        for row in FileInput:
            roof_list_string.append(row) # (file converted to a list)
    roof_list = []
    for k in range(0, len(roof_list_string), 1):
        print('STEP 2: Building final list, with roof plane k = ', k)
        point_list = []
        roof_list.append(point_list)
        for p in range(0, len(roof_list_string[k]), 1):
            roof_list[k].append(roof_list_string[k][p].split("[")[1].split("]")[0].split(", "))
    return roof_list

def f3Daccuracy(inputcsv_GT, inputcsv_MOD, width, height):
    print('STEP 1: Building list_GT from fRoofCornerLabeling_CSVreader()')
    L_GT = fRoofCornerLabeling_CSVreader(inputcsv_GT)
    print('STEP 2: Building list_MOD from fRoofCornerLabeling_CSVreader()')
    L_MOD = fRoofCornerLabeling_CSVreader(inputcsv_MOD)
    M_GT = np.zeros((width, height, 2))
    M_MOD = np.zeros((width, height, 2))
    # We first fill two np arrays of size height x width (8580 x 9900)
    for k in range(0, len(L_GT), 1):
        if (k % 100 == 0): print('STEP 3: Building numpy array from list_GT, with roof plane k=', k)
        for p in range(0, len(L_GT[k]), 1):
            M_GT[int(L_GT[k][p][0])][int(L_GT[k][p][1])][0] = L_GT[k][p][2] # z value at M[i][j][0]
            M_GT[int(L_GT[k][p][0])][int(L_GT[k][p][1])][1] = L_GT[k][p][4] # dataset value at M[i][j][1]
    for k in range(0, len(L_MOD), 1):
        if (k % 100 == 0): print('STEP 4: Building numpy array from list_MOD, with roof plane k=', k)
        for p in range(0, len(L_MOD[k]), 1):
            M_MOD[int(L_MOD[k][p][0])][int(L_MOD[k][p][1])][0] = L_MOD[k][p][2] # z value at M[i][j][0]
            M_MOD[int(L_MOD[k][p][0])][int(L_MOD[k][p][1])][1] = L_MOD[k][p][4] # dataset value at M[i][j][1]
    # Now computing the 3D accuracy metric between the two
    z_delta_total = 0
    z_delta_total_nr = 0
    z_delta_train = 0
    z_delta_train_nr = 0
    z_delta_val = 0
    z_delta_val_nr = 0
    z_delta_test = 0
    z_delta_test_nr = 0
    for i in range(0, width, 1):
        if (i % 100 == 0): print('STEP 5: Computing 3D statistic: i = ', i, '/', width)
        for j in range(0, height, 1):
            if (M_GT[i][j][0] >= 0.1) and (M_MOD[i][j][0] >= 0.1): # Compute 3D accuracy only when there is an IoU overlap between GT and MOD
                z_delta_total += abs(M_MOD[i][j][0] - M_GT[i][j][0]) * 100.0 / M_GT[i][j][0]
                z_delta_total_nr += 1
                if (M_MOD[i][j][1] == 0):  # Compute train set accuracy
                    z_delta_train += abs(M_MOD[i][j][0] - M_GT[i][j][0]) * 100.0 / M_GT[i][j][0]
                    z_delta_train_nr += 1
                if (M_MOD[i][j][1] == 1):  # Compute val set accuracy
                    z_delta_val += abs(M_MOD[i][j][0] - M_GT[i][j][0]) * 100.0 / M_GT[i][j][0]
                    z_delta_val_nr += 1
                if (M_MOD[i][j][1] == 2):  # Compute test set accuracy
                    z_delta_test += abs(M_MOD[i][j][0] - M_GT[i][j][0]) * 100.0 / M_GT[i][j][0]
                    z_delta_test_nr += 1
    result_total = 100 - (z_delta_total / z_delta_total_nr)
    result_train = 100 - (z_delta_train / z_delta_train_nr)
    result_val = 100 - (z_delta_val / z_delta_val_nr)
    result_test = 100 - (z_delta_test / z_delta_test_nr)
    print('3D statistic accuracy: result_total = ', result_total, '%, result_train = ', result_train, '%, result_val = ', result_val, '%, result_test =', result_test)
    with open('/home/jlussang/Desktop/Grid/detectron2_results/f3Daccuracy.txt', 'w') as f:
        f.write('result_total = ')
        f.write(str(result_total))
        f.write(' %')
        f.write('\n')
        f.write('result_train = ')
        f.write(str(result_train))
        f.write(' %')
        f.write('\n')
        f.write('result_val = ')
        f.write(str(result_val))
        f.write(' %')
        f.write('\n')
        f.write('result_test = ')
        f.write(str(result_test))
        f.write(' %')
        f.write('\n')

# This takes a GT raster with red roof planes, brown roof lines, white background, and change it to bw
def fGT_filtering(input, output, brown_threshold):
    Raster = PIL.Image.open(input)
    Raster_rgb = Raster.convert("RGB")
    Raster_np = np.asarray(Raster_rgb)
    height, width = Raster.size
    Result = np.zeros((width, height, 3), dtype=np.uint8)
    for i in range(1, width, 1):
        if (i % 100 == 0): print('Filtering GT raster at i = ', i, '/', width)
        for j in range(1, height, 1):
            cond_white = (Raster_np[i][j][0] >= brown_threshold) and (Raster_np[i][j][1] >= brown_threshold) and (Raster_np[i][j][2] >= brown_threshold)
            cond_red = (Raster_np[i][j][0] > brown_threshold) and (Raster_np[i][j][1] <= 50) and (Raster_np[i][j][2] <= 50)
            cond_brown = (Raster_np[i][j][0] <= brown_threshold) and (Raster_np[i][j][1] <= brown_threshold) and (Raster_np[i][j][2] <= brown_threshold)
            if cond_white:
                continue
            if cond_red:
                #print('Red pixel: ', Raster_np[i][j])
                Result[i][j][0] = 0
                Result[i][j][1] = 0
                Result[i][j][2] = 210 # switch to blue recall 200, 210, 220 corresponds to train, val, test sets (of no utility here since GT)
                continue
            '''
            if cond_brown:
                #print('Brown pixel: ', Raster_np[i][j])
                Result[i][j][0] = 0 # blackens this pixel
                Result[i][j][1] = 0 # blackens this pixel
                Result[i][j][2] = 0 # blackens this pixel
                continue
            '''
    res = Image.fromarray(Result, 'RGB')
    res.save(output)


def fSquare_dissociate(input, output):
    Raster = PIL.Image.open(input)
    Raster_rgb = Raster.convert("RGB")
    Raster_np = np.asarray(Raster_rgb)
    height, width = Raster.size
    for i in range(1, width-1, 1):
        if (i % 200 == 0): print('Square dissociating at i = ', i, '/', width)
        for j in range(1, height-1, 1):
            cond = (Raster_np[i][j][0] >= 1) and (Raster_np[i][j][2] >= 1)
            cond_up = (Raster_np[i-1][j][0] >= 1) and (Raster_np[i-1][j][2] >= 1) and (Raster_np[i][j][0] != Raster_np[i-1][j][0])
            cond_down = (Raster_np[i + 1][j][0] >= 1) and (Raster_np[i + 1][j][2] >= 1) and (Raster_np[i][j][0] != Raster_np[i+1][j][0])
            cond_left = (Raster_np[i][j-1][0] >= 1) and (Raster_np[i][j-1][2] >= 1) and (Raster_np[i][j][0] != Raster_np[i][j-1][0])
            cond_right = (Raster_np[i][j+1][0] >= 1) and (Raster_np[i][j+1][2] >= 1) and (Raster_np[i][j][0] != Raster_np[i][j+1][0])
            cond_up_left = (Raster_np[i - 1][j-1][0] >= 1) and (Raster_np[i - 1][j-1][2] >= 1) and (Raster_np[i][j][0] != Raster_np[i-1][j-1][0])
            cond_up_right = (Raster_np[i - 1][j+1][0] >= 1) and (Raster_np[i - 1][j+1][2] >= 1) and (Raster_np[i][j][0] != Raster_np[i-1][j+1][0])
            cond_down_left = (Raster_np[i+1][j - 1][0] >= 1) and (Raster_np[i+1][j - 1][2] >= 1) and (Raster_np[i][j][0] != Raster_np[i+1][j-1][0])
            cond_down_right = (Raster_np[i+1][j + 1][0] >= 1) and (Raster_np[i+1][j + 1][2] >= 1) and (Raster_np[i][j][0] != Raster_np[i+1][j+1][0])
            if cond and (cond_up or cond_down or cond_left or cond_right or cond_up_left or cond_up_right or cond_down_left or cond_down_right):
                Raster_np[i][j][0] = 0
    res = Image.fromarray(Raster_np, 'RGB')
    res.save(output)

def fGrinding(input_dir, output_dir):
    for file in sorted(os.listdir(input_dir)):
        if file.endswith('.png'):
            filename = os.path.splitext(os.path.basename(file))[0]
            print(filename)
            finput = PIL.Image.open(input_dir + file)
            finput_rgb = finput.convert("RGB")
            finput_np = np.asarray(finput_rgb)
            height, width = finput.size
            Result = np.zeros((width, height, 3), dtype=np.uint8)
            for i in range(1, width - 1, 1):
                if (i % 100 == 0): print('Grinding GT roof plane at i = ', i, '/', width)
                for j in range(1, height - 1, 1):
                    cond0 = finput_np[i][j][0] > 200 # white pixel
                    cond1 = finput_np[i-1][j-1][0] > 200 # black pixel
                    cond2 = finput_np[i-1][j][0] > 200 # black pixel
                    cond3 = finput_np[i-1][j+1][0] > 200 # black pixel
                    cond4 = finput_np[i][j-1][0] > 200 # black pixel
                    cond5 = finput_np[i][j+1][0] > 200 # black pixel
                    cond6 = finput_np[i+1][j-1][0] > 200 # black pixel
                    cond7 = finput_np[i+1][j][0] > 200 # black pixel
                    cond8 = finput_np[i+1][j+1][0] > 200 # black pixel
                    if cond0 and cond1 and cond2 and cond3 and cond4 and cond5 and cond6 and cond7 and cond8:
                        Result[i][j][0] = finput_np[i][j][0]
                        Result[i][j][1] = finput_np[i][j][1]
                        Result[i][j][2] = finput_np[i][j][2]
            res = Image.fromarray(Result, 'RGB')
            res.save(output_dir + filename + '.png')

def fGrinding_green(input_dir, output_dir):
    for file in sorted(os.listdir(input_dir)):
        if file.endswith('.png'):
            filename = os.path.splitext(os.path.basename(file))[0]
            print(filename)
            finput = PIL.Image.open(input_dir + file)
            finput_rgb = finput.convert("RGB")
            finput_np = np.asarray(finput_rgb)
            height, width = finput.size
            Result = np.asarray(finput_rgb)
            for i in range(1, width - 1, 1):
                if (i % 100 == 0): print('Grinding GT roof plane at i = ', i, '/', width)
                for j in range(1, height - 1, 1):
                    cond0 = finput_np[i][j][0] > 200 # white pixel
                    cond1 = finput_np[i-1][j-1][0] < 200 # black pixel
                    cond2 = finput_np[i-1][j][0] < 200 # black pixel
                    cond3 = finput_np[i-1][j+1][0] < 200 # black pixel
                    cond4 = finput_np[i][j-1][0] < 200 # black pixel
                    cond5 = finput_np[i][j+1][0] < 200 # black pixel
                    cond6 = finput_np[i+1][j-1][0] < 200 # black pixel
                    cond7 = finput_np[i+1][j][0] < 200 # black pixel
                    cond8 = finput_np[i+1][j+1][0] < 200 # black pixel
                    if cond0 and (cond1 or cond2 or cond3 or cond4 or cond5 or cond6 or cond7 or cond8):
                        Result[i][j][0] = 0
                        Result[i][j][1] = 255
                        Result[i][j][2] = 0
            res = Image.fromarray(Result, 'RGB')
            res.save(output_dir + filename + '.png')

# Function that colors 2Dsegmentation bw results as {0,0,value} according to the tile being in the train, val, or test set
def f2Dsegmentation_color_new(input_dir, output_dir, value):
    for file in sorted(os.listdir(input_dir)):
        if file.endswith('.png'):
            filename = os.path.splitext(os.path.basename(file))[0]
            print(filename)
            finput = PIL.Image.open(input_dir + file)
            finput_rgb = finput.convert("RGB")
            finput_np = np.asarray(finput_rgb)
            height, width = finput.size
            #Result = np.zeros((width, height, 3), dtype=np.uint8)
            Result = np.asarray(finput_rgb)
            for i in range(0, width, 1):
                #if (i % 100 == 0): print('Dataset coloring GT at i = ', i, '/', width)
                for j in range(0, height, 1):
                    if finput_np[i][j][2] > 200:
                        Result[i][j][0] = 0
                        Result[i][j][1] = 0
                        Result[i][j][2] = value
                        #print('Result[i][j]=', Result[i][j])
            res = Image.fromarray(Result, 'RGB')
            res.save(output_dir + filename + '.png')


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Warning: First mkdir ShapefileInput, RasterOutput, Rasterized, MergedRasterOutput
#Command_mkdir = 'mkdir ' + RootPath
#os.system(Command_mkdir + 'ShapefileInput'), os.system(Command_mkdir + 'RasterOutput'), os.system(Command_mkdir + 'Rasterized'), os.system(Command_mkdir + 'MergedRasterOutput')
#RootPath = '/user/jlussang/home/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/0/'
#fShapefileInput_InputPath = RootPath + 'RESIDENTIAL_2A_WV03-2020-08-13-10-48-49.shp'
#fShapefileInput_OutputPath = RootPath + 'ShapefileInput/{}.shp'
#fRasterOutput_InputPath = RootPath + 'ShapefileInput/*.shp'
fRasterOutput_OutputPath = '/home/jlussang/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/dsm.tif'
#fTifToJpgConvert_InputPath = RootPath + 'RasterOutput/'
#fShapefileInput(fShapefileInput_InputPath, fShapefileInput_OutputPath)
#fRasterOutput(fRasterOutput_InputPath, fRasterOutput_OutputPath, fTifToJpgConvert_InputPath)
#fTifToJpgConvert(fTifToJpgConvert_InputPath)
#VectorCoordInput = '/user/jlussang/home/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/0/ShapefileInput/1.shp'
#fVectorCoordOutput(VectorCoordInput)
VectorCoordInputAll = '/user/jlussang/home/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/0/ShapefileInput/'
#fVectorCoordOutputAll(VectorCoordInputAll)
#fShapefileInput('/user/jlussang/home/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/0/MergedVector.shp', '/user/jlussang/home/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/0/ShapefileInput/{}.shp')
finput1 = '/home/jlussang/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/0/WV03-2020-08-13-10-48-49_RGB_PSH.tif'
finput2 = '/home/jlussang/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/0/WV03-2020-08-13-10-49-51_RGB_PSH.tif'
finput3 = '/home/jlussang/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/3/17MAY17110802_RGBIr_Short.tif'
#output1 = '/user/jlussang/home/Desktop/Capsule/RawData/BasicA.tif'
#output2 = '/user/jlussang/home/Desktop/Capsule/RawData/BasicB.tif'
xmin = 598419.6 # new 597844.5901
ymin = 5441166.4 # new 5440977.226
xmax = 601156.115 # old 601213.8
ymax = 5444294.639 # old 5444347.6


#fClipRaster(finput2, output2, xmin, ymin, xmax, ymax)
#fResizeRaster('/user/jlussang/home/Desktop/Capsule/RawData/BasicA.tif', '/user/jlussang/home/Desktop/Capsule/RawData/BasicAx.tif', 768-1, 768-1)
#fResizeRaster('/user/jlussang/home/Desktop/Capsule/RawData/BasicB.tif', '/user/jlussang/home/Desktop/Capsule/RawData/BasicBx.tif', 768-1, 768-1)
#fMetrics(finput3)
#fRasterize(fRasterOutput_OutputPath1, '/user/jlussang/home/Desktop/')
#fTrainingSetGenerator()
#fGridify('/user/jlussang/home/Desktop/000/Basicus.tif', '/user/jlussang/home/Desktop/000/x/', 300)
#fRasterize2('/user/jlussang/home/Desktop/Grid/Others/00000000000295.tif', '/user/jlussang/home/Desktop/')
#fTrainingSetGeneratorCOCO('/user/jlussang/home/Desktop/Grid/Full/', '00000000000306')
#26s per image for a full scale of 8687x9890 raster is 26*8687*9890/300/300/3600=6.9h computation time
#fPlaneCoeffStats('/user/jlussang/home/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/0/ShapefileInput/')
#fTrainingSetGeneratorCOCOAll('/user/jlussang/home/Desktop/Grid/Full/')
#fCleanCocoDataset('/user/jlussang/home/Desktop/Grid/Full_masks/')
#fRenameCocoDataset('/user/jlussang/home/Desktop/Grid/Full_masks/')
#fCocoAnnotationGen2('/user/jlussang/home/Desktop/Grid/Full_masks/')
#fCocoAnnotation_CATEGORIES('/user/jlussang/home/Desktop/LuxCarta/ProjectFirstdata/Mourmelon/0/ShapefileInput/')
#fRenameCocoDataset2('/user/jlussang/home/Desktop/Grid/Full_masks/')

#fGridify('/home/jlussang/Desktop/Grid/Basicus.tif', '/home/jlussang/Desktop/x/', 300)
#fUngridify('/home/jlussang/Desktop/res/', '/home/jlussang/Desktop/BasicusMasked.tif', 300)
#fClipRaster(fRasterOutput_OutputPath, '/home/jlussang/Desktop/Grid/detectron2_results/Dsm_clip.tif', xmin, ymin, xmax, ymax)
#fResizeRaster('/home/jlussang/Desktop/Grid/detectron2_results/Dsm_clip.tif', '/home/jlussang/Desktop/Grid/detectron2_results/Dsm_clip_resized.tif', 8687, 9890)
#fDsming(100.0, '/home/jlussang/Desktop/Grid/detectron2_results/Dsm_clip_resized.tif', '/home/jlussang/Desktop/Grid/detectron2_results/BasicusMasked_bw.tif', '/home/jlussang/Desktop/Grid/detectron2_results/BasicusResult_100bwfloat32.tif')
#fRoofLabeling('/home/jlussang/Desktop/Grid/detectron2_results/BasicusResult_100bwfloat32.tif', '/home/jlussang/Desktop/Grid/detectron2_results/BasicusResult_FloodFill.png')
#fcgal_write()
#fcgal_read('/home/jlussang/cgal/cgal_output2.csv')



# BIYOLO 230
# Make a set of 230x230 raster tiles
#fGridifyPadding_raster('/home/jlussang/Desktop/Grid/detectron2_results/finput2_clip_resized.tif', '/home/jlussang/Desktop/Grid/detectron2_results/Gridify_finput2_230/', 230, 10)
# Make 2Dsegmentation masks from these
#fTrainingSetGeneratorCOCOAll('/user/jlussang/home/Desktop/Grid/detectron2_results/Gridify_finput2_230/')
# Generate json files with pycococreator
#fCocoAnnotation_NameChange('/user/jlussang/home/Desktop/pycococreator2/examples/shapes/train/annotationsgeo_train/')
#fCocoAnnotation_NameChange('/user/jlussang/home/Desktop/pycococreator2/examples/shapes/train/annotationsgeo_val/')
#fCocoAnnotation_NameChange('/user/jlussang/home/Desktop/pycococreator2/examples/shapes/train/annotationsgeo_test/')
# Train Mask-RCNN on these for three days
# Generate the output of mask-RCNN on this 2Dsegmentation task, put it in folder 'results_2Dsegmentation_150', and then blend these as opaque masks with the rasters in folder 'Gridify_finput2_150'
# Black(230, 230) # for demo6.py
# Blend these masks as opaque B channel onto rasters (this will be the input, and the GT will the roof corners with their height)
#fBlender_all("/home/jlussang/Desktop/Grid/detectron2_results/Gridify_finput2_230_random/", "/home/jlussang/Desktop/Grid/detectron2_results/results_2Dsegmentation_230_random/")
# First gridify the rdtm_clip_resized in padded 230x230 tiles
#fGridifyPadding_dtm('/home/jlussang/Desktop/Grid/detectron2_results/dtm_clip_resized.tif', '/home/jlussang/Desktop/Grid/detectron2_results/Gridify_dtm_230/', 230, 10)
# Create the GT for these blended opaque inputs
#fTrainingSetGeneratorCOCOAll('/user/jlussang/home/Desktop/Grid/detectron2_results/Gridify_finput2_230_random/')
# Build json files with pycococreator and then vizualize these via the NEF

# Randomize the train, val, test reps
#fRandomizeCOCOdatasets('/user/jlussang/home/Desktop/Grid/detectron2_results/Gridify_finput2_230_random/', '/user/jlussang/home/Desktop/Grid/detectron2_results/Masks_biyolo_230_random/', 170, 1755)
# Change 'roof' class label for '0'
#fCocoAnnotation_NameChange('/user/jlussang/home/Desktop/Grid/detectron2_results/Masks_2Dsegmentation_230/')
#fCocoAnnotation_NameChange('/user/jlussang/home/Desktop/pycococreator2/examples/shapes/train/annotationsgeo_train/')
#fCocoAnnotation_NameChange('/user/jlussang/home/Desktop/pycococreator2/examples/shapes/train/annotationsgeo_val/')
#fCocoAnnotation_NameChange('/user/jlussang/home/Desktop/pycococreator2/examples/shapes/train/annotationsgeo_test/')

input_all = '/user/jlussang/home/Desktop/Grid/detectron2_results/Gridify_biyolo_230_random/'
matrix_train = '/user/jlussang/home/Desktop/pycococreator2/examples/shapes/train/BACKUP/2Dsegmentation_230_random/roofs_train2021_train/'
goal_train = '/user/jlussang/home/Desktop/pycococreator2/examples/shapes/train/roofs_train2021_train/'
matrix_val = '/user/jlussang/home/Desktop/pycococreator2/examples/shapes/train/BACKUP/2Dsegmentation_230_random/roofs_train2021_val/'
goal_val = '/user/jlussang/home/Desktop/pycococreator2/examples/shapes/train/roofs_train2021_val/'
matrix_test = '/user/jlussang/home/Desktop/pycococreator2/examples/shapes/train/BACKUP/2Dsegmentation_230_random/roofs_train2021_test/'
goal_test = '/user/jlussang/home/Desktop/pycococreator2/examples/shapes/train/roofs_train2021_test/'

# After the issue of not copying the right input files for pycococreator, we built fCopycat
# The goal is to take each file name in matrix_xxx and copy this file name from input_all to goal_xxx
#fCopycat(matrix_train, input_all, goal_train)
#fCopycat(matrix_val, input_all, goal_val)
#fCopycat(matrix_test, input_all, goal_test)

# We now color our 2Dsegmentation results as such: {0,0,200}, {0,0,210}, or {0,0,220} for the tile raster image being from the train, val, or test set, respectively
# The results will be placed in output_all
output_all = '/user/jlussang/home/Desktop/Grid/detectron2_results/results_2Dsegmentation_230_random_colored/'
input_all = '/user/jlussang/home/Desktop/Grid/detectron2_results/results_2Dsegmentation_230_random/'
#f2Dsegmentation_color(input_all, output_all, matrix_train, 200)
#f2Dsegmentation_color(input_all, output_all, matrix_val, 210)
#f2Dsegmentation_color(input_all, output_all, matrix_test, 220)

# We now blend the color_coded 2Dsegmentation results with the 3Dreconstruction results
input_2Dsegmentation = '/user/jlussang/home/Desktop/Grid/detectron2_results/results_2Dsegmentation_230_random_colored/'
input_3Dreconstruction = '/user/jlussang/home/Desktop/Grid/detectron2_results/results_biyolo_230_random/results_6d_all/'
output2D3D = '/user/jlussang/home/Desktop/Grid/detectron2_results/results_blend2D3D/'
output2D3D_red = '/user/jlussang/home/Desktop/Grid/detectron2_results/results_blend2D3D/results_blend2D3D_red/'
output2D3D_sets = '/user/jlussang/home/Desktop/Grid/detectron2_results/results_blend2D3D/results_blend2D3D_sets/'
#fBlend_2D3D(input_2Dsegmentation, input_3Dreconstruction, output2D3D_transparent)

# This function replaces the red square (i.e. roof corners) in the tiles by black if over black
# Or by pink ({200+z,0,200}) if over pixel {0,0,200} (train set), by cyan ({0,210+z,210}) if over pixel {0,0,210} (val set), by gray ({220+z,220+z,220}) if over pixel {0,0,220} (test set)
output_colorcoded = '/user/jlussang/home/Desktop/Grid/detectron2_results/results_blend2D3D_transparent/smallcorners_colorcoded/'
#fCorners_colorcode(output2D3D, output_colorcoded)

# We now paint the pixel borders of each tile so as not to have black lines on the fully reconstructed image at next step
input_borders = '/user/jlussang/home/Desktop/Grid/detectron2_results/results_blend2D3D_transparent/smallcorners_colorcoded/'
output_borders = '/user/jlussang/home/Desktop/Grid/detectron2_results/results_blend2D3D_borders/'
#fBorders_fill(output2D3D, output_borders)

#We now stick all these individual tile images to reconstruct the whole raster
output_ungridify = '/user/jlussang/home/Desktop/Grid/detectron2_results/finput2_2D3D_smallcorners_unvisible.png'
#fUngridify(output_borders, output_ungridify, 230, 10)

# We expand roof corners squares to extreme neighbor roof planes
input_roofcornerenlarge = '/user/jlussang/home/Desktop/Grid/detectron2_results/finput2_2D3D_unvisible.png'
output_roofcornerenlarge = '/user/jlussang/home/Desktop/Grid/detectron2_results/finput2_2D3D_unvisible_roofcornerenlarge.png'
#input_roofcornerenlarge_test = '/user/jlussang/home/Desktop/Grid/detectron2_results/test/0000000000016.png'
#fRoofCornerEnlarge(output_ungridify, output_roofcornerenlarge)

# We remove redundant pixels of roof corners
output_cornershrink = '/user/jlussang/home/Desktop/Grid/detectron2_results/finput2_2D3D_unvisible_smallcorners.png'
#fRoofCornerShrinking('/user/jlussang/home/Desktop/Grid/detectron2_results/finput2_2D3D_colorcoded_smallcorners.png', output_cornershrink) # test
#fRoofCornerShrinking(output_roofcornerenlarge, output_cornershrink)

# We remove diagonal pixels linking two roof planes
output_separator = '/user/jlussang/home/Desktop/Grid/detectron2_results/finput2_2D3D_unvisible_smallcorners_separator.png'
#fRoofSeparator(output_cornershrink, output_separator)

# We remove roof corner pixels that are over black background
output_blackening = '/user/jlussang/home/Desktop/Grid/detectron2_results/finput2_2D3D_unvisible_smallcorners_separator_blackening.png'
#fRoofBlackening(output_separator, output_blackening)

# We keep only roof corner pixels with most black pixel neighbors in 8-connectivity
output_extremeshrinking = '/user/jlussang/home/Desktop/Grid/detectron2_results/finput2_2D3D_unvisible_smallcorners_separator_blackening_extremeshrinking.png'
#fRoofCornerExtremeShrinking(output_blackening, output_extremeshrinking) # Check ZYX if red visual is activated or not

# Now our whole finput2_2D3D.png is made of 2D roof shapes as {0,0,200} for train,{0,0,210} for train, {0,0,220} for test,
# and 3D small corners {z,0,200} over train roofs, {0,z,210} over val roofs, {z,z,220} over test roofs
# Output list_frag.csv by fRoofCornerLabeling(), where each line is the roofs id, as {x,y,z,roof_id,dataset_id} (z=0 for all non-corners and dataset_id=0,1,2 for train,val,test resp.)
# Note: This latter function uses flood_fill() with tolerance of 70, see https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_floodfill.html
#fGridifyPadding_raster(output_cornershrink, '/home/jlussang/Desktop/Grid/detectron2_results/Gridify_test/', 768, 0) # To test fRoofCornerLabeling
#fTifToPngConvert('/home/jlussang/Desktop/Grid/detectron2_results/Gridify_test/')
#fRoofCornerLabeling('/user/jlussang/home/Desktop/Grid/detectron2_results/results_blend2D3D_borders/smallcorners_colorcoded/00000000000582.png') # test 1
#fRoofCornerLabeling('/user/jlussang/home/Desktop/Grid/detectron2_results/0000000000031.png') # test 2
#fRoofCornerLabeling(output_extremeshrinking) # Check on ZZZ for roof regularization procedure

# CGAL pipeline
#os.system('make /home/jlussang/cgal/mulin2')
#os.system('/home/jlussang/cgal/./mulin2') # Found in ProjectCrop5 directory





# Now compare with ground truth (GT)
# Re-generate 2D segmentation GT (because it was lost) and assign its blue color 200, 210, 220 for train, val, test sets, resp.
#fTrainingSetGeneratorCOCOAll('/user/jlussang/home/Desktop/Grid/detectron2_results/Gridify_finput2_230_random/')
# The goal will be to take each file name in matrix_xxx and copy this file name from input_all to goal_xxx
input_all = '/user/jlussang/home/Desktop/Grid/detectron2_results/Masks_2Dsegmentation_230_random/'
matrix_train = '/user/jlussang/home/Desktop/pycococreator2/examples/shapes/train/BACKUP/2Dsegmentation_230_random/roofs_train2021_train/'
goal_train = '/user/jlussang/home/Desktop/Grid/detectron2_results/Masks_2Dsegmentation_230_random/train/'
matrix_val = '/user/jlussang/home/Desktop/pycococreator2/examples/shapes/train/BACKUP/2Dsegmentation_230_random/roofs_train2021_val/'
goal_val = '/user/jlussang/home/Desktop/Grid/detectron2_results/Masks_2Dsegmentation_230_random/val/'
matrix_test = '/user/jlussang/home/Desktop/pycococreator2/examples/shapes/train/BACKUP/2Dsegmentation_230_random/roofs_train2021_test/'
goal_test = '/user/jlussang/home/Desktop/Grid/detectron2_results/Masks_2Dsegmentation_230_random/test/'

matrix2_train = '/user/jlussang/home/Desktop/Grid/detectron2_results/Masks_2Dsegmentation_230_random/train/'
matrix2_val = '/user/jlussang/home/Desktop/Grid/detectron2_results/Masks_2Dsegmentation_230_random/val/'
matrix2_test = '/user/jlussang/home/Desktop/Grid/detectron2_results/Masks_2Dsegmentation_230_random/test/'
# First we change the individual GT tiles of 2D segmentation roof planes by shrinking them of one pixel turned green on their borders so as to prevent overlaping
#fGrinding_green('/user/jlussang/home/Desktop/Grid/detectron2_results/Masks_2Dsegmentation_230_random/raw_ungrinded/train/', matrix2_train)
#fGrinding_green('/user/jlussang/home/Desktop/Grid/detectron2_results/Masks_2Dsegmentation_230_random/raw_ungrinded/val/', matrix2_val)
#fGrinding_green('/user/jlussang/home/Desktop/Grid/detectron2_results/Masks_2Dsegmentation_230_random/raw_ungrinded/test/', matrix2_test)

# We now color our 2Dsegmentation results as such: {0,0,200}, {0,0,210}, or {0,0,220} for the tile raster image being from the train, val, or test set, respectively
output_all = '/user/jlussang/home/Desktop/Grid/detectron2_results/Masks_2Dsegmentation_230_random_colored/'
#f2Dsegmentation_color_new('/user/jlussang/home/Desktop/Grid/detectron2_results/Masks_2Dsegmentation_230_random/train/', output_all, 200)
#f2Dsegmentation_color_new('/user/jlussang/home/Desktop/Grid/detectron2_results/Masks_2Dsegmentation_230_random/val/', output_all, 210)
#f2Dsegmentation_color_new('/user/jlussang/home/Desktop/Grid/detectron2_results/Masks_2Dsegmentation_230_random/test/', output_all, 220)

# After the issue of not copying the right input files for pycococreator, we built fCopycat
#fCopycat(matrix_train, input_all, goal_train)
#fCopycat(matrix_val, input_all, goal_val)
#fCopycat(matrix_test, input_all, goal_test)
# We now color our 2Dsegmentation results as such: {0,0,200}, {0,0,210}, or {0,0,220} for the tile raster image being from the train, val, or test set, respectively
input_all = '/user/jlussang/home/Desktop/Grid/detectron2_results/Masks_2Dsegmentation_230_random/'
#f2Dsegmentation_color(goal_train, output_all, matrix2_train, 200)
#f2Dsegmentation_color(goal_val, output_all, matrix2_val, 210)
#f2Dsegmentation_color(goal_test, output_all, matrix2_test, 220)

# Put two black pixels around each roof plane
output_shrink = '/user/jlussang/home/Desktop/Grid/detectron2_results/Masks_2Dsegmentation_230_random_colored_shrank/'
#fShrink_2D2D(output_all, output_shrink)

# Blend these 2D segmentation with the 3D reconstruction GT (as points of red heights z)
input_corners_train = '/user/jlussang/home/Desktop/pycococreator2/examples/shapes/train/annotationsgeo_train/'
input_corners_val = '/user/jlussang/home/Desktop/pycococreator2/examples/shapes/train/annotationsgeo_val/'
input_corners_test = '/user/jlussang/home/Desktop/pycococreator2/examples/shapes/train/annotationsgeo_test/'
input_all_3D = '/user/jlussang/home/Desktop/Grid/detectron2_results/Masks_biyolo_230_random/'
output_2Dblend = '/user/jlussang/home/Desktop/Grid/detectron2_results/Masks_2Dsegmentation_230_random_blend/'
output_3Dblend = '/user/jlussang/home/Desktop/Grid/detectron2_results/Masks_3Dreconstruction_230_random_blend/'
output_GT = '/user/jlussang/home/Desktop/Grid/detectron2_results/Masks_GT/'

# For each raster tile id, fetch all 2D segmentation results in output, and blend them to output_2Dblend
#fBlend_2D2D(matrix_train, output_all, 230, 230, output_2Dblend)
#fBlend_2D2D(matrix_val, output_all, 230, 230, output_2Dblend)
#fBlend_2D2D(matrix_test, output_all, 230, 230, output_2Dblend)

# For each raster tile id, fetch all 3D reconstruction masks in input_all_3D, and blend them to output_3Dblend (changing their letter-code for z as interger)
#fBlend_3D3D(matrix_train, input_all_3D, 230, 230, output_3Dblend)
#fBlend_3D3D(matrix_val, input_all_3D, 230, 230, output_3Dblend)
#fBlend_3D3D(matrix_test, input_all_3D, 230, 230, output_3Dblend)



# Now working on the GT generation
# For each raster tile id, fetch each of these 2D segmentation and 3D reconstruction results, and match and blend them to output_GT
#fBlend_2D3D_GT(output_2Dblend, output_3Dblend, output_GT)

# We now paint the pixel borders of each tile so as not to have black lines on the fully reconstructed image at next step
output_GTborders = '/user/jlussang/home/Desktop/Grid/detectron2_results/output_GT_borders/'
#fBorders_fill(output_GT, output_GTborders)

# Then ungridify all these individual tile images to reconstruct the whole raster
output_ungridify = '/user/jlussang/home/Desktop/Grid/detectron2_results/finput2_GT_new.png'
#fUngridify(output_GTborders, output_ungridify, 230, 10)

# Remove the green roof lines as black
output_greenblackening = '/user/jlussang/home/Desktop/Grid/detectron2_results/finput2_GT_new_greenless.png'
#fGreenBlackening(output_ungridify, output_greenblackening)

# We expand the roof corner as squares of 14x14 pixels
output_square = '/user/jlussang/home/Desktop/Grid/detectron2_results/finput2_GT_new_greenless_squared.png'
#fSquare_expand(output_greenblackening, output_square, 7)

output_linkdel = '/user/jlussang/home/Desktop/Grid/detectron2_results/finput2_GT_new_greenless_squared_linkdel.png'
#fLink_delete(output_square, output_linkdel)

# We remove diagonal pixels linking two roof planes
output_separator = '/user/jlussang/home/Desktop/Grid/detectron2_results/finput2_GT_new_greenless_squared_linkdel_separator.png'
#fRoofSeparator(output_linkdel, output_separator)

# We remove roof corner pixels that are over black background
output_blackening = '/user/jlussang/home/Desktop/Grid/detectron2_results/finput2_GT_new_greenless_squared_linkdel_separator_blackening.png'
#fRoofBlackening(output_separator, output_blackening)

# Prevent homogenizing of two different corners
output_squaredissociate = '/user/jlussang/home/Desktop/Grid/detectron2_results/finput2_GT_new_greenless_squared_linkdel_separator_blackening_squaredissociate.png'
#fSquare_dissociate(output_blackening, output_squaredissociate)

# We keep only roof corner pixels with most black pixel neighbors in 8-connectivity
output_extremeshrinking = '/user/jlussang/home/Desktop/Grid/detectron2_results/finput2_GT_new_greenless_squared_linkdel_separator_blackening_squaredissociate_shrinking.png'
#fRoofCornerExtremeShrinking(output_squaredissociate, output_extremeshrinking) # Check ZYX if red visual is activated or not



#fRoofCornerLabeling(output_extremeshrinking)
#os.system('cp /user/jlussang/home/Desktop/Grid/detectron2_results/list_frag2.csv /user/jlussang/home/Desktop/Grid/detectron2_results/list_fragGT.csv')
#fRoofCornerLabeling(output_modelredening)
#os.system('cp /user/jlussang/home/Desktop/Grid/detectron2_results/list_frag2.csv /user/jlussang/home/Desktop/Grid/detectron2_results/list_fragMOD.csv')

# CGAL pipeline
#os.system('make /home/jlussang/cgal/mulin2')
#os.system('/home/jlussang/cgal/./mulin2') # Found in ProjectCrop5 directory

# We now compute accuracy statistics
output_modelredening = '/user/jlussang/home/Desktop/Grid/detectron2_results/finput2_2D3D_unvisible_smallcorners_separator_blackening_extremeshrinking_red.png'
#f2Daccuracy(output_extremeshrinking, output_modelredening) # total = 88.8 %, result_train = 88.6 %, result_val = 87.0 %, result_test = 88.6 %
inputcsv_test = '/user/jlussang/home/Desktop/Grid/detectron2_results/test_list2.csv'
inputcsv_GT = '/user/jlussang/home/Desktop/Grid/detectron2_results/list_fragGT.csv'
inputcsv_MOD = '/user/jlussang/home/Desktop/Grid/detectron2_results/list_fragMOD.csv'
#L = fRoofCornerLabeling_CSVreader(inputcsv_test)
#f3Daccuracy(inputcsv_test, inputcsv_test, 9900, 8580)
#f3Daccuracy(inputcsv_GT, inputcsv_MOD, 9900, 8580) # total = 75.8 %, train = 76.9 %, val = 74.0 %, test = 74.8 %

# Finalize results for visualization
output_MOD = '/user/jlussang/home/Desktop/Grid/detectron2_results/result_MOD.png'
output_GT = '/user/jlussang/home/Desktop/Grid/detectron2_results/result_GT.png'
#fResultComparison(output_extremeshrinking, output_GT, 'green') # GT
#fResultComparison(output_modelredening, output_MOD, 0) # MOD


'''
# Our GT is not well reconstructed, so we redo everything out of the raw shapefiles directly
# We first use QGIS to generate the raster Raster_from_shapefileGT out of the raw shapefiles (cutting it at xmin, xmax, ymin, ymax above), and then we resize it
#fResizeRaster('/home/jlussang/Desktop/Grid/detectron2_results/Raster_from_shapefileGT.tiff', '/home/jlussang/Desktop/Grid/detectron2_results/Raster_from_shapefileGT_resized.tiff', 8580, 9900)
#fTifToPngConvert('/home/jlussang/Desktop/Grid/detectron2_results/temp/') # Put previous output there temporarily, and change temporarily fTifToPngConvert() to apply to tiff files
# We change its colors
output_filtering = '/home/jlussang/Desktop/Grid/detectron2_results/Raster_from_shapefileGT_resized_filtered.png'
#fGT_filtering('/home/jlussang/Desktop/Grid/detectron2_results/Raster_from_shapefileGT_resized.png', output_filtering, 200)

# We first gridify this result so as to overcome issues of padding reconstruction
#fGridifyPadding_raster(output_filtering, '/home/jlussang/Desktop/Grid/detectron2_results/temp2D/', 230, 10) # To test fRoofCornerLabeling # To test fRoofCornerLabeling() below

# We then ungridify these to recover a proper 2D segmentation GT at the whole scale level
output_ungridify = '/user/jlussang/home/Desktop/Grid/detectron2_results/Raster_from_shapefileGT_resized_filtered_rescaled.png'
#fUngridify('/home/jlussang/Desktop/Grid/detectron2_results/temp2D/', output_ungridify, 230, 10)

# We then blend the latter (placed in rep temp2Db) with the whole raster of 3D reconstruction masks (placed in temp3D) and output it to rep temp2D3DGT
#fBlend_2D3D_GT('/user/jlussang/home/Desktop/Grid/detectron2_results/temp2Db/', '/user/jlussang/home/Desktop/Grid/detectron2_results/temp3D/', '/user/jlussang/home/Desktop/Grid/detectron2_results/temp2D3DGT/')

# Normalizing results to red visible
#fNormalizingResults('/user/jlussang/home/Desktop/Grid/detectron2_results/0000000000017.png', output_modelredening)
#fNormalizingResults(output_extremeshrinking_model, output_modelredening)

# Generate overall 3D pipeline reconstruction
#fGridifyPadding_raster(output_extremeshrinking_GT, '/home/jlussang/Desktop/Grid/detectron2_results/test2GT/', 768, 0) # To test fRoofCornerLabeling # To test fRoofCornerLabeling() below
#fGridifyPadding_raster(output_extremeshrinking_model, '/home/jlussang/Desktop/Grid/detectron2_results/test2/', 768, 0) # To test fRoofCornerLabeling # To test fRoofCornerLabeling() below
#fRoofCornerLabeling('/user/jlussang/home/Desktop/Grid/detectron2_results/0000000000017GT.png') # test2
#fRoofCornerLabeling('/user/jlussang/home/Desktop/Grid/detectron2_results/0000000000017.png') # test2
'''

