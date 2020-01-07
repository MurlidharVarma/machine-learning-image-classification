import glob
import os
from os import listdir, walk

import cv2

HOME_PATH = "./"
MASTER_PATH = "master"
DEST_PATH = "data/{}"
resize_width = 800

# function to resize an image
def resizeImage(image, width):
    ratio = resize_width * 1.0 / image.shape[1]
    dim = (resize_width, int(image.shape[0]*ratio))
    resized = cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
    return resized

def rotateImage(image, degree):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, degree, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return resizeImage(rotated, resize_width)

# create all rotational type images
def createRotationTypeImages(label, sourceFilePath, destFolderPath, fileIdx):
    image = cv2.imread(sourceFilePath)
    # creating directory
    # os.mkdir(destFolderPath)
    # file name appended with directory
    validatePercent=10
    # Loop through each degree and then write image
    for deg in range(0,360,5):

        subFolder=""
        if (deg%validatePercent == 0):
            subFolder=TEST_PATH
        else:
            subFolder=TRAIN_PATH

        IMAGE_DEST_FOLDER = subFolder+"/"+ label

        if not os.path.exists(IMAGE_DEST_FOLDER):
            os.mkdir(IMAGE_DEST_FOLDER)

        destFilePathTemplate = IMAGE_DEST_FOLDER +"/"+ label +"_{}_{}.png"
        cv2.imwrite(destFilePathTemplate.format(fileIdx,deg),rotateImage(image,deg))
        # print('Created image file: '+destFilePathTemplate.format(fileIdx,deg))


print(os.listdir(MASTER_PATH))

TRAIN_PATH= HOME_PATH + DEST_PATH.format("train")
TEST_PATH = HOME_PATH + DEST_PATH.format("test")

if not os.path.exists(TRAIN_PATH):
    print('TRAIN_PATH: '+TRAIN_PATH)
    os.mkdir(TRAIN_PATH)

if not os.path.exists(TEST_PATH):
    print('TEST_PATH: '+TEST_PATH)
    os.mkdir(TEST_PATH)

totalPath=HOME_PATH+MASTER_PATH

fileCount=0
for (dirpath, dirnames, filenames) in walk(totalPath):
    for filename in filenames:
        fileCount = fileCount + 1
        sourceFilePath = dirpath+"\\"+filename
        destFolderPath = dirpath.replace(MASTER_PATH,DEST_PATH)

        # extracting the folder name from master as folder name = app name
        folderName = (dirpath.split("\\"))[-1]

        print(dirpath+"\\"+filename)
        fn = str(filename)

        if (fn.find(".JPG") > -1 or fn.find(".jpg") > -1 or fn.find(".png") > -1 or fn.find(".PNG") > -1):
            createRotationTypeImages(folderName, sourceFilePath,destFolderPath, fileCount)
