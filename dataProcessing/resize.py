import os, glob
from PIL import Image, ImageFilter
from scipy import ndimage, datasets
import numpy as np

dirList = glob.glob(r'D:/Aayan/pixelHop4D/PixelHopGPU-Test-run_multiprocessing/srcLoop/resultsResize/*/')

print(dirList)


for file in dirList:
    imgList = glob.glob(file + "*.png")
    for img in imgList:
        im = Image.open(img)
        im2 = Image.fromarray(np.uint8(ndimage.median_filter(im, size=2)))
        im2 = im2.filter(ImageFilter.UnsharpMask(radius= 3, percent=200,threshold=3))
        img = img.replace("png", "tif")
        img = img.replace("resultsResize","resultsMedian2")
        testingPath = os.path.dirname(img)
        exists = os.path.exists(testingPath)
        if not exists:
            os.makedirs(testingPath)
        im2.save(img)