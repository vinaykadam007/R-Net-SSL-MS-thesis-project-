
import subprocess
import os
import sys
import tifftools
import vtk
import numpy
import cv2
import re
import argparse
import pyvista as pv
from os import system, name
import getopt


def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hd: s: ",["sfile = "])
    except getopt.GetoptError:
        print("Pythonfile.py -s <segmentedimages>")
        sys.exit(2)
    
      
    for opt, arg in opts:
        if opt == "-s":
            path = arg
     
        else:
            print("Pythonfile.py -s <segmentedimages>")
            sys.exit()
        
    IMG_HEIGHT = 64

    path = path

    dimensions = IMG_HEIGHT 

    finalDirectory = resource_path('')
    finalName = "unity"
    extension = "vti"
    finalDirectory = finalDirectory + "/Export files/vti/" + finalName

   

    def sorted_alphanumeric(data):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(data, key=alphanum_key)

    dirlist = sorted_alphanumeric(os.listdir(path))

    def PIL2array(img):
        img = cv2.resize(img, (dimensions, dimensions))
  
        
        return numpy.array(img, numpy.uint8)

    FRAMES = []
    FIRST_SIZE = None

    for fn in dirlist:
        img = cv2.imread((os.path.join(path, fn)))
        if FIRST_SIZE is None:
            FIRST_SIZE = img.size
        if img.size == FIRST_SIZE:
            FRAMES.append(PIL2array(img))
        else:
            print("Discard:", fn, img.size, "<>", FIRST_SIZE)



    binary = True 

    if bool == "True":
        FRAMES[FRAMES > 0] = 1


    Stack = numpy.dstack(FRAMES)
    data = pv.wrap(Stack)

    j = 512 
    i = 12


    finDir = finalDirectory
    data.spacing = (j, j, i )
    finDir += ( "." + extension )
    data.save(finDir, binary = True)
    print("Your file has been saved to the following location {}".format(finDir))
    
if __name__ == "__main__":
      main(sys.argv[1:])