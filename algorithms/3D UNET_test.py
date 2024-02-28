

from unittest.mock import patch
import tensorflow as tf
import keras
from skimage import io
from patchify import patchify, unpatchify
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
from PIL import Image
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.metrics import MeanIoU
from PIL import Image
import cv2
import random
import tifffile
from keras.utils.np_utils import normalize
from keras.models import load_model
from tifffile import imsave
import re
import os, sys
from skimage import color
import os,re, cv2, numpy as np, patchify, imgaug
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from skimage import color, io
from PIL import Image as im
import ipyplot
from patchify import patchify, unpatchify
import getopt

from keras.losses import binary_crossentropy
beta = 0.25
alpha = 0.25
gamma = 2
epsilon = 1e-5
smooth = 1


IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_DEPTH = 128
IMG_CHANNELS = 3
step_size = 128


def main(argv):

    try:
        opts, args = getopt.getopt(argv,"hd: r: m: s:",["rfile = ","mfile = ", "sfile = "])
    except getopt.GetoptError:
        print("Pythonfile.py -r <rawimages> -m <modelfile> -s <savefiles>(getopterror)")
        sys.exit(2)
        
        
    for opt, arg in opts:
        if opt == "-r":
            path = arg
        elif opt in ("-m", "--mfile"):
            mpath = arg
        elif opt in ("-s", "--sfile"):
            spath = arg
        else:
            print("Pythonfile.py -r <rawimages> -m <modelfile> -s <savefiles>")
            sys.exit()
        

    def dice_coefficient(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f*y_pred_f)
        return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


    def dice_coefficient_loss(y_true, y_pred):
        return 1 - dice_coefficient(y_true, y_pred)


    
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    IMG_DEPTH = 128
    IMG_CHANNELS = 3
    step_size = 128

        

        
    dependencies = {
        'dice_coefficient': dice_coefficient,
        'dice_coefficient_loss': dice_coefficient_loss,
        
    }


    my_model = load_model(mpath, custom_objects = dependencies)

        
    path = path

    print(path)
    dimensions = IMG_HEIGHT


    def PIL2array(img):
        img = cv2.resize(img, (dimensions, dimensions))
        img = np.array(img, dtype=np.uint8)
        return(img)



    def sorted_alphanumeric(data):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(data, key=alphanum_key)


    dirlist = sorted_alphanumeric(os.listdir(path))
    FRAMES = []
    FIRST_SIZE = None

    #   if allImages:
    for fn in dirlist:
        img = cv2.imread((os.path.join(path, fn)))
        if FIRST_SIZE is None:
            FIRST_SIZE = img.size
        if img.size == FIRST_SIZE:
            FRAMES.append(PIL2array(img))
            # #Horizontal Flip
            input_img = PIL2array(img)
            hflip = iaa.Fliplr(p=1.0)
            input_hf = hflip.augment_image(input_img)
            # _ = color.rgb2gray(input_hf)
            FRAMES.append(np.array(input_hf, np.uint8))
            #imgLabels.append("Horiz Flip")

            # Vertical Flip
            vflip = iaa.Flipud(p=1.0)
            input_vf = vflip.augment_image(input_img)
            # _ = color.rgb2gray(input_vf)
            FRAMES.append(np.array(input_vf, np.uint8))
            #imgLabels.append("Vertical Flip")

            # Rotation
            rot1 = iaa.Affine(rotate=(-90, 90))
            input_rot1 = rot1.augment_image(input_img)
            # _ = color.rgb2gray(input_rot1)
            FRAMES.append(np.array(input_rot1, np.uint8))
            #imgLabels.append("Rotation")

            ## Gaussian
            transform = iaa.AdditiveGaussianNoise(scale=(0, 0.009125 * 255))
            transformedImg = transform.augment_image(image=input_img)
            FRAMES.append(np.array(transformedImg, np.uint8))
            #imgLabels.append("Gaussian Noise")

            ## Laplace Noise

            laplace = iaa.AdditiveLaplaceNoise(scale=(0, 0.0008 * 255))
            laplaceImg = laplace.augment_image(image=input_img)
            FRAMES.append(np.array(laplaceImg, np.uint8))
            #imgLabels.append("Laplace Noise")

            ## Poisson Noise
            poisson = iaa.AdditivePoissonNoise((0, 3))
            poissonImg = poisson.augment_image(image=input_img)
            FRAMES.append(np.array(poissonImg, np.uint8))
            #imgLabels.append("Poisson Noise")

            ## Image Shearing
            shear = iaa.Affine(shear=(-40, 40))
            input_shear = shear.augment_image(input_img)
            # _ = color.rgb2gray(input_shear)
            FRAMES.append(np.array(input_shear, np.uint8))
            #imgLabels.append("Shearing Image")
            #print(type(imgList))
            print(fn, ": has been added")
        else:
            print("Discard:", fn, img.size, "<>", FIRST_SIZE)




    FRAMES = np.array(FRAMES)

    def rgb2grayPersonal(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    binary = True#input("Is this for binary segmentation? Enter True or False >>> ")

    if binary:
        gray = rgb2grayPersonal(FRAMES)
        FRAMES = gray



    temp = FRAMES
    img = temp

    print(img.shape)
    img_patches = patchify(img, (IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH), step=step_size)

    print(img_patches.shape)






    predicted_patches = []
    # cnt = 0
    for i in range(img_patches.shape[0]):
        for j in range(img_patches.shape[1]):
            for k in range(img_patches.shape[2]):
                single_patch = img_patches[i,j,k, :,:,:]
                single_patch_3ch = np.stack((single_patch,)*3, axis=-1) #make rgb
                single_patch_3ch = normalize(single_patch_3ch, axis=1)
                single_patch_3ch_input = np.expand_dims(single_patch_3ch, axis=0) #expand dimensions
                single_patch_prediction = my_model.predict(single_patch_3ch_input)
                single_patch_prediction_argmax = np.argmax(single_patch_prediction, axis=4)[0,:,:,:]
                predicted_patches.append(single_patch_prediction_argmax)





    #Convert list to numpy array
    predicted_patches = np.array(predicted_patches)



    #Reshape to the shape we had after patchifying
    predicted_patches_reshaped = np.reshape(predicted_patches, 
                                            (img_patches.shape[0], img_patches.shape[1], img_patches.shape[2],
                                             img_patches.shape[3], img_patches.shape[4], img_patches.shape[5]) )





    # Repach individual patches into the orginal volume shape
    reconstructed_image = unpatchify(predicted_patches_reshaped, img.shape)


    #Convert to uint8 so we can open image in most image viewing software packages
    reconstructed_image=reconstructed_image.astype(np.uint8)

    print('now saving')
    
    #Now save it as segmented volume.

    imsave(spath + "/segmented.tif", reconstructed_image)

    print('done')

if __name__ == "__main__":
      
      main(sys.argv[1:])
      

            
            
            
        
        