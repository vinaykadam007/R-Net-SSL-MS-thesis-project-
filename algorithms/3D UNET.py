

from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout, Lambda

from keras.layers import Activation, MaxPool2D, Concatenate
from unittest.mock import patch

import tensorflow as tf
import keras
from skimage import io
from patchify import patchify, unpatchify
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.metrics import MeanIoU
from PIL import Image
import cv2
import random
import tifffile
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import glob
import tifffile as tiff
import os
import argparse
import sys, os
import getopt

import os,re, cv2, numpy as np, patchify, imgaug
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from skimage import color, io
from PIL import Image as im
import ipyplot
from patchify import patchify, unpatchify
                            
from tensorflow.keras.optimizers import Adam


os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'


#--------------------------------------------------------------Build UNET-------------------------------------------------------------------
kernel_initializer =  'he_uniform'
def unet(n_filters, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
    #Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv3D(n_filters, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(s)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv3D(n_filters, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)
    
    c2 = Conv3D(n_filters*2, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.1)(c2)
    c2 = Conv3D(n_filters*2, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)
     
    c3 = Conv3D(n_filters*4, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.1)(c3)
    c3 = Conv3D(n_filters*4, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)
     
    c4 = Conv3D(n_filters*8, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.1)(c4)
    c4 = Conv3D(n_filters*8, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)
     
    c5 = Conv3D(n_filters*16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.1)(c5)
    c5 = Conv3D(n_filters*16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)
    c5 = BatchNormalization()(c5)
    
    #Expansive path 
    u6 = Conv3DTranspose(n_filters*8, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv3D(n_filters*8, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(0.1)(c6)
    c6 = Conv3D(n_filters*8, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)
    c6 = BatchNormalization()(c6)
     
    u7 = Conv3DTranspose(n_filters*4, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(n_filters*4, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.1)(c7)
    c7 = Conv3D(n_filters*4, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)
    c7 = BatchNormalization()(c7)
     
    u8 = Conv3DTranspose(n_filters*2, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(n_filters*2, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv3D(n_filters*2, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)
    c8 = BatchNormalization()(c8)
     
    u9 = Conv3DTranspose(n_filters, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv3D(n_filters, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv3D(n_filters, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)
    c9 = BatchNormalization()(c9)
    
    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    #compile model outside of this function to make it flexible. 
    # model.summary()
    
    return model




def main(argv):
  
  #---------------------------------------------------------------Training UNET--------------------------------------------------------------------

  # my_model = build_unet((64,64,64,3), n_classes=4)
  IMG_HEIGHT = 128
  IMG_WIDTH = 128
  IMG_DEPTH = 128
  IMG_CHANNELS = 3

  step_size = 128
  
  
  try:
    opts, args = getopt.getopt(argv,"hd: r: g: c: e:",["rfile = ","gfile = ","cfile = ","efile = "])
  except getopt.GetoptError:
      print("Pythonfile.py -r <rawimages> -g <groundtruths> -c <classes> -e <epochs>(getopterror)")
      sys.exit(2)
      
      
  for opt, arg in opts:
      if opt == "-r":
          path = arg
      elif opt in ("-g", "--gfile"):
          gpath = arg
      elif opt in ("-c", "--cfile"):
          classes = arg
      elif opt in ("-e", "--efile"):
          epochs = arg
      else:
          print("Pythonfile.py -r <rawimages> -g <groundtruths> -c <classes> -e <epochs>")
          sys.exit()
        
        
  path = path #input("Please enter the path of your tif images (raw images)>>>> ")
  dimensions = IMG_HEIGHT#int(input("Please enter the dimensions of the new tif array (length and width is same number) >>> "))
  allImages = True#input("Do you want to get all images (if yes, type in True otherwise False) >>> ")
  if allImages == "False":
      imageCount = int(input(("How many images do you want from the stack?>>> ")))


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

  if allImages:
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
            #   print(fn, ": has been added")
          else:
              print("Discard:", fn, img.size, "<>", FIRST_SIZE)


  else:
    imageCount = int(imageCount)
    remove = int(len(dirlist) ) - imageCount
    remove = int(remove)
    boblist = dirlist[ int( (remove/2) )  :  int ( ((remove/2) + imageCount) )]
    for fn in boblist:
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
            # imgLabels.append("Horiz Flip")

            # Vertical Flip
            vflip = iaa.Flipud(p=1.0)
            input_vf = vflip.augment_image(input_img)
            # _ = color.rgb2gray(input_vf)
            FRAMES.append(np.array(input_vf, np.uint8))
            # imgLabels.append("Vertical Flip")

            # # Rotation
            rot1 = iaa.Affine(rotate=(-90, 90))
            input_rot1 = rot1.augment_image(input_img)
            # # _ = color.rgb2gray(input_rot1)
            FRAMES.append(np.array(input_rot1, np.uint8))
            # # imgLabels.append("Rotation")
            #
            # ## Gaussian
            transform = iaa.AdditiveGaussianNoise(scale=(0, 0.009125 * 255))
            transformedImg = transform.augment_image(image=input_img)
            FRAMES.append(np.array(transformedImg, np.uint8))
            # # imgLabels.append("Gaussian Noise")
            #
            # ## Laplace Noise
            #
            laplace = iaa.AdditiveLaplaceNoise(scale=(0, 0.0008 * 255))
            laplaceImg = laplace.augment_image(image=input_img)
            FRAMES.append(np.array(laplaceImg, np.uint8))
            # # imgLabels.append("Laplace Noise")
            #
            # ## Poisson Noise
            poisson = iaa.AdditivePoissonNoise((0, 3))
            poissonImg = poisson.augment_image(image=input_img)
            FRAMES.append(np.array(poissonImg, np.uint8))
            # # imgLabels.append("Poisson Noise")

            ## Image Shearing
            shear = iaa.Affine(shear=(-40, 40))
            input_shear = shear.augment_image(input_img)
            # _ = color.rgb2gray(input_shear)
            FRAMES.append(np.array(input_shear, np.uint8))
            # imgLabels.append("Shearing Image")
            # print(type(imgList))
            # print(fn, ": has been added")
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
  img_patches = patchify(img, (IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH), step=step_size)





  #-------------------------------------------------------------------------------------------------------------------------



  gpath = gpath#input("Please enter the path of your tif images (Ground truth)>>>> ")

  dimensions = IMG_HEIGHT#int(input("Please enter the dimensions of the new tif array (length and width is same number) >>> "))
  allImages = True#input("Do you want to get all images (if yes, type in True otherwise False) >>> ")
  if allImages == "False":
      imageCount = int(input(("How many images do you want from the stack?>>> ")))


  def PIL2array(img):
      img = cv2.resize(img, (dimensions, dimensions))
      img = np.array(img, dtype=np.uint8)
      return(img)



  def sorted_alphanumeric(data):
      convert = lambda text: int(text) if text.isdigit() else text.lower()
      alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
      return sorted(data, key=alphanum_key)


  dirlist = sorted_alphanumeric(os.listdir(gpath))
  FRAMES = []
  FIRST_SIZE = None

  if allImages:
      for fn in dirlist:
          img = cv2.imread((os.path.join(gpath, fn)))
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
            #   print(fn, ": has been added")
          else:
              print("Discard:", fn, img.size, "<>", FIRST_SIZE)


  else:
    imageCount = int(imageCount)
    remove = int(len(dirlist) ) - imageCount
    remove = int(remove)
    boblist = dirlist[ int( (remove/2) )  :  int ( ((remove/2) + imageCount) )]
    for fn in boblist:
        img = cv2.imread((os.path.join(gpath, fn)))
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
            # imgLabels.append("Horiz Flip")

            # Vertical Flip
            vflip = iaa.Flipud(p=1.0)
            input_vf = vflip.augment_image(input_img)
            # _ = color.rgb2gray(input_vf)
            FRAMES.append(np.array(input_vf, np.uint8))
            # imgLabels.append("Vertical Flip")

            # # Rotation
            rot1 = iaa.Affine(rotate=(-90, 90))
            input_rot1 = rot1.augment_image(input_img)
            # # _ = color.rgb2gray(input_rot1)
            FRAMES.append(np.array(input_rot1, np.uint8))
            # # imgLabels.append("Rotation")
            #
            # ## Gaussian
            transform = iaa.AdditiveGaussianNoise(scale=(0, 0.009125 * 255))
            transformedImg = transform.augment_image(image=input_img)
            FRAMES.append(np.array(transformedImg, np.uint8))
            # # imgLabels.append("Gaussian Noise")
            #
            # ## Laplace Noise
            #
            laplace = iaa.AdditiveLaplaceNoise(scale=(0, 0.0008 * 255))
            laplaceImg = laplace.augment_image(image=input_img)
            FRAMES.append(np.array(laplaceImg, np.uint8))
            # # imgLabels.append("Laplace Noise")
            #
            # ## Poisson Noise
            poisson = iaa.AdditivePoissonNoise((0, 3))
            poissonImg = poisson.augment_image(image=input_img)
            FRAMES.append(np.array(poissonImg, np.uint8))
            # # imgLabels.append("Poisson Noise")

            ## Image Shearing
            shear = iaa.Affine(shear=(-40, 40))
            input_shear = shear.augment_image(input_img)
            # _ = color.rgb2gray(input_shear)
            FRAMES.append(np.array(input_shear, np.uint8))
            # imgLabels.append("Shearing Image")
            # print(type(imgList))
            # print(fn, ": has been added")
        else:
            print("Discard:", fn, img.size, "<>", FIRST_SIZE)


 




  FRAMES = np.array(FRAMES)

  def rgb2grayPersonal(rgb):
      return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

  binary =True#input("Is this for binary segmentation? Enter True or False >>> ")

  if binary:
      gray = rgb2grayPersonal(FRAMES)
      FRAMES = gray



  temp = FRAMES
  mask = temp
  mask_patches = patchify(mask, (IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH), step=step_size)
  mask_patches[mask_patches > 0] = 1






  input_img = np.reshape(img_patches, (-1, img_patches.shape[3], img_patches.shape[4], img_patches.shape[5]))
  input_mask = np.reshape(mask_patches, (-1, mask_patches.shape[3], mask_patches.shape[4], mask_patches.shape[5]))


  # Convert grey image to 3 channels by copying channel 3 times.
  # We do this as our unet model expects 3 channel input. 
  from keras.utils.np_utils import normalize

  train_img = np.stack((input_img,)*3, axis=-1)
  # train_img = train_img / 255.
  train_img = normalize(train_img, axis=1)
  train_mask = np.expand_dims(input_mask, axis=4)
  train_mask_cat = to_categorical(train_mask, num_classes=2)
  print(np.unique(train_mask_cat))


  X_train, X_test, y_train, y_test = train_test_split(train_img, train_mask_cat, test_size = 0.25)


    
  def dice_coefficient(y_true, y_pred):
      smoothing_factor = 0.0001
      flat_y_true = K.flatten(y_true)
      flat_y_pred = K.flatten(y_pred)
      return (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)

  def dice_coefficient_loss(y_true, y_pred):
      return 1 - dice_coefficient(y_true, y_pred)







  model = unet(64, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, int(classes))
  print(model.input_shape)
  print(model.output_shape)



  # # model = get_model()
  model.compile(optimizer='Adam', loss=dice_coefficient_loss, metrics=['accuracy',dice_coefficient])
  model.summary()



  callbacks = [
      EarlyStopping(patience=10, verbose=1),
      ReduceLROnPlateau(monitor='val_loss',factor=0.1, patience=5, min_lr=0.00001, verbose=1),
      ModelCheckpoint('models/model-zebrafish.h5', verbose=1, save_best_only=True)
  ]



  history = model.fit(X_train, y_train, 
                      batch_size = 1, 
                      verbose=1, 
                      epochs=int(epochs), 
                      validation_data=(X_test, y_test), 
                      callbacks = callbacks)



  y_pred=model.predict(X_test)
  y_pred_argmax=np.argmax(y_pred, axis=4)
  y_test_argmax=np.argmax(y_test, axis=4)


  n_classes = int(classes)
  IOU_keras = MeanIoU(num_classes=n_classes)  
  IOU_keras.update_state(y_test_argmax, y_pred_argmax)
  print("Mean IoU =", IOU_keras.result().numpy())
  
  with open(r'D:\Vinay\UI\logs/outputlog.txt','w') as f:
    f.write('Loss = ' + str(history.history['val_loss'][-1]))
    f.write('\n')
    f.write("Mean IoU = " + str(IOU_keras.result().numpy()))






if __name__ == "__main__":
      
      main(sys.argv[1:])
      

