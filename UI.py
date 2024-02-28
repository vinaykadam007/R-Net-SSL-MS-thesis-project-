import os
import multiprocessing
from multiprocessing import Process, freeze_support
multiprocessing.freeze_support()
import glob, random
import subprocess
import PyQt5
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtPrintSupport import *
import os, platform
import subprocess
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QGroupBox, QDialog, QVBoxLayout, QLineEdit, QMessageBox, QGridLayout, QInputDialog
from PyQt5.QtGui import QIcon
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from PyQt5 import QtGui, QtWidgets, QtCore
import shutil
import webbrowser 
import re
import IPython
import nbformat.v4 as convert
import json
import time


def replace_line(file_name, originalText, newText):
    print(file_name)
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if originalText in line:
                lines[i] = newText
                break
    with open (file_name, "w") as f:
        f.writelines(lines)
    

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


print(resource_path(''))
random.seed(10)

pyPath = ""

for key,value in os.environ.items():
    if (key == "PATH"):
        for everypath in value.split(";"):
            if 'Python311' in everypath.split('\\'):
                if "Scripts" not in everypath.split('\\'):
                    print(everypath)
                    pyPath = everypath
                    print(pyPath)

# class ProgressBar(QProgressBar):
    
#     def __init__(self, *args, **kwargs):
#         super(ProgressBar, self).__init__(*args, **kwargs)
#         self.setValue(0)
#         if self.minimum() != self.maximum():
#             self.timer = QTimer(self, timeout=self.onTimeout)
#             self.timer.start(random.randint(1, 3) * 1000)

#     def onTimeout(self):
#         if self.value() >= 100:
#             self.timer.stop()
#             self.timer.deleteLater()
#             del self.timer
#             return
#         self.setValue(self.value() + 1)
        
        
def automatic_brightness_and_contrast(image, clip_hist_percent=5):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)
   
   
class Thread(QThread):
    _signal = pyqtSignal(int)
    def __init__(self):
        super(Thread, self).__init__()

    def __del__(self):
        self.wait()

    def run(self):
        with open(resource_path('logs/log.txt'),'r') as f:
            content = f.read()
            print(content.split(' ')[-1])
        noofepochs = content.split(' ')[-1]
        if int(noofepochs) <= 100:
            print('under 100')
            print((int(noofepochs)*100)-((int(noofepochs)-1)*100))
            if int(noofepochs) <= 10: 
                for i in range((int(noofepochs)*100)-((int(noofepochs)-1)*100)):
                    print(i)
                    time.sleep(2)
                    self._signal.emit(i) 
                # pbar.setValue(int(noofepochs.text())*100/100)
            elif int(noofepochs) > 10:
                for i in range((int(noofepochs)*100)//int(noofepochs)):
                    time.sleep(2)
                    self._signal.emit(i) 



PixelHopNonMulti = """
import multiprocessing
from multiprocessing import Process, freeze_support
multiprocessing.freeze_support()
import numpy as np 
from numba import cuda
import os, glob, sys
from framework.layer import *
from framework.pixelhop import *
from utils import loadImage
import imageio
from division import div
import timeit
import xgboost as xgb
from sklearn import preprocessing
import pickle
import getopt
from sklearn.feature_selection import f_classif, SelectPercentile
from datetime import timedelta
from skimage import io



SAVE={}
img_size = 1024
patch_size = 64#16
delta_x = patch_size # non overlap






def run(argv):
    try:
        opts, args = getopt.getopt(argv, "hd: r: g: c: e: v: n:",["rfile = ","gfile = ","cfile = ","efile = ", "vfile = ", "nfile = "])
    except getopt.GetoptError:
        print("Pythonfile.py -r <rawimages> -g <groundtruths> -c <classes> -e <epochs>(getopterror)")
        sys.exit(2)
        
        
    for opt, arg in opts:
        if opt == "-r":
            print(arg)
            path = arg
        elif opt in ("-g", "--gfile"):
            gpath = arg
        elif opt in ("-c", "--cfile"):
            classes = arg
        elif opt in ("-e", "--efile"):
            epochs = arg
        elif opt in ("-v", "--vfile"):
            variance = float(arg)
        elif opt in ("-n", "--nfile"):
            num_training_imgs = int(arg)
            
        else:
            print("Pythonfile.py -r <rawimages> -g <groundtruths> -c <classes> -e <epochs>")
            sys.exit()






    #num_training_imgs = 1
    train_img_path = (path + '/*.png')
    test_img_path =  (gpath + '/*.png')

    train_img_addrs = glob.glob(train_img_path)
    test_img_addrs = glob.glob(test_img_path)

    print(train_img_path)
    print(train_img_addrs)
    start_train = timeit.default_timer()

    print("----------------------TRAINING-------------------------")
    # Initialize
    mask_patch_list, img_patch_list = [], []

    # Control the num of training images
    count = 0
    for train_img_addr in train_img_addrs:
        if 'raw' in train_img_addr: # only want to load training images that are raw
            count += 1
            if count > num_training_imgs: break
            print('Adding {} for training................'.format(train_img_addr))

            # Add raw images
            img = loadImage(train_img_addr) # Load image
            #img = io.imread(img, as_gray = True)
            print(train_img_addr)

            # Add mask images
            mask = loadImage(train_img_addr.replace('raw', 'seg'))
            print(mask.shape)

            if len(img.shape) != 3:
                img = np.expand_dims(img, axis=2)
                print(img.shape)

            # Create patches for training images
            for i in range(0, img_size, delta_x):
                for j in range(0, img_size, delta_x):
                    if i+patch_size <= img_size and j+patch_size <= img_size:
                        img_patch = img[i:i+patch_size, j:j+patch_size, :]
                        mask_patch = mask[i:i+patch_size, j:j+patch_size]

                    elif i+patch_size > img_size and j+patch_size <= img_size:
                        temp_size = img[i:, j:j+patch_size, :].shape
                        img_patch = np.lib.pad(img[i:, j:j+patch_size, :], ((0,patch_size-temp_size[0]),(0,0), (0,0)), 'edge')
                        mask_patch = np.lib.pad(mask[i:, j:j+patch_size], ((0,patch_size-temp_size[0]),(0,0)), 'edge')

                    elif i+patch_size <= img_size and j+patch_size > img_size:
                        temp_size = img[i:i+patch_size, j:, :].shape
                        img_patch = np.lib.pad(img[i:i+patch_size, j:, :], ((0,0),(0,patch_size-temp_size[1]), (0,0)), 'edge')
                        mask_patch = np.lib.pad(mask[i:i+patch_size, j:], ((0,0),(0,patch_size-temp_size[1])), 'edge')

                    else: 
                        temp_size = img[i:, j:, :].shape
                        img_patch = np.lib.pad(img[i:, j:, :], ((0,patch_size-temp_size[0]),(0,patch_size-temp_size[1]), (0,0)), 'edge')
                        mask_patch = np.lib.pad(mask[i:, j:], ((0,patch_size-temp_size[0]),(0,patch_size-temp_size[1])), 'edge')

                    assert (img_patch.shape[0], img_patch.shape[1]) == (patch_size,patch_size)

                    # Save each patch to list
                    img_patch_list.append(img_patch)
                    mask_patch_list.append(mask_patch)
                    
                    
    # Convert list to numpy array
    img_patches = np.asarray(img_patch_list)
    #img_patches[img_patches > 0] = 1
    mask_patches = np.array(mask_patch_list)
    #mask_patches[mask_patches > 0] =1
    print(img_patches.shape)
    print(mask_patches.shape)

    print('--------------------------------------')
    # Number of classes
    print('Number of classes: {}'.format(np.unique(mask_patches)))
    

    ################################################## PIXELHOP UNIT 1 ####################################################
        
    train_feature1=PixelHop_Unit_GPU(img_patches, dilate=1, pad='reflect', weight_name='pixelhop1.pkl', getK=1, energypercent=variance)
    
    ################################################ PIXELHOP UNIT 2 ####################################################

    train_featurem1 = MaxPooling(train_feature1)
    train_feature2=PixelHop_Unit_GPU(train_featurem1, dilate=1, pad='reflect',  weight_name='pixelhop2.pkl', getK=1, energypercent=variance)
   
    
    print(train_feature1.shape)
    print(train_feature2.shape)
    
    # Upsample the pixelhop feature 
    train_feature_reduce_unit1 = train_feature1 
    train_feature_reduce_unit2 = myResize(train_feature2, img_patches.shape[1], img_patches.shape[2])
    print(train_feature_reduce_unit1.shape)
    print(train_feature_reduce_unit2.shape)

    # Reshape the pixelhop feature
    train_feature_unit1 = train_feature_reduce_unit1.reshape(train_feature_reduce_unit1.shape[0]*train_feature_reduce_unit1.shape[1]*train_feature_reduce_unit1.shape[2], -1)
    train_feature_unit2 = train_feature_reduce_unit2.reshape(train_feature_reduce_unit2.shape[0]*train_feature_reduce_unit2.shape[1]*train_feature_reduce_unit2.shape[2], -1)
   
    del train_feature_reduce_unit1, train_feature_reduce_unit2
    
    print(train_feature_unit1.shape)
    print(train_feature_unit2.shape)
    
    ### NEW CODE ###
    count = num_training_imgs
    patch_ind = count * (img_size//delta_x) * (img_size//delta_x)
    np_img_patch_list = np.array(img_patch_list)

    ## get parameters for threading
    depth1 = np_img_patch_list.shape[3] + 2 + train_feature_unit1.shape[1]
    depth2 = np_img_patch_list.shape[3] + 2 + train_feature_unit2.shape[1]
    depth = max(depth1, depth2)
    feature_shape1 = (count, img_size//delta_x, img_size//delta_x, patch_size, patch_size, depth1)
    feature_shape2 = (count, (img_size//delta_x), (img_size//delta_x), patch_size, patch_size, depth2)

    ## allocate and transfer data to device
    d_img_patch_list = cuda.to_device(np.ascontiguousarray(img_patch_list))
    d_train_feature_unit1 = cuda.to_device(np.ascontiguousarray(train_feature_unit1))
    d_train_feature_unit2 = cuda.to_device(np.ascontiguousarray(train_feature_unit2))
    d_feature_list_unit1 = cuda.device_array(feature_shape1)
    d_feature_list_unit2 = cuda.device_array(feature_shape2)

    ## setup thread dimensions
    threadDimensions = np.ascontiguousarray([count, img_size//delta_x, img_size//delta_x, patch_size, patch_size, depth])
    d_threadDimensions = cuda.to_device(threadDimensions)
    totalThreads = threadDimensions.prod()
    threadsPerBlock = 64
    blocksPerGrid = math.ceil(totalThreads/threadsPerBlock)
    
    ## run device kernel
    GPU_Feature[blocksPerGrid, threadsPerBlock](d_img_patch_list, d_train_feature_unit1, d_train_feature_unit2, d_feature_list_unit1, d_feature_list_unit2, d_threadDimensions, delta_x, patch_size, depth1, depth2)

    ## transfer results back
    feature_list_unit1 = d_feature_list_unit1.copy_to_host().reshape(count * (img_size//delta_x) * (img_size//delta_x) * patch_size * patch_size, -1)
    feature_list_unit2 = d_feature_list_unit2.copy_to_host().reshape(count * (img_size//delta_x) * (img_size//delta_x) * patch_size * patch_size, -1)

    ## get gt_list from mask_patch_list directly
    gt_list = np.array(mask_patch_list).flatten()
    ### END NEW CODE ###

    print(feature_list_unit1.shape)
    print(feature_list_unit2.shape)
    
    print(gt_list.shape)
    
    # F-test to get top 80% features
    fs1 = SelectPercentile(score_func=f_classif, percentile=80)
    fs1.fit(feature_list_unit1, gt_list)
    new_features1 = fs1.transform(feature_list_unit1)
    print(new_features1.shape)
    print(fs1.scores_)
    
    fs2 = SelectPercentile(score_func=f_classif, percentile=80)
    fs2.fit(feature_list_unit2, gt_list)
    new_features2 = fs2.transform(feature_list_unit2)
    print(new_features2.shape)
    print(fs2.scores_)
    
    # Concatenate all the features together
    concat_features  = np.concatenate((new_features1, new_features2), axis=1)
    print(concat_features.shape)
    
    del feature_list_unit1, new_features1, feature_list_unit2, new_features2
    
    # Preprocessing (standardize features by removing the mean and scaling to unit variance)
    scaler1=preprocessing.StandardScaler().fit(concat_features)
    feature = scaler1.transform(concat_features) 
    print(feature.shape)

    # Define and train XGBoost algorithm
    xgb_model = xgb.XGBClassifier(tree_method='gpu_hist',objective="binary:logistic", verbosity=3)
    #le = LabelEncoder()
    
    #gt_list_copy = le.fit_transform(gt_list)
    
    clf = xgb_model.fit(feature, gt_list)

    pathToResultSave = os.path.abspath(epochs) + (chr(92))
    print(pathToResultSave)
    # Save all the model files
    pickle.dump(clf, open("{}classifier.sav".format(pathToResultSave),'wb')) 
    pickle.dump(scaler1,open("{}scaler1.sav".format(pathToResultSave),'wb') ) 
    pickle.dump(fs1, open("{}fs1.sav".format(pathToResultSave),'wb')) 
    pickle.dump(fs2, open("{}fs2.sav".format(pathToResultSave),'wb')) 
    print('All models saved!!!')

    stop_train = timeit.default_timer()

    print('Total Time: ' + str(timedelta(seconds=stop_train-start_train)))
    f = open('{}train_time.txt'.format(pathToResultSave),'w+')
    f.write('Total Time: ' + str(timedelta(seconds=stop_train-start_train)))
    f.close()

    patch_ind = 0

    start_test = timeit.default_timer()
    print("----------------------TESTING-------------------------")

    test_img_patch, test_img_patch_list = [], []
    count = 0
    for test_img_addr in test_img_addrs:
        if 'raw' in test_img_addr: # only want to load testing images that are raw
            count += 1
            print('Processing {}............'.format(test_img_addr))

            img = loadImage(test_img_addr) 

            if len(img.shape) != 3:
                img = np.expand_dims(img, axis=2)

            # Initialzing
            predict_0or1 = np.zeros((img_size, img_size, 2))

            predict_mask = np.zeros(img.shape)

            # Create patches for test image
            for i in range(0, img_size, delta_x):
                for j in range(0, img_size, delta_x):
                    if i+patch_size <= img_size and j+patch_size <= img_size:
                        test_img_patch = img[i:i+patch_size, j:j+patch_size, :]

                    elif i+patch_size > img_size and j+patch_size <= img_size:
                        temp_size = img[i:, j:j+patch_size, :].shape
                        test_img_patch = np.lib.pad(img[i:, j:j+patch_size, :], ((0,patch_size-temp_size[0]),(0,0), (0,0)), 'edge')

                    elif i+patch_size <= img_size and j+patch_size > img_size:
                        temp_size = img[i:i+patch_size, j:, :].shape
                        test_img_patch = np.lib.pad(img[i:i+patch_size, j:, :], ((0,0),(0,patch_size-temp_size[1]), (0,0)), 'edge')

                    else: 
                        temp_size = img[i:, j:, :].shape
                        test_img_patch = np.lib.pad(img[i:, j:, :], ((0,patch_size-temp_size[0]),(0,patch_size-temp_size[1]), (0,0)), 'edge')
                    assert (test_img_patch.shape[0], test_img_patch.shape[1]) == (patch_size,patch_size)
                    
                    test_img_patch_list.append(test_img_patch)
                    

                    # convert list to numpy
                    test_img_subpatches = np.asarray(test_img_patch_list)
                    print(test_img_subpatches.shape)

                    
                    ################################################## PIXELHOP UNIT 1 ####################################################
        
                    test_feature1=PixelHop_Unit_GPU(test_img_subpatches, dilate=1, pad='reflect', weight_name='pixelhop1.pkl', getK=0, energypercent=variance)

                    ################################################# PIXELHOP UNIT 2 ####################################################
                    test_featurem1 = MaxPooling(test_feature1)
                    test_feature2=PixelHop_Unit_GPU(test_featurem1, dilate=1, pad='reflect', weight_name='pixelhop2.pkl', getK=0, energypercent=variance)
                    
    
                    test_feature_reduce_unit1 = test_feature1 
                    test_feature_reduce_unit2 = myResize(test_feature2, test_img_subpatches.shape[1], test_img_subpatches.shape[2])
                    print(test_feature_reduce_unit1.shape)
                    print(test_feature_reduce_unit2.shape)

                    
                    test_feature_unit1 = test_feature_reduce_unit1.reshape(test_feature_reduce_unit1.shape[0]*test_feature_reduce_unit1.shape[1]*test_feature_reduce_unit1.shape[2], -1)
                    test_feature_unit2 = test_feature_reduce_unit2.reshape(test_feature_reduce_unit2.shape[0]*test_feature_reduce_unit2.shape[1]*test_feature_reduce_unit2.shape[2], -1)
                    
                    print(test_feature_unit1.shape)
                    print(test_feature_unit2.shape)
    
                    
                    #--------lag unit--------------
                    test_feature_list_unit1, test_feature_list_unit2, test_feature_list_unit3, test_feature_list_unit4 = [], [], [], []
                    
                    for k in range(patch_size):
                        for l in range(patch_size):
                            ######################################
                            # get features
                            feature = np.array([])
                            # patch_ind = (div(k,patch_size))*(div(patch_size,patch_size)) + div(l,patch_size) # int div
                            # subpatch_ind = (div(k,subpatch_size))*(div(patch_size,subpatch_size)) + div(l,subpatch_size) # int div
                            feature = np.append(feature, test_img_patch[k,l,:])
                            feature = np.append(feature, [div(patch_size,2) - abs(i+k-div(patch_size,2)), div(patch_size,2) - abs(j+l-div(patch_size,2))]) #int div
                            
                            # feature = np.append(feature, test_feature12[0,:]) #takes only first few comps
                            feature1 = np.append(feature, test_feature_unit1[patch_ind,:])
                            feature2 = np.append(feature, test_feature_unit2[patch_ind,:])

                            test_feature_list_unit1.append(feature1)
                            test_feature_list_unit2.append(feature2)
                    

                    feature_list_unit1 = np.array(test_feature_list_unit1)
                    feature_list_unit2 = np.array(test_feature_list_unit2)

                    print(feature_list_unit1.shape)
                    print(feature_list_unit2.shape)


                    test_feature_red1= fs1.transform(feature_list_unit1)
                    test_feature_red2= fs2.transform(feature_list_unit2)


                    test_concat_features  =np.concatenate((test_feature_red1, test_feature_red2), axis=1)
                    print(test_concat_features.shape)
                    

                    feature_test = scaler1.transform(test_concat_features)
                    print(feature_test.shape)
                    

                    pre_list = clf.predict(feature_test)
                    print(pre_list.shape)
                    print(np.unique(pre_list))

                    # Generate predicted result
                    for k in range(patch_size):
                        for l in range(patch_size):
                            if i+k >= img_size or j+l >= img_size: break

                            # Binary
                            if pre_list[k*patch_size + l] > 0.5:
                                predict_0or1[i+k, j+l, 1] += 1
                            else:
                                predict_0or1[i+k, j+l, 0] += 1

                            # Multi-class
                            # if pre_list[k*patch_size + l] == 85.0:
                            #     predict_0or1[i+k, j+l, 1] += 1
                            # if pre_list[k*patch_size + l] == 170.0:
                            #     predict_0or1[i+k, j+l, 2] += 1
                            # if pre_list[k*patch_size + l] == 255.0:
                            #     predict_0or1[i+k, j+l, 3] += 1
                            # else:
                            #     predict_0or1[i+k, j+l, 0] += 1

            print('*************************************************************************************')
            print('one predicted mask')
            predict_mask = np.argmax(predict_0or1, axis=2)
            
            imageName = pathToResultSave + os.path.basename(test_img_addr)
            print(imageName)
            imageio.imsave(imageName, (predict_mask * 255).astype(np.uint8))
            del predict_0or1, predict_mask

    
    stop_test = timeit.default_timer()

    print('Total Time: ' + str(timedelta(seconds=stop_test-start_test)))
    f = open('{}test_time.txt'.format(pathToResultSave),'w+')
    f.write('Total Time: ' + str(timedelta(seconds=stop_test-start_test)))
    f.close()

@cuda.jit
def GPU_Feature(d_img_patch_list, d_train_feature_unit1, d_train_feature_unit2, d_feature_list_unit1, d_feature_list_unit2, d_threadDimensions, delta_x, patch_size, depth1, depth2):
	threadIdx = cuda.grid(1)
	t, i, j, k, l, d = indices6(threadIdx, d_threadDimensions)
	if t < d_threadDimensions[0]:
		i3 = d_img_patch_list.shape[3]
		patch_ind = t * (img_size//delta_x) * (img_size//delta_x) + i * (img_size//delta_x) + j
		if d < i3:
			d_feature_list_unit1[t, i, j, k, l, d] = d_img_patch_list[patch_ind,k,l,d]
			d_feature_list_unit2[t, i, j, k, l, d] = d_img_patch_list[patch_ind,k,l,d]
		elif d == i3:
			d_feature_list_unit1[t, i, j, k, l, d] = patch_size//2 - abs(i*delta_x+k-patch_size//2)
			d_feature_list_unit2[t, i, j, k, l, d] = patch_size//2 - abs(i*delta_x+k-patch_size//2)
		elif d == i3 + 1:
			d_feature_list_unit1[t, i, j, k, l, d] = patch_size//2 - abs(j*delta_x+l-patch_size//2)
			d_feature_list_unit2[t, i, j, k, l, d] = patch_size//2 - abs(j*delta_x+l-patch_size//2)
		elif d < depth1 and depth2:
			d_feature_list_unit1[t, i, j, k, l, d] = d_train_feature_unit1[patch_ind, d - (i3 + 2)]
			d_feature_list_unit2[t, i, j, k, l, d] = d_train_feature_unit2[patch_ind, d - (i3 + 2)]
		elif d < depth2:
			d_feature_list_unit2[t, i, j, k, l, d] = d_train_feature_unit2[patch_ind, d - (i3 + 2)]

@cuda.jit(device=True)
def indices6(m, threadDimensions):
    t = m // (threadDimensions[1] * threadDimensions[2] * threadDimensions[3] * threadDimensions[4] * threadDimensions[5])
    m -= t * (threadDimensions[1] * threadDimensions[2] * threadDimensions[3] * threadDimensions[4] * threadDimensions[5])
    i = m // (threadDimensions[2] * threadDimensions[3] * threadDimensions[4] * threadDimensions[5])
    m -= i * (threadDimensions[2] * threadDimensions[3] * threadDimensions[4] * threadDimensions[5])
    j = m // (threadDimensions[3] * threadDimensions[4] * threadDimensions[5])
    m -= j * (threadDimensions[3] * threadDimensions[4] * threadDimensions[5])
    k = m // (threadDimensions[4] * threadDimensions[5])
    m -= k * (threadDimensions[4] * threadDimensions[5])
    l = m // (threadDimensions[5])
    m -= l * (threadDimensions[5])
    d = m
    return t, i, j, k, l, d

    
if __name__=="__main__":
    freeze_support()
    run(sys.argv[1:])


"""

PixelHopMulti = """

import multiprocessing as mp
mp.freeze_support()
import numpy as np 
from numba import cuda
import os, glob, sys
from framework.layer import *
from framework.pixelhop import *
from utils import loadImage
import imageio
from division import div
import timeit
import xgboost as xgb
from sklearn import preprocessing
import pickle
import getopt
from sklearn.feature_selection import f_classif, SelectPercentile
from datetime import timedelta
from skimage import io
import numpy as np 
from numba import cuda
import os, glob
from framework.layer import *
from framework.pixelhop import *
from utils import loadImage
import imageio
from division import div
import timeit
import xgboost as xgb
from sklearn import preprocessing
import pickle
from sklearn.feature_selection import f_classif, SelectPercentile
from datetime import timedelta
import sys
import multiprocessing as mp
from multiprocessing.pool import Pool
from multiprocessing import freeze_support


numCPUCoresToUse = mp.cpu_count() - 1 if mp.cpu_count() > 1 else 1 

SAVE={}
img_size = 1024
patch_size = 64#16
delta_x = patch_size # non overlap






def run(argv):
    try:
        opts, args = getopt.getopt(argv, "hd: r: g: c: e: v: n:",["rfile = ","gfile = ","cfile = ","efile = ", "vfile = ", "nfile = "])
    except getopt.GetoptError:
        print("Pythonfile.py -r <rawimages> -g <groundtruths> -c <classes> -e <epochs>(getopterror)")
        sys.exit(2)
        
        
    for opt, arg in opts:
        if opt == "-r":
            print(arg)
            path = arg
        elif opt in ("-g", "--gfile"):
            gpath = arg
        elif opt in ("-c", "--cfile"):
            classes = arg
        elif opt in ("-e", "--efile"):
            epochs = arg
        elif opt in ("-v", "--vfile"):
            variance = float(arg)
        elif opt in ("-n", "--nfile"):
            num_training_imgs = int(arg)
            
        else:
            print("Pythonfile.py -r <rawimages> -g <groundtruths> -c <classes> -e <epochs>")
            sys.exit()






    #num_training_imgs = 1
    train_img_path = (path + '/*.png')
    test_img_path =  (gpath + '/*.png')

    train_img_addrs = glob.glob(train_img_path)
    test_img_addrs = glob.glob(test_img_path)

    print(train_img_path)
    print(train_img_addrs)
    start_train = timeit.default_timer()
    print("----------------------TRAINING-------------------------")
    # Initialize
    mask_patch_list, img_patch_list = [], []

    # Control the num of training images
    count = 0
    for train_img_addr in train_img_addrs:
        if 'raw' in train_img_addr: # only want to load training images that are raw
            count += 1
            if count > num_training_imgs: break
            print('Adding {} for training................'.format(train_img_addr))

            # Add raw images
            img = loadImage(train_img_addr) # Load image
            print(train_img_addr)

            # Add mask images
            mask = loadImage(train_img_addr.replace('raw', 'seg'))
            print(mask.shape)

            if len(img.shape) != 3:
                img = np.expand_dims(img, axis=2)
                print(img.shape)

            # Create patches for training images
            for i in range(0, img_size, delta_x):
                for j in range(0, img_size, delta_x):
                    if i+patch_size <= img_size and j+patch_size <= img_size:
                        img_patch = img[i:i+patch_size, j:j+patch_size, :]
                        mask_patch = mask[i:i+patch_size, j:j+patch_size]

                    elif i+patch_size > img_size and j+patch_size <= img_size:
                        temp_size = img[i:, j:j+patch_size, :].shape
                        img_patch = np.lib.pad(img[i:, j:j+patch_size, :], ((0,patch_size-temp_size[0]),(0,0), (0,0)), 'edge')
                        mask_patch = np.lib.pad(mask[i:, j:j+patch_size], ((0,patch_size-temp_size[0]),(0,0)), 'edge')

                    elif i+patch_size <= img_size and j+patch_size > img_size:
                        temp_size = img[i:i+patch_size, j:, :].shape
                        img_patch = np.lib.pad(img[i:i+patch_size, j:, :], ((0,0),(0,patch_size-temp_size[1]), (0,0)), 'edge')
                        mask_patch = np.lib.pad(mask[i:i+patch_size, j:], ((0,0),(0,patch_size-temp_size[1])), 'edge')

                    else: 
                        temp_size = img[i:, j:, :].shape
                        img_patch = np.lib.pad(img[i:, j:, :], ((0,patch_size-temp_size[0]),(0,patch_size-temp_size[1]), (0,0)), 'edge')
                        mask_patch = np.lib.pad(mask[i:, j:], ((0,patch_size-temp_size[0]),(0,patch_size-temp_size[1])), 'edge')

                    assert (img_patch.shape[0], img_patch.shape[1]) == (patch_size,patch_size)

                    # Save each patch to list
                    img_patch_list.append(img_patch)
                    mask_patch_list.append(mask_patch)
                    
                    
    # Convert list to numpy array
    img_patches = np.asarray(img_patch_list)
    mask_patches = np.array(mask_patch_list)
    print(img_patches.shape)
    print(mask_patches.shape)

    print('--------------------------------------')
    # Number of classes
    print('NUmber of classes: {}'.format(np.unique(mask_patches)))
    

    ################################################## PIXELHOP UNIT 1 ####################################################
        
    train_feature1=PixelHop_Unit_GPU(img_patches, dilate=1, pad='reflect', weight_name='pixelhop1.pkl', getK=1, energypercent=0.97)
    
    ################################################ PIXELHOP UNIT 2 ####################################################

    train_featurem1 = MaxPooling(train_feature1)
    train_feature2=PixelHop_Unit_GPU(train_featurem1, dilate=1, pad='reflect',  weight_name='pixelhop2.pkl', getK=1, energypercent=0.97)
   
    
    print(train_feature1.shape)
    print(train_feature2.shape)
    
    # Upsample the pixelhop feature 
    train_feature_reduce_unit1 = train_feature1 
    train_feature_reduce_unit2 = myResize(train_feature2, img_patches.shape[1], img_patches.shape[2])
    print(train_feature_reduce_unit1.shape)
    print(train_feature_reduce_unit2.shape)

    # Reshape the pixelhop feature
    train_feature_unit1 = train_feature_reduce_unit1.reshape(train_feature_reduce_unit1.shape[0]*train_feature_reduce_unit1.shape[1]*train_feature_reduce_unit1.shape[2], -1)
    train_feature_unit2 = train_feature_reduce_unit2.reshape(train_feature_reduce_unit2.shape[0]*train_feature_reduce_unit2.shape[1]*train_feature_reduce_unit2.shape[2], -1)
   
    del train_feature_reduce_unit1, train_feature_reduce_unit2
    
    print(train_feature_unit1.shape)
    print(train_feature_unit2.shape)
    
    ### NEW CODE ###
    count = num_training_imgs
    patch_ind = count * (img_size//delta_x) * (img_size//delta_x)
    np_img_patch_list = np.array(img_patch_list)

    ## get parameters for threading
    depth1 = np_img_patch_list.shape[3] + 2 + train_feature_unit1.shape[1]
    depth2 = np_img_patch_list.shape[3] + 2 + train_feature_unit2.shape[1]
    depth = max(depth1, depth2)
    feature_shape1 = (count, img_size//delta_x, img_size//delta_x, patch_size, patch_size, depth1)
    feature_shape2 = (count, (img_size//delta_x), (img_size//delta_x), patch_size, patch_size, depth2)

    ## allocate and transfer data to device
    d_img_patch_list = cuda.to_device(np.ascontiguousarray(img_patch_list))
    d_train_feature_unit1 = cuda.to_device(np.ascontiguousarray(train_feature_unit1))
    d_train_feature_unit2 = cuda.to_device(np.ascontiguousarray(train_feature_unit2))
    d_feature_list_unit1 = cuda.device_array(feature_shape1)
    d_feature_list_unit2 = cuda.device_array(feature_shape2)

    ## setup thread dimensions
    threadDimensions = np.ascontiguousarray([count, img_size//delta_x, img_size//delta_x, patch_size, patch_size, depth])
    d_threadDimensions = cuda.to_device(threadDimensions)
    totalThreads = threadDimensions.prod()
    threadsPerBlock = 64
    blocksPerGrid = math.ceil(totalThreads/threadsPerBlock)
    
    ## run device kernel
    GPU_Feature[blocksPerGrid, threadsPerBlock](d_img_patch_list, d_train_feature_unit1, d_train_feature_unit2, d_feature_list_unit1, d_feature_list_unit2, d_threadDimensions, delta_x, patch_size, depth1, depth2)

    ## transfer results back
    feature_list_unit1 = d_feature_list_unit1.copy_to_host().reshape(count * (img_size//delta_x) * (img_size//delta_x) * patch_size * patch_size, -1)
    feature_list_unit2 = d_feature_list_unit2.copy_to_host().reshape(count * (img_size//delta_x) * (img_size//delta_x) * patch_size * patch_size, -1)

    ## get gt_list from mask_patch_list directly
    gt_list = np.array(mask_patch_list).flatten()
    ### END NEW CODE ###

    print(feature_list_unit1.shape)
    print(feature_list_unit2.shape)
    
    print(gt_list.shape)
    
    # F-test to get top 80% features
    fs1 = SelectPercentile(score_func=f_classif, percentile=80)
    fs1.fit(feature_list_unit1, gt_list)
    new_features1 = fs1.transform(feature_list_unit1)
    print(new_features1.shape)
    print(fs1.scores_)
    
    fs2 = SelectPercentile(score_func=f_classif, percentile=80)
    fs2.fit(feature_list_unit2, gt_list)
    new_features2 = fs2.transform(feature_list_unit2)
    print(new_features2.shape)
    print(fs2.scores_)
    
    # Concatenate all the features together
    concat_features  = np.concatenate((new_features1, new_features2), axis=1)
    print(concat_features.shape)
    
    del feature_list_unit1, new_features1, feature_list_unit2, new_features2
    
    # Preprocessing (standardize features by removing the mean and scaling to unit variance)
    scaler1=preprocessing.StandardScaler().fit(concat_features)
    feature = scaler1.transform(concat_features) 
    print(feature.shape)

    # Define and train XGBoost algorithm
    xgb_model = xgb.XGBClassifier(tree_method='gpu_hist', objective="binary:logistic", verbosity=3)
    clf = xgb_model.fit(feature, gt_list)

    # Save all the model files
    
    pathToResultSave = os.path.abspath(epochs) + (chr(92))
    print(pathToResultSave)
    # Save all the model files
    pickle.dump(clf, open("{}classifier.sav".format(pathToResultSave),'wb')) 
    pickle.dump(scaler1,open("{}scaler1.sav".format(pathToResultSave),'wb') ) 
    pickle.dump(fs1, open("{}fs1.sav".format(pathToResultSave),'wb')) 
    pickle.dump(fs2, open("{}fs2.sav".format(pathToResultSave),'wb')) 
    print('All models saved!!!')

    stop_train = timeit.default_timer()

    print('Total Time: ' + str(timedelta(seconds=stop_train-start_train)))
    f = open('{}train_time.txt'.format(pathToResultSave),'w+')
    f.write('Total Time: ' + str(timedelta(seconds=stop_train-start_train)))
    f.close()
    
    print(patch_ind)
    patch_ind = 0
    # delete unused variables to free up memory
    del img_patch_list, img, img_patches, img_patch, np_img_patch_list
    del gt_list
    del feature, concat_features
    del mask_patch, mask, mask_patch_list, mask_patches
    del train_feature1, train_feature2, train_featurem1, train_feature_unit1, train_feature_unit2
    del d_feature_list_unit1, d_feature_list_unit2, d_train_feature_unit1, d_train_feature_unit2, d_img_patch_list

    start_test = timeit.default_timer()
    print("----------------------TESTING-------------------------")
    
    #for i in range(len(test_img_addrs)):
    #    testImage(i, scaler1, clf, fs1, fs2, patch_ind)
    
    numCores = numCPUCoresToUse
    numImages = len(test_img_addrs)
    chunkSize = math.ceil(numImages/numCores)

    with Pool(numCores, initializer=init_pool, initargs=(path, test_img_addrs, patch_size, scaler1, clf, fs1, fs2, patch_ind,)) as p:
        print(p.map(testImage, range(numImages), chunksize=chunkSize))

    stop_test = timeit.default_timer()

    #print('Total Time: ' + str(timedelta(seconds=stop_test-start_test)))
    #f = open( path +'results/test_time.txt','w+')
    #f.write('Total Time: ' + str(timedelta(seconds=stop_test-start_test))
    #f.close()

def init_pool(a_path, a_test_img_addrs, a_patch_size, a_scaler1, a_clf, a_fs1, a_fs2, a_patch_ind):
    global path
    global test_img_addrs
    global patch_size
    global scaler1, clf, fs1, fs2
    global patch_ind

    path = a_path
    test_img_addrs = a_test_img_addrs
    patch_size = a_patch_size
    scaler1 = a_scaler1
    clf = a_clf
    fs1 = a_fs1
    fs2 = a_fs2
    patch_ind = a_patch_ind

@cuda.jit
def GPU_Feature(d_img_patch_list, d_train_feature_unit1, d_train_feature_unit2, d_feature_list_unit1, d_feature_list_unit2, d_threadDimensions, delta_x, patch_size, depth1, depth2):
	threadIdx = cuda.grid(1)
	t, i, j, k, l, d = indices6(threadIdx, d_threadDimensions)
	if t < d_threadDimensions[0]:
		i3 = d_img_patch_list.shape[3]
		patch_ind = t * (img_size//delta_x) * (img_size//delta_x) + i * (img_size//delta_x) + j
		if d < i3:
			d_feature_list_unit1[t, i, j, k, l, d] = d_img_patch_list[patch_ind,k,l,d]
			d_feature_list_unit2[t, i, j, k, l, d] = d_img_patch_list[patch_ind,k,l,d]
		elif d == i3:
			d_feature_list_unit1[t, i, j, k, l, d] = patch_size//2 - abs(i*delta_x+k-patch_size//2)
			d_feature_list_unit2[t, i, j, k, l, d] = patch_size//2 - abs(i*delta_x+k-patch_size//2)
		elif d == i3 + 1:
			d_feature_list_unit1[t, i, j, k, l, d] = patch_size//2 - abs(j*delta_x+l-patch_size//2)
			d_feature_list_unit2[t, i, j, k, l, d] = patch_size//2 - abs(j*delta_x+l-patch_size//2)
		elif d < depth1 and depth2:
			d_feature_list_unit1[t, i, j, k, l, d] = d_train_feature_unit1[patch_ind, d - (i3 + 2)]
			d_feature_list_unit2[t, i, j, k, l, d] = d_train_feature_unit2[patch_ind, d - (i3 + 2)]
		elif d < depth2:
			d_feature_list_unit2[t, i, j, k, l, d] = d_train_feature_unit2[patch_ind, d - (i3 + 2)]

@cuda.jit(device=True)
def indices6(m, threadDimensions):
    t = m // (threadDimensions[1] * threadDimensions[2] * threadDimensions[3] * threadDimensions[4] * threadDimensions[5])
    m -= t * (threadDimensions[1] * threadDimensions[2] * threadDimensions[3] * threadDimensions[4] * threadDimensions[5])
    i = m // (threadDimensions[2] * threadDimensions[3] * threadDimensions[4] * threadDimensions[5])
    m -= i * (threadDimensions[2] * threadDimensions[3] * threadDimensions[4] * threadDimensions[5])
    j = m // (threadDimensions[3] * threadDimensions[4] * threadDimensions[5])
    m -= j * (threadDimensions[3] * threadDimensions[4] * threadDimensions[5])
    k = m // (threadDimensions[4] * threadDimensions[5])
    m -= k * (threadDimensions[4] * threadDimensions[5])
    l = m // (threadDimensions[5])
    m -= l * (threadDimensions[5])
    d = m
    return t, i, j, k, l, d

def testImage(test_img_index):#, scaler1, clf, fs1, fs2, patch_ind):
    test_img_patch_list = []

    test_img_addr = test_img_addrs[test_img_index]

    if 'raw' in test_img_addr: # only want to load testing images that are raw
        #print('Processing {}............'.format(test_img_addr))

        img = loadImage(test_img_addr) 

        if len(img.shape) != 3:
            img = np.expand_dims(img, axis=2)

        # Initialzing
        predict_0or1 = np.zeros((img_size, img_size, 4))

        predict_mask = np.zeros(img.shape)

        # Create patches for test image
        for i in range(0, img_size, delta_x):
            for j in range(0, img_size, delta_x):
                if i+patch_size <= img_size and j+patch_size <= img_size:
                    test_img_patch = img[i:i+patch_size, j:j+patch_size, :]

                elif i+patch_size > img_size and j+patch_size <= img_size:
                    temp_size = img[i:, j:j+patch_size, :].shape
                    test_img_patch = np.lib.pad(img[i:, j:j+patch_size, :], ((0,patch_size-temp_size[0]),(0,0), (0,0)), 'edge')

                elif i+patch_size <= img_size and j+patch_size > img_size:
                    temp_size = img[i:i+patch_size, j:, :].shape
                    test_img_patch = np.lib.pad(img[i:i+patch_size, j:, :], ((0,0),(0,patch_size-temp_size[1]), (0,0)), 'edge')

                else: 
                    temp_size = img[i:, j:, :].shape
                    test_img_patch = np.lib.pad(img[i:, j:, :], ((0,patch_size-temp_size[0]),(0,patch_size-temp_size[1]), (0,0)), 'edge')
                assert (test_img_patch.shape[0], test_img_patch.shape[1]) == (patch_size,patch_size)
                    
                test_img_patch_list.append(test_img_patch)
                    

                # convert list to numpy
                test_img_subpatches = np.asarray(test_img_patch_list)
                print(test_img_subpatches.shape)

                    
                ################################################## PIXELHOP UNIT 1 ####################################################
        
                test_feature1=PixelHop_Unit_GPU(test_img_subpatches, dilate=1, pad='reflect', weight_name='pixelhop1.pkl', getK=0, energypercent=0.97)

                ################################################# PIXELHOP UNIT 2 ####################################################
                test_featurem1 = MaxPooling(test_feature1)
                test_feature2=PixelHop_Unit_GPU(test_featurem1, dilate=1, pad='reflect', weight_name='pixelhop2.pkl', getK=0, energypercent=0.97)
                    
    
                test_feature_reduce_unit1 = test_feature1 
                test_feature_reduce_unit2 = myResize(test_feature2, test_img_subpatches.shape[1], test_img_subpatches.shape[2])
                print(test_feature_reduce_unit1.shape)
                print(test_feature_reduce_unit2.shape)

                    
                test_feature_unit1 = test_feature_reduce_unit1.reshape(test_feature_reduce_unit1.shape[0]*test_feature_reduce_unit1.shape[1]*test_feature_reduce_unit1.shape[2], -1)
                test_feature_unit2 = test_feature_reduce_unit2.reshape(test_feature_reduce_unit2.shape[0]*test_feature_reduce_unit2.shape[1]*test_feature_reduce_unit2.shape[2], -1)
                    
                print(test_feature_unit1.shape)
                print(test_feature_unit2.shape)
    
                    
                #--------lag unit--------------
                test_feature_list_unit1, test_feature_list_unit2, test_feature_list_unit3, test_feature_list_unit4 = [], [], [], []
                    
                for k in range(patch_size):
                    for l in range(patch_size):
                        ######################################
                        # get features
                        feature = np.array([])
                        patch_ind = (div(k,patch_size))*(div(patch_size,patch_size)) + div(l,patch_size) # int div
                        # subpatch_ind = (div(k,subpatch_size))*(div(patch_size,subpatch_size)) + div(l,subpatch_size) # int div
                        feature = np.append(feature, test_img_patch[k,l,:])
                        feature = np.append(feature, [div(patch_size,2) - abs(i+k-div(patch_size,2)), div(patch_size,2) - abs(j+l-div(patch_size,2))]) #int div
                            
                        # feature = np.append(feature, test_feature12[0,:]) #takes only first few comps
                        feature1 = np.append(feature, test_feature_unit1[patch_ind,:])
                        feature2 = np.append(feature, test_feature_unit2[patch_ind,:])

                        test_feature_list_unit1.append(feature1)
                        test_feature_list_unit2.append(feature2)
                        
                del test_feature_unit1, test_feature_unit2

                feature_list_unit1 = np.array(test_feature_list_unit1)
                feature_list_unit2 = np.array(test_feature_list_unit2)

                print(feature_list_unit1.shape)
                print(feature_list_unit2.shape)


                test_feature_red1= fs1.transform(feature_list_unit1)
                test_feature_red2= fs2.transform(feature_list_unit2)


                test_concat_features  =np.concatenate((test_feature_red1, test_feature_red2), axis=1)
                print(test_concat_features.shape)
                    

                feature_test = scaler1.transform(test_concat_features)
                print(feature_test.shape)
                    

                pre_list = clf.predict(feature_test)
                print(pre_list.shape)
                print(np.unique(pre_list))

                # Generate predicted result
                for k in range(patch_size):
                    for l in range(patch_size):
                        if i+k >= img_size or j+l >= img_size: break

                        # Binary
                        if pre_list[k*patch_size + l] > 0.5:
                            predict_0or1[i+k, j+l, 1] += 1
                        else:
                            predict_0or1[i+k, j+l, 0] += 1

                        # Multi-class
                        # if pre_list[k*patch_size + l] == 1.0:
                        #     predict_0or1[i+k, j+l, 1] += 1
                        #     print('atria')
                        # elif pre_list[k*patch_size + l] == 2.0:
                        #     predict_0or1[i+k, j+l, 2] += 1
                        #     print('trabaculae')
                        # elif pre_list[k*patch_size + l] == 3.0:
                        #     predict_0or1[i+k, j+l, 3] += 1
                        #     print('ventricle')
                        # else:
                        #     predict_0or1[i+k, j+l, 0] += 1
                                
                        # if pre_list[k*patch_size + l] == 85.0:
                        #     predict_0or1[i+k, j+l, 1] += 1
                        # if pre_list[k*patch_size + l] == 170.0:
                        #     predict_0or1[i+k, j+l, 2] += 1
                        # if pre_list[k*patch_size + l] == 255.0:
                        #     predict_0or1[i+k, j+l, 3] += 1
                        # else:
                        #     predict_0or1[i+k, j+l, 0] += 1

        #print('*************************************************************************************')
        #print('one predicted mask')
        predict_mask = np.argmax(predict_0or1, axis=2)
        imageName = pathToResultSave + os.path.basename(test_img_addr)
        print(imageName)
        imageio.imsave(imageName, (predict_mask * 255).astype(np.uint8))
        #print('Test image '+test_img_index+' complete...')

        return 1
    
if __name__=="__main__":
    freeze_support()
    run(sys.argv[1:])
    

"""




# print(unet)

algorithms_list=['PixelHop']
# lossfunctions_list = ['Binary Cross Entropy', 'Cateogorical Cross Entropy', 'Customize']
        
class Input_tab(QWidget):
    
    def __init__(self):
        super().__init__()
        global Train, pbar, proc, rawimagepathtextbox, groundtruthtextbox, algorithms, rawimagelabel, groundtruthlabel, consoletextbox, noofclasses, noofepochs, variance, trainImgNumber, useMulti,  testImgNumber

        useMulti = False
       
    
        vlayout = QVBoxLayout()
        
        #-----------------------------------------------------INPUTS-----------------------------------------------------
        
        hlayout = QHBoxLayout()
        hlayout1 = QHBoxLayout()
        hlayout2 = QHBoxLayout()
        hlayout3 = QHBoxLayout()
        hlayout4 = QHBoxLayout()
        
        
        
        rawimagepathlabel = QLabel('Training Images ', self)
        rawimagepathlabel.setFont(QFont('Segoe UI', 7))
    
  
        rawimagepathtextbox = QLineEdit(self, placeholderText='Training Images folder')
        rawimagepathtextbox.setFont(QFont('Segoe UI', 7))
        rawimagepathtextbox.resize(600,30)
        
        
        rawimagepathbrowse = QPushButton('Select', self)
        rawimagepathbrowse.clicked.connect(self.on_click_rawimagefolder)
        rawimagepathbrowse.setToolTip('Select the Training Images Folder')
        rawimagepathbrowse.setStyleSheet('QPushButton {  border-color: white; border-width: 0px; padding: 10px; border-style: outset; border-radius: 6px; background-color : #3375ec; font-weight: regular; font-size: 8pt; font-family: Seoge UI; color: white; }')#setStyleSheet('QPushButton {  background-color : #92374d; color: white; }')
        
        
        
        rawimagelabel = QLabel(self)
        groundtruthlabel = QLabel(self)
        
        
        
    
        
        algorithmslabel = QLabel('Select algorithm', self)
        algorithmslabel.setFont(QFont('Segoe UI', 7))
        
        groundtruthpathlabel = QLabel('Testing Images', self)
        groundtruthpathlabel.setFont(QFont('Segoe UI', 7))
    
      
        groundtruthtextbox = QLineEdit(self, placeholderText='Testing Images folder')
        groundtruthtextbox.setFont(QFont('Segoe UI', 7))
        groundtruthtextbox.resize(600,30)
        
        groundtruthpathbrowse = QPushButton('Select', self)
        groundtruthpathbrowse.clicked.connect(self.on_click_groundtruthfolder)
        groundtruthpathbrowse.setToolTip('Select the Testing Images Folder')
        groundtruthpathbrowse.setStyleSheet('QPushButton {  border-color: white; border-width: 0px; padding: 10px; border-style: outset; border-radius: 6px; background-color : #3375ec; font-weight: regular; font-size: 8pt; font-family: Seoge UI; color: white; }')#setStyleSheet('QPushButton {  background-color : #92374d; color: white; }')
        
      
        algorithms = QComboBox()
        algorithms.setFont(QFont('Segoe UI', 7))
        algorithms.addItems(algorithms_list)
        # algorithms.addItem("2D Unet")
        # algorithms.addItem("Custom")
        algorithms.currentIndexChanged.connect(self.algo_change)
        
        classeslabel = QLabel('Number of Classes', self)
        classeslabel.setFont(QFont('Segoe UI',7))
        
        noofclasses = QLineEdit(self)
        noofclasses.setFont(QFont('Segoe UI', 7 ))
        noofclasses.setFixedWidth(30)
        noofclasses.setText('2')
        
        variancelabel = QLabel('Variance 0.01 to 0.99', self)
        variancelabel.setFont(QFont('Segoe UI', 7))
        
        variance = QLineEdit(self)
        variance.setFont(QFont('Segoe UI', 7))
        variance.setFixedWidth(30)
        variance.setText('.98')
        
        trainImgNumlabel = QLabel('Number of Training Images', self)
        trainImgNumlabel.setFont(QFont('Segoe UI', 7))
        
        trainImgNumber = QLineEdit(self)
        trainImgNumber.setFont(QFont('Segoe UI', 7))
        trainImgNumber.setFixedWidth(30)
        
        testImgNumlabel = QLabel('More than 1 test image?', self)
        testImgNumlabel.setFont(QFont('Segoe UI', 7))
        
        testImgNumber = QCheckBox('', self)
        testImgNumber.stateChanged.connect(self.checkedc)
        testImgNumber.setChecked(useMulti)
        
        
        
        
        epochlabel =  QLabel('Folder to Save Outputs', self)
        epochlabel.setFont(QFont('Segoe UI', 7))
        
        noofepochs = QLineEdit(self)
        noofepochs.setFont(QFont('Segoe UI', 7))
        #noofepochs.setFixedWidth(30)
        noofepochs.resize(600,30)

        noefepochsbrowse = QPushButton('Select', self)
        noefepochsbrowse.clicked.connect(self.on_click_noefepochs)
        noefepochsbrowse.setToolTip('Select the Result Save Folder')
        noefepochsbrowse.setStyleSheet('QPushButton {  border-color: white; border-width: 0px; padding: 10px; border-style: outset; border-radius: 6px; background-color : #3375ec; font-weight: regular; font-size: 8pt; font-family: Seoge UI; color: white; }')#setStyleSheet('QPushButton {  background-color : #92374d; color: white; }')
        
        
        
        
        
        
        
        
        # lossfunctionslabel = QLabel('Select loss function', self)
        # lossfunctionslabel.setFont(QFont('Segoe UI', 10))
        
        # lossfunctions = QComboBox()
        # lossfunctions.setFont(QFont('Segoe UI', 8))
        # lossfunctions.addItems(lossfunctions_list)
        # # lossfunctions.addItem("Binary Cross Entropy")
        # # lossfunctions.addItem("Cateogorical Cross Entropy")
        # lossfunctions.currentIndexChanged.connect(self.loss_change)
        
        
        
        
        
        consoletextbox = QTextEdit(self)
        consoletextbox.setFont(QFont('Segoe UI', 9))
        #consoletextbox.setFixedHeight(200)
        consoletextbox.setStyleSheet('QTextEdit{ border:0; }')
        #consoletextbox.setText("For Help visit our GitHub github.com/the905/R-NET")
        
        helpButton = QPushButton('Help', self)
        helpButton.clicked.connect(self.on_click_helpButton)
        helpButton.setToolTip('Click here for help and our github')
        helpButton.setStyleSheet('QPushButton {  border-color: white; border-width: 0px; padding: 10px; border-style: outset; border-radius: 6px; background-color : #3375ec; font-weight: regular; font-size: 8pt; font-family: Seoge UI; color: white; }')#setStyleSheet('QPushButton {  background-color : #92374d; color: white; }')
        
        
        
        
        
        
        pbar = QProgressBar(self, textVisible=False)
        pbar.setValue(0)
        pbar.setFixedHeight(10)
        # pbar.setAlignment(Qt.AlignCenter)
        pbar.setStyleSheet("""QProgressBar {
border-radius: 5px;
}
QProgressBar::chunk 
{
background-color: green;
border-radius :5px;
}    
                          """)
        
        # pbar.resize(300, 100)
        
        Train = QPushButton('Run', self)
        Train.clicked.connect(self.on_click_run)
        Train.setToolTip('Train the model')
        Train.setStyleSheet('QPushButton {  border-color: black; border-width: 0.2px; padding: 14px; border-style: outset; border-radius: 20px; background-color : lightgray; font-weight: regular; font-size: 12pt; font-family: Seoge UI; color: black; } }')
       
       
       
       
       
               
      
       
        hlayout.addWidget(rawimagepathlabel)
        hlayout.addWidget(rawimagepathtextbox)
        hlayout.addWidget(rawimagepathbrowse)
        # hlayout.addWidget(algorithms)
        
        hlayout1.addWidget(groundtruthpathlabel)
        hlayout1.addWidget(groundtruthtextbox)
        hlayout1.addWidget(groundtruthpathbrowse)
        
        hlayout4.addWidget(rawimagelabel)
        hlayout4.addWidget(groundtruthlabel)
        
        
        
        hlayout2.addWidget(classeslabel)
        hlayout2.addWidget(noofclasses,1)
        
        hlayout2.addWidget(variancelabel)
        hlayout2.addWidget(variance, 1)
        
        hlayout2.addWidget(trainImgNumlabel)
        hlayout2.addWidget(trainImgNumber, 1)
        
        hlayout2.addWidget(testImgNumlabel)
        hlayout2.addWidget(testImgNumber, 1)

        hlayout2.addWidget(QWidget())
        hlayout2.addWidget(QWidget())
        hlayout2.addWidget(QWidget())
        hlayout2.addWidget(QWidget())
        hlayout2.addWidget(QWidget())
        hlayout2.addWidget(QWidget())
        hlayout2.addWidget(QWidget())
        hlayout2.addWidget(QWidget())
        hlayout2.addWidget(QWidget())
        hlayout2.addWidget(QWidget())
        hlayout2.addWidget(QWidget())
        hlayout2.addWidget(QWidget())
        
        
        
        hlayout3.addWidget(epochlabel)
        hlayout3.addWidget(noofepochs,1)
        hlayout3.addWidget(noefepochsbrowse)
        #hlayout3.addWidget(helpButton)
        
     
        hlayout3.addWidget(QWidget())
        hlayout3.addWidget(QWidget())
        hlayout3.addWidget(QWidget())
        hlayout3.addWidget(QWidget())
        hlayout3.addWidget(QWidget())
        hlayout3.addWidget(QWidget())
        hlayout3.addWidget(QWidget())
        hlayout3.addWidget(QWidget())
        hlayout3.addWidget(QWidget())
        hlayout3.addWidget(QWidget())
        hlayout3.addWidget(QWidget())
        hlayout3.addWidget(QWidget())
      
        vlayout.addLayout(hlayout)
        vlayout.addLayout(hlayout1)
        vlayout.addLayout(hlayout4)
        vlayout.addWidget(QWidget())
        vlayout.addLayout(hlayout2)
        vlayout.addLayout(hlayout3)
        
        vlayout.addWidget(QWidget())
        vlayout.addWidget(algorithmslabel)
        vlayout.addWidget(algorithms)
        vlayout.addWidget(QWidget())
        # vlayout.addWidget(lossfunctionslabel)
        # vlayout.addWidget(lossfunctions)
        
        vlayout.addWidget(QWidget())
        # vlayout.addWidget(QWidget())
        vlayout.addWidget(QWidget())
        vlayout.addWidget(QWidget())
        
        vlayout.addWidget(consoletextbox)
        vlayout.addWidget(QWidget())
        #vlayout.addWidget(helpButton)
        vlayout.addWidget(pbar)
        vlayout.addWidget(helpButton, alignment =QtCore.Qt.AlignLeft)
        vlayout.addWidget(Train)
       
        
        
        self.setLayout(vlayout)
        

    
    @pyqtSlot()
    
    def checkedc(self):
        global testImgNumber
        if testImgNumber.isChecked():
            useMulti = True
            print(useMulti)
            print("Multi is being used")
            
        else:
            useMulti = False   
            print ("Multi is not being used")    
            
    def on_click_helpButton(self):
         urL='http://github.com/the905/R-NET'
         #chrome_path ="C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe %s"
         #webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(chrome_path))
         webbrowser.open_new_tab(urL)
            
    
    
    
    def on_click_rawimagefolder(self):
        global rawimages
        rawimagepathtextbox.setText('')
        foldername = QFileDialog.getExistingDirectory(self, 'Select Folder')
        
        
        
        rawimagepathtextbox.setText(foldername)
        
        rawimages = glob.glob(foldername+'/*.*')
        
        print(rawimages)
        
        
        
        for x in range (int(len(rawimages)) ):
            currentImg = str((os.path.basename(rawimages[x])))
            print(currentImg)
            if("raw" in currentImg):
                indexOfImage = x
                break
        
            
        # randomimage = random.randint(0, len(groundtruthimages))
        # print(rawimages[randomimage])
        # print(groundtruthimages[0])
        
        print(str(rawimages[indexOfImage]))
        
        r_img = cv2.imread(rawimages[indexOfImage])
        cols, rows, channel = r_img.shape
        print(r_img.shape)
        brightness= np.sum(r_img)/(255*cols*rows)
        minimum_brightness = 0.33
        ratio = brightness/minimum_brightness
        
      

        r_bright_img = cv2.convertScaleAbs(r_img, alpha = 1/ratio, beta=0)
        
        
        bytesperline = 3 * cols
        r_qimg = QImage(r_bright_img,cols,rows,bytesperline,QImage.Format_RGB888)        
        pixmap = QPixmap(r_qimg)
        pixmap4 = pixmap.scaled(400, 400, QtCore.Qt.KeepAspectRatio)
        rawimagelabel.setPixmap(pixmap4)
        # rawimagelabel.setFixedSize(100,100)
        
        # randomimage = random.choice(rawimages)#QPixmap(rawimages)
        # print(randomimage)
        
    def on_click_noefepochs(self):
        noofepochs.setText('')
        foldername = QFileDialog.getExistingDirectory(self, 'Select Folder')
        noofepochs.setText(foldername)
        
        rawimages = glob.glob(foldername+'/*.*')
        randomimage = random.choice(rawimages)#QPixmap(rawimages)
        # print(randomimage)    
    
    def on_click_groundtruthfolder(self):
        global rawimages, groundtruthimages
        groundtruthtextbox.setText('')
        foldername = QFileDialog.getExistingDirectory(self, 'Select Folder')
        groundtruthimages = glob.glob(foldername+'/*.*')
        print(groundtruthimages)
        # randomgroundtruthimage = random.choice(groundtruthimages)#QPixmap(rawimages)
        # print(randomgroundtruthimage)
        groundtruthtextbox.setText(foldername)
        
        indexOfImage = 0
        #image_list = os.listdir(foldername)
        #print(image_list)
        
        
        g_img = cv2.imread(groundtruthimages[0])
        # g_img = cv2.resize(g_img, (400,400), interpolation=cv2.INTER_AREA)
        cols, rows, channel = g_img.shape
        print(g_img.shape)
        brightness= np.sum(g_img)/(255*cols*rows)
        minimum_brightness = 0.33
        ratio = brightness/minimum_brightness

        g_bright_img = cv2.convertScaleAbs(g_img, alpha = 1/ratio, beta=0) 
        
        
        g_qimg = QImage(g_bright_img,cols,rows,QImage.Format_RGB888) 
        
        pixmap = QPixmap(g_qimg)
        pixmap5 = pixmap.scaled(400, 400, QtCore.Qt.KeepAspectRatio)
        groundtruthlabel.setPixmap(pixmap5)
        # groundtruthlabel.setFixedSize(100,100)
        
       
        
    def algo_change(self, i):
        print(algorithms.currentText())
        
        if algorithms.currentText() == "Customize":
            text, okPressed = QInputDialog.getText(self, "Algorithm","Algorithm name:", QLineEdit.Normal, "")
            if okPressed and text != '':
                print(text)
                algorithms.clear()
                algorithms_list.append(text)
                algorithms_list.sort(key = 'Customize'.__eq__)
                # algorithms_list.append(algorithms_list.pop(algorithms_list.index(len(algorithms_list)-1))) 
                algorithms.addItems(algorithms_list)
                algorithms.setCurrentText(text)

            

        
        
        
        
    # def loss_change(self, i):
    #     print(lossfunctions.currentText())
    #     if lossfunctions.currentText() == "Customize":
    #         text, okPressed = QInputDialog.getText(self, "Loss function","Loss function name:", QLineEdit.Normal, "")
    #         if okPressed and text != '':
    #             print(text)
    #             lossfunctions.clear()
    #             lossfunctions_list.append(text)
    #             lossfunctions_list.sort(key = 'Customize'.__eq__)
    #             # algorithms_list.append(algorithms_list.pop(algorithms_list.index(len(algorithms_list)-1))) 
    #             lossfunctions.addItems(lossfunctions_list)
    #             lossfunctions.setCurrentText(text)
        
        
    def on_click_run(self):
        global proc, pbar, Train, noofepochs, testImgNumber
        print("Run")
        consoletextbox.clear()
        with open(resource_path('src\\algorithms\\example.py'),'w') as f:
            print(useMulti)
            if testImgNumber.isChecked():
                print("Using Multi")
                f.write(PixelHopMulti)
            else:
                print("Using Non Multi")
                f.write(PixelHopNonMulti)
            print('algorithim written')
        
        #with open(resource_path('logs/log.txt'),'w') as f:
            #f.write('Numbar of epochs: ' + str(noofepochs.text()))
        
        
        
        # self.thread = Thread()
        # self.thread._signal.connect(self.signal_accept)
        # self.thread.start()
        # Train.setEnabled(False)
        # pbar.setValue(0)
        files = glob.glob(resource_path('src//algorithms//*.*'))
        print(files)
        for f in files:
            if f.endswith('example.py'):
                #print(pyPath+ "python.exe" + " \"algorithms/3D UNET.py\" -r \""+rawimagepathtextbox.text()+"\" -g \""+groundtruthtextbox.text()+"\" -c \""+noofclasses.text()+"\" -e \""+noofepochs.text()+"\"")
                
                # execution = []
                # execution.append("python")
                # execution.append (resource_path("src/algorithms/example.py" ).replace("\\","/") )
                # execution.append("-r")
                # execution.append(rawimagepathtextbox.text())
                # execution.append("-g")
                # execution.append(groundtruthtextbox.text())
                # execution.append("-c")
                # execution.append(noofclasses.text() )
                # execution.append("-e")
                # execution.append(noofepochs.text())
               
                # # execution = repr(execution)
                # print(execution)
                proc = subprocess.Popen(pyPath+ "python.exe" + " \"src/algorithms/example.py\" -r \""+rawimagepathtextbox.text()+"\" -g \""+groundtruthtextbox.text()+"\" -c \""+noofclasses.text()+"\" -e \""+noofepochs.text()+"\" -v \"" + variance.text() +"\" -n \""+trainImgNumber.text()+"\"",shell= False ) 
                # proc.wait()
                # os.remove(resource_path('algorithms/3D UNET.py'))
            # elif f.split('\\')[1] == '2D UNET.py':
            #     proc = subprocess.Popen(pyPath+ "python.exe" + " \"algorithms/2D UNET.py\" -r \""+rawimagepathtextbox.text()+"\" -g \""+groundtruthtextbox.text()+"\" -c \""+noofclasses.text()+"\" -e \""+noofepochs.text()+"\"")
            else:
                print('no script')
    
    def signal_accept(self, msg):
        
        #if proc.poll() is None:
            # if int(noofepochs.text()) < 100:
            pbar.setValue(int(msg))
            # elif int(noofepochs.text()) > 100:
                # pbar.setValue(int(noofepochs.text())*100/100)
        #else:
            pbar.setValue(100)
            print('Done')
            #with open(resource_path('logs/outputlog.txt'),'r') as f:
                #score = f.read()
            
            #consoletextbox.appendPlainText('Results:\n')    
            #consoletextbox.appendPlainText('Loss: ' + score.split('\n')[0].split(' ')[2])
            #consoletextbox.appendPlainText('IOU Score: ' + score.split(' ')[-1])
                    
            # Train.setEnabled(True)
            # self.thread.terminate()
        

    
        
class Test_tab(QWidget):
    
    def __init__(self):
        super().__init__()
        global algorithms, lossfunctions, testimagespathtextbox, modelpathtextbox, modelpathtextbox2, savepredictedtextbox, classifierbrowsetextbox, scalerbrowsetextbox, featurebrowse1textbox, featurebrowse2textbox
        
        
        hlayout2 = QHBoxLayout()
        hlayout3 = QHBoxLayout()
        hlayout4 = QHBoxLayout()
        hlayout5 = QHBoxLayout()
        vlayout4 = QVBoxLayout()
        
         
        testimageslabel = QLabel('Test Images  ', self)
        testimageslabel.setFont(QFont('Segoe UI', 10))
    
  
        testimagespathtextbox = QLineEdit(self, placeholderText='Test images folder')
        testimagespathtextbox.setFont(QFont('Segoe UI', 8))
        testimagespathtextbox.resize(600,30)
        
        
        testimagespathbrowse = QPushButton('Select', self)
        testimagespathbrowse.clicked.connect(self.test_image)
        testimagespathbrowse.setToolTip('Select the Test Images Folder')
        testimagespathbrowse.setStyleSheet('QPushButton {  border-color: white; border-width: 0px; padding: 10px; border-style: outset; border-radius: 6px; background-color : #3375ec; font-weight: regular; font-size: 8pt; font-family: Seoge UI; color: white; }')#setStyleSheet('QPushButton {  background-color : #92374d; color: white; }')
        
  
        
        modellabel = QLabel('Main Model Files ', self)
        modellabel.setFont(QFont('Segoe UI', 10))
    
  
        modelpathtextbox = QLineEdit(self, placeholderText='Model file 1')
        modelpathtextbox.setFont(QFont('Segoe UI', 8))
        modelpathtextbox.resize(600,30)
        
        
        modelfilebrowse = QPushButton('Select', self)
        modelfilebrowse.clicked.connect(self.model_file)
        modelfilebrowse.setToolTip('Select the Model File')
        modelfilebrowse.setStyleSheet('QPushButton {  border-color: white; border-width: 0px; padding: 10px; border-style: outset; border-radius: 6px; background-color : #3375ec; font-weight: regular; font-size: 8pt; font-family: Seoge UI; color: white; }')#setStyleSheet('QPushButton {  background-color : #92374d; color: white; }')
        
        modelpathtextbox2 = QLineEdit(self, placeholderText='Model file 2')
        modelpathtextbox2.setFont(QFont('Segoe UI', 8))
        modelpathtextbox2.resize(600,30)
        
        
        modelfilebrowse2 = QPushButton('Select', self)
        modelfilebrowse2.clicked.connect(self.model_file2)
        modelfilebrowse2.setToolTip('Select the Model File')
        modelfilebrowse2.setStyleSheet('QPushButton {  border-color: white; border-width: 0px; padding: 10px; border-style: outset; border-radius: 6px; background-color : #3375ec; font-weight: regular; font-size: 8pt; font-family: Seoge UI; color: white; }')#setStyleSheet('QPushButton {  background-color : #92374d; color: white; }')
        
        
        savepredictedlabel = QLabel('Save Predicted Images  ', self)
        savepredictedlabel.setFont(QFont('Segoe UI', 10))
        
        
        
        
        othermodelfileslabel = QLabel('Extra Model Files ', self)
        othermodelfileslabel.setFont(QFont('Segoe UI', 10))
        
        classifierbrowsetextbox = QLineEdit(self, placeholderText='Classifier')
        classifierbrowsetextbox.setFont(QFont('Segoe UI', 8))
        classifierbrowsetextbox.resize(200,30)
        
        classifierbrowse = QPushButton('Select', self)
        classifierbrowse.clicked.connect(self.classBrows)
        classifierbrowse.setToolTip('Select the classifier .sav File')
        classifierbrowse.setStyleSheet('QPushButton {  border-color: white; border-width: 0px; padding: 10px; border-style: outset; border-radius: 6px; background-color : #3375ec; font-weight: regular; font-size: 8pt; font-family: Seoge UI; color: white; }')#setStyleSheet('QPushButton {  background-color : #92374d; color: white; }')
    
        scalerbrowsetextbox = QLineEdit(self, placeholderText='Scaler')
        scalerbrowsetextbox.setFont(QFont('Segoe UI', 8))
        scalerbrowsetextbox.resize(200,30)
    
        scalerbrowse = QPushButton('Select', self)
        scalerbrowse.clicked.connect(self.scalerBrows)
        scalerbrowse.setToolTip('Select the scaler .sav File')
        scalerbrowse.setStyleSheet('QPushButton {  border-color: white; border-width: 0px; padding: 10px; border-style: outset; border-radius: 6px; background-color : #3375ec; font-weight: regular; font-size: 8pt; font-family: Seoge UI; color: white; }')#setStyleSheet('QPushButton {  background-color : #92374d; color: white; }')
        
        featurebrowse1textbox = QLineEdit(self, placeholderText='Feature 1')
        featurebrowse1textbox.setFont(QFont('Segoe UI', 8))
        featurebrowse1textbox.resize(200,30)
        
        featurebrowse1 = QPushButton('Select', self)
        featurebrowse1.clicked.connect(self.featureBrows)
        featurebrowse1.setToolTip('Select the feature file')
        featurebrowse1.setStyleSheet('QPushButton {  border-color: white; border-width: 0px; padding: 10px; border-style: outset; border-radius: 6px; background-color : #3375ec; font-weight: regular; font-size: 8pt; font-family: Seoge UI; color: white; }')#setStyleSheet('QPushButton {  background-color : #92374d; color: white; }')
        
        featurebrowse2textbox = QLineEdit(self, placeholderText='Feature 2')
        featurebrowse2textbox.setFont(QFont('Segoe UI', 8))
        featurebrowse2textbox.resize(200,30)
        
        featurebrowse2 = QPushButton('Select', self)
        featurebrowse2.clicked.connect(self.featureBrows2)
        featurebrowse2.setToolTip('Select the feature file')
        featurebrowse2.setStyleSheet('QPushButton {  border-color: white; border-width: 0px; padding: 10px; border-style: outset; border-radius: 6px; background-color : #3375ec; font-weight: regular; font-size: 8pt; font-family: Seoge UI; color: white; }')#setStyleSheet('QPushButton {  background-color : #92374d; color: white; }')
        
        
  
        savepredictedtextbox = QLineEdit(self, placeholderText='Save Predicted Images folder')
        savepredictedtextbox.setFont(QFont('Segoe UI', 8))
        savepredictedtextbox.resize(600,30)
        
        
        savepredictedbrowse = QPushButton('Select', self)
        savepredictedbrowse.clicked.connect(self.savepredicted_image)
        savepredictedbrowse.setToolTip('Select the Save Predicted Images Folder')
        savepredictedbrowse.setStyleSheet('QPushButton {  border-color: white; border-width: 0px; padding: 10px; border-style: outset; border-radius: 6px; background-color : #3375ec; font-weight: regular; font-size: 8pt; font-family: Seoge UI; color: white; }')#setStyleSheet('QPushButton {  background-color : #92374d; color: white; }')
        
        
        Test = QPushButton('Test', self)
        Test.clicked.connect(self.testing)
        Test.setToolTip('Test the model')
        Test.setStyleSheet('QPushButton {  border-color: black; border-width: 0.2px; padding: 14px; border-style: outset; border-radius: 20px; background-color : lightgray; font-weight: regular; font-size: 12pt; font-family: Seoge UI; color: black; } }')
       
       
       
       
        hlayout2.addWidget(testimageslabel)
        hlayout2.addWidget(testimagespathtextbox)
        hlayout2.addWidget(testimagespathbrowse)
        
        hlayout3.addWidget(modellabel)
        hlayout3.addWidget(modelpathtextbox)
        hlayout3.addWidget(modelfilebrowse)
        

        hlayout3.addWidget(modelpathtextbox2)
        hlayout3.addWidget(modelfilebrowse2)
               
        hlayout4.addWidget(savepredictedlabel)
        hlayout4.addWidget(savepredictedtextbox)
        hlayout4.addWidget(savepredictedbrowse)
      
        
        hlayout5.addWidget(othermodelfileslabel)
        
        hlayout5.addWidget(classifierbrowsetextbox)
        hlayout5.addWidget(classifierbrowse)
        
        hlayout5.addWidget(scalerbrowsetextbox)
        hlayout5.addWidget(scalerbrowse)
        
        hlayout5.addWidget(featurebrowse1textbox)
        hlayout5.addWidget(featurebrowse1)
        
        hlayout5.addWidget(featurebrowse2textbox)
        hlayout5.addWidget(featurebrowse2)
        
        
        vlayout4.addLayout(hlayout2)
        vlayout4.addLayout(hlayout3)
        vlayout4.addLayout(hlayout5)
        vlayout4.addLayout(hlayout4)
        
        
        
        vlayout4.addWidget(QWidget())
        vlayout4.addWidget(QWidget())
        vlayout4.addWidget(QWidget())
        
        
        vlayout4.addWidget(Test)
        
        
        self.setLayout(vlayout4)
        
        
        
       
    @pyqtSlot()
    def savepredicted_image(self):
        global savepredictedtextbox
        print("select predicted images folder")
        savepredictedtextbox.setText('')
        foldername = QFileDialog.getExistingDirectory(self, 'Select Folder')
        savepredictedtextbox.setText(foldername)
        
    def test_image(self):
        global testimagespathtextbox
        print("select test images folder")
        testimagespathtextbox.setText('')
        foldername = QFileDialog.getExistingDirectory(self, 'Select Folder')
        testimagespathtextbox.setText(foldername)
    
    def model_file(self):
        global modelpathtextbox
        print("select model file")
        #modelpathtextbox.setText("")
        filename = QFileDialog.getOpenFileName(self, 'Select File')
        modelpathtextbox.setText(filename[0])
    
    def model_file2(self):
        global modelpathtextbox2
        print("select model file")
        #modelpathtextbox.setText("")
        filename = QFileDialog.getOpenFileName(self, 'Select File')
        modelpathtextbox2.setText(filename[0])
    
    def classBrows(self):
        global classifierbrowsetextbox
        #print("select model file")
        #modelpathtextbox.setText("")
        filename = QFileDialog.getOpenFileName(self, 'Select File')
        classifierbrowsetextbox.setText(filename[0])
    
    def scalerBrows(self):
        global scalerbrowsetextbox
        #print("select model file")
        #modelpathtextbox.setText("")
        filename = QFileDialog.getOpenFileName(self, 'Select File')
        scalerbrowsetextbox.setText(filename[0])
    
    def featureBrows(self):
        global featurebrowse1textbox
        #print("select model file")
        #modelpathtextbox.setText("")
        filename = QFileDialog.getOpenFileName(self, 'Select File')
        featurebrowse1textbox.setText(filename[0])
        
    def featureBrows2(self):
        global featurebrowse2textbox
        #print("select model file")
        #modelpathtextbox.setText("")
        filename = QFileDialog.getOpenFileName(self, 'Select File')
        featurebrowse2textbox.setText(filename[0])
        
    
    def testing(self):
        global testimagespathtextbox,modelpathtextbox, modelpathtextbox2, savepredictedtextbox, classifierbrowsetextbox, scalerbrowsetextbox, featurebrowse1textbox, featurebrowse2textbox

        unet_test = """
        
        
import multiprocessing as mp
mp.freeze_support()
import numpy as np 
from numba import cuda
import os, glob, sys
from framework.layer import *
from framework.pixelhop import *
from utils import loadImage
import imageio
from division import div
import timeit
import xgboost as xgb
from sklearn import preprocessing
import pickle
import getopt
from sklearn.feature_selection import f_classif, SelectPercentile
from datetime import timedelta
from skimage import io
import numpy as np 
from numba import cuda
import os, glob
from framework.layer import *
from framework.pixelhop import *
from utils import loadImage
import imageio
from division import div
import timeit
import xgboost as xgb
from sklearn import preprocessing
import pickle
from sklearn.feature_selection import f_classif, SelectPercentile
from datetime import timedelta
import sys
import multiprocessing as mp
from multiprocessing.pool import Pool
from multiprocessing import freeze_support
import pickle


numCPUCoresToUse = mp.cpu_count() - 1 if mp.cpu_count() > 1 else 1 

SAVE={}
img_size = 1024
patch_size = 64#16
delta_x = patch_size # non overlap




patch_ind = 0

def run(argv):
    try:
        opts, args = getopt.getopt(argv, "hd: r: m: c: s: g: e: v: n:",["rfile = ","mfile = ","cfile = ","sfile = ", "gfile = ", "efile =", "vfile = ", "nfile = "])
        #opts, args = getopt.getopt(argv, "hd: r: g: c: e: v: n:",["rfile = ","gfile = ","cfile = ","efile = ", "vfile = ", "nfile = "])
    except getopt.GetoptError:
        print("error")
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == "-r":
            print(arg)
            test_img_addrs = arg
        
        elif opt in ("-m", "--mfile"):
            
            model1 = arg
            print(model1)
        
        elif opt in ("-c", "--cfile"):
            
            model2 = arg  
            print(model2)
        
        elif opt in ("-s", "--sfile"):
            savePath = arg
        
        elif opt in ("-g", "--gfile"):
            print(arg)
            file = open(arg, 'rb')
            clfFile = pickle.load(file)
            
        elif opt in ("-e", "--efile"):
            print(arg)
            file = open(arg, 'rb')
            scalerFile = pickle.load(file)
            
            
        elif opt in ("-v", "--vfile"):
            print(arg)
            file = open(arg, 'rb')
            arg2 = pickle.load(file)
            fs1File = arg2
            
        elif opt in ("-n", "--nfile"):
            print(arg)
            file = open(arg, 'rb')
            arg2 = pickle.load(file)
            fs2File = arg2
                    
        else:
            print("something is incorrect")
            sys.exit()

    clf = clfFile
    scaler1 = scalerFile
    fs1 = fs1File
    fs2 = fs2File

        
    print("----------------------TESTING-------------------------")

    test_img_patch, test_img_patch_list = [], []
    count = 0
    basePath = test_img_addrs
    test_img_addrs = os.listdir(test_img_addrs)
    print(test_img_addrs)
    for test_img_addr in test_img_addrs:
        if 'raw' in test_img_addr: # only want to load testing images that are raw
            count += 1
            print('Processing {}............'.format(test_img_addr))

            img = loadImage(basePath + "//" + test_img_addr) 

            if len(img.shape) != 3:
                img = np.expand_dims(img, axis=2)

            # Initialzing
            predict_0or1 = np.zeros((img_size, img_size, 2))

            predict_mask = np.zeros(img.shape)

            # Create patches for test image
            for i in range(0, img_size, delta_x):
                for j in range(0, img_size, delta_x):
                    if i+patch_size <= img_size and j+patch_size <= img_size:
                        test_img_patch = img[i:i+patch_size, j:j+patch_size, :]

                    elif i+patch_size > img_size and j+patch_size <= img_size:
                        temp_size = img[i:, j:j+patch_size, :].shape
                        test_img_patch = np.lib.pad(img[i:, j:j+patch_size, :], ((0,patch_size-temp_size[0]),(0,0), (0,0)), 'edge')

                    elif i+patch_size <= img_size and j+patch_size > img_size:
                        temp_size = img[i:i+patch_size, j:, :].shape
                        test_img_patch = np.lib.pad(img[i:i+patch_size, j:, :], ((0,0),(0,patch_size-temp_size[1]), (0,0)), 'edge')

                    else: 
                        temp_size = img[i:, j:, :].shape
                        test_img_patch = np.lib.pad(img[i:, j:, :], ((0,patch_size-temp_size[0]),(0,patch_size-temp_size[1]), (0,0)), 'edge')
                    assert (test_img_patch.shape[0], test_img_patch.shape[1]) == (patch_size,patch_size)
                    
                    test_img_patch_list.append(test_img_patch)
                    

                    # convert list to numpy
                    test_img_subpatches = np.asarray(test_img_patch_list)
                    print(test_img_subpatches.shape)

                    
                    ################################################## PIXELHOP UNIT 1 ####################################################
        
                    test_feature1=PixelHop_Unit_GPU(test_img_subpatches, dilate=1, pad='reflect', weight_name= model1, getK=0, energypercent=.98)

                    ################################################# PIXELHOP UNIT 2 ####################################################
                    test_featurem1 = MaxPooling(test_feature1)
                    test_feature2=PixelHop_Unit_GPU(test_featurem1, dilate=1, pad='reflect', weight_name= model2, getK=0, energypercent=.98)
                    
    
                    test_feature_reduce_unit1 = test_feature1 
                    test_feature_reduce_unit2 = myResize(test_feature2, test_img_subpatches.shape[1], test_img_subpatches.shape[2])
                    print(test_feature_reduce_unit1.shape)
                    print(test_feature_reduce_unit2.shape)

                    
                    test_feature_unit1 = test_feature_reduce_unit1.reshape(test_feature_reduce_unit1.shape[0]*test_feature_reduce_unit1.shape[1]*test_feature_reduce_unit1.shape[2], -1)
                    test_feature_unit2 = test_feature_reduce_unit2.reshape(test_feature_reduce_unit2.shape[0]*test_feature_reduce_unit2.shape[1]*test_feature_reduce_unit2.shape[2], -1)
                    
                    print(test_feature_unit1.shape)
                    print(test_feature_unit2.shape)
    
                    
                    #--------lag unit--------------
                    test_feature_list_unit1, test_feature_list_unit2, test_feature_list_unit3, test_feature_list_unit4 = [], [], [], []
                    
                    for k in range(patch_size):
                        for l in range(patch_size):
                            ######################################
                            # get features
                            feature = np.array([])
                            # patch_ind = (div(k,patch_size))*(div(patch_size,patch_size)) + div(l,patch_size) # int div
                            # subpatch_ind = (div(k,subpatch_size))*(div(patch_size,subpatch_size)) + div(l,subpatch_size) # int div
                            feature = np.append(feature, test_img_patch[k,l,:])
                            feature = np.append(feature, [div(patch_size,2) - abs(i+k-div(patch_size,2)), div(patch_size,2) - abs(j+l-div(patch_size,2))]) #int div
                            
                            # feature = np.append(feature, test_feature12[0,:]) #takes only first few comps
                            feature1 = np.append(feature, test_feature_unit1[patch_ind,:])
                            feature2 = np.append(feature, test_feature_unit2[patch_ind,:])

                            test_feature_list_unit1.append(feature1)
                            test_feature_list_unit2.append(feature2)
                    

                    feature_list_unit1 = np.array(test_feature_list_unit1)
                    feature_list_unit2 = np.array(test_feature_list_unit2)

                    print(feature_list_unit1.shape)
                    print(feature_list_unit2.shape)


                    test_feature_red1= fs1.transform(feature_list_unit1)
                    test_feature_red2= fs2.transform(feature_list_unit2)


                    test_concat_features  =np.concatenate((test_feature_red1, test_feature_red2), axis=1)
                    print(test_concat_features.shape)
                    

                    feature_test = scaler1.transform(test_concat_features)
                    print(feature_test.shape)
                    

                    pre_list = clf.predict(feature_test)
                    print(pre_list.shape)
                    print(np.unique(pre_list))

                    # Generate predicted result
                    for k in range(patch_size):
                        for l in range(patch_size):
                            if i+k >= img_size or j+l >= img_size: break

                            # Binary
                            if pre_list[k*patch_size + l] > 0.5:
                                predict_0or1[i+k, j+l, 1] += 1
                            else:
                                predict_0or1[i+k, j+l, 0] += 1

                            # Multi-class
                            # if pre_list[k*patch_size + l] == 85.0:
                            #     predict_0or1[i+k, j+l, 1] += 1
                            # if pre_list[k*patch_size + l] == 170.0:
                            #     predict_0or1[i+k, j+l, 2] += 1
                            # if pre_list[k*patch_size + l] == 255.0:
                            #     predict_0or1[i+k, j+l, 3] += 1
                            # else:
                            #     predict_0or1[i+k, j+l, 0] += 1

            print('*************************************************************************************')
            print('one predicted mask')
            predict_mask = np.argmax(predict_0or1, axis=2)
            
            imageName = savePath + os.path.basename(test_img_addr)
            print(imageName)
            imageio.imsave(imageName, (predict_mask * 255).astype(np.uint8))
            del predict_0or1, predict_mask

    
    #stop_test = timeit.default_timer()

    #print('Total Time: ' + str(timedelta(seconds=stop_test-start_test)))
    #f = open('{}test_time.txt'.format(pathToResultSave),'w+')
    #f.write('Total Time: ' + str(timedelta(seconds=stop_test-start_test)))
    #f.close()

@cuda.jit
def GPU_Feature(d_img_patch_list, d_train_feature_unit1, d_train_feature_unit2, d_feature_list_unit1, d_feature_list_unit2, d_threadDimensions, delta_x, patch_size, depth1, depth2):
	threadIdx = cuda.grid(1)
	t, i, j, k, l, d = indices6(threadIdx, d_threadDimensions)
	if t < d_threadDimensions[0]:
		i3 = d_img_patch_list.shape[3]
		patch_ind = t * (img_size//delta_x) * (img_size//delta_x) + i * (img_size//delta_x) + j
		if d < i3:
			d_feature_list_unit1[t, i, j, k, l, d] = d_img_patch_list[patch_ind,k,l,d]
			d_feature_list_unit2[t, i, j, k, l, d] = d_img_patch_list[patch_ind,k,l,d]
		elif d == i3:
			d_feature_list_unit1[t, i, j, k, l, d] = patch_size//2 - abs(i*delta_x+k-patch_size//2)
			d_feature_list_unit2[t, i, j, k, l, d] = patch_size//2 - abs(i*delta_x+k-patch_size//2)
		elif d == i3 + 1:
			d_feature_list_unit1[t, i, j, k, l, d] = patch_size//2 - abs(j*delta_x+l-patch_size//2)
			d_feature_list_unit2[t, i, j, k, l, d] = patch_size//2 - abs(j*delta_x+l-patch_size//2)
		elif d < depth1 and depth2:
			d_feature_list_unit1[t, i, j, k, l, d] = d_train_feature_unit1[patch_ind, d - (i3 + 2)]
			d_feature_list_unit2[t, i, j, k, l, d] = d_train_feature_unit2[patch_ind, d - (i3 + 2)]
		elif d < depth2:
			d_feature_list_unit2[t, i, j, k, l, d] = d_train_feature_unit2[patch_ind, d - (i3 + 2)]

@cuda.jit(device=True)
def indices6(m, threadDimensions):
    t = m // (threadDimensions[1] * threadDimensions[2] * threadDimensions[3] * threadDimensions[4] * threadDimensions[5])
    m -= t * (threadDimensions[1] * threadDimensions[2] * threadDimensions[3] * threadDimensions[4] * threadDimensions[5])
    i = m // (threadDimensions[2] * threadDimensions[3] * threadDimensions[4] * threadDimensions[5])
    m -= i * (threadDimensions[2] * threadDimensions[3] * threadDimensions[4] * threadDimensions[5])
    j = m // (threadDimensions[3] * threadDimensions[4] * threadDimensions[5])
    m -= j * (threadDimensions[3] * threadDimensions[4] * threadDimensions[5])
    k = m // (threadDimensions[4] * threadDimensions[5])
    m -= k * (threadDimensions[4] * threadDimensions[5])
    l = m // (threadDimensions[5])
    m -= l * (threadDimensions[5])
    d = m
    return t, i, j, k, l, d


run(sys.argv[1:])        
      
      
        """
        
        
        
        with open(resource_path('src/testing/pixelHopTest.py'),'w') as f:
            f.write(unet_test)
        
        # with open(resource_path('log.txt'),'w') as f:
        #     f.write('Numbar of epochs: ' + str(noofepochs.text()))
        
        
        
        # self.thread = Thread()
        # self.thread._signal.connect(self.signal_accept)
        # self.thread.start()
        # Train.setEnabled(False)
        # # pbar.setValue(0)
        print(pyPath+ "python.exe" + " \"src/testing/pixelHopTest.py\" " +"-r \""+ testimagespathtextbox.text() +"\""+ " -m \""+ 
              modelpathtextbox.text()+"\""+" -c \""+ modelpathtextbox2.text()+"\""+" -s \""+savepredictedtextbox.text()+"\""
              +" -g \""+classifierbrowsetextbox.text() +"\""
              +" -e \""+scalerbrowsetextbox.text()+"\""
              +" -v \""+featurebrowse1textbox.text()+"\""
              +" -n \""+featurebrowse2textbox.text()+"\""
              )
        
        
        
        files = glob.glob('src/testing/*.*')
        print(files)
        for f in files:
            if f.endswith('pixelHopTest.py'):
                # print(pyPath+ "python.exe" + " \"algorithms/3D UNET.py\" -r \""+rawimagepathtextbox.text()+"\" -g \""+groundtruthtextbox.text()+"\" -c \""+noofclasses.text()+"\" -e \""+noofepochs.text()+"\"")
                proc = subprocess.Popen(pyPath+ "python.exe" + " \"src/testing/pixelHopTest.py\" " +
                                        "-r \""+ testimagespathtextbox.text() +"\""+ 
                                        " -m \""+ modelpathtextbox.text()+"\""+
                                        " -c \""+ modelpathtextbox2.text()+"\""+
                                        " -s \""+savepredictedtextbox.text()+"\""
                                        +" -g \""+classifierbrowsetextbox.text() +"\""
                                        +" -e \""+scalerbrowsetextbox.text()+"\""
                                        +" -v \""+featurebrowse1textbox.text()+"\""
                                        +" -n \""+featurebrowse2textbox.text()+"\"")
                proc.wait()
                # os.remove(resource_path('algorithms/3D UNET_test.py'))
            # elif f.split('\\')[1] == '2D UNET_test.py':
            #     proc = subprocess.Popen(pyPath+ "python.exe" + " \"algorithms/2D UNET_test.py\" -r \""+rawimagepathtextbox.text()+"\" -g \""+groundtruthtextbox.text()+"\" -c \""+noofclasses.text()+"\" -e \""+noofepochs.text()+"\"")
            else:
                print('no script')
        
        
class About_tab(QWidget):
    def __init__(self):
        super().__init__()
        aboutLayout = QGridLayout()
        logoPic = QPixmap(resource_path("utd logo circular.png"))
        label = QLabel(self)
        label.setPixmap(logoPic)
        # label.setText("Created by Aayan Rahat and Vinay Kadam, Researchers at UTD and the Ding Incubator")
        label.setAlignment(QtCore.Qt.AlignCenter)
        aboutLayout.addWidget(label)
        message = QLabel(
            'Reconstruct-Net: An Open Source Platform Created by Aayan Rahat and Vinay Kadam, Researchers at UTD and the Ding Incubator')
        message.setAlignment(QtCore.Qt.AlignCenter)
        aboutLayout.addWidget(message)
        message2 = QLabel('For all code and latest revisions, please visit our Github')
        message2.setAlignment(QtCore.Qt.AlignCenter)
        aboutLayout.addWidget(message2)
        self.setLayout(aboutLayout)

class Help_tab(QWidget):
    def __init__(self):
        super().__init__()
        
        helpLayout =  QGridLayout()
        helpButton = QPushButton('Help', self)
        helpButton.clicked.connect(self.on_click_helpButton)
        helpButton.setToolTip('Click here for help and our github')
        helpButton.setStyleSheet('QPushButton {  border-color: white; border-width: 0px; padding: 10px; border-style: outset; border-radius: 6px; background-color : #3375ec; font-weight: regular; font-size: 8pt; font-family: Seoge UI; color: white; }')#setStyleSheet('QPushButton {  background-color : #92374d; color: white; }')
        
        helpLayout.addWidget(helpButton)
        self.setLayout(helpLayout)
    def on_click_helpButton(self):
         urL='http://github.com/the905/R-NET'
         #chrome_path ="C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe %s"
         #webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(chrome_path))
         webbrowser.open_new_tab(urL)


class Export_tab(QWidget):
    
    def __init__(self):
        super().__init__()
        global rawimagepathtextbox, groundtruthtextbox, algorithms, lossfunctions, segmentedimagespathtextbox, spacing, height, width, useVTI, exportAlgo
        
        useVTI = True
        
        hlayout4 = QHBoxLayout()
        vlayout5 = QVBoxLayout()
        
        Segmentedimageslabel = QLabel('Segmented Images ', self)
        Segmentedimageslabel.setFont(QFont('Segoe UI', 10))
    
  
        segmentedimagespathtextbox = QLineEdit(self, placeholderText='Segmented images folder')
        segmentedimagespathtextbox.setFont(QFont('Segoe UI', 8))
        segmentedimagespathtextbox.resize(600,30)
        
        spacinglabel = QLabel('Spacing (mm)', self)
        spacinglabel.setFont(QFont('Segoe UI', 10))
    
  
        spacing = QLineEdit(self)
        spacing.setFont(QFont('Segoe UI', 8))
        spacing.resize(30,30)
        #spacing.setText('0.00375')
        
        heightlabel = QLabel('Height', self)
        heightlabel.setFont(QFont('Segoe UI', 10))
    
  
        height = QLineEdit(self)
        height.setFont(QFont('Segoe UI', 8))
        height.resize(30,30)
        #height.setText('512')
        
        widthlabel = QLabel('Width', self)
        widthlabel.setFont(QFont('Segoe UI', 10))
    
  
        width = QLineEdit(self)
        width.setFont(QFont('Segoe UI', 8))
        width.resize(30,30)
        #width.setText('512')
        
        
        exportAlgoLabel = QLabel('VTI Method', self)
        exportAlgoLabel.setFont(QFont('Segoe UI', 7))
        
        exportAlgo= QCheckBox('', self)
        exportAlgo.stateChanged.connect(self.checked)
        exportAlgo.setChecked(useVTI)
        
        segmentedimagespathbrowse = QPushButton('Select', self)
        segmentedimagespathbrowse.clicked.connect(self.segmented_images)
        segmentedimagespathbrowse.setToolTip('Select the Segmented Images Folder')
        segmentedimagespathbrowse.setStyleSheet('QPushButton {  border-color: white; border-width: 0px; padding: 10px; border-style: outset; border-radius: 6px; background-color : #3375ec; font-weight: regular; font-size: 8pt; font-family: Seoge UI; color: white; }')#setStyleSheet('QPushButton {  background-color : #92374d; color: white; }')
        

        
        
        
        
  
        export = QPushButton('Export 3D', self)
        export.clicked.connect(self.export)
        export.setToolTip('Export the 3D model')
        export.setStyleSheet('QPushButton {  border-color: black; border-width: 0.2px; padding: 14px; border-style: outset; border-radius: 20px; background-color : lightgray; font-weight: regular; font-size: 12pt; font-family: Seoge UI; color: black; } }')
       
        # export4 = QPushButton('Export 4D', self)
        # export4.clicked.connect(self.export)
        # export4.setToolTip('Export the 4D model')
        # export4.setStyleSheet('QPushButton {  border-color: black; border-width: 0.2px; padding: 14px; border-style: outset; border-radius: 20px; background-color : lightgray; font-weight: regular; font-size: 12pt; font-family: Seoge UI; color: black; } }')
       
        hlayout4.addWidget(Segmentedimageslabel)
        hlayout4.addWidget(segmentedimagespathtextbox)
        hlayout4.addWidget(segmentedimagespathbrowse)
       
        hlayout4.addWidget(spacinglabel)
        hlayout4.addWidget(spacing)
        
        hlayout4.addWidget(heightlabel)
        hlayout4.addWidget(height)
        
        hlayout4.addWidget(widthlabel)
        hlayout4.addWidget(width)
        
        hlayout4.addWidget(exportAlgoLabel)
        hlayout4.addWidget(exportAlgo)
       
        vlayout5.addLayout(hlayout4)
       
        vlayout5.addWidget(QWidget())
        vlayout5.addWidget(QWidget())
        vlayout5.addWidget(QWidget())
        vlayout5.addWidget(export)
        
        
        self.setLayout(vlayout5)
  
    @pyqtSlot()
    def segmented_images(self):
        global segmentedimagespathtextbox
        print('selected segmented images folder')
        segmentedimagespathtextbox.setText('')
        foldername = QFileDialog.getExistingDirectory(self, 'Select Folder')
        segmentedimagespathtextbox.setText(foldername)
    
    def checked(self):
        if exportAlgo.isChecked():
            useVTI = True
            print(useVTI)
            print("VTI is being used")
            
        else:
            useVTI = False   
            print ("VTI is not being used")  

        
        
    def export(self):
        print("export 3D")
        
        export_vti = """
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
import multiprocessing
from multiprocessing import Process, freeze_support



def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hd: s: g: a: b:",["sfile = ", "gfile = ", "afile = ", "bfile = "])
    except getopt.GetoptError:
        print("Pythonfile.py -s <segmentedimages>")
        sys.exit(2)
    
      
    for opt, arg in opts:
        if opt == "-s":
            path = arg 
        elif opt in ("-g", "--gfile"):
            print(arg + " val" )
            spacing = arg
        elif opt in ("-a", "--afile"):
            print(arg + " val" )
            height = arg
        elif opt in ("-b", "--bfile"):
            print(arg + " val" )
            width = arg
     
        else:
            print("Pythonfile.py -s <segmentedimages>")
            sys.exit()
        
    IMG_HEIGHT = 64

    path = path

    dimensions = IMG_HEIGHT 

    finalDirectory = resource_path('')
    finalName = "unity"
    extension = "vti"
    finalDirectory = finalDirectory + "" + finalName

   

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

   


    finDir = finalDirectory
    data.spacing = (int(height), int(width), float(spacing) )
    finDir += ( "." + extension )
    data.save(finDir)
    print("Your file has been saved to the following location {}".format(finDir))
    
if __name__ == "__main__":
      freeze_support()
      main(sys.argv[1:])
        
        """

        with open(resource_path('bin/export_vti.py'),'w') as f:
            if(exportAlgo.isChecked()):
                useVTI = True
                f.write(export_vti)
                print("writing vti")
            else:
                useVTI =  False    
        
        # with open(resource_path('log.txt'),'w') as f:
        #     f.write('Numbar of epochs: ' + str(noofepochs.text()))
        
        
        
        # self.thread = Thread()
        # self.thread._signal.connect(self.signal_accept)
        # self.thread.start()
        # Train.setEnabled(False)
        # pbar.setValue(0)
        print(pyPath+ "python.exe" + " \""+resource_path('')+"bin/export_vti.py\" " +"-s \""+ segmentedimagespathtextbox.text() +"\"")
        
        
        
        files = glob.glob(resource_path('bin/*.*'))
        print(files)
        for f in files:
            if f.split('\\')[-1] == 'export_vti.py':
                print("its there")
                if(useVTI):
                    print("writing vti")
                    print(pyPath+ "python.exe" + " \""+resource_path('')+"bin/export_vti.py\" " +"-s \""+ segmentedimagespathtextbox.text() +"\" -g \""+spacing.text()+"\" -a \""+ height.text() + "\" -b \""+ width.text() +"\"")
                    proc = subprocess.Popen(pyPath+ "python.exe" + " \""+resource_path('')+"bin/export_vti.py\" " +"-s \""+ segmentedimagespathtextbox.text() +"\" -g \""+spacing.text()+"\" -a \""+ height.text() + "\" -b \""+ width.text() +"\"")
                    #proc = subprocess.Popen(pyPath+ "python.exe" + " \"src/algorithms/example.py\" -r \""+rawimagepathtextbox.text()+"\" -g \""+groundtruthtextbox.text()+"\" -c \""+noofclasses.text()+"\" -e \""+noofepochs.text()+"\" -v \"" + variance.text() +"\" -n \""+trainImgNumber.text()+"\"",shell= True ) 
                    proc.wait()
                    #os.remove(resource_path('')+"bin/export_vti.py")
            else:
                print('no script')
       
       
        docker_script = """
import os
import sys
import subprocess
import time


userprofile = os.environ['USERPROFILE']

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

if os.path.exists(resource_path('Slicer_docker.dll')):
    print('slicerdocker already install')
else:
    #install docker image and create container
    install = subprocess.Popen("docker run -d --name slicernotes -p 8888:8888 -p49053:49053 -v {}:/home/sliceruser/work --rm -ti slicer/slicer-notebook:latest".format(userprofile), shell = False)
    install.wait()
    time.sleep(2)
    
    #install jupyter nbconvert into docker
    install_nbconvert = subprocess.Popen('winpty docker exec -it slicernotes bash -c \"./Slicer-4.11.0-2020-05-11-linux-amd64/Slicer --launch PythonSlicer -m pip install jupyter nbconvert\"')
    install_nbconvert.wait()

    time.sleep(2)

    #make directory inside docker
    make_obj_directory = subprocess.Popen('winpty docker exec -it slicernotes bash -c \"mkdir -p obj\"')
    make_obj_directory.wait()

    time.sleep(2)
    with open(resource_path('Slicer_docker.dll'), 'w') as f:
        f.write("Slicer for Ding's Lab")

# copy files to docker
vti_filename = resource_path("unity.vti")
vtifile = "unity.vti"
jupyter_filename = resource_path("bin/main.ipynb")
jupyterfile = "main.ipynb"

print("docker cp {0} slicernotes:./home/sliceruser/{1}".format(vti_filename,vtifile))
vti_copy = subprocess.Popen("docker cp {0} slicernotes:/home/sliceruser/{1}".format(vti_filename,vtifile))
vti_copy.wait()

jupyter_copy = subprocess.Popen("docker cp {0} slicernotes:/home/sliceruser/{1}".format(jupyter_filename, jupyterfile))
jupyter_copy.wait()

time.sleep(2)

#run jupyter command inside docker
run_jupyter = subprocess.Popen('docker exec -it slicernotes bash -c \"jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute main.ipynb\"')
run_jupyter.wait()

# time.sleep(`0`)

# copy files from docker
obj_filename= resource_path("UIGeneratedModels")
obj_copy_docker = subprocess.Popen("docker cp slicernotes:/home/sliceruser/obj/. {}".format(obj_filename))
obj_copy_docker.wait()




"""


        docker_script_non_VTI = """



import os
import sys
import subprocess
import time
import getopt

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hd: s:",["sfile = "])
    except getopt.GetoptError:
        print("Pythonfile.py -s <segmentedimages>")
        sys.exit(2)
        
        
    for opt, arg in opts:
        if opt == "-s":
            path = arg 
        else:
            print("Pythonfile.py -s <segmentedimages>")
            sys.exit()

    userprofile = os.environ['USERPROFILE']

    def resource_path(relative_path):
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(os.path.abspath("."), relative_path)

    if os.path.exists(resource_path('Slicer_docker.dll')):
        print('slicerdocker already install')
    else:
        #install docker image and create container
        install = subprocess.Popen("docker run -d --name slicernotes -p 8888:8888 -p49053:49053 -v {}:/home/sliceruser/work --rm -ti slicer/slicer-notebook:latest".format(userprofile), shell = False)
        install.wait()
        time.sleep(2)

        
        #install jupyter nbconvert into docker
        install_nbconvert = subprocess.Popen('docker exec -it slicernotes bash -c \"./Slicer-4.11.0-2020-05-11-linux-amd64/Slicer --launch PythonSlicer -m pip install jupyter nbconvert\"')
        install_nbconvert.wait()
        install_nbconvert2 = subprocess.Popen('docker exec -it slicernotes bash -c \"./Slicer-4.11.0-2020-05-11-linux-amd64/Slicer --launch PythonSlicer -m pip install slicer\"')
        install_nbconvert2.wait()
        time.sleep(2)

        #make directory inside docker
        make_obj_directory = subprocess.Popen('docker exec -it slicernotes bash -c \"mkdir -p obj\"')
        make_obj_directory.wait()
        
        make_volume_directory = subprocess.Popen('docker exec -it slicernotes bash -c \"mkdir -p volumeFolder\"')
        make_volume_directory.wait()

        time.sleep(2)
        with open(resource_path('Slicer_docker.dll'), 'w') as f:
            f.write("Slicer for Ding's Lab")

    # copy files to docker
    volumeFolder = path + "/."
    jupyter_filename = resource_path("bin/mainNonVTI.ipynb")
    print(jupyter_filename)
    jupyterfile = "mainNonVTI.ipynb"
    folderName = "volumeFolder"
    print(volumeFolder)

    print("docker cp {0} slicernotes:/home/sliceruser/{1}".format(volumeFolder,folderName))
    folder_copy = subprocess.Popen("docker cp {0} slicernotes:/home/sliceruser/{1}".format(volumeFolder,folderName))
    folder_copy.wait()
    
    #changeOwnership = subprocess.Popen("chmod 777 {}".format(jupyter_filename) )

    jupyter_copy = subprocess.Popen("docker cp {0} slicernotes:/home/sliceruser/{1}".format(jupyter_filename, jupyterfile))
    jupyter_copy.wait()
    print("copied")
    
    time.sleep(2)

    #run jupyter command inside docker
    run_jupyter = subprocess.Popen('docker exec -it slicernotes bash -c \"jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute mainNonVTI.ipynb\"')
    run_jupyter.wait()

    # time.sleep(`0`)

    # # copy files from docker
    
    obj_filename= resource_path("UIGeneratedModels")
    


    obj_copy_docker = subprocess.Popen("docker cp slicernotes:/home/sliceruser/obj/. {}".format(obj_filename))
    obj_copy_docker.wait()




    # mtl_filename= "Segmentation.mtl"
    # mtl_copy_docker = subprocess.Popen("docker cp slicernotes:./home/sliceruser/obj/{0} {1}".format(mtl_filename, mtl_filename))

    # mtl_copy_docker.wait()

main(sys.argv[1:])


"""

        mainNonVTI = """



# import JupyterNotebooksLib as slicernb
import vtk
import slicer
import os
import glob
import getopt

# Clear scene
slicer.mrmlScene.Clear(False)


print("hello there")

path = '/home/sliceruser/volumeFolder/*.tif'
savePath = "obj"
fileNames = glob.glob(path)
print(fileNames)

for files in fileNames :
    # Load from local file
    imagename = str(files)
    print(imagename)
    # Clear scene
    slicer.mrmlScene.Clear(False)
    masterVolumeNode = slicer.util.loadVolume(imagename)
    outputVolumeSpacingMm = [height, width, depth]
    masterVolumeNode.SetSpacing(outputVolumeSpacingMm)


    # Create segmentation
    segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    nodename = "vol_{}".format(imagename)
        
    segmentationNode.SetName(nodename)
    slicer.modules.markups.widgetRepresentation().onRenameAllWithCurrentNameFormatPushButtonClicked()
    segmentationNode.CreateDefaultDisplayNodes() # only needed for display
    segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(masterVolumeNode)

    # Create temporary segment editor to get access to effects
    segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
    segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
    segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
    segmentEditorWidget.setSegmentationNode(segmentationNode)
    segmentEditorWidget.setMasterVolumeNode(masterVolumeNode)

    # Create segments by thresholding
    cellAll = []
    for i in range(1, 4):
        cellDic = []   
        cellDic.append("node_"+str(i))
        cellDic.append(i)
        cellDic.append(i)
        cellAll.append(cellDic)
    # segmentsFromHounsfieldUnits = [
    #     ["cell_1", 1, 1],
    #     ["cell_2", 2, 2],
    #     ["cell_3", 3, 3] ]
    segmentsFromHounsfieldUnits = cellAll

    for segmentName, thresholdMin, thresholdMax in segmentsFromHounsfieldUnits:
        # Create segment
        addedSegmentID = segmentationNode.GetSegmentation().AddEmptySegment(segmentName)
        segmentEditorNode.SetSelectedSegmentID(addedSegmentID)
        # Fill by thresholding
        segmentEditorWidget.setActiveEffectByName("Threshold")
        effect = segmentEditorWidget.activeEffect()
        effect.setParameter("MinimumThreshold",str(thresholdMin))
        effect.setParameter("MaximumThreshold",str(thresholdMax))
        effect.self().onApply()
        segmentId = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segmentName)
        print(segmentId)
        print(segmentName)
        if thresholdMin < 256:
            segmentationNode.GetSegmentation().GetSegment(segmentId).SetColor(thresholdMin/255,1-thresholdMin/255,0)
        elif thresholdMin > 255 and thresholdMin < 512:
            segmentationNode.GetSegmentation().GetSegment(segmentId).SetColor(0,(thresholdMin-255)/255,1-(thresholdMin-255)/255)
        elif thresholdMin > 511 and thresholdMin < 768:
            segmentationNode.GetSegmentation().GetSegment(segmentId).SetColor(1-(thresholdMin-511)/255,0,(thresholdMin-511)/255)

    # Delete temporary segment editor
    segmentEditorWidget = None
    slicer.mrmlScene.RemoveNode(segmentEditorNode)
    segmentationNode.CreateClosedSurfaceRepresentation()

    print("pre checkpoint")
   
    slicer.modules.segmentations.logic().ExportSegmentsClosedSurfaceRepresentationToFiles(savePath, segmentationNode, None,"OBJ")

print("post checkpoint")




"""
    
        tifLocation = ( segmentedimagespathtextbox.text() )
        saveFolderPath = "'" + segmentedimagespathtextbox.text() + "/models" + "'"
        
        #mainNonVTI = ("path = {}".format(tifLocation)) + '\n' + mainNonVTI
        
        #mainNonVTI = ("saveFolder = {}".format(saveFolderPath))+ '\n' + mainNonVTI
       
        
        global mainFile
        with open(resource_path('bin/export_model.py'),'w') as f:
            if(useVTI):
                f.write(docker_script)
            else:
                
                mainNonVTI = ("height = float({})".format(height.text())) + '\n' + mainNonVTI
                mainNonVTI = ("width = float({})".format(width.text())) + '\n' + mainNonVTI
                mainNonVTI = ("depth = float({})".format(spacing.text())) + '\n' + mainNonVTI
                print(mainNonVTI)
                f.write(docker_script_non_VTI)
                
                with open(resource_path('bin/mainNonVTI.py'),'w') as g:
                    g.write(mainNonVTI)
                    command = resource_path('bin/mainNonVTI.py') + " " + resource_path('bin/mainNonVTItransfer.ipynb')
                    time.sleep(3)
                    bob = subprocess.Popen("ipynb-py-convert {}".format(command))
    


                with open(resource_path('bin/main.ipynb'), mode = "r",  encoding= "utf-8" ) as g:
                    mainFile = json.loads(g.read())
                    time.sleep(3)
                    print(mainFile)
                    with open(resource_path('bin/mainNonVTItransfer.ipynb'), mode = "r",  encoding= "utf-8" ) as b:
                        codeFile = json.loads(b.read())
                        print(codeFile)
                        time.sleep(3)
                        mainFile['cells'][0]['source'] =  codeFile['cells'][0]['source']
                        print(mainFile)
                        time.sleep(5)

                with open(resource_path('bin/mainNonVTI.ipynb'), 'w') as outfile:
                    json.dump(mainFile, outfile)   
                                    
   
        print(pyPath+ "python.exe" + " \""+resource_path('')+"bin/export_model.py\"")
        
        files = glob.glob(resource_path('bin/*.*'))
        print(files)
        for f in files:
            print(f.split('\\')[-1])
            if f.split('\\')[-1] == 'export_model.py':
                print("its there")
                try:
                    shutil.move(resource_path('export files/vti/unity.vti'),resource_path('bin/unity.vti'))
                except:
                    print('no vti found')
                    
                if(useVTI == False):
                    proc = subprocess.Popen(pyPath+ "python.exe" + " \""+resource_path('')+"bin/export_model.py\" " 
                                            + " -s \""+tifLocation + "\"")
                    proc.wait()
                else:
                    proc = subprocess.Popen(pyPath+ "python.exe" + " \""+resource_path('')+"bin/export_model.py\"" )
                    proc.wait()
                #os.remove(resource_path('')+"bin/export_model.py")
                # os.remove(resource_path('')+"bin/unity.vti")
                try:
                    shutil.move(resource_path('Segmentation.obj'),resource_path('export files/models/Segmentation.obj'))
                    shutil.move(resource_path('Segmentation.mtl'),resource_path('export files/models/Segmentation.mtl'))
                except:
                    print('no files found')
                
            else:
                print('no script')
       
    


        
class TabBar(QTabBar):
    def tabSizeHint(self, index):
        s = QTabBar.tabSizeHint(self, index)
        s.transpose()
        return s

    def paintEvent(self, event):
        painter = QStylePainter(self)
        opt = QStyleOptionTab()

        for i in range(self.count()):
            self.initStyleOption(opt, i)
            painter.drawControl(QStyle.CE_TabBarTabShape, opt)
            painter.save()

            s = opt.rect.size()
            s.transpose()
            r = QtCore.QRect(QtCore.QPoint(), s)
            r.moveCenter(opt.rect.center())
            opt.rect = r

            c = self.tabRect(i).center()
            painter.translate(c)
            painter.rotate(90)
            painter.translate(-c)
            painter.drawControl(QStyle.CE_TabBarTabLabel, opt)
            painter.restore()

class VerticalTabWidget(QTabWidget):
    def __init__(self, *args, **kwargs):
        QTabWidget.__init__(self, *args, **kwargs)
        self.setTabBar(TabBar())
        self.setTabPosition(QtWidgets.QTabWidget.West)       
        
        
class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        
        
        self.setWindowTitle('Image Segmentation')
        self.setFixedWidth(1024)
        self.setFixedHeight(960)
        self.setWindowIcon(QIcon(str(resource_path('logo.png'))))
        # self.setGeometry(self.top, self.left, self.width, self.height)
        frameGm = self.frameGeometry()
        screen = PyQt5.QtWidgets.QApplication.desktop().screenNumber(PyQt5.QtWidgets.QApplication.desktop().cursor().pos())
        centerPoint = PyQt5.QtWidgets.QApplication.desktop().screenGeometry(screen).center()
        frameGm.moveCenter(centerPoint)
        self.move(frameGm.topLeft())
        layout = QGridLayout()
        self.setLayout(layout)
        # label1 = QLabel("Widget in Tab 1.")
        # label2 = QLabel("Widget in Tab 2.")
        tabwidget = VerticalTabWidget()
        tabwidget.update()
        tabwidget.setCursor(QCursor(Qt.PointingHandCursor))
        
        # tabwidget.setTabShape(QTabWidget.Rounded)
        # tabwidget.setTabPosition(QTabWidget.West)
        tabwidget.addTab(About_tab(), "About")
        tabwidget.addTab(Input_tab(), "Train/Test")
        tabwidget.addTab(Test_tab(), "Test")
        tabwidget.addTab(Export_tab(), "Export")
        #tabwidget.addTab(Help_tab(), "Help/Info")
        # tabwidget.setDocumentMode(True)
        tabwidget.setStyleSheet('''QTabBar::tab { height: 100px; width: 50px; font-size: 8pt; font-family: Segoe UI; font-weight: Bold;}
                                ''')
 
        layout.addWidget(tabwidget, 0, 0)
        
        
        self.show()
        
        
if __name__ == '__main__':
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    ex = MainApp()
   
    sys.exit(app.exec_())