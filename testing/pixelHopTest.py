## Changes in 120 and 122 depending on if list or just single image

import re
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
import pickle, subprocess


numCPUCoresToUse = mp.cpu_count() - 1 if mp.cpu_count() > 1 else 1 

SAVE={}
img_size = 1024
patch_size = 64#16
delta_x = patch_size # non overlap


def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


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
    #test_img_addrs = #os.listdir(test_img_addrs)
   # test_img_addrs = sorted_nicely(test_img_addrs)
    print(test_img_addrs)
    addrs = []
    addrs.append(basePath)
    for test_img_addr in addrs:
        if 'raw' in test_img_addr: # only want to load testing images that are raw
            count += 1
            print('Processing {}............'.format(test_img_addr))

            #img = loadImage(basePath + "//" + test_img_addr) 
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
        
                    test_feature1=PixelHop_Unit_GPU(test_img_subpatches, dilate=1, pad='reflect', weight_name= model1, getK=0, energypercent=.95)

                    ################################################# PIXELHOP UNIT 2 ####################################################
                    test_featurem1 = MaxPooling(test_feature1)
                    test_feature2=PixelHop_Unit_GPU(test_featurem1, dilate=1, pad='reflect', weight_name= model2, getK=0, energypercent=.95)
                    
    
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

                            # # Binary
                            # if pre_list[k*patch_size + l] > 0.5:
                            #     predict_0or1[i+k, j+l, 1] += 1
                            # else:
                            #     predict_0or1[i+k, j+l, 0] += 1

                            #  ##Multi-class
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

                            # Multi-class
                            if pre_list[k*patch_size + l] == 1:
                                predict_0or1[i+k, j+l, 1] += 1
                            if pre_list[k*patch_size + l] == 2:
                                predict_0or1[i+k, j+l, 2] += 1
                            if pre_list[k*patch_size + l] == 3:
                                predict_0or1[i+k, j+l, 3] += 1
                            else:
                                predict_0or1[i+k, j+l, 0] += 1

            print('*************************************************************************************')
            print('one predicted mask')
            predict_mask = np.argmax(predict_0or1, axis=2)
            
            imageName = savePath + r'/'+ os.path.basename(test_img_addr)
            print(imageName)
            imageio.imsave(imageName, (predict_mask).astype(np.uint8))
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
# import re
# import multiprocessing as mp
# mp.freeze_support()
# import numpy as np 
# from numba import cuda
# import os, glob, sys
# from framework.layer import *
# from framework.pixelhop import *
# from utils import loadImage
# import imageio
# from division import div
# import timeit
# import xgboost as xgb
# from sklearn import preprocessing
# import pickle
# import getopt
# from sklearn.feature_selection import f_classif, SelectPercentile
# from datetime import timedelta
# from skimage import io
# import numpy as np 
# from numba import cuda
# import os, glob
# from framework.layer import *
# from framework.pixelhop import *
# from utils import loadImage
# import imageio
# from division import div
# import timeit
# import xgboost as xgb
# from sklearn import preprocessing
# import pickle
# from sklearn.feature_selection import f_classif, SelectPercentile
# from datetime import timedelta
# import sys
# import multiprocessing as mp
# from multiprocessing.pool import Pool
# from multiprocessing import freeze_support
# import pickle, subprocess


# numCPUCoresToUse = mp.cpu_count() - 1 if mp.cpu_count() > 1 else 1 

# SAVE={}
# img_size = 800
# patch_size = 64#16
# delta_x = patch_size # non overlap


# def sorted_nicely( l ): 
#     """ Sort the given iterable in the way that humans expect.""" 
#     convert = lambda text: int(text) if text.isdigit() else text 
#     alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
#     return sorted(l, key = alphanum_key)


# patch_ind = 0

# def run(argv):
#     try:
#         opts, args = getopt.getopt(argv, "hd: r: m: c: s: g: e: v: n:",["rfile = ","mfile = ","cfile = ","sfile = ", "gfile = ", "efile =", "vfile = ", "nfile = "])
#         #opts, args = getopt.getopt(argv, "hd: r: g: c: e: v: n:",["rfile = ","gfile = ","cfile = ","efile = ", "vfile = ", "nfile = "])
#     except getopt.GetoptError:
#         print("error")
#         sys.exit(2)
        
#     for opt, arg in opts:
#         if opt == "-r":
#             print(arg)
#             test_img_addrs = arg
        
#         elif opt in ("-m", "--mfile"):
            
#             model1 = arg
#             print(model1)
        
#         elif opt in ("-c", "--cfile"):
            
#             model2 = arg  
#             print(model2)
        
#         elif opt in ("-s", "--sfile"):
#             savePath = arg
        
#         elif opt in ("-g", "--gfile"):
#             print(arg)
#             file = open(arg, 'rb')
#             clfFile = pickle.load(file)
            
#         elif opt in ("-e", "--efile"):
#             print(arg)
#             file = open(arg, 'rb')
#             scalerFile = pickle.load(file)
            
            
#         elif opt in ("-v", "--vfile"):
#             print(arg)
#             file = open(arg, 'rb')
#             arg2 = pickle.load(file)
#             fs1File = arg2
            
#         elif opt in ("-n", "--nfile"):
#             print(arg)
#             file = open(arg, 'rb')
#             arg2 = pickle.load(file)
#             fs2File = arg2
                    
#         else:
#             print("something is incorrect")
#             sys.exit()

#     clf = clfFile
#     scaler1 = scalerFile
#     fs1 = fs1File
#     fs2 = fs2File

        
#     print("----------------------TESTING-------------------------")

#     test_img_patch, test_img_patch_list = [], []
#     count = 0
#     basePath = test_img_addrs
#     test_img_addrs = os.listdir(test_img_addrs)
#     test_img_addrs = sorted_nicely(test_img_addrs)
#     print(test_img_addrs)
#     for test_img_addr in test_img_addrs:
#         if 'slice' in test_img_addr: # only want to load testing images that are raw
#             count += 1
#             print('Processing {}............'.format(test_img_addr))

#             img = loadImage(basePath + "//" + test_img_addr) 

#             if len(img.shape) != 3:
#                 img = np.expand_dims(img, axis=2)

#             # Initialzing
#             predict_0or1 = np.zeros((img_size, img_size, 2))

#             predict_mask = np.zeros(img.shape)

#             # Create patches for test image
#             for i in range(0, img_size, delta_x):
#                 for j in range(0, img_size, delta_x):
#                     if i+patch_size <= img_size and j+patch_size <= img_size:
#                         test_img_patch = img[i:i+patch_size, j:j+patch_size, :]

#                     elif i+patch_size > img_size and j+patch_size <= img_size:
#                         temp_size = img[i:, j:j+patch_size, :].shape
#                         test_img_patch = np.lib.pad(img[i:, j:j+patch_size, :], ((0,patch_size-temp_size[0]),(0,0), (0,0)), 'edge')

#                     elif i+patch_size <= img_size and j+patch_size > img_size:
#                         temp_size = img[i:i+patch_size, j:, :].shape
#                         test_img_patch = np.lib.pad(img[i:i+patch_size, j:, :], ((0,0),(0,patch_size-temp_size[1]), (0,0)), 'edge')

#                     else: 
#                         temp_size = img[i:, j:, :].shape
#                         test_img_patch = np.lib.pad(img[i:, j:, :], ((0,patch_size-temp_size[0]),(0,patch_size-temp_size[1]), (0,0)), 'edge')
#                     assert (test_img_patch.shape[0], test_img_patch.shape[1]) == (patch_size,patch_size)
                    
#                     test_img_patch_list.append(test_img_patch)
                    

#                     # convert list to numpy
#                     test_img_subpatches = np.asarray(test_img_patch_list)
#                     print(test_img_subpatches.shape)

                    
#                     ################################################## PIXELHOP UNIT 1 ####################################################
        
#                     test_feature1=PixelHop_Unit_GPU(test_img_subpatches, dilate=1, pad='reflect', weight_name= model1, getK=0, energypercent=.98)

#                     ################################################# PIXELHOP UNIT 2 ####################################################
#                     test_featurem1 = MaxPooling(test_feature1)
#                     test_feature2=PixelHop_Unit_GPU(test_featurem1, dilate=1, pad='reflect', weight_name= model2, getK=0, energypercent=.98)
                    
    
#                     test_feature_reduce_unit1 = test_feature1 
#                     test_feature_reduce_unit2 = myResize(test_feature2, test_img_subpatches.shape[1], test_img_subpatches.shape[2])
#                     print(test_feature_reduce_unit1.shape)
#                     print(test_feature_reduce_unit2.shape)

                    
#                     test_feature_unit1 = test_feature_reduce_unit1.reshape(test_feature_reduce_unit1.shape[0]*test_feature_reduce_unit1.shape[1]*test_feature_reduce_unit1.shape[2], -1)
#                     test_feature_unit2 = test_feature_reduce_unit2.reshape(test_feature_reduce_unit2.shape[0]*test_feature_reduce_unit2.shape[1]*test_feature_reduce_unit2.shape[2], -1)
                    
#                     print(test_feature_unit1.shape)
#                     print(test_feature_unit2.shape)
    
                    
#                     #--------lag unit--------------
#                     test_feature_list_unit1, test_feature_list_unit2, test_feature_list_unit3, test_feature_list_unit4 = [], [], [], []
                    
#                     for k in range(patch_size):
#                         for l in range(patch_size):
#                             ######################################
#                             # get features
#                             feature = np.array([])
#                             # patch_ind = (div(k,patch_size))*(div(patch_size,patch_size)) + div(l,patch_size) # int div
#                             # subpatch_ind = (div(k,subpatch_size))*(div(patch_size,subpatch_size)) + div(l,subpatch_size) # int div
#                             feature = np.append(feature, test_img_patch[k,l,:])
#                             feature = np.append(feature, [div(patch_size,2) - abs(i+k-div(patch_size,2)), div(patch_size,2) - abs(j+l-div(patch_size,2))]) #int div
                            
#                             # feature = np.append(feature, test_feature12[0,:]) #takes only first few comps
#                             feature1 = np.append(feature, test_feature_unit1[patch_ind,:])
#                             feature2 = np.append(feature, test_feature_unit2[patch_ind,:])

#                             test_feature_list_unit1.append(feature1)
#                             test_feature_list_unit2.append(feature2)
                    

#                     feature_list_unit1 = np.array(test_feature_list_unit1)
#                     feature_list_unit2 = np.array(test_feature_list_unit2)

#                     print(feature_list_unit1.shape)
#                     print(feature_list_unit2.shape)


#                     test_feature_red1= fs1.transform(feature_list_unit1)
#                     test_feature_red2= fs2.transform(feature_list_unit2)


#                     test_concat_features  =np.concatenate((test_feature_red1, test_feature_red2), axis=1)
#                     print(test_concat_features.shape)
                    

#                     feature_test = scaler1.transform(test_concat_features)
#                     print(feature_test.shape)
                    

#                     pre_list = clf.predict(feature_test)
#                     print(pre_list.shape)
#                     print(np.unique(pre_list))

#                     # Generate predicted result
#                     for k in range(patch_size):
#                         for l in range(patch_size):
#                             if i+k >= img_size or j+l >= img_size: break

#                             # Binary
#                             if pre_list[k*patch_size + l] > 0.5:
#                                 predict_0or1[i+k, j+l, 1] += 1
#                             else:
#                                 predict_0or1[i+k, j+l, 0] += 1

#                             # Multi-class
#                             # if pre_list[k*patch_size + l] == 85.0:
#                             #     predict_0or1[i+k, j+l, 1] += 1
#                             # if pre_list[k*patch_size + l] == 170.0:
#                             #     predict_0or1[i+k, j+l, 2] += 1
#                             # if pre_list[k*patch_size + l] == 255.0:
#                             #     predict_0or1[i+k, j+l, 3] += 1
#                             # else:
#                             #     predict_0or1[i+k, j+l, 0] += 1

#             print('*************************************************************************************')
#             print('one predicted mask')
#             predict_mask = np.argmax(predict_0or1, axis=2)
            
#             imageName = savePath + r'/'+ os.path.basename(test_img_addr)
#             print(imageName)
#             imageio.imsave(imageName, (predict_mask).astype(np.uint8))
#             del predict_0or1, predict_mask

    
#     #stop_test = timeit.default_timer()

#     #print('Total Time: ' + str(timedelta(seconds=stop_test-start_test)))
#     #f = open('{}test_time.txt'.format(pathToResultSave),'w+')
#     #f.write('Total Time: ' + str(timedelta(seconds=stop_test-start_test)))
#     #f.close()

# @cuda.jit
# def GPU_Feature(d_img_patch_list, d_train_feature_unit1, d_train_feature_unit2, d_feature_list_unit1, d_feature_list_unit2, d_threadDimensions, delta_x, patch_size, depth1, depth2):
# 	threadIdx = cuda.grid(1)
# 	t, i, j, k, l, d = indices6(threadIdx, d_threadDimensions)
# 	if t < d_threadDimensions[0]:
# 		i3 = d_img_patch_list.shape[3]
# 		patch_ind = t * (img_size//delta_x) * (img_size//delta_x) + i * (img_size//delta_x) + j
# 		if d < i3:
# 			d_feature_list_unit1[t, i, j, k, l, d] = d_img_patch_list[patch_ind,k,l,d]
# 			d_feature_list_unit2[t, i, j, k, l, d] = d_img_patch_list[patch_ind,k,l,d]
# 		elif d == i3:
# 			d_feature_list_unit1[t, i, j, k, l, d] = patch_size//2 - abs(i*delta_x+k-patch_size//2)
# 			d_feature_list_unit2[t, i, j, k, l, d] = patch_size//2 - abs(i*delta_x+k-patch_size//2)
# 		elif d == i3 + 1:
# 			d_feature_list_unit1[t, i, j, k, l, d] = patch_size//2 - abs(j*delta_x+l-patch_size//2)
# 			d_feature_list_unit2[t, i, j, k, l, d] = patch_size//2 - abs(j*delta_x+l-patch_size//2)
# 		elif d < depth1 and depth2:
# 			d_feature_list_unit1[t, i, j, k, l, d] = d_train_feature_unit1[patch_ind, d - (i3 + 2)]
# 			d_feature_list_unit2[t, i, j, k, l, d] = d_train_feature_unit2[patch_ind, d - (i3 + 2)]
# 		elif d < depth2:
# 			d_feature_list_unit2[t, i, j, k, l, d] = d_train_feature_unit2[patch_ind, d - (i3 + 2)]

# @cuda.jit(device=True)
# def indices6(m, threadDimensions):
#     t = m // (threadDimensions[1] * threadDimensions[2] * threadDimensions[3] * threadDimensions[4] * threadDimensions[5])
#     m -= t * (threadDimensions[1] * threadDimensions[2] * threadDimensions[3] * threadDimensions[4] * threadDimensions[5])
#     i = m // (threadDimensions[2] * threadDimensions[3] * threadDimensions[4] * threadDimensions[5])
#     m -= i * (threadDimensions[2] * threadDimensions[3] * threadDimensions[4] * threadDimensions[5])
#     j = m // (threadDimensions[3] * threadDimensions[4] * threadDimensions[5])
#     m -= j * (threadDimensions[3] * threadDimensions[4] * threadDimensions[5])
#     k = m // (threadDimensions[4] * threadDimensions[5])
#     m -= k * (threadDimensions[4] * threadDimensions[5])
#     l = m // (threadDimensions[5])
#     m -= l * (threadDimensions[5])
#     d = m
#     return t, i, j, k, l, d




# run(sys.argv[1:]) 