import os, subprocess, sys, glob, re
from subprocess import Popen
from itertools import islice
import numpy as np
import itertools
from itertools import chain
pyPath = ""

for key,value in os.environ.items():
    if (key == "PATH"):
        for everypath in value.split(";"):
            if 'Python311' in everypath.split('\\'):
                if "Scripts" not in everypath.split('\\'):
                    print(everypath)
                    pyPath = everypath
                    print(pyPath)     
      

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

path = r'/'.join(resource_path('').split('\\')[:-2]) + '/'

def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


dir_list = []
dir_list = glob.glob(r"Z:/Aayan/PixelHopGPU-Test-run_multiprocessing/data/wholeMouse1024x1024/*/*.png")


directory_list = sorted_nicely(dir_list)

print("here")
print(directory_list)
#print(len(directory_list)/80)

commandList = []

for file in directory_list:  
        testimagespath = os.path.abspath(file)
        modelpathtext =r'pixelhop1.pkl'
        modelpathtextbox2 = r'pixelhop2.pkl'
       
        folderName =  os.path.basename(os.path.dirname(os.path.normpath(file)))
        path2 = os.path.abspath((r'Z:/Aayan/PixelHopGPU-Test-run_multiprocessing/srcLoop/resultsMultiClass/' + folderName))
        #print(path2)
        exists = os.path.exists(path2)
        if not exists:
            os.makedirs(path2)
        
        savepredictedtextbox = path2
        classifierbrowse = path + r'results/classifier.sav'
        scalerbrowsetextbox =path + r'results/scaler1.sav'
        featurebrowse1textbox= path + r'results/fs1.sav'
        featurebrowse2textbox= path + r'results/fs2.sav'


        commandList.append(pyPath + "python" + " \"pixelHopTest.py\" " +
                                                "-r \""+ testimagespath+"\""+ 
                                                " -m \""+ modelpathtext+"\""+
                                                " -c \""+ modelpathtextbox2 +"\""+
                                                " -s \""+savepredictedtextbox+"\""
                                                +" -g \""+classifierbrowse+"\""
                                                +" -e \""+scalerbrowsetextbox+"\""
                                                +" -v \""+featurebrowse1textbox+"\""
                                                +" -n \""+featurebrowse2textbox+"\"")


# ### List splitting
a_list = commandList

a_list = np.array_split(np.array(a_list), 12)

for lists in a_list:
    processes = [Popen(cmd, shell = True) for cmd in lists]
    exit_codes = [p.wait() for p in processes]
    print(exit_codes)
    continue
    




# import os, subprocess, sys, glob, re
# from subprocess import Popen

# pyPath = ""

# for key,value in os.environ.items():
#     if (key == "PATH"):
#         for everypath in value.split(";"):
#             if 'Python311' in everypath.split('\\'):
#                 if "Scripts" not in everypath.split('\\'):
#                     print(everypath)
#                     pyPath = everypath
#                     print(pyPath)     
      

# def resource_path(relative_path):
#     if hasattr(sys, '_MEIPASS'):
#         return os.path.join(sys._MEIPASS, relative_path)
#     return os.path.join(os.path.abspath("."), relative_path)

# path = r'/'.join(resource_path('').split('\\')[:-2]) + '/'

# def sorted_nicely( l ): 
#     """ Sort the given iterable in the way that humans expect.""" 
#     convert = lambda text: int(text) if text.isdigit() else text 
#     alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
#     return sorted(l, key = alphanum_key)


# dir_list = []
# for root, dirs, files in os.walk(r"Z:\Aayan\PixelHopGPU-Test-run_multiprocessing\srcLoop\data\xinyuanDataRawPNG"):
#     dir_list.append(root)


# directory_list = sorted_nicely(dir_list)

# print("here")
# print(directory_list)

# commandList = []

# for file in directory_list:  
#         testimagespath = file
#         modelpathtext =r'pixelhop1.pkl'
#         modelpathtextbox2 = r'pixelhop2.pkl'
       
#         folderName =  os.path.basename(os.path.normpath(file)) 
#         path2 = (r'Z:\Aayan\PixelHopGPU-Test-run_multiprocessing\srcLoop\results/' + folderName)
#         print(path2)
#         exists = os.path.exists(path2)
#         if not exists:
#             os.makedirs(path2)
        
#         savepredictedtextbox = path2
#         classifierbrowse = path + r'results/classifier.sav'
#         scalerbrowsetextbox =path + r'results/scaler1.sav'
#         featurebrowse1textbox= path + r'results/fs1.sav'
#         featurebrowse2textbox= path + r'results/fs2.sav'


#         commandList.append(pyPath + "python" + " \"pixelHopTest.py\" " +
#                                                 "-r \""+ testimagespath+"\""+ 
#                                                 " -m \""+ modelpathtext+"\""+
#                                                 " -c \""+ modelpathtextbox2 +"\""+
#                                                 " -s \""+savepredictedtextbox+"\""
#                                                 +" -g \""+classifierbrowse+"\""
#                                                 +" -e \""+scalerbrowsetextbox+"\""
#                                                 +" -v \""+featurebrowse1textbox+"\""
#                                                 +" -n \""+featurebrowse2textbox+"\"")



# a_list = commandList
# half_length = len(a_list) // 2
# first_half, second_half = a_list[:half_length], a_list[half_length:]

# thirdQuarter, fourthQuarter = second_half[:(len(second_half)//2)], second_half[(len(second_half)//2):]
# firstQuarter, secondQuarter = first_half[:(len(first_half)//2)], first_half[(len(first_half)//2):]

# processes = [Popen(cmd, shell = True) for cmd in a_list]

# for p in processes: p.wait()

# # processes = [Popen(cmd, shell = True) for cmd in secondQuarter]

# # for p in processes: p.wait()

# # for item in commandList:
# #     proc = subprocess.Popen(item)
# #     proc.wait()
