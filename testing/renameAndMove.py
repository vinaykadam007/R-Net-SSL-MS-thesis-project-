import os

path = r''

destPath = r''
counter = 0

dir_list =  os.listdir(path)
dir_list = [directory for directory in os.listdir(path) if os.path.isdir(path+directory)]

for dir in dir_list:
    files = os.listdir(dir)
    for filename in files:
        if filename.startswith("Segmentation.obj"):
            os.rename(filename, (destPath + "Segmentation{}.obj".format(counter)) )
        if filename.startswith("Segmentation.mtl"):
            os.rename(filename, (destPath + "Segmentation{}.mtl".format(counter)) )
           