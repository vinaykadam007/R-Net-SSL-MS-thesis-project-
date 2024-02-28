# Introduction
## What is R-NET?
[R-NET](https://drive.google.com/file/d/11LwXUnXKbCn8VSTV6v6DrAAxvTXvq2IE/view?usp=sharing) is an open source program designed to segment LSFM images and then volumetrically reconstruct those images into 3D representations. R-NET was designed by Aayan Rahat and Vinay Kadam under the guidance of Dr.Yichen Ding at the Ding-Incubator @UTDallas. 

### Where can I find more information?
All of the latest code updates and issue updates can be found on our [github page](https://www.github.com/orgs/D-Incubator/repositories)

# Folder/Data Structure
## Folder Setup
Most of these folders will come pre populated with the download. If not, make sure your file structure and folder structure matches the one below (case sensitive). Also make sure that no part of your file path has spaces (will cause issues in docker commands)

![fileStructure](https://github.com/D-Incubator/R-NET/blob/exeFile/fileStructure.PNG)



# Dependencies + Install
## Python
Firstly, make sure that you have python installed on your computer. We recommend installing python 3.11.3 and adding it to your [path](https://www.machinelearningplus.com/python/add-python-to-path-how-to-add-python-to-the-path-environment-variable-in-windows/) and following the basic setup expectations. This will make sure all the libraries run accurately and are compatible. 

### Associated Libraries 
Most libraries will come pre-installed within the exe file and should run properly. If not, make sure to have the following libraries. You can use the pip install command to install these libraries. You can follow this [tutorial](https://packaging.python.org/en/latest/tutorials/installing-packages/) if you are unsure about “pip install”. 
Here are needed libraries: **os, multiprocessing, glob, random, subprocess, PyQt5, platform, sys, numpy, opencv-python, matplotlib, shutil, time, webbrowser, re, IPython, nbformat, json, ipynb-py-convert (INSTALL THIS BY HAND)**

Make sure, you install these variables to the python interpreter or within the directory of the folder. You can type “cmd” into the address bar of the directory to do this. 

# Train Tab
![trainTab](https://github.com/D-Incubator/R-NET/blob/exeFile/images/train%20page.PNG)
## Training Images
These are the images that you will use to train your model. Make sure, your images are all in png format. If they are not in the correct format, you can use this program ([Fiji]("https://imagej.net/software/fiji/downloads")) to save these images as png format [tutorial](https://www.youtube.com/watch?v=6OlIAsoUdj0). Also make sure that your images are in grayscale format, not RGB. [tutorial here](https://www.linkedin.com/advice/0/what-benefits-converting-image-grayscale-imagej-skills-imagej). 

Once your images are in this format, make sure that they are named with this format (image_raw_##.png or image_seg_##.png) within the same folder. Make sure they are named this way, case sensitively. Go into the folder completely and select it: if you have an error, make sure you are completely into the folder.
## Testing Images
These are the images that you will use to test your model. Follow all the same conventions and data formatting as above. 
## Fields 
**Number of Classes**: You can choose the number of classes to use. For now, we have only tested binary segmentation so use 2 classes for now. 

**Variance**: This is the variation of the generated model or the “energy percentage”, you can customize this but we recommend somewhere around 0.96 to 0.98

**Number of Training Images**: This determines how many images your model will be trained on, the more images you have, the more accurate your model will be (until overfitting) but it will take much more time

**More than 1 image?**: This will choose whether or not multiprocessing (GPU based computation) will be used. If you are using multiple images or testing a large model, we recommend clicking this option and implementing the multiprocessing PixelHop algorithm 

**Folder to Save Outputs**: This is where the test images’ mask(s) will be outputted as well as the model files for the test files

**Select Algorithm**: For now, we only have the PixelHop algorithm. Once we develop more algorithms, we will add more options 

# Test Tab
![testTab](https://github.com/D-Incubator/R-NET/blob/exeFile/images/dedicated%20test%20page.PNG)
## Fields 

**Test Images**: This is the folder that you will use to take raw images (same format as mentioned above) that will now become masks

**Main Model Files**: The “Train/Test” tab will output two files called “pixelhop1.pkl” and “pixelhop2.pkl”, upload both of these files to the model files folders 

**Extra Model Files**: These are the files found within the “Folder to Save Outputs” from before. You can see each file that needs to be uploaded based on the text within the address bar. 

**Save Predicted Images**: This is where you want you test images’ masks to be outputted  

# Export Tab
![exportTab](https://github.com/D-Incubator/R-NET/blob/exeFile/images/export%20page.PNG)
### Prerequisites:
Make sure you have docker desktop installed and you are logged in and set up. After that, make sure docker is open in the background. Make sure the “Slicerdocker.dll” file is also not in your current directory so that the code will run correctly. 

### Fields 
**Segmented Images**: This is the folder of your segmented images (.tif format)

**Spacing/Height/Width**: This is the dimensions/spacing (Z, Y, X) of your model. Please try to match this as much as possible with your original image dimensions or with the dimensions of what your model to be. This may take some experimentation. 

**VTI Method**: We have two methods of reconstruction. The first “VTI Method” is based on reconstruction that will create a VTI model and then convert that into an OBJ model. We recommend this option for large datasets or “cell data” as you can easily change the spacing and scaling of the model. The “Non VTI Method” is based on cell based reconstruction. This method will output a model for every image (95 images means 95 .obj and .mtl files) but the “VTI Method” will only output one model file for the whole image.
