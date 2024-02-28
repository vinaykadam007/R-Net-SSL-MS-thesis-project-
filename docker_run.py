
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
    with open(resource_path('Slicer_docker.dll'), 'w') as f:
        f.write("Slicer for Ding's Lab")

# copy files to docker
vti_filename = resource_path("/export files/vti/unity.vti")
vtifile = "unity.vti"
jupyter_filename = resource_path("main.ipynb")
jupyterfile = "main.ipynb"


vti_copy = subprocess.Popen("docker cp \""+vti_filename+"\" slicernotes:./home/sliceruser/"+vtifile+"")
vti_copy.wait()

jupyter_copy = subprocess.Popen("docker cp \""+jupyter_filename+"\" slicernotes:./home/sliceruser/"+jupyterfile+"")
jupyter_copy.wait()

time.sleep(2)

#install jupyter nbconvert into docker
install_nbconvert = subprocess.Popen("winpty docker exec -it slicernotes bash -c \"./Slicer-4.11.0-2020-05-11-linux-amd64/Slicer --launch PythonSlicer -m pip install jupyter nbconvert\n\"")
install_nbconvert.wait()

time.sleep(2)

#make directory inside docker
make_obj_directory = subprocess.Popen("winpty docker exec -it slicernotes bash -c \"mkdir -p obj\n\"")
make_obj_directory.wait()

time.sleep(2)

#run jupyter command inside docker
run_jupyter = subprocess.Popen("winpty docker exec -it slicernotes bash -c \"jupyter nbconvert --to notebook --inplace --execute main.ipynb --ExecutePreprocessor.timeout=-1\n\"")
run_jupyter.wait()

# time.sleep(`0`)

# copy files from docker
obj_filename= "Segmentation.obj"
obj_copy_docker = subprocess.Popen("docker cp slicernotes:./home/sliceruser/obj/"+obj_filename+" "+"\""+obj_filename+"\"")
obj_copy_docker.wait()




mtl_filename= "Segmentation.mtl"
mtl_copy_docker = subprocess.Popen("docker cp slicernotes:./home/sliceruser/obj/"+mtl_filename+" "+"\""+mtl_filename+"\"")

mtl_copy_docker.wait()
