



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
        install_nbconvert = subprocess.Popen('docker exec -it slicernotes bash -c "./Slicer-4.11.0-2020-05-11-linux-amd64/Slicer --launch PythonSlicer -m pip install jupyter nbconvert"')
        install_nbconvert.wait()
        install_nbconvert2 = subprocess.Popen('docker exec -it slicernotes bash -c "./Slicer-4.11.0-2020-05-11-linux-amd64/Slicer --launch PythonSlicer -m pip install slicer"')
        install_nbconvert2.wait()
        time.sleep(2)

        #make directory inside docker
        make_obj_directory = subprocess.Popen('docker exec -it slicernotes bash -c "mkdir -p obj"')
        make_obj_directory.wait()
        
        make_volume_directory = subprocess.Popen('docker exec -it slicernotes bash -c "mkdir -p volumeFolder"')
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
    run_jupyter = subprocess.Popen('docker exec -it slicernotes bash -c "jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute mainNonVTI.ipynb"')
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


