import json
import sys, os, subprocess, time

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)



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

path = './home/sliceruser/volumeFolder/tif_data'
savePath = "obj"
fileNames = glob.glob(path + "/*.tif")
print(fileNames)

for files in fileNames :
    # Load from local file
    imagename = str(files)
    print(imagename)
    # Clear scene
    slicer.mrmlScene.Clear(False)
    masterVolumeNode = slicer.util.loadVolume(imagename)
    outputVolumeSpacingMm = [0.000325, 0.000325, 0.001]
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
    for i in range(1, 256):
        cellDic = []   
        cellDic.append("cell_"+str(i))
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

    
# with open('main.ipynb', mode = "r",  encoding= "utf-8" ) as g:
#         baseFile = json.loads(g.read())
#         with open(r'D:\Aayan\UI_TESTING\bin\mainNonVTI.ipynb', mode = "r",  encoding= "utf-8" ) as b:
#             codeFile = json.loads(b.read())
#             baseFile['cells'][0]['source'] =  codeFile['cells'][0]['source']
            
# with open(r'D:\Aayan\UI_TESTING\bin\mainNonVTI.ipynb', 'w') as outfile:
#     json.dump(baseFile, outfile)            