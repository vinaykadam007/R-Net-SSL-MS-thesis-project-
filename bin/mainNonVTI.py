depth = float(1)
width = float(4.6)
height = float(4.6)




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




