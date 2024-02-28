import tifftools, glob, re

dirList = glob.glob(r'D:/Aayan/pixelHop4D/PixelHopGPU-Test-run_multiprocessing/srcLoop/resultsMedian2/*/')

print(dirList)

state = 1
for dir in dirList:
    tff_lst = glob.glob(dir + "*.tif")
    print(tff_lst)
    tff = tifftools.read_tiff(tff_lst[0])
    for other in tff_lst[1:]:
        othertff = tifftools.read_tiff(other)
        tff['ifds'].extend(othertff['ifds'])
    tifftools.write_tiff(tff, ('{0}{1}.tiff'.format(r"D:\Aayan\pixelHop4D\PixelHopGPU-Test-run_multiprocessing\srcLoop\resultsMedian2TIF/",str(state))))
    state +=1