import pyvista as pv
import os
import re
def sorted_nicely( l ): 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

my_dir = (r'D:\Aayan\pixelHop4D\obj smoothing\input')

root, dirs, files = next(os.walk(my_dir, topdown=True))
files = [ os.path.join(root, f) for f in files ]
files = sorted_nicely(files)


counter = 1
for file in files:
    if file.endswith(".obj"):
        print(file)
        reader = pv.get_reader(file)
        mesh = reader.read()
        mesh = mesh.clean()
        mesh = mesh.smooth_taubin(n_iter = 20, pass_band = 0.1, feature_smoothing = True)
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(mesh)
        plotter.export_obj('smoothed/smoothed_vol_{}.obj'.format(str(counter)))
        counter +=1 
