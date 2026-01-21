#!/usr/bin/env python3
import argparse
import os
import polyscope as ps
import numpy as np
from symmetries.symmetries import SymmetryPlane

"""
Execute as: python plot_shapenet_symmetry.py --file /home/ncaytuir/data/Datasets/ShapeNetCore.v3.PC15k/02691156/train/2d43c1430df8194ace5721ccacba16.npy --sym_file sym_example.txt
"""

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default=1, help='Number of shape to plot')
parser.add_argument('--sym_file', type=str, default=1, help='Number of shape to plot')

#Add boolean argument
parser.add_argument("--no_sym", action='store_true', help='Use this option if you only want to plot the shape without symmetries')

opt = parser.parse_args()
sym = opt.sym_file

#points = np.load(opt.file)["points"] # Load .npz files
points = np.load(opt.file)
print(points)

if not opt.no_sym:
    symmetry_list = []
    #Read symmetries
    #with open(os.path.join(opt.path, sym)) as f:
    with open(sym) as f:
        num_symmetries = int(f.readline().strip())
        print(num_symmetries)
    
        for i in range(num_symmetries):
            L = f.readline().strip().split()
            if L[0]=="plane":
                L = L[1:]
                L = [float(x) for x in L]
                print(L[0:3])
                symmetry_list.append(SymmetryPlane(point=np.array(L[3:6]), normal=np.array(L[0:3])))

ps.init()
ps.set_up_dir("y_up")
ps.register_point_cloud("PC", points)

if not opt.no_sym:
    for i, sym in enumerate(symmetry_list):
        if isinstance(sym, SymmetryPlane):
            mesh = ps.register_surface_mesh("sym_plane_"+str(i), sym.coords, sym.trianglesBase)
            mesh.set_transparency(0.8)
        
ps.show()