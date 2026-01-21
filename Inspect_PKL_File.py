import torch
import numpy as np
import open3d as o3d
import polyscope as ps

import os

data = torch.load("/home/ncaytuir/data/Datasets/Resultados_XCube/Airplane/result_dict_794.pkl", map_location='cpu')

#print(type(data))
#print(data.keys() if hasattr(data, "keys") else data[:5])

# dict_keys(['coarse_xyz', 'coarse_normal', 'fine_xyz', 'fine_normal'])
#points = data['coarse_xyz']
#points = data['coarse_normal']
points = data['fine_xyz']
#points = data['fine_normal']

# Polyscope Visualization
ps.init()
ps.register_point_cloud("puntos", points)
ps.show()

# Open3D Visualization
""" pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
o3d.visualization.draw_geometries([pcd]) """