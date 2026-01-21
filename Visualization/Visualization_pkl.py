import sys
import types   # <--- ESTO FALTABA
import torch
import open3d as o3d
import polyscope as ps
import numpy as np
import pickle

# ------------------------------------------------------

def farthest_point_sampling(points, n_samples):
    # Create a point cloud of Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # compute FPS
    downpcd_farthest = pcd.farthest_point_down_sample(n_samples)

    return np.asarray(downpcd_farthest.points)

# Cargar el .pkl
data = torch.load(
    "/home/ncaytuir/data/Datasets/Resultados_XCube/OverHalf/plane_2025-12-19_13-22-18/result_dict_5.pkl",
    map_location="cpu"
)

print(data.keys())
print(data)

# Extraer la nube de puntos
points = data['fine_xyz']  # o 'coarse_xyz', según el caso
points2 = data['coarse_xyz']  # o 'coarse_xyz', según el caso

print(points.shape)
print(points2.shape)

sampled_points = farthest_point_sampling(points, 2048)

# Init polyscope
ps.init()

# Polyscope set-ups
ps.set_up_dir("x_up")            # -Z Up
ps.set_front_dir("y_front")          # Y Front
ps.set_view_projection_mode("orthographic")
ps.set_screenshot_extension(".png") 
ps.set_ground_plane_mode("none")  # Desactiva la cuadrícula/base

pc_original = ps.register_point_cloud("pc Original", sampled_points)
pc_original.set_color((0,0,1))

pc_original.add_scalar_quantity("X", sampled_points[:, 0], enabled=True)

# Visualize
ps.show()