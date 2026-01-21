import polyscope as ps
import numpy as np
import argparse
import open3d as o3d
import sys
import os

# Compute FPS
def farthest_point_sampling(points, n_samples=2048):
    # Create a point cloud of Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # compute FPS
    downpcd_farthest = pcd.farthest_point_down_sample(n_samples)

    return np.asarray(downpcd_farthest.points)

# Añadir el directorio padre al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Symmetry_Computation.Householder_transform import householder_transformation

# Dir path
#file2 = "/home/ncaytuir/data/Datasets/ShapeNer.v2/airplane/ff13be97bdfa45f8254dc1d04198881.npy"

# Argument parser
parser = argparse.ArgumentParser(description="Visualize a point cloud")
parser.add_argument("original", type=str, help="Path to the point cloud file")
parser.add_argument("--mirrored", type=str, help="Path to the point cloud file")

args = parser.parse_args()
file1 = args.original
#file2 = args.mirrored

# Load point cloud
original_points = np.load(file1)
#mirrored_points = np.load(file2)

print("original", original_points)

mirrored_points = householder_transformation(original_points)
print("trans", mirrored_points)
#np.save("/home/ncaytuir/data/Datasets/Scripts/Visualization/Output/archivoreflejado.npy", mirrored_points)

offset = np.array([1.0, 0.0, 0.0])
mirrored_points += offset

# Init polyscope
ps.init()

# Polyscope set-ups
ps.set_up_dir("neg_z_up")            # -Z Up
ps.set_front_dir("y_front")          # Y Front
ps.set_view_projection_mode("orthographic")
ps.set_screenshot_extension(".png") 
ps.set_ground_plane_mode("none")  # Desactiva la cuadrícula/base

# Register point clouds

# - Original
opc_sampled = farthest_point_sampling(original_points, 2048)

pc_original = ps.register_point_cloud("pc Original", opc_sampled)
pc_original.set_color((0,0,1))
pc_original.add_scalar_quantity("Original", opc_sampled[:, 0], enabled=True)

# Agregar flechas desde puntos originales hacia reflejados
#vectors = mirrored_points - opc_sampled
#pc_original.add_vector_quantity("Reflected vectors", vectors, enabled=True)

# - Reflected
rpc_sampled = farthest_point_sampling(mirrored_points, 2048)

pc_reflejado = ps.register_point_cloud("pc reflejado", rpc_sampled)
pc_reflejado.set_color((1,0,0))
pc_reflejado.add_scalar_quantity("Mirrored", -rpc_sampled[:, 0], enabled=True)

try:
    ps.screenshot("/home/ncaytuir/data/Datasets/Scripts/Visualization/planes.png")
    print("done")
except:
    print("No") 

# Visualize
ps.show()
ps.screenshot("/home/ncaytuir/data/Datasets/Scripts/Visualization/planes.png")