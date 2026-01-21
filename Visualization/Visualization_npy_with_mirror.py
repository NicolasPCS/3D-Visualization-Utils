import polyscope as ps
import numpy as np
import argparse
import open3d as o3d
import torch

# Compute FPS
def farthest_point_sampling(points, n_samples):
    # Create a point cloud of Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # compute FPS
    downpcd_farthest = pcd.farthest_point_down_sample(n_samples)

    return np.asarray(downpcd_farthest.points)

def dynamic_shift(pc, k):
    """
    pc: [1, N, 3]
    Returns the best shift value
    """
    print(f"pc: {pc.shape}")
    x = pc[0, :, 0] # [N]
    print(f"x: {x.shape}")
    # Fewest k = topk over -x
    #k = min(k, x.shape[0])
    #print(f"Dynamic shift: k={k}")
    vals, _ = torch.topk(x, k=k, largest=False)
    print(f"Dynamic shift: vals={vals}")
    mean_k_smallest = vals.mean()
    print(f"Dynamic shift: mean_k_smallest={mean_k_smallest}")
    #print(f"Dynamic shift: {mean_k_smallest / 100}")
    return mean_k_smallest / 100

# Dir path
#file2 = "/home/ncaytuir/Datasets/ShapeNetCore.v3.PC15k/02691156/train/1a6ad7a24bb89733f412783097373bdc.npy"

# Argument parser
parser = argparse.ArgumentParser(description="Visualize a point cloud")
parser.add_argument("file", type=str, help="Path to the point cloud file")

args = parser.parse_args()
file = args.file

# Load point cloud
points = np.load(file)

# Mover todo a x > 0
points[:, 0] -= (np.min(points[:, 0]))
d_shift = dynamic_shift(torch.tensor(points).unsqueeze(0), k=64)
print(f"Dynamic shift applied: {d_shift}")
shift = abs(np.min(points[:, 0])) - abs(float(d_shift)) # 0.01
points[:, 0] += shift

# Reflejar la nube
points_mirrored = points.copy()
points_mirrored[:, 0] *= -1

# Unir con la nube original
points = np.concatenate([points, points_mirrored], axis=0)

print(points)
print(points.shape)

points = farthest_point_sampling(points, n_samples=2048)
print(points.shape)

# Init polyscope
ps.init()

# Polyscope set-ups
ps.set_up_dir("neg_z_up")            # -Z Up - Y Up
ps.set_front_dir("y_front")          # Y Front - -Y front
ps.set_view_projection_mode("orthographic")
ps.set_screenshot_extension(".png") 
ps.set_ground_plane_mode("none")  # Desactiva la cuadrícula/base

""" # Mover todo a x > 0
shift = abs(np.min(points[:, 0])) - 0.01
points[:, 0] += shift

# Reflejar la nube
points_mirrored = points.copy()
points_mirrored[:, 0] *= -1

# Unir con la nube original
points = np.concatenate([points, points_mirrored], axis=0)
 """

# Register point clouds

# - Original
pc_original = ps.register_point_cloud("pc Original", points)
pc_original.set_color((0,0,1))

# Agregar flechas desde puntos originales hacia reflejados
#vectors = points2 - points
#pc_original.add_vector_quantity("Original: To Reflected", vectors, enabled=True)

pc_original.add_scalar_quantity("X", points[:, 0], enabled=True)

# - Reflected
#pc_reflejado = ps.register_point_cloud("pc reflejado", points)
#pc_reflejado.set_color((1,0,0))

#pc_reflejado.add_vector_quantity("Reflected", points, enabled=True)

"""
points2 = np.load(file2)
pc_reflejado.add_scalar_quantity("X", points2[:, 0], enabled=True)
diff = points[:, 0] + points2[:, 0]  # Debe ser cercano a 0 si reflejó bien (x + (-x) ≈ 0

print("Suma de X + reflejado X:", np.mean(np.abs(diff)))"""

#print("Puntos originales X positivos:", np.sum(points[:, 0] > 0))
#print("Puntos reflejados X positivos:", np.sum(points2[:, 0] > 0))

# Visualize
ps.show()