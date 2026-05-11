import polyscope as ps
import numpy as np
import argparse

# Dir path
file1 = "/home/ncaytuir/LION_necs/MyScripts/latent_results/original_pc_2.npy"
file2 = "/home/ncaytuir/LION_necs/MyScripts/latent_results/mirrored_pc_2.npy"
#file3 = "/home/ncaytuir/data/Datasets/Resultados_LION/Airplane/complete_point_clouds/pc_351.npy"

# Load point cloud
points1 = np.load(file1)
points2 = np.load(file2)
#points3 = np.load(file3)

offset = np.array([0.1, 0.0, 0.0])
points2 += offset
#points3 += offset * 2

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
pc_original1 = ps.register_point_cloud("pc Original 1", points1)
pc_original1.set_color((0,0,1))

pc_original2 = ps.register_point_cloud("pc Original 2", points2)
pc_original2.set_color((0,0,1))

#pc_original3 = ps.register_point_cloud("pc Original 3", points3)
#pc_original3.set_color((0,0,1))

# Agregar flechas desde puntos originales hacia reflejados
#vectors = points2 - points
#pc_original1.add_vector_quantity("Original: To Reflected", vectors, enabled=True)

pc_original1.add_scalar_quantity("X", points1[:, 0], enabled=True)
pc_original2.add_scalar_quantity("X", points2[:, 0], enabled=True)
#pc_original3.add_scalar_quantity("X", points3[:, 0], enabled=True)

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