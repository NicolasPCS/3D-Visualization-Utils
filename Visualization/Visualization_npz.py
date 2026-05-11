import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import polyscope as ps

# =====================
# PUT YOUR FILE NAME
# =====================
#input_file = "/home/ncaytuir/data/Datasets/shapenet_psr/shapenet_psr/02691156/1a04e3eab45ca15dd86060f189eb133/pointcloud.npz"
#input_file = "/home/ncaytuir/data/gca_necs/data/shapenet_sdf/chair/train/1a6f615e8b1b5ae4dbbc9440457e303e.npz"
input_file = "/home/ncaytuir/data/Datasets/Scripts.v5/noises/latent_noise_98.npz"
NPZ_FILE = input_file

# ---------------------
# Load file
# ---------------------
data = np.load(NPZ_FILE)
print("\nAvailable keys in npz file:")
for k in data.files:
    print(f"  {k}: shape = {data[k].shape}")

#print("points", data['points'])
#print("normals", data['normals'])
#print("loc", data['loc'])
#print("scale", data['scale'])

points = data['coord']
#features = data["feat"] 
#normals = data['normals']

# Init polyscope
ps.init()

# Polyscope set-ups
ps.set_up_dir("neg_z_up")            # -Z Up
ps.set_front_dir("y_front")          # Y Front
ps.set_view_projection_mode("orthographic")
ps.set_screenshot_extension(".png") 
ps.set_ground_plane_mode("none")  # Desactiva la cuadrícula/base

# - Original
pc_original = ps.register_point_cloud("pc Original", points)
pc_original.set_color((0,0,1))

#pc_original.add_scalar_quantity("X", features, enabled=True) # points[:, 0]

#pc_original.add_vector_quantity("Normals", normals, enabled=True, length=0.02, radius=0.001)

# Visualize
ps.show()