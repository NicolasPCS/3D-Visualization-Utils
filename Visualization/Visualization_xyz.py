import numpy as np
import polyscope as ps

# Leer archivo .xyz con 6 columnas: x y z r g b
data = np.loadtxt("/home/ncaytuir/self-sample/demo-results/airplane/targets/export_iter:10000.xyz")

positions = data[:, :3]
normals = data[:, 3:6]

print(data.shape)
print(positions)
print(normals)

# Inicializar Polyscope
ps.init()

# Polyscope set-ups
ps.set_up_dir("x_up")            # X Up
ps.set_front_dir("y_front")          # Y Front
ps.set_view_projection_mode("orthographic")
ps.set_screenshot_extension(".png") 
ps.set_ground_plane_mode("none")  # Desactiva la cuadrícula/base

# Registrar nube
cloud = ps.register_point_cloud("nube_xyz", positions)
#cloud.add_vector_quantity("normals", normals)

cloud.add_scalar_quantity("X", data[:, 0], enabled=True)

# Mostrar
ps.show()
