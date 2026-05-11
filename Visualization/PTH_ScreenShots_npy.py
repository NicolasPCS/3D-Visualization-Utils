import polyscope as ps
import numpy as np
import argparse
import open3d as o3d
import torch
import random
import os

def farthest_point_sampling(points, n_samples):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    downpcd_farthest = pcd.farthest_point_down_sample(n_samples)
    return np.asarray(downpcd_farthest.points)

# --- Argumentos ---
parser = argparse.ArgumentParser(description="Renderizar nubes de puntos a imágenes")
parser.add_argument("file", type=str, help="Ruta al archivo .pth")
parser.add_argument("out_dir", type=str, help="Carpeta de salida para los renders")
parser.add_argument("clase", type=str, help="clase")
parser.add_argument("--num_images", type=int, default=10, help="Cantidad de imágenes a guardar")
parser.add_argument("--samples", type=int, default=2048, help="Puntos tras FPS")

args = parser.parse_args()

# Crear carpeta de salida si no existe
os.makedirs(args.out_dir, exist_ok=True)

# --- Carga de Datos ---
data = torch.load(args.file)
if isinstance(data, dict):
    all_points = data.get('points', None)
    if torch.is_tensor(all_points):
        all_points = all_points.cpu().numpy()
else:
    all_points = data.cpu().numpy() if torch.is_tensor(data) else np.array(data)

num_total = all_points.shape[0]
print(f"Total disponible: {num_total}. Generando {args.num_images} renders...")

# --- Inicializar Polyscope ---
ps.init()
ps.set_view_projection_mode("orthographic")
ps.set_ground_plane_mode("none")
ps.set_screenshot_extension(".png")

# Selección aleatoria de índices
indices = random.sample(range(num_total), min(args.num_images, num_total))

# Lógica de orientación basada en el nombre del archivo
is_chair = "chair" in args.file.lower()
if is_chair:
    ps.set_up_dir("y_up")
    ps.set_front_dir("neg_z_front")
else:
    ps.set_up_dir("neg_z_up")
    ps.set_front_dir("y_front")

# --- Bucle de Renderizado ---
for i, idx in enumerate(indices):
    points = all_points[idx]
    
    # Limpiar la escena anterior
    ps.remove_all_structures()
    
    # FPS
    if points.shape[0] > args.samples:
        points = farthest_point_sampling(points, n_samples=args.samples)
    
    # Registrar nube
    pc = ps.register_point_cloud(f"render_{idx}", points)
    
    # Configurar look similar a tu imagen (Gradiente en X)
    pc.add_scalar_quantity("X_Grad", points[:, 0], enabled=True, cmap='viridis')
    
    # Guardar imagen
    save_path = os.path.join(args.out_dir, f"{args.clase}_cloud_{idx:04d}.png")
    
    ps.frame_tick()

    # Renderizar un frame y guardar
    ps.screenshot(save_path, transparent_bg=True)
    
    print(f"[{i+1}/{args.num_images}] Guardado: {save_path}")

print("\nProceso finalizado con éxito.")

