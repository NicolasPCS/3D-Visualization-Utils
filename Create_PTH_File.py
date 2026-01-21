import os
import json
import torch
import numpy as np
import open3d as o3d

input_path = "/home/ncaytuir/data/Datasets/Resultados_PVD/Over_Half_Objects/airplane/ckpt_2899/complete_pcs"
output_path = "/home/ncaytuir/data/Datasets/Resultados_PVD/Over_Half_Objects/airplane/ckpt_2899/2899_complete_shape_samples_airplane.pth"
reference_data = "/home/ncaytuir/data/Datasets/ValDataForLION/ref_val_airplane.pt"

def load_reference_data(ref_path):
    ref_data = torch.load(ref_path)

    #ref_pcs = ref_data['ref']
    mean = ref_data['mean'].float()
    std = ref_data['std'].float()

    return mean, std 

# Compute FPS
def farthest_point_sampling(points, n_samples):
    # Create a point cloud of Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # compute FPS
    downpcd_farthest = pcd.farthest_point_down_sample(n_samples)

    return np.asarray(downpcd_farthest.points)

# Listar y ordenar archivos .npy
file_list = sorted([f for f in os.listdir(input_path) if f.endswith(".npy")]) #[:250] # 250 could be adjusted

all_pcs = []
num_points = 2048

for filename in file_list:
    file_path = os.path.join(input_path, filename)
    pc = np.load(file_path)

    sampled_points = farthest_point_sampling(pc, num_points)

    print(f"La nube tenia {pc.shape[0]} ahora tiene {sampled_points.shape[0]}")

    all_pcs.append(sampled_points)

# Convetir a tensor
ref = torch.from_numpy(np.stack(all_pcs)).float() # (N, 2048, 3)
#mean = ref.mean(dim=1, keepdim=True) # (N, 1, 3)
#std = ref.std(dim=1, keepdim=True).mean(dim=2, keepdim=True) # (N, 1, 1)

mean, std = load_reference_data(reference_data)

ref = ref * std + mean

# Save pt
#torch.save({'ref': ref, 'mean': mean, 'std': std}, output_path)
torch.save(ref, output_path)
print(f"Guardado {ref.shape[0]} nubes en {output_path}")

# Inspección final
data = torch.load(output_path)

# Ver qué contiene
print(type(data))
#print(data.keys())