import numpy as np
import argparse
from pathlib import Path
import torch
import open3d as o3d

# Argument parser
parser = argparse.ArgumentParser(description="Compute symmetry")
parser.add_argument("path", type=str, help="Path to the directory")
parser.add_argument("file_ext", type=str, default='npy', help='File extension (i.e., .npy)')

args = parser.parse_args()
path = Path(args.path)
file_ext = args.file_ext

# Compute FPS
def farthest_point_sampling(points, n_samples=2048):
    # Create a point cloud of Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # compute FPS
    downpcd_farthest = pcd.farthest_point_down_sample(n_samples)

    return np.asarray(downpcd_farthest.points)

# Count number of points in a point cloud
def count(files, file_ext):
    #for p in files:
    cont = 0
    for p in files:
        if file_ext == 'npy':
            # Load point cloud
            original_points = np.load(p)
            #print(original_points.shape)
            return original_points.shape[0]
        elif file_ext == 'pkl':
            # Load point cloud
            data = torch.load(p, map_location='cpu')
            original_points = data['fine_xyz']
            #print(original_points.shape)
            sampled_points = farthest_point_sampling(original_points, 2048)
            np.save(f"/home/ncaytuir/data/Datasets/Scripts/XCube_2048_FPS_test/nube{cont}.npy", sampled_points)
            if cont == 4:
                break
            cont += 1
            return original_points.shape[0]
        elif file_ext == 'xyz':
            # Load point cloud
            data = np.loadtxt(p)
            return data.shape[0]
    return

if file_ext == 'npy':
    files = list(path.glob("*.npy"))
    #count(files, file_ext)
    no_points = count(files, file_ext)
    print("Number of points: ", no_points)

elif file_ext == 'pkl':
    files = list(path.glob("*.pkl"))
    count(files, file_ext)
    no_points = count(files, file_ext)
    print("Number of points: ", no_points)

elif file_ext == 'xyz':
    files = list(path.glob("*.xyz"))
    no_points = count(files, file_ext)
    print("Number of points: ", no_points)

else:
    print("Invalid file extension.")

