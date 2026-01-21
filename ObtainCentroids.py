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

def is_centered_x(points, tolerance=1e-3):
    centroid_x = np.mean(points[:, 0])
    return abs(centroid_x) < tolerance, centroid_x

def center_in_x(points, centroid_x):
    points[:, 0] -= centroid_x
    return points

# Count number of points in a point cloud
def count(files, file_ext):
    #for p in files:
    for p in files:
        if file_ext == 'npy':
            # Load point cloud
            points = np.load(p)
            #print(points.shape)

            is_centered, cx = is_centered_x(points)

            if is_centered:
                print(f"Centrado en X: {is_centered} (centroide X = {cx:.5f})")
                original_points = points
            else:
                centered_points = center_in_x(points, cx)

                is_centered2, cx2 = is_centered_x(centered_points)

                print(f"No era centrado en X. Pero ahora: {is_centered2} (centroide X = {cx2:.5f})")
                original_points = centered_points

            #return original_points.shape[0]
        elif file_ext == 'pkl':
            # Load point cloud
            data = torch.load(p, map_location='cpu')
            original_points = data['fine_xyz']
            #print(original_points.shape)

            is_centered, cx = is_centered_x(original_points)
            print(f"Centrado en X: {is_centered} (centroide X = {cx:.5f})")

            #return original_points.shape[0]
        elif file_ext == 'xyz':
            # Load point cloud
            data = np.loadtxt(p)
            original_points = data[:, :3]

            is_centered, cx = is_centered_x(original_points)
            print(f"Centrado en X: {is_centered} (centroide X = {cx:.5f})")

            #return data.shape[0]
    return

if file_ext == 'npy':
    files = list(path.glob("*.npy"))
    count(files, file_ext)
    #no_points = count(files, file_ext)
    #print("Number of points: ", no_points)

elif file_ext == 'pkl':
    files = list(path.glob("*.pkl"))
    count(files, file_ext)
    #no_points = count(files, file_ext)
    #print("Number of points: ", no_points)

elif file_ext == 'xyz':
    files = list(path.glob("*.xyz"))
    count(files, file_ext)
    #no_points = count(files, file_ext)
    #print("Number of points: ", no_points)

else:
    print("Invalid file extension.")

