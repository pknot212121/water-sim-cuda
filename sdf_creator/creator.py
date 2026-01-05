import point_cloud_utils as pcu
import numpy as np

v, f = pcu.load_mesh_vf("../Glass_Cup.obj")

center = v.mean(axis=0)
v -= center

scale = np.abs(v).max()
print(scale)
v = v / scale * 0.9
print(np.max(v))

resolution = 128
x = np.linspace(-1, 1, resolution)
y = np.linspace(-1, 1, resolution)
z = np.linspace(-1, 1, resolution)
grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
query_pts = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1)

dist, face_id, closest_pts = pcu.signed_distance_to_mesh(query_pts, v, f)

sdf_matrix = dist.reshape(resolution, resolution, resolution)
sdf_matrix.astype(np.float32).tofile('model_cup.sdf')