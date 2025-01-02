"""
********************************************************************************
MIT License
Copyright (c) 2024 Shaimaa Monem

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
********************************************************************************
"""
import numpy as np
from numpy import linalg as npla
import csv 
#---------------------------------
# # polyscope and potpourri3d includes
import polyscope as ps
ps.init()
import potpourri3d as pp3d
#---------------------------------
# # libigl includes
import igl
import os
from utils.utils import  read_mesh_file
import matplotlib.pyplot as plt

root_folder = os.getcwd()
# #root_folder = os.path.join(os.getcwd(), "tutorial")

def readOriginalMesh(filePath, tri=False):
	## Load a mesh in OFF format
	#root_folder = os.path.join(os.getcwd(), filePath)
	if tri:
		v0, f0 = igl.read_triangle_mesh(filePath)
	else :
		v0, _, f0 = read_mesh_file(filePath)
	## Print the vertices and faces matrices
	print("Vertices: ", len(v0))
	print("Faces: ", len(f0))
	GauCur = igl.gaussian_curvature(v0, f0)
	print("GaussianCurvature: ", len(GauCur))  # per vertex scalar
	return v0, f0

def compute_accuracy(orig_mesh, is_surface, start, end, step,  full_files_path, max_reduction, reduced_files_path, reduced_ext, type="tetstrain", writer=None):
	_, f = readOriginalMesh(orig_mesh, False)


	headerComp = ['numComponent', 'Frame', 'minError', 'mnaxError', 'maxAccumError', 'minAccumError', 'sumAccumError']

	with open('PODerrorsVal.csv', 'w', encoding='UTF8') as errorsPODFile:
		writer = csv.writer(errorsPODFile)
		writer.writerow(headerComp)
		accum_error_on_all_verts = []
		accum_per_vert_min =[]
		accum_per_vert_max = []
		normal_angles_r=[]
		normal_angles = []
		global frame, reduction, frames_range
		frame = 1
		reduction = 10
		frames_range = []

		def callback():
			global frame, reduction, frames_range

			k = frame
			r = reduction
			f1 =full_files_path + str(k) + ".off"
			f2 = reduced_files_path+ str(r) + reduced_ext + str(k) +".off"
			(v, _) = pp3d.read_mesh(f1)
			(v_r, _) = pp3d.read_mesh(f2)

			min_corner = np.min(v_r, axis=0)
			max_corner = np.max(v_r, axis=0)
			center = (min_corner + max_corner) / 2
			target_position = center  # Look at the center of the bounding box
			ps.look_at((1.5,1.5,-2), target_position)

			frame_err = ((v - v_r)**2).sum(axis=1)
			frames_range.append(frame_err)

			n = igl.per_vertex_normals(v, f)
			n_r = igl.per_vertex_normals(v_r, f)
			normal_angles_r.append(abs(n - n_r).sum(axis=1))

			#ps_mesh2 = ps.register_surface_mesh("mesh", v, f)
			ps_mesh2 = ps.register_surface_mesh("mesh_r", v_r, f)

			ps_cloud = ps.register_point_cloud("reduced mesh verts", v_r)
			ps_cloud.add_vector_quantity("Full Normals", n , length=0.06, color=(0.2, 0.5, 0.5), enabled=True)
			ps_cloud.add_vector_quantity("Reduced Normals", n_r, length=0.06, enabled=True)
			ps_mesh2.add_scalar_quantity("Error", frame_err, defined_on='vertices', cmap='jet', vminmax=(2, 20), enabled=True)
			ps_mesh2.add_scalar_quantity("Norms diff", abs(n - n_r).sum(axis=1), defined_on='vertices', cmap='jet', vminmax=(0.0002, 2.0), enabled=True)

			frame +=1
			if frame == end + 1:
				accum_error_on_all_verts.append(np.array(frames_range).sum())
				accum_per_vert_max.append(np.array(frames_range).max())
				accum_per_vert_min.append(np.array(frames_range).min())
				normal_angles.append(np.array(normal_angles_r).min())

				frame = 1
				reduction += 10
				frames_range = []

			if reduction  >= max_reduction + 1:
				ps.unshow()

	# Update the Polyscope viewer
	ps.set_user_callback(callback)
	ps.show()

	plt.figure('Error measures for DEIM' + type , figsize=(20, 10))
	plt.xlabel('Reduction Dimension (r)')
	plt.ylabel('Error measure')
	rows = 2
	cols = 2
	plt.subplot(rows, cols, 1)
	plt.plot(accum_error_on_all_verts, label='$accum on all verts')
	plt.subplot(rows, cols, 2)
	plt.plot(accum_per_vert_max, label='$ accum max on per vert')
	plt.subplot(rows, cols, 3)
	plt.plot(accum_per_vert_min, label='$ accum min on per vert')
	plt.subplot(rows, cols, 4)
	plt.plot(normal_angles, label='$ angle max on per vert')

	plt.legend()
	plt.show()
