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
from polyscope import screenshot

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

def compute_accuracy(orig_mesh, snapsFormat, frame_start, frame_end, r_start, r_step, r_end,
					 full_files_path, reduced_files_path, reduced_ext, type, dir, case="_test_on_training_set"):

	global f, num_verts, denom
	v0, f = readOriginalMesh(orig_mesh, False)
	denom = np.sqrt(3 * (frame_end - frame_start) * v0.shape[0])
	num_verts = v0.shape[0]
	del v0
	def angle_between_row_vectors(vectors1, vectors2):
		# Ensure vectors are NumPy arrays
		vectors1 = np.array(vectors1)
		vectors2 = np.array(vectors2)
		# Calculate dot products for corresponding pairs
		dot_products = np.einsum('ij,ij->i', vectors1, vectors2)
		# Calculate norms for each vector
		norms1 = np.linalg.norm(vectors1, axis=1)
		norms2 = np.linalg.norm(vectors2, axis=1)
		# Calculate cosine of angle for each pair of vectors
		cos_thetas = dot_products / (norms1 * norms2)
		# Ensure the cosine values are within the valid range [-1, 1] to avoid NaNs from floating point issues
		cos_thetas = np.clip(cos_thetas, -1.0, 1.0)
		# Calculate angles in radians
		angles_radians = np.arccos(cos_thetas)
		# Convert to degrees
		angles_degrees = np.degrees(angles_radians)
		return angles_degrees

	def write_to_file():
		global num_verts

		headerComp = ['numComponent', 'norm_error_min', 'norm_error_mean', 'norm_error_max', 'norm_error_sum',
						'angle_error_min', 'angle_error_mean', 'angle_error_max', 'angle_error_sum',
						"accum_norm_min","accum_norm_meann", "accum_norm_max" ,
						"accum_angle_min","accum_angle_mean","accum_angle_max" ]

		with open(os.path.join(dir, type + "_on_mesh_measures"+case) +'.csv', 'w', encoding='UTF8') as acc_file:
			writer = csv.writer(acc_file)
			writer.writerow(headerComp)
			r_list=[]
			for r in range(r_start, r_end, r_step):
				accum_norm_mat = np.zeros(num_verts)
				accum_angle_mat = np.zeros(num_verts)
				frames_err_r = []
				normal_angles_r = []
				full_mesh_error =[]
				for k in range(frame_start, frame_end, 1):
					f1 = full_files_path + str(k) + snapsFormat
					f2 = reduced_files_path + str(r) + reduced_ext + str(k) + snapsFormat
					(v, f) = pp3d.read_mesh(f1)
					(v_r, _) = pp3d.read_mesh(f2)

					# pointwise error
					frame_err = ((v - v_r) ** 2).sum(axis=1) / ((v) ** 2).sum(axis=1)/denom
					full_frame_error = npla.norm(v-v_r)/npla.norm(v)/denom
					full_mesh_error.append(full_frame_error)
					frames_err_r.append(frame_err)
					accum_norm_mat += frame_err

					n = igl.per_vertex_normals(v, f)
					n_r = igl.per_vertex_normals(v_r, f)
					normal_angles_r.append(angle_between_row_vectors(n, n_r))
					accum_angle_mat += angle_between_row_vectors(n, n_r)

				plt.plot(full_mesh_error, label= r)
				plt.show()
				plt.plot(np.array(frames_err_r).sum(axis=1), label= r)
				plt.legend()
				plt.show()
			# 	writer.writerow([r, np.array(frames_err_r).min(),  np.array(frames_err_r).mean(),
			# 					 np.array(frames_err_r).max(),np.array(frames_err_r).sum(),
			# 					 np.array(normal_angles_r).min(), np.array(normal_angles_r).mean(),
			# 					 np.array(normal_angles_r).max(), np.array(normal_angles_r).sum(),
			# 					accum_norm_mat.min(), accum_norm_mat.mean(), accum_norm_mat.max(),
			# 					accum_angle_mat.min(), accum_angle_mat.mean(), accum_angle_mat.max()])
			#
			# 	r_list.append(([r, np.array(frames_err_r).min(),  np.array(frames_err_r).mean(),
			# 					 np.array(frames_err_r).max(),np.array(frames_err_r).sum(),
			# 					 np.array(normal_angles_r).min(), np.array(normal_angles_r).mean(),
			# 					 np.array(normal_angles_r).max(), np.array(normal_angles_r).sum(),
			# 					accum_norm_mat.min(), accum_norm_mat.mean(), accum_norm_mat.max(),
			# 					accum_angle_mat.min(), accum_angle_mat.mean(), accum_angle_mat.max()]))
			#
			# r_list = np.array(r_list)
			# for col  in range(1,r_list.shape[1]):
			# 	plt.plot(r_list[:,0], r_list[:, col], label=col)
			# 	plt.legend()
			# 	plt.show()
		acc_file.close()
		print("Accuracy measures written to file", acc_file)
	def visualize(overlap=True, show_angles_norm=True):

		global frame, reduction, frames_range, denom, f, normal_angles_r,\
			normal_angles, accum_per_vert_min, screenshot_dir


		accum_per_vert_min = []
		normal_angles = []
		frame = frame_start
		reduction = r_start
		frames_range = []
		normal_angles_r = []

		screenshot_dir = os.path.join(dir, "error_screenshots"+case)
		if not os.path.exists(screenshot_dir):
			os.makedirs(screenshot_dir)
			print("Directory is created to store screenshots!", screenshot_dir)

		def callback():
			global frame, reduction, frames_range, accum_per_vert_min, \
				normal_angles_r, normal_angles, denom, f, screenshot_dir

			k = frame
			r = reduction
			if overlap:

				screenshot_file = os.path.join(screenshot_dir, type + "_comp_overlap_" + str(r) + "_fr_" +str(k))
			else:
				screenshot_file = os.path.join(screenshot_dir, type + "_comp_nonoverlap_" + str(r) + "_fr_" +str(k))

			f1 =full_files_path + str(k) + snapsFormat
			f2 = reduced_files_path+ str(r) + reduced_ext + str(k) +snapsFormat
			(v, _) = pp3d.read_mesh(f1)
			(v_r, _) = pp3d.read_mesh(f2)

			if overlap:
				min_corner = np.min(v, axis=0)
				max_corner = np.max(v, axis=0)
				center = (min_corner + max_corner) / 2
				bounding_box_size = np.linalg.norm(max_corner - min_corner)

				# Set the camera position in front of the object
				front_camera_position = (center[0], center[1], center[2] + 2 * bounding_box_size)

				# Fix the camera to look at the object's center
				ps.look_at(front_camera_position, center)
			else:
				v[:, 0] += 2
				min_corner = np.min(np.vstack([v_r,v]), axis=0)
				max_corner = np.max(np.vstack([v_r,v]), axis=0)
				center = (min_corner + max_corner) / 2
				target_position = center  # Look at the center of the bounding box

				size = max_corner - min_corner
				camera_distance = 1.1 * np.max(size)  # Adjust multiplier based on your specific needs
				camera_position = center + np.array([0, 0, camera_distance])
				ps.look_at(camera_position, center)

			frame_err = ((v - v_r)**2).sum(axis=1)/((v)**2).sum(axis=1)


			n = igl.per_vertex_normals(v, f)
			n_r = igl.per_vertex_normals(v_r, f)

			ps_mesh1 = ps.register_surface_mesh("mesh", v, f, color=(227, 156, 28))
			ps_mesh2 = ps.register_surface_mesh("mesh_r", v_r, f)

			# ps_cloud = ps.register_point_cloud("reduced mesh verts", v_r)

			if show_angles_norm:
				ps_mesh2.add_scalar_quantity("Norms diff",angle_between_row_vectors(n, n_r), defined_on='vertices',
											 cmap='jet', vminmax=(0, 360), enabled=True)
			else:
				ps_mesh2.add_scalar_quantity("Error", frame_err, defined_on='vertices', cmap='jet', vminmax=(0, 10),
											 enabled=True)
			ps.set_ground_plane_mode("shadow_only")
			ps.screenshot(screenshot_file)
			frame +=1
			if frame == frame_end + 1:
				frame = 1
				reduction += r_step


			if reduction  >= r_end + 1:
				ps.unshow()

		# Update the Polyscope viewer
		ps.set_user_callback(callback)
		ps.show()

	write_to_file()
	visualize(overlap=True, show_angles_norm=True)
	visualize(overlap=True, show_angles_norm=False)
