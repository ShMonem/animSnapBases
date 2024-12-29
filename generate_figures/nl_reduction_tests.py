import numpy as np
import matplotlib.pyplot as plt
from utils.utils import log_time
from config.config import Config_parameters
import csv
from utils.utils import testSparsity, test_linear_dependency
from utils.support import compute_edge_incidence_matrix_on_tets, extract_sub_vertices_and_edges, extract_sub_vertices_and_tet_edges
import os
from snapbases.constraintsComponents import constraintsComponents

angle = 0
frame = 1


def tets_plots_deim(nlConst_bases: constraintsComponents, param:Config_parameters, writer=None):
    """
    Plots different reconstruction errors for varying reduction dimensions "r".
    :param f: Original tensor (T, N, 3)
    :param V_f: Basis tensor (N, max_r, 3)
    :param max_r: Maximum reduction dimension (number of modes)
    """
    constProj_output_directory = param.constProj_output_directory
    @log_time(constProj_output_directory)
    def run_tests(nlConst_bases: constraintsComponents, constProj_output_directory, param:Config_parameters, writer=None):
        k = nlConst_bases.numComp
        p = nlConst_bases.nonlinearSnapshots.constraintsSize
        rp_values = range(p, k * p + 1, p)
        r_values = range(1, k + 1)

        # PCA tests --------------------------------------------------------------------------------------------------------
        plt.figure('Error measures for PCA.', figsize=(20, 10))

        store_kp_singVals = True

        rows = 1
        cols = 3
        plt.subplot(rows, cols, 1)

        # singular values at 'K' largest deformation blocks during PCA bases extarction
        # singVals starts from entry 4 in the measures_at_largeDeforVerts array
        mark=['bo', 'ro', 'go']
        for i in range(p):
            plt.plot(r_values, nlConst_bases.measures_at_largeDeforVerts[:, 3+i] /
                     nlConst_bases.measures_at_largeDeforVerts[:, 3+i].max(), mark[i], ls='-.', label=f'$\sigma_{{{i}}}$')

        plt.legend(loc='upper center')
        plt.xlabel('Reduction Dimension (r)')
        plt.ylabel('Normalized $\sigma$')
        plt.title("singVals at larg_deformation points")
        # plt.yscale("log")
        # plt.xticks(np.arange(1, k+1, 1))
        plt.legend()

        plt.subplot(rows, cols, 2)
        # residual_norm values at 'K' largest deformation blocks: norm(R) is expected to be the 3rd entry
        plt.plot(r_values, nlConst_bases.measures_at_largeDeforVerts[:, 2], 'rv', ls='-', label='$\| R_{pca} \|_F$ blocks')
        plt.legend(loc='upper center')
        plt.xlabel('Reduction Dimension (r)')
        plt.ylabel('Fro. nom')
        plt.title("norm(R) at K PCA bases extraction")
        # plt.yscale("log")
        # plt.xticks(np.arange(1, k + 1, 1))
        plt.legend()

        plt.subplot(rows, cols, 3)
        # singular vals for the bases over full Kp range
        if store_kp_singVals:
            header_ = ['x', 'y', 'z']
            file_name_ = os.path.join(constProj_output_directory, "deim_Kp_singVals")
            with open(file_name_ + '.csv', 'w', encoding='UTF8') as dataFile_:
                writer_ = csv.writer(dataFile_)
                writer_.writerow(header_)

                s = nlConst_bases.test_basesSingVals(writer_)

            dataFile_.close()
        else:
            s = nlConst_bases.test_basesSingVals()

        values = range(1, k*p + 1, 1)
        plt.plot(values, s[:, 0], 'bo', ls='--', label='$\sigma_x$')
        plt.plot(values, s[:, 1], 'ro', ls='--', label='$\sigma_y$')
        plt.plot(values, s[:, 2], 'go', ls='--', label='$\sigma_z$')
        plt.legend(loc='upper center')
        plt.xlabel('Reduction Dimension (r)')
        plt.ylabel('Fro. nom')
        plt.title("Normalized singVal(bases), full Kp range")
        # plt.yscale("log")
        # plt.xticks(values)
        plt.legend()
        fig_name = os.path.join(constProj_output_directory, 'pca_extraction_tests')
        plt.savefig(fig_name)
        # End of PCA tests -------------------------------------------------------------------------------------------------

        # After post-process tests
        testSparsity(nlConst_bases.comps)
        test_linear_dependency(nlConst_bases.comps, 3,
                               nlConst_bases.numComp * nlConst_bases.nonlinearSnapshots.constraintsSize)

        if param.constProj_orthogonal:
            nlConst_bases.is_utmu_orthogonal()  # test U^T M U = I (Kp x Kp)

        # DEIM tests -------------------------------------------------------------------------------------------------------

        frobenius_errors = []
        max_errors = []
        relative_errors_x = []
        relative_errors_y = []
        relative_errors_z = []
        best_num_element_to_plot = 0
        f = nlConst_bases.nonlinearSnapshots.snapTensor
        for r in r_values:
            # Reconstruct the tensor for the current r
            f_reconstructed = nlConst_bases.deim_constructed(r)

            # Compute various errors
            fro_error = nlConst_bases.frobenius_error(f, f_reconstructed)
            max_err = nlConst_bases.max_pointwise_error(f, f_reconstructed)
            rel_errors = nlConst_bases.relative_error_per_component(f, f_reconstructed)

            # Store errors
            frobenius_errors.append(fro_error)
            max_errors.append(max_err)
            relative_errors_x.append(rel_errors[0])
            relative_errors_y.append(rel_errors[1])
            relative_errors_z.append(rel_errors[2])

            if writer is not None:
                writer.writerow([r, fro_error, max_err, rel_errors[0], rel_errors[1], rel_errors[2]])

        # Plot Frobenius and inf norm error
        plt.figure('Error measures for DEIM ', figsize=(20, 10))

        plt.subplot(1, 2, 1)
        plt.plot(frobenius_errors, label='Frobenius Error', marker='o')
        plt.plot(r_values, max_errors, label='Inf Error', marker='o')
        plt.xlabel('Reduction Dimension (r)')
        plt.ylabel('Error')
        plt.title('Frobenius Norm')
        plt.yscale("log")
        # Set x-ticks to integers only
        # plt.xticks(np.arange(1, k + 1, 1)) # range 0 <= r <= numComponents
        plt.legend()

        # Plot Relative Errors for each component (x, y, z)
        plt.subplot(1, 2, 2)
        relative_error = np.sum(np.array([relative_errors_x, relative_errors_y, relative_errors_z]), axis=0)
        plt.plot(r_values, relative_errors_x, label='Relative Error X', marker='o')
        plt.plot(r_values, relative_errors_y, label='Relative Error Y', marker='x')
        plt.plot(r_values, relative_errors_z, label='Relative Error Z', marker='s')
        plt.plot(r_values, relative_error, label='sumRelative Error', marker='v')
        plt.xlabel('Reduction Dimension (r)')
        plt.ylabel('Relative Error')
        plt.title('Relative Errors per Component (X, Y, Z)')
        plt.yscale("log")
        # plt.xticks(np.arange(1, k+1, 1))
        plt.legend()

        #plt.tight_layout()
        fig_name = os.path.join(constProj_output_directory, 'deim_convergence_tests')
        plt.savefig(fig_name)

        plt.figure('Number of constrained elements in DEIM ', figsize=(20, 10))
        plt.subplot(1, 1, 1)
        plt.plot(nlConst_bases.deim_alpha_ranges, 'bo', ls='--', label=' 0 < elements < e')
        plt.xlabel('Reduction Dimension (r)')
        plt.ylabel('number of elements')
        plt.title('Number of constrained elements in DEIM ')

        fig_name = os.path.join(constProj_output_directory, 'deim_numberOfElements')
        plt.legend()
        plt.savefig(fig_name)

        if param.visualize_deim_elements:
            visualize_interpolation_elements(nlConst_bases, param.visualize_deim_elements_at_K, constProj_output_directory)

    run_tests(nlConst_bases, constProj_output_directory, param, writer)
    # End of DEIM tests ------------------------------------------------------------------------------------------------

    # plt.show()


def visualize_interpolation_elements(nlConst_bases: constraintsComponents, visualize_deim_elements_at_K,
                                     constProj_output_directory, ele_color=(0.5, 0.8, 0.5), num_frames = 20, file_prefix = "frame"):
    """
    Highlights specific elements (vertices, tetrahedra, faces) in a tetrahedral mesh using Polyscope.

    Parameters:
    - vertices: np.ndarray, array of vertex positions.
    - tets: np.ndarray, array of tetrahedral indices.
    - highlight_verts: list[int], indices of vertices to highlight.
    - highlight_tets: list[int], indices of tetrahedra to highlight.
    - highlight_faces: list[tuple], specific faces (triplets of vertex indices) to highlight.
    """
    import polyscope as ps
    ps.init()
    deim_verts = nlConst_bases.deim_interpol_verts[:visualize_deim_elements_at_K]
    highlight_elements = nlConst_bases.deim_alpha[:nlConst_bases.deim_alpha_ranges[visualize_deim_elements_at_K - 1]]
    highlight_type = nlConst_bases.nonlinearSnapshots.ele_type

    # Register the mesh
    ps.register_volume_mesh("Tet Mesh", nlConst_bases.nonlinearSnapshots.verts, nlConst_bases.nonlinearSnapshots.tets,
                            transparency=0.1, color=(0.89, 0.807, 0.565))
    ps.register_point_cloud("deim Vertices", nlConst_bases.nonlinearSnapshots.verts[deim_verts], enabled=True,
                            color=(0.9, 0.1, 0.25), radius=0.008)

    # Highlight vertices
    if highlight_type == "_verts":
        ps.register_point_cloud("Highlighted Vertices", nlConst_bases.nonlinearSnapshots.verts[highlight_elements],
                                enabled=True, color=ele_color)

    # Highlight tetrahedra
    elif highlight_type == "_tets":
        ps.register_volume_mesh("Highlighted Tets", nlConst_bases.nonlinearSnapshots.verts,
                                nlConst_bases.nonlinearSnapshots.tets[highlight_elements], transparency=0.8,
                                color=ele_color)

    # Highlight faces
    elif highlight_type == "_faces":
        faces = nlConst_bases.nonlinearSnapshots.tris[highlight_elements]
        ps.register_surface_mesh("Highlighted Faces", nlConst_bases.nonlinearSnapshots.verts, faces)

    # Highlight edges
    elif highlight_type == "_triEdges":
        edges = nlConst_bases.nonlinearSnapshots.edges[highlight_elements]
        sub_verts, sub_edges = extract_sub_vertices_and_edges(nlConst_bases.nonlinearSnapshots.verts, edges)
        ps.register_curve_network("Highlighted Tri- Edges", sub_verts, sub_edges)

    elif highlight_type == "_tetEdges":
        # TODO: check if required!
        edges = compute_edge_incidence_matrix_on_tets(nlConst_bases.nonlinearSnapshots.tets)[highlight_elements]
        sub_verts, sub_edges = extract_sub_vertices_and_tet_edges(nlConst_bases.nonlinearSnapshots.verts, edges)
        ps.register_curve_network("Highlighted Tet-Edges", sub_verts, sub_edges)

    #ps.show()
    output_dir = os.path.join(constProj_output_directory, "rotation_scene_snapshots")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Compute the bounding box of the vertices
    min_corner = np.min(nlConst_bases.nonlinearSnapshots.verts, axis=0)
    max_corner = np.max(nlConst_bases.nonlinearSnapshots.verts, axis=0)
    center = (min_corner + max_corner) / 2
    bounding_box_size = np.linalg.norm(max_corner - min_corner)

    # Determine camera distance (e.g., 2x bounding box size for full view)
    camera_distance = 1.1 * bounding_box_size
    ps.set_ground_plane_mode("none")

    global angle, frame
    angle = 360.0 / num_frames
    frame = 1

    def callback():
        # Rotate the view incrementally
        global angle, frame
        angle += angle
        camera_position = (
            center[0] + camera_distance * np.sin(np.radians(angle)),
            center[1],
            center[2] + camera_distance * np.cos(np.radians(angle)),
        )
        target_position = center  # Look at the center of the bounding box
        ps.look_at(camera_position, target_position)

        if frame < num_frames:
            # Capture the screenshot
            filename = os.path.join(output_dir, f"{file_prefix}_{frame:03d}.png")
            ps.screenshot(filename, transparent_bg=False)
            frame +=1
    # Update the Polyscope viewer
    ps.set_user_callback(callback)
    ps.show()

    print(f"Captured {num_frames} frames in {output_dir}")


