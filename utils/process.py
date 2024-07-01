# This file is part of the animSnapBases project (https://github.com/ShMonem/animSnapBases).
# Copyright animSnapBases Shaimaa Monem, Peter Bener and Christian Lessig. All rights reserved.
# License: Apache-2.0

from os import path
from glob import glob
from io import StringIO
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import re

from mayavi import mlab
from itertools import count
import numpy as np
import h5py
from traits.api import HasTraits, Range, Instance, Bool, Int, on_trait_change
from traitsui.api import View, Item, HGroup, RangeEditor
from tvtk.api import tvtk
from tvtk.pyface.scene_editor import SceneEditor
from tvtk.common import configure_input, configure_input_data
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene
from pyface.timer.api import Timer


'''
The following functions are borrowed from: https://github.com/tneumann/splocs
Copyright (c) 2013 Thomas Neumann
'''
def load_ply(filename):
    try:
        from enthought.tvtk.api import tvtk
    except ImportError:
        try:
            from tvtk.api import tvtk
        except ImportError:
            print ("Reading PLY files requires TVTK. The easiest way is to install mayavi2")
            print ("(e.g. on Ubuntu: apt-get install mayavi2)")
            raise
    reader = tvtk.PLYReader(file_name=filename)
    reader.update()
    polys = reader.output.polys.to_array().reshape((-1, 4))
    assert np.all(polys[:,0] == 3)
    return reader.output.points.to_array(), polys[:,1:]

def load_off(filename, no_colors=False):
    lines = open(filename).readlines()
    lines = [line for line in lines if line.strip() != '' and line[0] != '#']
    assert lines[0].strip() in ['OFF', 'COFF'], 'OFF header missing'
    has_colors = lines[0].strip() == 'COFF'
    n_verts, n_faces, _ = map(int, lines[1].split())
    vertex_data = np.loadtxt(
        StringIO(''.join(lines[2:2 + n_verts])),
        dtype=float)
    if n_faces > 0:
        faces = np.loadtxt(StringIO(''.join(lines[2+n_verts:])), dtype=int)[:,1:]
    else:
        faces = None
    if has_colors:
        colors = vertex_data[:,3:].astype(np.uint8)
        vertex_data = vertex_data[:,:3]
    else:
        colors = None
    if no_colors:
        return vertex_data, faces
    else:
        return vertex_data, colors, faces

def convert_sequence_to_hdf5(filename_pattern, loader_function, hdf_output_file, max_frames, icreament):
    verts_all = []
    tris = None
    files = glob(path.expanduser(filename_pattern))
    sort_nicely(files)
    count = 0
    for i, f in enumerate(files):
        if i % icreament == 0 and count < max_frames:
            print ("loading file %d/%d [%s]" % (i+1, len(files), f) )
            verts, new_tris = loader_function(f)
            if tris is not None and new_tris.shape != tris.shape and new_tris != tris:
                raise ValueError("inconsistent topology between meshes of different frames")
            tris = new_tris
            verts_all.append(verts)
            count +=1

    verts_all = np.array(verts_all, np.float32)
    verts_all, tris, _, verts_mean, verts_scale = preprocess_mesh_animation(verts_all, tris)

    with h5py.File(hdf_output_file, 'w') as f:
        f.create_dataset('verts', data=verts_all, compression='gzip')
        f['tris'] = tris
        f.attrs['mean'] = verts_mean
        f.attrs['scale'] = verts_scale

    print ("saved as %s" % hdf_output_file )

def filter_reindex(condition, target):
    """
    >>> indices = np.array([1, 4, 1, 4])
    >>> condition = np.array([False, True, False, False, True])
    >>> filter_reindex(condition, indices).tolist()
    [0, 1, 0, 1]
    """
    if condition.dtype != bool:
        raise ValueError( "condition must be a binary array" )
    reindex = np.cumsum(condition) - 1
    return reindex[target]
def preprocess_mesh_animation(verts, tris):
    """
    Preprocess the mesh animation:
        - removes zero-area triangles
        - keep only the biggest connected component in the mesh
        - normalize animation into -0.5 ... 0.5 cube
    """
    print ("Vertices: ", verts.shape)
    print ("Triangles: ", verts.shape)
    assert verts.ndim == 3
    assert tris.ndim == 2
    # check for zero-area triangles and filter
    e1 = verts[0, tris[:,1]] - verts[0, tris[:,0]]
    e2 = verts[0, tris[:,2]] - verts[0, tris[:,0]]
    n = np.cross(e1, e2)
    tris = tris[veclen(n) > 1.e-8]
    # remove unconnected vertices
    ij = np.r_[np.c_[tris[:,0], tris[:,1]],
               np.c_[tris[:,0], tris[:,2]],
               np.c_[tris[:,1], tris[:,2]]]
    G = csr_matrix((np.ones(len(ij)), ij.T), shape=(verts.shape[1], verts.shape[1]))
    n_components, labels = connected_components(G, directed=False)
    if n_components > 1:
        size_components = np.bincount(labels)
        if len(size_components) > 1:
            print ("[warning] found %d connected components in the mesh, keeping only the biggest one" % n_components)
            print ("component sizes: " )
            print (size_components)
        keep_vert = labels == size_components.argmax()
    else:
        keep_vert = np.ones(verts.shape[1], bool)
    verts = verts[:, keep_vert, :]
    tris = filter_reindex(keep_vert, tris[keep_vert[tris].all(axis=1)])
    # normalize triangles to -0.5...0.5 cube
    verts_mean = verts.mean(axis=0).mean(axis=0)
    verts -= verts_mean
    verts_scale = np.abs(verts.ptp(axis=1)).max()
    verts /= verts_scale
    print ("after preprocessing:")
    print ("Vertices: ", verts.shape)
    print ("Triangles: ", verts.shape)
    return verts, tris, ~keep_vert, verts_mean, verts_scale

def veclen(vectors):
    """ return L2 norm (vector length) along the last axis, for example to compute the length of an array of vectors """
    return np.sqrt(np.sum(vectors**2, axis=-1))

def normalized(vectors):
    """ normalize array of vectors along the last axis """
    return vectors / veclen(vectors)[..., np.newaxis]

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def tryint(s):
    try:
        return int(s)
    except:
        return s

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

def homogenize(v, value=1):
    """ returns v as homogeneous vectors by inserting one more element into the last axis
    the parameter value defines which value to insert (meaningful values would be 0 and 1)
    >>> homogenize([1, 2, 3]).tolist()
    [1, 2, 3, 1]
    >>> homogenize([1, 2, 3], 9).tolist()
    [1, 2, 3, 9]
    >>> homogenize([[1, 2], [3, 4]]).tolist()
    [[1, 2, 1], [3, 4, 1]]
    """
    v = np.asanyarray(v)
    return np.insert(v, v.shape[-1], value, axis=-1)

def dehomogenize(a):
    """ makes homogeneous vectors inhomogenious by dividing by the last element in the last axis
    >>> dehomogenize([1, 2, 4, 2]).tolist()
    [0.5, 1.0, 2.0]
    >>> dehomogenize([[1, 2], [4, 4]]).tolist()
    [[0.5], [1.0]]
    """
    a = np.asfarray(a)
    return a[...,:-1] / a[...,np.newaxis,-1]

def transform(v, M, w=1):
    """ transforms vectors in v with the matrix M
    if matrix M has one more dimension then the vectors
    this will be done by homogenizing the vectors
    (with the last dimension filled with w) and
    then applying the transformation """
    if M.shape[0] == M.shape[1] == v.shape[-1] + 1:
        v1 = homogenize(v, value=w)
        return dehomogenize(np.dot(v1.reshape((-1,v1.shape[-1])), M.T)).reshape(v.shape)
    else:
        return np.dot(v.reshape((-1,v.shape[-1])), M.T).reshape(v.shape)

def find_rbm_procrustes(frompts, topts, rigid):
    """
    Finds a rigid body transformation M that moves points in frompts to the points in topts
    that is, it finds a rigid body motion [ R | t ] with R \in SO(3)

    This algorithm first approximates the rotation by solving
    the orthogonal procrustes problem.
    """
    # center data
    t0 = frompts.mean(0)
    t1 = topts.mean(0)
    frompts_local = frompts - t0
    topts_local = topts - t1
    # find best rotation - procrustes problem
    M = np.dot(topts_local.T, frompts_local)
    U, s, Vt = np.linalg.svd(M)
    R = np.dot(U, Vt)
    if np.linalg.det(R) < 0:
        R *= -1
    T0 = np.eye(4)
    if rigid:
        T0[:3, 3] = t1 - np.dot(R, t0)
    return T0
def align(input_hdf5_file, output_hdf5_file, rigid):
    data = h5py.File(input_hdf5_file, 'r')
    verts = data['verts'][()]#.value
    tris = data['tris'][()] #.value

    v0 = verts[0]
    verts_new = []
    for i, v in enumerate(verts):
        print ("frame %d/%d" % (i+1, len(verts)) )
        M = find_rbm_procrustes(v, v0, rigid)
        verts_new.append(transform(v, M))
    verts = np.array(verts_new, np.float32)

    with h5py.File(output_hdf5_file, 'w') as f:
        f.create_dataset('verts', data=verts, compression='gzip')
        f['tris'] = tris


def view_anim_file(hdf5_animation_file):
    weights = None
    with h5py.File(hdf5_animation_file, 'r') as f:
        verts = f['verts'][()]  # .value
        tris = f['tris'][()]  # .value
        if 'weights' in f:
            weights = f['weights'][()]  # .value

    pd = tvtk.PolyData(points=verts[0], polys=tris)
    normals = tvtk.PolyDataNormals(splitting=False)
    configure_input_data(normals, pd)

    # choose position and orientation
    actor = tvtk.Actor(mapper=tvtk.PolyDataMapper(), position=(0, 0, 0), orientation=(0, -90, 0))
    configure_input(actor.mapper, normals)
    actor.property.set(edge_color=(0.5, 0.5, 0.5), ambient=0.0,
                       specular=0.15, specular_power=128., shading=True, diffuse=0.8)

    fig = mlab.figure(bgcolor=(1, 1, 1), size=(1024, 1024))
    fig.scene.add_actor(actor)

    # Choose a view angle, and display the figure
    # mlab.view(90, 45, 7.5, [1.5, 0, 0])
    # mlab.savefig(filename='photofile.png')

    @mlab.animate(delay=40, ui=True)
    def animation():
        for i in count():
            if weights is not None:
                w_str = ",".join(["%0.2f"] * weights.shape[1])
                print("Frame %d Weights = " + w_str) % tuple([i] + weights[i].tolist())
            frame = i % len(verts)
            pd.points = verts[frame]
            fig.scene.render()
            yield

    a = animation()
    fig.scene.z_minus_view()
    mlab.show()

class Visualization(HasTraits):
    component = Int(0)
    _max_component_index = Int()
    activation = Range(-1., 1.)  # or fix -0.5
    oscillate = Bool(True)
    allow_negative = Bool(False)
    pd = Instance(tvtk.PolyData)
    normals = Instance(tvtk.PolyDataNormals)
    actor = Instance(tvtk.Actor)
    scene = Instance(MlabSceneModel, (), kw=dict(background=(1, 1, 1)))
    timer = Instance(Timer)

    def __init__(self, Xmean, tris, components):
        HasTraits.__init__(self)
        self._components = components
        self._max_component_index = len(components)
        self._Xmean = Xmean
        self.pd = tvtk.PolyData(points=Xmean, polys=tris)
        self.normals = tvtk.PolyDataNormals(splitting=False)
        configure_input_data(self.normals, self.pd)
        mapper = tvtk.PolyDataMapper()  # immediate_mode_rendering=True)
        self.actor = tvtk.Actor(mapper=mapper)
        configure_input(self.actor.mapper, self.normals)
        self.actor.mapper.lookup_table = tvtk.LookupTable(
            hue_range=(0.45, 0.6),  # blue  ## for red use (0.01, 0.03),
            saturation_range=(0., 0.8),
            # value_range = (.6, 1.),    # just adds more color to the object
        )
        self.scene.add_actor(self.actor)
        self.timer = Timer(40, self.animate)  # .next)

    def animate(self):
        for i in count():
            if self.oscillate:
                frame = i % 30
                alpha = np.sin(frame / 30. * np.pi * 2)
                if not self.allow_negative:
                    alpha = np.abs(alpha)
                self.activation = alpha
                next(self.animate())
            yield

    def next_animate(self):
        return next(self.animate)

    @on_trait_change('activation, component')
    def update_plot(self):
        c = self._components[self.component]
        self.pd.points = self._Xmean + self.activation * c
        magnitude = veclen(c)
        self.pd.point_data.scalars = magnitude
        self.actor.mapper.scalar_range = (0, magnitude.max())

        self.scene.render()

    view = View(
        Item('scene', editor=SceneEditor(scene_class=MayaviScene),
             height=600, width=800, show_label=False),
        HGroup(
            Item('component', editor=RangeEditor(
                is_float=False, low=0, high_name='_max_component_index', mode='spinner')),
            'activation',
            'oscillate',
            'allow_negative',
        ),
        resizable=True, title="View SPLOC's",
    )

def view_components(component_hdf5_file):
    Xmean, tris, components, names = load_splocs(component_hdf5_file)

    visualization = Visualization(Xmean, tris, components)
    visualization.configure_traits()

def load_splocs(component_hdf5_file):
    with h5py.File(component_hdf5_file, 'r') as f:
        tris = f['tris'][()] # .value
        Xmean = f['default'][()] # .value
        names = sorted(list(set(f.keys()) - set(['tris', 'default'])))
        components = np.array([
            f[name][()] - Xmean
            for name in names])
    return Xmean, tris, components, names
