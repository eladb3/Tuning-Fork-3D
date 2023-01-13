import trimesh, cv2
import numpy as np
import plotly.graph_objects as go
import open3d as o3d

def stl2np(path_stl_file):
  mesh = trimesh.load_mesh(path_stl_file)
  assert(mesh.is_watertight) # you cannot build a solid if your volume is not tight
  volume = mesh.voxelized(pitch=0.05)
  mat = volume.matrix # matrix of boolean
  return mat

def np2stl(mat, output):
  volume = trimesh.voxel.base.VoxelGrid(mat)
  with open(output, "wb") as f:
    volume.marching_cubes.export(f, "stl")

def gen_voxel(m, prong_length, thickness, close_sides=False, sides_thickness=15, handle_length=None):
  """
  Args:
    m: 2d numpy array 
    prong_length: length of prong
    thickness: thickness of the base
    close_sides: open sides let you see the inside
    sides_thickness:
    handle_length: 
  """

  H, W = m.shape
  if handle_length is None:
    handle_length= H * 3

  array = np.zeros([H, W*3, handle_length])

  def _put_rec(a, p1, p2, color, thickness):
    if thickness < 0:
      ds=[0]
    else:
      ds = range(thickness)
    
    (x1,y1), (x2,y2) = p1, p2
    for d in ds:
      t = 1 if thickness > 0 else -1
      a = cv2.rectangle(a, (x1+d,y1+d), (x2-d,y2-d), (255,0,0), t)
    return a
  # handle
  for i in range(handle_length):
    a=np.stack([array[:, :, i] for _ in range(3)], axis=2)
    _thickness = thickness
    if close_sides and i < sides_thickness:
      _thickness = -1
    array[:, :, i] = _put_rec(a, (W, 0), (2*W-1, H-1), (255,0,0), _thickness)[:, :, 0]


  left_prong = [(0, 0), (W-1, H-1)]
  right_prong = [(2*W, 0), (3*W-1, H-1)]
  start_prong = int(0.75 * handle_length)
  for i in range(start_prong, handle_length):
    a=np.stack([array[:, :, i] for _ in range(3)], axis=2)
    _thickness = thickness
    if close_sides and i < start_prong + sides_thickness or i > handle_length - sides_thickness - 1:
      _thickness = -1
      a = np.zeros_like(array[:, :, i]) + 255
    else:
      a = _put_rec(a, *left_prong, (255,0,0), _thickness)
      a = _put_rec(a, *right_prong, (255,0,0), _thickness)
      a = a[:, :, 0]
    array[:, :, i] = a

  prongs = np.zeros([H, W*3, prong_length])
  single_prong = np.stack([m for _ in range(prong_length)], axis = 2)
  (x1,y1), (x2,y2) = left_prong
  prongs[y1:y2+1, x1:x2+1, :] = single_prong
  (x1,y1), (x2,y2) = right_prong
  prongs[y1:y2+1, x1:x2+1, :] = single_prong
  array=np.concatenate([array, prongs], axis = 2)
  return array

def show_voxel(mat=None, path=None, axis_visible=True, color=None, opacity=0.60):

  if path is None:
    assert mat is not None
    path = "_tmp.stl"
    np2stl(mat, path)

  mesh = o3d.io.read_triangle_mesh(path)
  assert not mesh.is_empty()
  if not mesh.has_vertex_normals(): mesh.compute_vertex_normals()
  if not mesh.has_triangle_normals(): mesh.compute_triangle_normals()
  triangles = np.asarray(mesh.triangles)
  vertices = np.asarray(mesh.vertices)
  colors = None
  if color is None and mesh.has_triangle_normals():
      colors = (0.5, 0.5, 0.5) + np.asarray(mesh.triangle_normals) * 0.5
      colors = tuple(map(tuple, colors))
  elif color is not None:
      colors = color
  else:
      colors = (1.0, 0.0, 0.0)
  fig = go.Figure(
      data=[
          go.Mesh3d(
              x=vertices[:,0],
              y=vertices[:,1],
              z=vertices[:,2],
              i=triangles[:,0],
              j=triangles[:,1],
              k=triangles[:,2],
              facecolor=colors,
              opacity=opacity)
      ],
      layout=dict(
          scene=dict(
              xaxis=dict(visible=axis_visible),
              yaxis=dict(visible=axis_visible),
              zaxis=dict(visible=axis_visible)
          )
      )
  )
  fig.update_layout(scene_aspectmode='data')
  fig.show()
  

# Example with random m
def sample_m(H, W):
    mask = (np.random.random([H,W]) > 0.5)
    m = np.zeros([H,W])
    m[mask] = 255
    thickness = 5
    m[:thickness] = 255
    m[-thickness:] = 255
    m[:, :thickness] = 255
    m[:, -thickness:] = 255
    return m
if __name__ == '__main__':
    H, W=[32,32] # H, W
    m = sample_m(H, W)
    thickness = 4
    prong_length = 128

    mat = gen_voxel(m, prong_length, thickness, close_sides=True, sides_thickness=5, handle_length=None)
    np2stl(mat, "tuning-fork.stl")
    # Not sure why but the plotly visualization changes the height / width ratio
    # You can download the tuning-fork.stl file and view it on your computer to get more accruate visualization
    show_voxel(path="tuning-fork.stl") 

