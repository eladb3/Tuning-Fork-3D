
import open3d as o3d

def simplify_mesh(path, target_number_of_triangles = None):
    mesh_in = o3d.io.read_triangle_mesh(path)
    mesh = mesh_in.remove_duplicated_triangles()
    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_non_manifold_edges()
    mesh = mesh.remove_unreferenced_vertices()
    mesh_in = mesh
    print(
        f'Input mesh has {len(mesh_in.vertices)} vertices and {len(mesh_in.triangles)} triangles'
    )
    if target_number_of_triangles is None:
        target_number_of_triangles = len(mesh_in.triangles) // 2
    mesh_smp = mesh_in.simplify_quadric_decimation(target_number_of_triangles=target_number_of_triangles)
    print(
        f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles'
    )

    # mesh_smp = mesh_in.simplify_quadric_decimation(target_number_of_triangles=1700)
    # print(
    #     f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles'
    # )
    return mesh_smp
