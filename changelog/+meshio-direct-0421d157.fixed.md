Fix a trimesh 5 `DeprecationWarning` when `ModelBuilder.add_urdf()` or `Mesh.create_from_file()` loads a Gmsh `.msh` mesh. Newton now reads these files with meshio directly.
