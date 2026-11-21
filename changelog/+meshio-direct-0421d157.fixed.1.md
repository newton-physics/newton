Fix `Mesh.create_from_file()` and `TetMesh.create_from_file()` exiting the Python process when meshio cannot read the file. They now raise `ValueError`.
