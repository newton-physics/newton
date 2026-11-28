Use the face normal for collision contacts at coplanar interior triangle-mesh
edges, preserving lateral normals on real boundaries and creases. This prevents
boxes from being deflected sideways while sliding across a flat triangulated
surface, and keeps support under slightly tilted shapes that straddle such
edges.
