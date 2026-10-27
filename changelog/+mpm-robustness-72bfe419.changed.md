Change `SolverImplicitMPM` strain modes, particle contacts, and Gauss-Seidel scheduling:

  - Orthogonalize P1d and Q1d strain modes in their nodal basis (experimental), keeping supported modes in partially filled cells and yield strengths consistent with P0 regardless of fill fraction. Existing `strain_basis` settings remain valid, but results can change in partially filled cells.
  - Solve particle contacts inside the Gauss-Seidel strain color loop by default for `gs` and `gs-soa` with `pic*` collider bases and discontinuous strain bases (experimental), which improves convergence at a fixed iteration budget. Set `Config.gs_colored_contacts = False` to keep contacts in a separate pass.
  - Activate particle contacts (`pic*` collider bases) within 0.25 voxels of colliders and let them close the remaining gap within the step instead of stopping material before it reaches the surface. Collider adhesion acts on these contacts once they touch, and grid-based collider nodes keep enforcing the full contact condition. No changes are required.
  - Speed up Gauss-Seidel solves on small scenes and with discontinuous strain bases. No changes are required.
