Add experimental cell-colored Gauss-Seidel particle contacts that reuse the
discontinuous strain-cell coloring for the selected velocity basis.
With GIMP integration, evaluate contacts at particle centers using velocity-basis
weights while retaining GIMP material transfers.
Update contacts after each strain color to reduce contact violations introduced
by the material solve.
Support GS and GS-SOA, including finite-mass colliders with rigidity feedback
applied once per full color sweep.
Enable colored contacts automatically for supported configurations, with an
explicit opt-out through `gs_colored_contacts=False`.
