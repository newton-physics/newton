Use automatic body-local voxel coverage for experimental body-pair contact reduction, sharing scale-independent grid sizing and indexing with SDF reduction. Keep varying-depth cells conservatively; pass an explicit `BodyPairReductionConfig.cell_size` to retain the previous projected-grid behavior.

Apply scale-independent voxel sizing and normalized boundary indexing to ordinary mesh/SDF reduction even when body-pair reduction is disabled. Retained contacts and trajectories can change for small or degenerate geometry; no API or scene migration is required.
