Fix `ControllerJointImpedance` producing wrong or out-of-bounds torque indices on
models whose joint coordinate and DOF counts differ (free, ball, or distance
joints). A single index array previously had to be valid in both spaces at once,
so no correct value existed. Index arrays are now validated for range, matched
coordinate/DOF pairing, controllable joint type, and per-articulation grouping;
the last of these previously allowed out-of-range and negative mass-matrix reads.
