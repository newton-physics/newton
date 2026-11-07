Fix coupled solvers refreshing particle material properties after `MODEL_PROPERTIES` notifications while preserving particle ownership and proxy effective masses.

Preserve implicit MPM particle-mask and material-volume allocations during same-size material updates so captured solver steps read the updated buffers.
