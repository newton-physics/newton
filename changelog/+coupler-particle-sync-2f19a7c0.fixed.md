Fix coupled solvers refreshing particle material properties after `MODEL_PROPERTIES` notifications while preserving particle ownership, explicitly configured child particle masses, and proxy effective masses. Refresh particle proxy masses on source solvers' declared effective-mass flags as well as material notifications.

Preserve implicit MPM particle-mask and material-volume allocations during same-size material updates so captured solver steps read the updated buffers.
