Fix URDF `continuous` joints taking the `lower` and `upper` of their `<limit>` element as position limits. They are now imported without position limits and keep their effort and velocity limits.
