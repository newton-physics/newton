Add `newton.exceptions.NewtonWarning` and `newton.exceptions.NewtonDeprecationWarning`
and emit every Newton warning with one of them, so applications can filter Newton
warnings by category. They subclass `UserWarning` and `DeprecationWarning`, so
existing warning filters keep matching.
