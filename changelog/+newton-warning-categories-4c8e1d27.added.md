Add `newton.NewtonWarning` and `newton.NewtonDeprecationWarning`
and emit every Newton warning with one of them, so applications can filter Newton
warnings by category. `NewtonWarning` subclasses `UserWarning` and
`NewtonDeprecationWarning` subclasses `DeprecationWarning`, so existing warning
filters keep matching the same warnings.
