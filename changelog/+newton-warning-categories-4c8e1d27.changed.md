Emit Newton warnings as `NewtonWarning` or `NewtonDeprecationWarning` instead
of plain `UserWarning` or `DeprecationWarning`, and drop the redundant
`Warning:` prefix from importer warning messages. Filters and `assertWarns`
checks on the builtin categories still match; replace exact category checks
such as `w.category is DeprecationWarning` with `issubclass()`, and update
message filters or output matching that expects the `Warning:` prefix or the
`UserWarning:` category name.
