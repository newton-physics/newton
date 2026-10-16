Print `newton` logger records by default: until the application configures
logging, `INFO` records go to stdout and warnings and errors to stderr. The
`newton` logger now defaults to `INFO`; set
`logging.getLogger("newton").setLevel(logging.WARNING)` to hide `INFO` output.
Messages that were previously logged at `INFO` but hidden by default, such as
dropped USD UVs and SDF cache version misses, are now logged at `DEBUG`.
