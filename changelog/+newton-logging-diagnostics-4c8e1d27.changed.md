Report Newton's console diagnostics, including `verbose=True` output and
importer warnings, through the `newton` logger instead of `print()`. Without
logging configuration, the same messages are still printed, but warnings and
errors now go to stderr instead of stdout. Configure the `newton` logger to
route, filter, or silence them. `SolverKamino` messages also use the `newton`
logger and no longer configure the root logger or add a colored `[KAMINO]`
header.
