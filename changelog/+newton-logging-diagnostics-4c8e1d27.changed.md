Report Newton's console diagnostics, including `verbose=True` output and
importer warnings, through the `newton` logger instead of `print()`. Without
logging configuration, the same messages are still printed, but warnings and
errors now go to stderr instead of stdout. Configure the `newton` logger to
route, filter, or silence them.
