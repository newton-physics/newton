from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("newton")
except PackageNotFoundError:
    __version__ = "unknown"
