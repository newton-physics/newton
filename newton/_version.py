try:
    from importlib.metadata import version

    __version__ = version("newton")
except Exception:
    __version__ = "unknown"
