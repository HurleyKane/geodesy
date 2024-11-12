try:
    import pygmt
except ImportError:
    raise ImportError("pygmt is not installed.")

pygmt.show_versions()