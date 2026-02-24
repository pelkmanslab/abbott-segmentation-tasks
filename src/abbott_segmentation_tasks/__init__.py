"""Package description."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("abbott_segmentation_tasks")
except PackageNotFoundError:
    __version__ = "uninstalled"
