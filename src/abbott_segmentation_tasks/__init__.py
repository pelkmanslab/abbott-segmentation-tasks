"""Package description."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("abbott_stardist_task")
except PackageNotFoundError:
    __version__ = "uninstalled"
