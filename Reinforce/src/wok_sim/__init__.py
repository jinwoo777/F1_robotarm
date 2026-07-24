"""Open-loop 웍질 물리 시뮬레이션 패키지."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("wok-sim")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
