"""Stand-in for the parts of ``pkg_resources`` this code used.

``pkg_resources`` was removed from setuptools (v81+), so ``import pkg_resources``
fails on current Python installs. This module offers the same three helpers on top
of ``packaging`` and ``importlib.metadata``.
"""

from importlib import metadata

from packaging.requirements import Requirement
from packaging.version import parse as parse_version


def parse_requirements(lines):
    """Yield ``Requirement`` objects from requirements.txt lines, skipping comments."""
    for line in lines:
        line = line.split('#', 1)[0].strip()
        if line and not line.startswith('-'):
            yield Requirement(line)


def require(spec):
    """Raise if the requirement ``spec`` (e.g. ``'numpy>=1.18'``) is not satisfied."""
    req = Requirement(spec)
    installed = metadata.version(req.name)  # raises PackageNotFoundError if missing
    if req.specifier and not req.specifier.contains(installed, prereleases=True):
        raise RuntimeError(f'{req.name} {installed} is installed but {spec} is required')
