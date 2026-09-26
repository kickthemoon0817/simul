"""Marker package so setuptools ships ``khemoo.simul/`` as data.

The bridge extension itself is loaded by Isaac Sim from
``<isaac-root>/extsUser/khemoo.simul/``, not by importing through
this package — the ``khemoo.simul`` directory contains a dot in
its name and cannot be a Python module path. This empty ``__init__``
exists only so that ``[tool.setuptools.package-data]`` can attach
``khemoo.simul/**/*`` to a real package and have those files
ship in both sdist and wheel.

``simul isaac install-bridge`` resolves the bundled source as
``Path(simul.__file__).parent / "bridge_ext" / "khemoo.simul"``,
which works for both editable installs and pip-installed wheels.
"""
