"""Marker package so setuptools ships ``khemoo.simul.mcp/`` as data.

The bridge extension itself is loaded by Isaac Sim from
``<isaac-root>/extsUser/khemoo.simul.mcp/``, not by importing through
this package — the ``khemoo.simul.mcp`` directory contains a dot in
its name and cannot be a Python module path. This empty ``__init__``
exists only so that ``[tool.setuptools.package-data]`` can attach
``khemoo.simul.mcp/**/*`` to a real package and have those files
ship in both sdist and wheel.

``simul-mcp isaac install-bridge`` resolves the bundled source as
``Path(simul_mcp.__file__).parent / "bridge_ext" / "khemoo.simul.mcp"``,
which works for both editable installs and pip-installed wheels.
"""
