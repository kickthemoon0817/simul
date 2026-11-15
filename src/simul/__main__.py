"""``python -m simul``: the ``simul`` command, for an environment whose scripts directory is not on PATH."""

from simul.cli.main import app

if __name__ == "__main__":
    app(prog_name="simul")
