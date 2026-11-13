"""Build package entrypoints and resolve their source without importing them."""

from pathlib import Path
import re


def module_command(python, module, *arguments):
    return [str(python), "-m", module, *(str(value) for value in arguments)]


def command_source(command, worktree):
    """Return the executable source and the offset of its application arguments."""
    if len(command) < 3 or command[1] != "-m":
        raise ValueError("Jobs must use a python -m utopia entrypoint")
    module = command[2]
    if not re.fullmatch(r"utopia(?:\.[A-Za-z_][A-Za-z_0-9]*)*", module):
        raise ValueError("Jobs must use a utopia package module")
    relative = (
        "utopia/__main__.py" if module == "utopia" else module.replace(".", "/") + ".py"
    )
    return Path(worktree) / relative, 3
