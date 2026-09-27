"""Load admission code from a frozen worktree without modifying sys.path."""

from contextlib import contextmanager
from pathlib import Path
import sys
from types import ModuleType


@contextmanager
def frozen_package(worktree):
    package_path = Path(worktree).resolve() / "utopia"
    if (
        not package_path.is_dir()
        or package_path.resolve().parent != Path(worktree).resolve()
    ):
        raise ValueError("Frozen utopia package is missing or escapes its worktree")
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "utopia" or name.startswith("utopia.")
    }
    old_bytecode = sys.dont_write_bytecode
    try:
        for name in saved:
            del sys.modules[name]
        package = ModuleType("utopia")
        package.__path__ = [str(package_path)]
        package.__file__ = str(package_path / "__init__.py")
        package.__package__ = "utopia"
        sys.modules["utopia"] = package
        sys.dont_write_bytecode = True
        yield
    finally:
        for name in list(sys.modules):
            if name == "utopia" or name.startswith("utopia."):
                del sys.modules[name]
        sys.modules.update(saved)
        sys.dont_write_bytecode = old_bytecode
