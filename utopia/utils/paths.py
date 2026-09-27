"""Locate a source checkout without depending on its directory name."""
from pathlib import Path


def project_root(source=__file__) -> Path:
    """Find the nearest checkout containing the package or a Git marker."""
    source = Path(source).resolve()
    directory = source if source.is_dir() else source.parent
    for candidate in (directory, *directory.parents):
        if (candidate / "utopia").is_dir() or (candidate / ".git").exists():
            return candidate
    raise FileNotFoundError(f"No ScienceUtopia checkout above {source}")


def check_cwd():
    """Require the project root, independent of the checkout's directory name."""
    if Path.cwd().resolve() != project_root():
        raise ValueError(f"Run from the project root: {project_root()}")
