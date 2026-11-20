"""Read-only access to resources inside a user-selected archive directory."""
from pathlib import Path, PureWindowsPath

try:                                  # Python 3.13+; PurePath.is_reserved goes in 3.15
    from ntpath import isreserved as _reserved
except ImportError:                   # Python 3.12
    def _reserved(part):
        return PureWindowsPath(part).is_reserved()


def resource(root: Path, name: str) -> Path:
    """Reject absolute, escaping, alternate-stream and redirected paths."""
    rel = PureWindowsPath(name)
    if (not name or rel.anchor or ".." in rel.parts or ":" in name
            or any(part.endswith((" ", ".")) or _reserved(part)
                   for part in rel.parts)):
        raise ValueError(f"Invalid archive resource: {name!r}")
    path = root
    for part in rel.parts:
        path = path / part
        if path.is_symlink() or path.is_junction():
            raise ValueError(f"Linked archive resource: {name!r}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Resource outside archive: {name!r}")
    return path
