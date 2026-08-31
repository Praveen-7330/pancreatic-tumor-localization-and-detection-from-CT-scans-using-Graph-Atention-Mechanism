from pathlib import Path

def resolve_path(*parts: str) -> Path:
    """Resolve a path relative to the repository root.
    Example: ``resolve_path('data', 'download_msd.py')`` returns a Path object
    pointing to ``<repo_root>/data/download_msd.py``.
    """
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root.joinpath(*parts)
