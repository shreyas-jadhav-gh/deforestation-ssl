"""Lets the toolkit run as  python src/forestwatch  from the repository root, or as
python -m forestwatch  when the package is installed (pip install -e .)."""

if __package__ in (None, ""):
    # Run as a folder: Python put this folder itself first on the path. Replace it with its
    # parent (src/), so the package's own modules are only importable as forestwatch.*
    import sys
    from pathlib import Path
    sys.path[0] = str(Path(__file__).resolve().parent.parent)
    from forestwatch.cli import main
else:
    from .cli import main

main()
