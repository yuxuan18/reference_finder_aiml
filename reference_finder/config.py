"""Load local collection credentials without shell evaluation or interpolation."""
from pathlib import Path


def load_environment(path=None):
    from dotenv import load_dotenv

    # Resolve relative to the project, independent of the caller's directory.
    # Shell environment values take precedence; passwords containing $ stay literal.
    path = Path(path) if path is not None else Path(__file__).resolve().parent.parent / ".env"
    if path.is_file():
        load_dotenv(path, override=False, interpolate=False)
