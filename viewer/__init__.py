"""Local static-viewer assets; no provider or server starts on import."""
from pathlib import Path


def asset_root() -> Path:
    return Path(__file__).parent
