"""Project environment/path helpers."""

from __future__ import annotations

from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
ENV_FILE = PROJECT_ROOT / ".env"


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)
