"""Pytest configuration for common_crawl_search_engine."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the repository root for tests that need file paths."""
    return Path(__file__).resolve().parent


@pytest.fixture(scope="session", autouse=True)
def _set_test_env() -> None:
    """Set sane defaults for tests to avoid accidental long-running work."""
    os.environ.setdefault("BRAVE_RESOLVE_ROWGROUP_SLICE_MODE", "off")
