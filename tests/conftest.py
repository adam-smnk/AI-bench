"""Shared pytest fixtures for the ai_bench test suite."""

import pytest

from ai_bench.config.settings import reset_settings


@pytest.fixture(autouse=True)
def _reset_settings_state():
    """Reset all settings state before and after every test.

    The settings module caches a single instance and holds process-level
    configuration (programmatic overrides and env-loaded state). Tests mutate
    ``AIBENCH_*`` environment variables and call ``configure()`` at runtime, so
    the full state must be reset between tests to keep them isolated (and to
    mirror the previous live ``os.environ`` behaviour).
    """
    reset_settings()
    yield
    reset_settings()
