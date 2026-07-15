"""Typed, centralized configuration for ai_bench.

Historically the ``AIBENCH_*`` environment variables were read directly through
scattered ``os.environ`` calls across the codebase. This module consolidates all
of that into a single, validated :class:`pydantic_settings.BaseSettings` model.

Environment variables remain the public configuration contract, so existing
users are unaffected. This module only centralizes *how* they are read:

* Values are typed and validated in one place.
* A typo in a field default surfaces immediately instead of silently failing.
* Consumers ask for :func:`get_settings` instead of touching ``os.environ``.

The module deliberately does **not** create a module-level singleton instance.
Instead, :func:`get_settings` lazily builds and caches a :class:`Settings`
object in the module-level ``_settings`` variable -- the only piece of
module-level state. Programmatic overrides (:func:`configure`) and the log
level live on that cached instance. Passing ``reload=True`` rebuilds it from the
current environment.
"""

import os

from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

_AIBENCH_PREFIX = "AIBENCH_"


class Settings(BaseSettings):
    """All ``AIBENCH_*`` configuration values.

    Each field maps to an environment variable formed by the ``AIBENCH_``
    prefix plus the upper-cased field name (e.g. ``specs_dir`` maps to
    ``AIBENCH_SPECS_DIR``).

    Path fields are intentionally kept as raw strings (not ``Path``) so that
    the existing precedence and existence-checking logic in
    :mod:`ai_bench.utils.finder` is preserved exactly. In particular, an empty
    string stays falsy just like ``os.environ.get(...)`` returned before.
    """

    model_config = SettingsConfigDict(
        env_prefix=_AIBENCH_PREFIX,
        case_sensitive=False,
        extra="ignore",
    )

    # --- Path configuration -------------------------------------------------
    specs_dir: str | None = None
    kernels_dir: str | None = None
    triton_kernels_dir: str | None = None
    helion_kernels_dir: str | None = None
    mlir_kernels_dir: str | None = None
    gluon_kernels_dir: str | None = None
    sycl_kernels_dir: str | None = None

    # --- MLIR backend -------------------------------------------------------
    mlir_lib_path: str | None = None
    mlir_dump: bool = False
    mlir_dump_obj: bool = False

    # --- SYCL backend -------------------------------------------------------
    sycl_compiler: str = "icpx"
    sycl_flags: str | None = None
    sycl_include: str | None = None
    sycl_target: str = ""

    # --- Benchmark tuning ---------------------------------------------------
    # ``None`` means "not overridden": the runner keeps its device-specific
    # default.
    warmup: int | None = None
    rep: int | None = None
    cpu_min_cache_nuke_mib: int = 0

    # --- Logging ------------------------------------------------------------
    log: str | int | None = None

    @property
    def mlir_lib_paths(self) -> list[str]:
        """``AIBENCH_MLIR_LIB_PATH`` parsed as a ``:``-separated list."""
        if not self.mlir_lib_path:
            return []
        return [p for p in self.mlir_lib_path.split(":") if p]

    @property
    def sycl_include_dirs(self) -> list[str]:
        """``AIBENCH_SYCL_INCLUDE`` parsed as a ``:``-separated list."""
        if not self.sycl_include:
            return []
        return [d for d in self.sycl_include.split(":") if d]

    @property
    def sycl_flag_list(self) -> list[str]:
        """``AIBENCH_SYCL_FLAGS`` parsed as a whitespace-separated list."""
        if not self.sycl_flags:
            return []
        return self.sycl_flags.split()


# The cached :class:`Settings` instance -- the only module-level state. Every
# mutable value (programmatic :func:`configure` overrides and the log level)
# lives on this instance, so discarding it resets all of them.
_settings: Settings | None = None


def get_settings(reload: bool = False) -> Settings:
    """Return a lazily-built, cached :class:`Settings` instance.

    Args:
        reload: If ``True``, rebuild the instance from the current environment
            (discarding any programmatic :func:`configure` overrides). Use this
            to pick up environment changes made after the settings were first
            built.

    Returns:
        The cached :class:`Settings` instance.
    """
    global _settings
    if reload or _settings is None:
        _settings = Settings()
    return _settings


def configure(**overrides: object) -> None:
    """Apply programmatic configuration overrides.

    Overrides mutate the cached :class:`Settings` instance in place, taking
    precedence over environment variables. ``None`` is a valid value and
    overrides any value read from the environment (only keys that are passed are
    affected).

    Args:
        **overrides: Field name to value mapping (e.g. ``specs_dir=...``). Keys
            must match :class:`Settings` field names.
    """
    settings = get_settings()
    for key, value in overrides.items():
        setattr(settings, key, value)


def reset_settings() -> None:
    """Discard the cached :class:`Settings` instance and all state on it.

    The next :func:`get_settings` call rebuilds it from the current environment.
    Backs :func:`ai_bench.utils.finder.reset_configuration`.
    """
    global _settings
    _settings = None


def reset_fields(*names: str) -> None:
    """Revert specific fields to their environment/default values.

    Only the named fields are touched; the rest of the cached settings (other
    programmatic overrides included) are left as-is. If the settings instance
    has not been built yet this is a no-op -- the values are read from the
    environment on next access anyway.

    Args:
        *names: :class:`Settings` field names to revert.
    """
    global _settings
    if _settings is None:
        return
    env_values = Settings()
    for name in names:
        setattr(_settings, name, getattr(env_values, name))


def setting_env_vars() -> dict[str, str]:
    """Return all project's setting environment variables currently set.

    These are read live from ``os.environ`` because the key set is open-ended
    (arbitrary run metadata such as ``AIBENCH_CARD`` or ``AIBENCH_SYSTEM``) and
    is therefore not part of the typed :class:`Settings` schema. Centralizing
    the read here keeps ``os.environ`` access out of the rest of the codebase.
    """
    return {k: v for k, v in os.environ.items() if k.startswith(_AIBENCH_PREFIX)}
