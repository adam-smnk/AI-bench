"""Path configuration for ai_bench.

When used as a library, paths must be configured via configure() or environment variables.
When used as CLI from project root, paths are auto-detected.
"""

from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

from ai_bench.config import settings

# Whether a .env file has been loaded into the process.
_env_loaded: bool = False

# Settings fields that hold path configuration.
_PATH_SETTINGS = (
    "specs_dir",
    "kernels_dir",
    "triton_kernels_dir",
    "helion_kernels_dir",
    "mlir_kernels_dir",
    "gluon_kernels_dir",
    "sycl_kernels_dir",
)


class ConfigurationError(Exception):
    """Raised when required paths are not configured."""

    pass


def load_env(env_path: Path | str | None = None, override: bool = False) -> bool:
    """Load configuration from .env file.

    Searches for .env file in the following order:
    1. Explicit path if provided
    2. Current working directory
    3. Project root directory

    Args:
        env_path: Explicit path to .env file (optional)
        override: If True, override existing environment variables

    Returns:
        True if .env file was loaded, False otherwise

    Example:
        >>> import ai_bench
        >>> ai_bench.load_env()  # Auto-find .env
        >>> ai_bench.load_env("/path/to/.env")  # Explicit path
        >>> ai_bench.load_env(override=True)  # Override existing vars
    """
    global _env_loaded
    if env_path is not None:
        path = Path(env_path)
        if path.is_file():
            load_dotenv(path, override=override)
            settings.reset_settings()
            _env_loaded = True
            return True
        return False

    # Search order: CWD, then project root
    search_paths = [
        Path.cwd() / ".env",
        project_root() / ".env",
    ]

    for path in search_paths:
        if path.is_file():
            load_dotenv(path, override=override)
            settings.reset_settings()
            _env_loaded = True
            return True

    return False


def is_env_loaded() -> bool:
    """Check if .env file has been loaded.

    Returns:
        True if load_env() successfully loaded a .env file
    """
    return _env_loaded


def configure(
    specs_dir: Path | str | None = None,
    kernels_dir: Path | str | None = None,
    triton_kernels_dir: Path | str | None = None,
    helion_kernels_dir: Path | str | None = None,
    mlir_kernels_dir: Path | str | None = None,
    gluon_kernels_dir: Path | str | None = None,
    sycl_kernels_dir: Path | str | None = None,
) -> None:
    """Configure library paths.

    Call this before using KernelBenchRunner when using ai_bench as a library.

    Args:
        specs_dir: Path to YAML spec files directory
        kernels_dir: Path to PyTorch kernel implementations
        triton_kernels_dir: Path to Triton kernel implementations
        helion_kernels_dir: Path to Helion kernel implementations
        mlir_kernels_dir: Path to MLIR kernel implementations
        gluon_kernels_dir: Path to Gluon kernel implementations
        sycl_kernels_dir: Path to SYCL kernel implementations

    Example:
        >>> import ai_bench
        >>> ai_bench.configure(
        ...     specs_dir="/path/to/specs",
        ...     kernels_dir="/path/to/kernels",
        ... )
    """
    overrides = {
        "specs_dir": specs_dir,
        "kernels_dir": kernels_dir,
        "triton_kernels_dir": triton_kernels_dir,
        "helion_kernels_dir": helion_kernels_dir,
        "mlir_kernels_dir": mlir_kernels_dir,
        "gluon_kernels_dir": gluon_kernels_dir,
        "sycl_kernels_dir": sycl_kernels_dir,
    }
    settings.configure(
        **{key: str(value) for key, value in overrides.items() if value is not None}
    )


def reset_configuration() -> None:
    """Reset the path configuration to environment/default values.

    Only the path settings are reverted (to ``None`` when not set via the
    environment); other settings (e.g. logging level, MLIR dump flags) are left
    untouched.
    """
    global _env_loaded
    _env_loaded = False
    settings.reset_fields(*_PATH_SETTINGS)


def _get_path(
    env_var: str,
    value: str | None,
    default_fn: Callable[[], Path],
    name: str,
) -> Path:
    """Get path from settings or default.

    Priority:
    1. Value resolved from central settings (``configure()`` override, then
       environment variable)
    2. Default (relative to project_root)

    Args:
        env_var: Environment variable name (used for error messages)
        value: Path value resolved from :mod:`ai_bench.config` settings
        default_fn: Function returning default path
        name: Human-readable name for error messages

    Returns:
        Resolved path

    Raises:
        ConfigurationError: If path cannot be determined or doesn't exist
    """
    # Priority 1: Value from settings (explicit override or environment variable)
    if value:
        path = Path(value)
        if not path.exists():
            raise ConfigurationError(f"{name} does not exist: {path}")
        return path

    # Priority 2: Default (project structure)
    try:
        return default_fn()
    except Exception:
        raise ConfigurationError(
            f"{name} not configured. Either:\n"
            f"  1. Call ai_bench.configure({name.lower().replace(' ', '_')}=...)\n"
            f"  2. Set {env_var} environment variable\n"
            f"  3. Run from project root with standard directory structure"
        )


def project_root() -> Path:
    """Path to the project's root directory.

    Returns:
        Project root path (parent of ai_bench package)
    """
    return Path(__file__).parent.parent.parent


def specs() -> Path:
    """Path to the problem specs directory.

    Can be configured via:
    - ai_bench.configure(specs_dir=...)
    - AIBENCH_SPECS_DIR environment variable
    - Auto-detected from project structure

    Returns:
        Path to specs directory

    Raises:
        ConfigurationError: If path cannot be determined
    """

    def default() -> Path:
        path = project_root() / "problems" / "specs"
        if not path.exists():
            raise FileNotFoundError(f"Default specs path not found: {path}")
        return path

    return _get_path(
        "AIBENCH_SPECS_DIR",
        settings.get_settings().specs_dir,
        default,
        "Specs directory",
    )


def kernel_bench_dir() -> Path:
    """Path to the KernelBench directory (PyTorch kernels).

    Can be configured via:
    - ai_bench.configure(kernels_dir=...)
    - AIBENCH_KERNELS_DIR environment variable
    - Auto-detected from project structure

    Returns:
        Path to PyTorch kernels directory

    Raises:
        ConfigurationError: If path cannot be determined
    """

    def default() -> Path:
        path = project_root() / "third_party" / "KernelBench"
        if not path.exists():
            raise FileNotFoundError(f"Default kernels path not found: {path}")
        return path

    return _get_path(
        "AIBENCH_KERNELS_DIR",
        settings.get_settings().kernels_dir,
        default,
        "Kernels directory",
    )


def triton_kernels_dir() -> Path:
    """Path to the Triton kernels directory.

    Can be configured via:
    - ai_bench.configure(triton_kernels_dir=...)
    - AIBENCH_TRITON_KERNELS_DIR environment variable
    - Auto-detected from project structure

    Returns:
        Path to Triton kernels directory

    Raises:
        ConfigurationError: If path cannot be determined
    """

    def default() -> Path:
        path = project_root() / "backends" / "triton"
        if not path.exists():
            raise FileNotFoundError(f"Default Triton kernels path not found: {path}")
        return path

    return _get_path(
        "AIBENCH_TRITON_KERNELS_DIR",
        settings.get_settings().triton_kernels_dir,
        default,
        "Triton kernels directory",
    )


def helion_kernels_dir() -> Path:
    """Path to the Helion kernels directory.

    Can be configured via:
    - ai_bench.configure(helion_kernels_dir=...)
    - AIBENCH_HELION_KERNELS_DIR environment variable
    - Auto-detected from project structure

    Returns:
        Path to Helion kernels directory

    Raises:
        ConfigurationError: If path cannot be determined
    """

    def default() -> Path:
        path = project_root() / "backends" / "helion"
        if not path.exists():
            raise FileNotFoundError(f"Default Helion kernels path not found: {path}")
        return path

    return _get_path(
        "AIBENCH_HELION_KERNELS_DIR",
        settings.get_settings().helion_kernels_dir,
        default,
        "Helion kernels directory",
    )


def mlir_kernels_dir() -> Path:
    """Path to the MLIR kernels directory.

    Can be configured via:
    - ai_bench.configure(mlir_kernels_dir=...)
    - AIBENCH_MLIR_KERNELS_DIR environment variable
    - Auto-detected from project structure

    Returns:
        Path to MLIR kernels directory

    Raises:
        ConfigurationError: If path cannot be determined
    """

    def default() -> Path:
        path = project_root() / "backends" / "mlir"
        if not path.exists():
            raise FileNotFoundError(f"Default MLIR kernels path not found: {path}")
        return path

    return _get_path(
        "AIBENCH_MLIR_KERNELS_DIR",
        settings.get_settings().mlir_kernels_dir,
        default,
        "MLIR kernels directory",
    )


def gluon_kernels_dir() -> Path:
    """Path to the Gluon kernels directory.

    Can be configured via:
    - ai_bench.configure(gluon_kernels_dir=...)
    - AIBENCH_GLUON_KERNELS_DIR environment variable
    - Auto-detected from project structure

    Returns:
        Path to Gluon kernels directory

    Raises:
        ConfigurationError: If path cannot be determined
    """

    def default() -> Path:
        path = project_root() / "backends" / "gluon"
        if not path.exists():
            raise FileNotFoundError(f"Default Gluon kernels path not found: {path}")
        return path

    return _get_path(
        "AIBENCH_GLUON_KERNELS_DIR",
        settings.get_settings().gluon_kernels_dir,
        default,
        "Gluon kernels directory",
    )


def sycl_kernels_dir() -> Path:
    """Path to the SYCL kernels directory.

    Can be configured via:
    - ai_bench.configure(sycl_kernels_dir=...)
    - AIBENCH_SYCL_KERNELS_DIR environment variable
    - Auto-detected from project structure

    Returns:
        Path to SYCL kernels directory

    Raises:
        ConfigurationError: If path cannot be determined
    """

    def default() -> Path:
        path = project_root() / "backends" / "sycl"
        if not path.exists():
            raise FileNotFoundError(f"Default SYCL kernels path not found: {path}")
        return path

    return _get_path(
        "AIBENCH_SYCL_KERNELS_DIR",
        settings.get_settings().sycl_kernels_dir,
        default,
        "SYCL kernels directory",
    )
