import ast
import hashlib
import importlib.machinery
import importlib.util
import logging
import os
from pathlib import Path
import sys
import types

# Prefix for the synthetic namespace that hosts package-based kernels. Kernels
# with relative imports are loaded underneath this namespace so their imports
# resolve without leaking the on-disk directory names (e.g. "triton", "cpu")
# into the global module namespace, where they could shadow installed packages
# or collide across kernel sets.
_KERNEL_NS = "_aibench_kernels"

_logger = logging.getLogger(__name__)

# Collision keys already reported, so each dual-import-path warning is emitted
# at most once per process.
_warned_collisions: set[tuple] = set()


def _warn_once(key: tuple, message: str, *args: object) -> None:
    """Emit a warning the first time a given collision key is seen."""
    if key in _warned_collisions:
        return
    _warned_collisions.add(key)
    _logger.warning(message, *args)


def _scan_relative_imports(
    file_path: Path,
) -> list[tuple[int, str | None, tuple[str, ...]]]:
    """Parse a source file and return its relative imports.

    The file is parsed, never executed. Each 'from ... import' that uses leading
    dots yields a '(level, module, names)' entry. An unparsable file yields an
    empty list.

    Args:
        file_path: Path to a Python source file
    Returns:
        Relative imports as '(level, module, imported names)' tuples
    """
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return []
    imports: list[tuple[int, str | None, tuple[str, ...]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            names = tuple(alias.name for alias in node.names)
            imports.append((node.level, node.module, names))
    return imports


def _load_standalone(module_name: str, file_path: Path) -> types.ModuleType:
    """Load a module from a file with no enclosing package.

    From importlib docs:
    https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _exposed_top_names(
    relative_imports: list[tuple[int, str | None, tuple[str, ...]]],
    package_parts: tuple[str, ...],
) -> set[str]:
    """Names the file's relative imports expose directly under the anchor.

    These are the top-level names (immediate children of the anchor directory)
    that the synthetic namespace makes importable and that could therefore also
    exist as a real, identically named package.

    Args:
        relative_imports: Output of '_scan_relative_imports'
        package_parts: Path from the anchor down to the file's own directory
    Returns:
        Set of top-level names reachable from the anchor
    """
    depth = len(package_parts)
    names: set[str] = set()
    for level, module, imported in relative_imports:
        # 'level - 1' directories are climbed from the file's directory. While
        # that stays below the anchor, every target shares the same first
        # component; once it reaches the anchor, the module/imported names are
        # themselves anchored children.
        if depth - (level - 1) > 0:
            names.add(package_parts[0])
        elif module:
            names.add(module.split(".")[0])
        else:
            names.update(imported)
    return names


def _spec_directory(spec: importlib.machinery.ModuleSpec) -> Path | None:
    """Resolve the on-disk directory a module spec points at, if any."""
    locations = getattr(spec, "submodule_search_locations", None)
    if locations:
        try:
            return Path(next(iter(locations))).resolve()
        except (OSError, RuntimeError, StopIteration, TypeError):
            return None
    origin = getattr(spec, "origin", None)
    if origin and origin not in ("built-in", "frozen", "namespace"):
        try:
            return Path(origin).resolve().parent
        except (OSError, RuntimeError):
            return None
    return None


def _warn_on_dual_import_paths(
    anchor: Path, top_names: set[str]
) -> None:
    """Warn when a kernel's helpers could be imported by a real name too.

    Modules hosted under the synthetic namespace are duplicated only if the same
    file is also reachable by its real dotted name. Two situations are flagged:

    1. A 'sys.path' entry sits inside the kernel package tree, so its modules are
       importable both relatively (isolated copy) and by their bare real name
       (separate copy). This catches, for example, a kernel/utils directory left
       on 'sys.path'.
    2. A relative-import target shares its name with an installed/importable
       package located elsewhere; the relative import resolves to the local
       copy, while absolute imports of that name resolve to the other module.

    Detection only - the warning lets authors fix the layout; the loader never
    silently redirects a relative import.

    Args:
        anchor: Directory the synthetic namespace is anchored at
        top_names: Top-level names exposed under the anchor by relative imports
    """
    for entry in sys.path:
        if not entry:
            continue
        try:
            resolved = Path(entry).resolve()
        except (OSError, RuntimeError, ValueError):
            continue
        if resolved == anchor or anchor in resolved.parents:
            _warn_once(
                ("syspath", str(anchor), str(resolved)),
                "Possible duplicate kernel modules: '%s' is on sys.path and lies"
                " within the kernel package tree at '%s'. Helpers there can be"
                " imported both via relative imports (isolated copy) and by"
                " their real name (separate copy); remove it from sys.path.",
                resolved,
                anchor,
            )

    for name in sorted(top_names):
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, AttributeError, ValueError):
            spec = None
        if spec is None:
            continue
        location = _spec_directory(spec)
        if location is not None and (
            location == anchor or anchor in location.parents
        ):
            # Same tree as the kernel - already covered by the sys.path check.
            continue
        _warn_once(
            ("shadow", str(anchor), name),
            "Relative import target '%s' (under '%s') shares its name with an"
            " importable package%s. The relative import uses the local copy;"
            " code importing '%s' absolutely will get a different module.",
            name,
            anchor,
            f" at '{location}'" if location else "",
            name,
        )


def import_from_path(
    module_name: str, file_path: str | os.PathLike
) -> types.ModuleType:
    """Import a module directly from a source file.

    Resolution is generic and driven entirely by the file itself, with no need
    for '__init__.py' marker files or for the caller to know the layout:

    - If the file uses relative imports, its deepest relative-import level marks
      where its package root lives (N leading dots reach N-1 directories above
      the file). The directory tree from that root down is exposed as a PEP 420
      namespace package under a unique synthetic prefix, and the file is
      imported as a member of it. This lets co-located helpers be reached with
      self-contained relative imports (e.g. 'from ...utils.helper import fn').
    - Otherwise the file is loaded directly as a standalone module.

    Only modules reached through *relative* imports live under the synthetic
    namespace. Absolute imports (e.g. 'import torch', 'import ai_bench.utils')
    are unaffected and resolve to their normal, single instance. A module is
    therefore duplicated only if the very same file is reachable both ways at
    once: relatively (under the synthetic prefix) and by its real dotted name
    because its package root is also on 'sys.path' / installed. To avoid that,
    use relative imports for private co-located helpers and absolute imports for
    genuinely shared or installed packages; do not place a kernel's helper
    directory on 'sys.path' and relative-import into it as well. Such a dual
    import path is detected on a best-effort basis and logged as a warning (a
    'sys.path' entry inside the kernel tree, or a relative-import name that also
    resolves to an importable package elsewhere).

    Args:
        module_name: Fallback module name used for standalone files
        file_path: Path to a Python source file
    Returns:
        Loaded module
    """
    if file_path is None:
        raise ValueError("Path to the module must be provided.")
    filepath = Path(file_path)
    if not filepath.exists():
        raise ValueError(f"Path to the module does not exist: {filepath}")
    filepath = filepath.resolve()

    relative_imports = _scan_relative_imports(filepath)
    level = max((lvl for lvl, _, _ in relative_imports), default=0)
    if level == 0:
        return _load_standalone(module_name, filepath)

    # The deepest relative import reaches 'level - 1' directories above the
    # file's own directory; that directory is the package root. Anchor a
    # namespace package there so every relative import in the file (and in the
    # helpers it pulls in) resolves against the real directory tree.
    ancestors = [filepath.parent, *filepath.parent.parents]
    if level > len(ancestors):
        raise ImportError(
            f"Relative import (level {level}) in '{filepath.name}' reaches "
            f"above the filesystem root"
        )
    anchor = ancestors[level - 1]

    package_parts = filepath.parent.relative_to(anchor).parts
    _warn_on_dual_import_paths(
        anchor, _exposed_top_names(relative_imports, package_parts)
    )

    # Hash the anchor so distinct kernel sets map to distinct namespaces
    # (avoiding collisions between identically named directories) while the
    # same set always maps to the same namespace (keeping relative imports and
    # module caching consistent).
    token = hashlib.sha1(str(anchor).encode()).hexdigest()[:12]
    base = f"{_KERNEL_NS}_{token}"
    if base not in sys.modules:
        namespace = types.ModuleType(base)
        namespace.__path__ = [str(anchor)]
        namespace.__package__ = base
        sys.modules[base] = namespace

    qualified = ".".join((base, *package_parts, filepath.stem))
    if qualified in sys.modules:
        return sys.modules[qualified]
    return importlib.import_module(qualified)
