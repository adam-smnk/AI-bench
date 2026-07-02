"""Pydantic schema for KernelBench-style problem spec YAML files.

This module provides optional, additive validation on top of the existing
dict-based spec APIs in :mod:`ai_bench.harness.core.specs`. It does not
change the YAML syntax/layout, the `SpecKey`/`InKey`/`VKey`/... enums, or any
of the existing getter functions - it only offers a typed, validated view of
the same data (useful for editor autocompletion and catching spec errors
before a kernel is run).

Example:
    >>> from ai_bench.harness.core import schema
    >>> spec = schema.load_spec_file(
    ...     "problems/specs/KernelBench/level1/1_Square_matrix_multiplication_.yaml"
    ... )
    >>> spec.inputs.keys()
    >>> spec.get_variants("ci")[0].dims
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Union

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
import torch

from .specs import InInitKey
from .specs import InInputKey

# Concrete dimension value(s) for a variant's 'dims' mapping.
DimValue = Union[int, float, bool, list[Union[int, float, bool]]]

# 'flop'/'mem_bytes' are either a number or a formula string evaluated over 'dims'.
FormulaValue = Union[str, float, int]

_VALID_INPUT_INITS = frozenset(str(key) for key in InInitKey)
_INHERIT = str(InInputKey.INHERIT)
# Variant category names known to be used by the existing YAML specs and/or
# 'ai_bench' CLI tools. Listed explicitly so editors can offer autocompletion
# for them; any other key is still accepted (see 'additionalProperties' in
# `generate_json_schema`) since variant category names are open-ended.
_KNOWN_VARIANT_CATEGORIES = ("ci", "simple-cpu", "bench-cpu", "bench-gpu")


def _flow_array_json_schema(json_schema: dict[str, Any]) -> None:
    """Hint editors to complete this array property as an empty flow-style
    sequence ('key: []') instead of a multi-line block sequence ('key:\n  -
    ').

    JSON Schema/yaml-language-server has no keyword to request flow style for
    a *pre-filled* array (it always renders non-empty 'default'/'defaultSnippets'
    values as a block sequence), so this only improves the empty-value case.
    """
    json_schema["default"] = []


def _drop_null_default_json_schema(json_schema: dict[str, Any]) -> None:
    """Drop a `None`/`null` 'default' from an optional field's exported schema.

    Pydantic sets `"default": null` for `Optional[X] = None` fields. Editors
    treat an explicit `null` default as a real value and insert it literally
    (e.g. accepting 'flop' + Enter produces 'flop: null' instead of leaving
    the value empty for the user to fill in). Removing it only affects the
    editor-facing schema - the field's Python default/Optional-ness is
    unchanged, still `None` when omitted.
    """
    if json_schema.get("default", False) is None:
        json_schema.pop("default", None)


def _check_torch_dtype(value: str) -> str:
    """Validate that a string names a real torch dtype.
    Args:
        value: Candidate dtype name
    Returns:
        The validated dtype name
    """
    dtype = getattr(torch, value, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unknown torch dtype: '{value}'")
    return value


def _check_torch_memory_format(value: str) -> str:
    """Validate that a string names a real torch memory format.
    Args:
        value: Candidate memory format name
    Returns:
        The validated memory format name
    """
    memory_format = getattr(torch, value, None)
    if not isinstance(memory_format, torch.memory_format):
        raise ValueError(f"Unknown torch memory_format: '{value}'")
    return value


def _known_torch_memory_format_names() -> list[str]:
    """List every real torch memory_format attribute name (e.g.
    'channels_last', 'contiguous_format', ...).
    Returns:
        Sorted torch memory_format names
    """
    return sorted(
        name
        for name in dir(torch)
        if isinstance(getattr(torch, name, None), torch.memory_format)
    )


def _memory_format_json_schema(json_schema: dict[str, Any]) -> None:
    """Offer every real torch memory format name as completion examples for
    'memory_format'.

    Uses 'examples' (not 'enum') so new torch memory formats don't require a
    schema update, matching `_check_torch_memory_format`'s dynamic
    (`getattr(torch, ...)`) check.
    """
    json_schema["examples"] = _known_torch_memory_format_names()
    _drop_null_default_json_schema(json_schema)


def _known_torch_dtype_names() -> list[str]:
    """List every real torch dtype attribute name (e.g. 'float32', 'int64', ...).
    Returns:
        Sorted torch dtype names
    """
    return sorted(
        name
        for name in dir(torch)
        if isinstance(getattr(torch, name, None), torch.dtype)
    )


# Basic/common torch dtypes: all floating point types, power-of-two-width
# integers (signed/unsigned), and bool.
# Deliberately excludes quantized types (qint8, quint4x2, ...), complex types,
# float8 variants, and sub-byte/experimental types (int1-int7, bits* etc.).
_BASIC_DTYPE_CANDIDATES = (
    "float",
    "int",
    "bool",
    "bfloat16",
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
)


def _basic_torch_dtype_names() -> list[str]:
    """List basic/common torch dtype names available in the installed torch.
    Returns:
        Subset of suggested dtypes
    """
    known = set(_known_torch_dtype_names())
    return sorted(name for name in _BASIC_DTYPE_CANDIDATES if name in known)


def _input_dtype_json_schema(json_schema: dict[str, Any]) -> None:
    """Inputs entry 'dtype' field suggestions.

    Uses 'examples' (not 'enum') so this only affects editor suggestions - it
    doesn't restrict validation to this exact list, matching
    `_check_torch_dtype`'s dynamic (`getattr(torch, ...)`) check.
    """
    json_schema["examples"] = [_INHERIT, *_basic_torch_dtype_names()]


def _no_required_json_schema(json_schema: dict[str, Any]) -> None:
    """Drop the 'required' list from a model's exported JSON schema.

    Editors auto-fill required object fields as soon as a new array item is
    started (e.g. accepting a new 'inits' list entry immediately inserts
    'dim: '). 'inits' is legitimately empty or absent in most real specs, so
    this keeps that noise out of the editor-facing schema. Runtime
    validation via pydantic (`parse_spec`/`load_spec_file`) still requires
    the field regardless - only the JSON schema's hinting is relaxed.
    """
    json_schema.pop("required", None)


class InitEntry(BaseModel):
    """Schema for a single entry of the top-level 'inits' list.

    Mirrors the fields read by `ai_bench.harness.core.get_inits`.
    """

    model_config = ConfigDict(
        extra="forbid", json_schema_extra=_no_required_json_schema
    )

    dim: str = Field(
        ...,
        description="Dimension name (from a variant's 'dims') to pass to the "
        "kernel's constructor, in declaration order.",
    )


class InputSpec(BaseModel):
    """Schema for a single entry of the top-level 'inputs' mapping.

    Mirrors the fields read by `ai_bench.harness.core.get_inputs`.
    """

    model_config = ConfigDict(extra="forbid")

    shape: list[str] = Field(
        ...,
        min_length=1,
        description="Dimension names composing the input's shape.",
        json_schema_extra=_flow_array_json_schema,
    )
    dtype: str = Field(
        default=_INHERIT,
        description="Torch dtype name (e.g. 'float32'), or 'inherit' to reuse "
        "the variant's dtype.",
        json_schema_extra=_input_dtype_json_schema,
    )
    range: list[Union[int, float, str]] | None = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="[low, high) value range for integer/bool inputs. Values "
        "may be numbers or dimension names.",
        json_schema_extra=_drop_null_default_json_schema,
    )
    inits: list[str] | None = Field(
        default=None,
        description="Initialization transforms (see InInitKey) applied in order.",
        json_schema_extra=_drop_null_default_json_schema,
    )

    @field_validator("dtype")
    @classmethod
    def _validate_dtype(cls, value: str) -> str:
        if value == _INHERIT:
            return value
        return _check_torch_dtype(value)

    @field_validator("inits")
    @classmethod
    def _validate_inits(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        unknown = [init for init in value if init not in _VALID_INPUT_INITS]
        if unknown:
            raise ValueError(
                f"Unknown input init(s) {unknown}. Expected one of: "
                f"{sorted(_VALID_INPUT_INITS)}"
            )
        return value


def _tolerance_json_schema(json_schema: dict[str, Any]) -> None:
    """Widen the generated JSON Schema for 'rtol'/'atol' to also accept numeric
    strings (e.g. 'inf'), matching `VariantEntry._coerce_tolerance` below.
    """
    json_schema.pop("anyOf", None)
    json_schema["anyOf"] = [
        {"type": "number"},
        {
            "type": "string",
            "description": "Numeric string (e.g. 'inf', '-inf', 'nan').",
        },
        {"type": "null"},
    ]
    _drop_null_default_json_schema(json_schema)


def _variant_dtype_json_schema(json_schema: dict[str, Any]) -> None:
    """Variant entry 'dtype' field suggestions.

    Uses 'examples' (not 'enum') so this only affects editor suggestions - it
    doesn't restrict validation to this exact list, matching
    `_check_torch_dtype`'s dynamic (`getattr(torch, ...)`) check.
    """
    json_schema["examples"] = [*_basic_torch_dtype_names()]


class VariantEntry(BaseModel):
    """Schema for a single entry of a variant category list (e.g. 'ci', 'bench-cpu').

    Mirrors the fields read via `VKey` by the various `get_*` helpers.
    """

    model_config = ConfigDict(extra="forbid")

    params: list[str] = Field(
        ...,
        min_length=1,
        description="Names of declared 'inputs' to construct and feed to this variant.",
        json_schema_extra=_flow_array_json_schema,
    )
    dtype: str | None = Field(
        ...,
        description="Torch dtype name applied to the model/variant.",
        json_schema_extra=_variant_dtype_json_schema,
    )
    memory_format: str | None = Field(
        default=None,
        description="Torch memory format name (e.g. 'channels_last') applied "
        "to the model/inputs.",
        json_schema_extra=_memory_format_json_schema,
    )
    dims: dict[str, DimValue] = Field(
        default_factory=dict,
        description="Dimension name -> concrete value(s) for this variant.",
    )
    flop: FormulaValue | None = Field(
        default=None,
        description="Number of FLOP, or a formula over 'dims'.",
        json_schema_extra=_drop_null_default_json_schema,
    )
    mem_bytes: FormulaValue | None = Field(
        default=None,
        description="Number of memory access bytes, or a formula over 'dims'.",
        json_schema_extra=_drop_null_default_json_schema,
    )
    rtol: float | None = Field(
        default=None,
        description="Relative tolerance override.",
        json_schema_extra=_tolerance_json_schema,
    )
    atol: float | None = Field(
        default=None,
        description="Absolute tolerance override.",
        json_schema_extra=_tolerance_json_schema,
    )

    @field_validator("dtype")
    @classmethod
    def _validate_dtype(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _check_torch_dtype(value)

    @field_validator("memory_format")
    @classmethod
    def _validate_memory_format(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _check_torch_memory_format(value)

    @field_validator("rtol", "atol", mode="before")
    @classmethod
    def _coerce_tolerance(cls, value: Any) -> Any:
        # Some existing specs use unquoted 'inf', which YAML parses as a str
        # (not a float) unless written as '.inf'. Accept it for compatibility.
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError as exc:
                raise ValueError(f"Invalid tolerance value: '{value}'") from exc
        return value


class KernelSpec(BaseModel):
    """Schema for a KernelBench-style problem spec YAML file.

    The 'inputs' and 'inits' keys are validated explicitly. Every other
    top-level key (e.g. 'ci', 'bench-cpu', 'bench-gpu', 'simple-cpu', or any
    custom variant name used via `--variant`) is treated as a variant
    category and validated as a list of `VariantEntry`, so new categories
    don't require changes to this schema or to `SpecKey`.
    """

    model_config = ConfigDict(extra="allow")

    name: str | None = None
    description: str | None = None
    inputs: dict[str, InputSpec] = Field(..., min_length=1)
    inits: list[InitEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_variant_categories(self) -> "KernelSpec":
        extra: dict[str, Any] = dict(self.__pydantic_extra__ or {})
        validated: dict[str, list[VariantEntry]] = {}
        for category, entries in extra.items():
            if not isinstance(entries, list):
                raise ValueError(
                    f"Variant category '{category}' must be a list, got "
                    f"{type(entries).__name__}"
                )
            try:
                validated[category] = [
                    VariantEntry.model_validate(entry) for entry in entries
                ]
            except Exception as exc:
                raise ValueError(
                    f"Invalid entry in variant category '{category}': {exc}"
                ) from exc

        for category, variants in validated.items():
            for variant in variants:
                self._cross_validate_variant(category, variant)

        object.__setattr__(self, "__pydantic_extra__", validated)
        return self

    def _cross_validate_variant(self, category: str, variant: VariantEntry) -> None:
        """Cross-check a variant against the spec's 'inputs' and 'inits'."""
        unknown_params = [p for p in variant.params if p not in self.inputs]
        if unknown_params:
            raise ValueError(
                f"Variant category '{category}' references undeclared input(s): "
                f"{unknown_params}"
            )

        for init_entry in self.inits:
            if init_entry.dim not in variant.dims:
                raise ValueError(
                    f"Variant category '{category}' is missing dim '{init_entry.dim}' "
                    "required by the top-level 'inits'"
                )

        for param in variant.params:
            input_spec = self.inputs[param]
            if input_spec.dtype == _INHERIT and variant.dtype is None:
                raise ValueError(
                    f"Variant category '{category}': input '{param}' uses 'inherit' "
                    "dtype but the variant defines no 'dtype'"
                )
            missing_dims = [dim for dim in input_spec.shape if dim not in variant.dims]
            if missing_dims:
                raise ValueError(
                    f"Variant category '{category}': input '{param}' needs dim(s) "
                    f"{missing_dims} not present in the variant's 'dims'"
                )

    def variant_categories(self) -> dict[str, list[VariantEntry]]:
        """Return all validated variant category lists.
        Returns:
            Mapping of category name (e.g. 'ci', 'bench-cpu', 'bench-gpu', ...)
            to its validated variant entries.
        """
        return dict(self.__pydantic_extra__ or {})

    def get_variants(self, category: str) -> list[VariantEntry]:
        """Return validated variant entries for a given category name.
        Args:
            category: Variant category name (e.g. `SpecKey.V_CI`, 'bench-cpu', ...)
        Returns:
            List of validated variant entries
        Raises:
            KeyError: If the category is not present in the spec.
        """
        variants = self.variant_categories()
        if category not in variants:
            raise KeyError(f"Spec has no variant category '{category}'")
        return variants[category]


def parse_spec(spec: dict[str, Any]) -> KernelSpec:
    """Validate a raw spec dict (as returned by `yaml.safe_load`) against the schema.
    Args:
        spec: Raw spec dictionary
    Returns:
        Validated spec model
    """
    return KernelSpec.model_validate(spec)


def load_spec_file(path: Path | str) -> KernelSpec:
    """Load and validate a spec YAML file.
    Args:
        path: Path to the spec '.yaml' file
    Returns:
        Validated spec model
    """
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)
    return parse_spec(raw)


def generate_json_schema() -> dict[str, Any]:
    """Build a standalone JSON Schema document for KernelBench spec YAML files.

    The result is suitable for editor YAML autocompletion/validation (e.g.
    VS Code's `redhat.vscode-yaml` extension via the `yaml.schemas` setting).
    It is derived from the same pydantic models used by `parse_spec`, so it
    stays in sync with the Python-side validation rules that don't require
    cross-field lookups (JSON Schema alone cannot express e.g. "a variant's
    'dims' must cover every dim name used in its inputs' shapes" - those
    checks still only run through `parse_spec`/`load_spec_file`).
    Returns:
        A JSON Schema (draft-07) document.
    """
    variant_list_schema = {
        "type": "array",
        "items": {"$ref": "#/$defs/VariantEntry"},
    }

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "KernelBench problem spec",
        "description": (
            "Schema for KernelBench-style problem spec YAML files (see "
            "ai_bench.harness.core.schema.KernelSpec). Generated from the "
            "pydantic models - do not edit by hand; regenerate with "
            "'uv run python -m ai_bench.harness.core.schema <output-path>'."
        ),
        "type": "object",
        "required": ["inputs"],
        "properties": {
            "name": {
                "type": "string",
                "description": "Optional, informational problem name.",
            },
            "description": {
                "type": "string",
                "description": "Optional, informational problem description.",
            },
            "inputs": {
                "type": "object",
                "minProperties": 1,
                "description": "Mapping of input name -> input spec.",
                "additionalProperties": {"$ref": "#/$defs/InputSpec"},
            },
            "inits": {
                "type": "array",
                "default": [],
                "description": (
                    "Dimension names passed positionally to the kernel's constructor."
                ),
                "items": {"$ref": "#/$defs/InitEntry"},
            },
            **{name: variant_list_schema for name in _KNOWN_VARIANT_CATEGORIES},
        },
        "additionalProperties": variant_list_schema,
        "$defs": {
            "InputSpec": InputSpec.model_json_schema(),
            "InitEntry": InitEntry.model_json_schema(),
            "VariantEntry": VariantEntry.model_json_schema(),
        },
    }


if __name__ == "__main__":
    import json
    import sys

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    text = json.dumps(generate_json_schema(), indent=2) + "\n"
    if out is None:
        print(text, end="")
    else:
        out.write_text(text)
