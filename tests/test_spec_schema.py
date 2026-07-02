"""Tests for the pydantic spec schema (ai_bench.harness.core.schema)."""

from pathlib import Path

from pydantic import ValidationError
import pytest

from ai_bench.harness.core import schema

_REPO_ROOT = Path(__file__).resolve().parents[1]
_KERNELBENCH_SPECS = _REPO_ROOT / "problems" / "specs" / "KernelBench"
_ALL_SPEC_FILES = sorted(_KERNELBENCH_SPECS.glob("level*/*.yaml"))


class TestRealSpecs:
    """Validate every real KernelBench spec file against the schema."""

    def test_specs_found(self):
        """Sanity check that spec discovery actually found files."""
        assert len(_ALL_SPEC_FILES) > 0

    @pytest.mark.parametrize(
        "spec_path",
        _ALL_SPEC_FILES,
        ids=[str(p.relative_to(_KERNELBENCH_SPECS)) for p in _ALL_SPEC_FILES],
    )
    def test_spec_validates(self, spec_path):
        """Every checked-in spec file must satisfy the schema."""
        spec = schema.load_spec_file(spec_path)
        assert spec.inputs
        assert spec.variant_categories()


class TestKernelSpecModel:
    """Unit tests for schema validation behavior."""

    def _base_spec(self, **overrides):
        spec = {
            "inputs": {
                "X": {"shape": ["BATCH", "N"], "dtype": "inherit"},
            },
            "inits": [{"dim": "N"}],
            "ci": [
                {
                    "params": ["X"],
                    "dtype": "float32",
                    "dims": {"BATCH": 2, "N": 4},
                }
            ],
        }
        spec.update(overrides)
        return spec

    def test_valid_spec_parses(self):
        spec = schema.parse_spec(self._base_spec())
        assert list(spec.inputs.keys()) == ["X"]
        variants = spec.get_variants("ci")
        assert variants[0].dims == {"BATCH": 2, "N": 4}

    def test_missing_inputs_rejected(self):
        bad = self._base_spec()
        del bad["inputs"]
        with pytest.raises(ValidationError):
            schema.parse_spec(bad)

    def test_unknown_input_field_rejected(self):
        bad = self._base_spec()
        bad["inputs"]["X"]["bogus"] = 1
        with pytest.raises(ValidationError):
            schema.parse_spec(bad)

    def test_invalid_dtype_rejected(self):
        bad = self._base_spec()
        bad["inputs"]["X"]["dtype"] = "not_a_real_dtype"
        with pytest.raises(ValidationError):
            schema.parse_spec(bad)

    def test_invalid_input_init_rejected(self):
        bad = self._base_spec()
        bad["inputs"]["X"]["inits"] = ["not_a_real_init"]
        with pytest.raises(ValidationError):
            schema.parse_spec(bad)

    def test_invalid_memory_format_rejected(self):
        bad = self._base_spec()
        bad["ci"][0]["memory_format"] = "not_a_real_format"
        with pytest.raises(ValidationError):
            schema.parse_spec(bad)

    def test_variant_category_must_be_list(self):
        bad = self._base_spec()
        bad["ci"] = {"not": "a list"}
        with pytest.raises(ValidationError):
            schema.parse_spec(bad)

    def test_undeclared_input_reference_rejected(self):
        bad = self._base_spec()
        bad["ci"][0]["params"] = ["UNKNOWN"]
        with pytest.raises(ValidationError):
            schema.parse_spec(bad)

    def test_missing_dim_for_input_shape_rejected(self):
        bad = self._base_spec()
        del bad["ci"][0]["dims"]["N"]
        with pytest.raises(ValidationError):
            schema.parse_spec(bad)

    def test_missing_dim_for_top_level_init_rejected(self):
        bad = self._base_spec(inits=[{"dim": "MISSING_DIM"}])
        with pytest.raises(ValidationError):
            schema.parse_spec(bad)

    def test_inherit_dtype_without_variant_dtype_rejected(self):
        bad = self._base_spec()
        del bad["ci"][0]["dtype"]
        with pytest.raises(ValidationError):
            schema.parse_spec(bad)

    def test_custom_variant_category_name_allowed(self):
        """Arbitrary category names (e.g. 'simple-cpu') are supported, matching
        real spec files even though they aren't in the SpecKey enum."""
        good = self._base_spec()
        good["simple-cpu"] = [
            {"params": ["X"], "dtype": "bfloat16", "dims": {"BATCH": 8, "N": 16}}
        ]
        spec = schema.parse_spec(good)
        assert "simple-cpu" in spec.variant_categories()
        assert spec.get_variants("simple-cpu")[0].dtype == "bfloat16"

    def test_get_variants_unknown_category_raises(self):
        spec = schema.parse_spec(self._base_spec())
        with pytest.raises(KeyError):
            spec.get_variants("does-not-exist")

    def test_string_inf_tolerance_coerced_to_float(self):
        """Some existing specs use unquoted 'inf' (parsed as str by YAML)."""
        good = self._base_spec()
        good["ci"][0]["rtol"] = "inf"
        spec = schema.parse_spec(good)
        assert spec.get_variants("ci")[0].rtol == float("inf")

    def test_name_and_description_are_optional_metadata(self):
        good = self._base_spec(name="my_kernel", description="does a thing")
        spec = schema.parse_spec(good)
        assert spec.name == "my_kernel"
        assert spec.description == "does a thing"
