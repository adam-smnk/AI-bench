"""Tests for ai_bench.config.settings."""

import os

from pydantic import ValidationError
import pytest

from ai_bench.config import settings as cfg
from ai_bench.config.settings import Settings
from ai_bench.config.settings import get_settings
from ai_bench.config.settings import reset_settings
from ai_bench.config.settings import setting_env_vars


@pytest.fixture
def clean_env(monkeypatch):
    """Remove every AIBENCH_* variable so defaults can be asserted."""
    for key in list(os.environ):
        if key.startswith("AIBENCH_"):
            monkeypatch.delenv(key, raising=False)
    reset_settings()
    return monkeypatch


def _fresh() -> Settings:
    """Rebuild the settings from the current environment."""
    return get_settings(reload=True)


class TestGetSettingsCache:
    """Tests for the cached settings getter."""

    def test_returns_settings_instance(self):
        assert isinstance(get_settings(), Settings)

    def test_cached_instance_is_reused(self):
        assert get_settings() is get_settings()

    def test_reset_cache_rebuilds_instance(self):
        first = get_settings()
        reset_settings()
        assert get_settings() is not first

    def test_reload_rebuilds_instance(self):
        first = get_settings()
        assert get_settings(reload=True) is not first

    def test_no_module_level_singleton(self):
        """The module must not eagerly instantiate settings at import time."""
        reset_settings()
        assert cfg._settings is None
        # It is only built on demand.
        get_settings()
        assert cfg._settings is not None


class TestDefaults:
    """Tests for default values when no AIBENCH_* variables are set."""

    def test_defaults(self, clean_env):
        s = _fresh()

        assert s.specs_dir is None
        assert s.kernels_dir is None
        assert s.triton_kernels_dir is None
        assert s.helion_kernels_dir is None
        assert s.mlir_kernels_dir is None
        assert s.gluon_kernels_dir is None
        assert s.sycl_kernels_dir is None
        assert s.mlir_lib_path is None
        assert s.mlir_dump is False
        assert s.mlir_dump_obj is False
        assert s.sycl_compiler == "icpx"
        assert s.sycl_flags is None
        assert s.sycl_include is None
        assert s.sycl_target == ""
        assert s.warmup is None
        assert s.rep is None
        assert s.cpu_min_cache_nuke_mib == 0
        assert s.log is None

    def test_derived_lists_default_empty(self, clean_env):
        s = _fresh()

        assert s.mlir_lib_paths == []
        assert s.sycl_include_dirs == []
        assert s.sycl_flag_list == []


class TestPathSettings:
    """Tests for the path configuration fields."""

    @pytest.mark.parametrize(
        "env_var,field",
        [
            ("AIBENCH_SPECS_DIR", "specs_dir"),
            ("AIBENCH_KERNELS_DIR", "kernels_dir"),
            ("AIBENCH_TRITON_KERNELS_DIR", "triton_kernels_dir"),
            ("AIBENCH_HELION_KERNELS_DIR", "helion_kernels_dir"),
            ("AIBENCH_MLIR_KERNELS_DIR", "mlir_kernels_dir"),
            ("AIBENCH_GLUON_KERNELS_DIR", "gluon_kernels_dir"),
            ("AIBENCH_SYCL_KERNELS_DIR", "sycl_kernels_dir"),
        ],
    )
    def test_path_field_from_env(self, clean_env, env_var, field):
        clean_env.setenv(env_var, "/some/path")

        assert getattr(_fresh(), field) == "/some/path"

    def test_paths_kept_as_str_not_path(self, clean_env):
        clean_env.setenv("AIBENCH_SPECS_DIR", "/some/path")

        assert isinstance(_fresh().specs_dir, str)

    def test_empty_string_stays_falsy(self, clean_env):
        """An empty value must remain falsy, matching os.environ.get behaviour."""
        clean_env.setenv("AIBENCH_SPECS_DIR", "")

        assert not _fresh().specs_dir


class TestMlirSettings:
    """Tests for MLIR backend settings."""

    def test_mlir_lib_paths_split(self, clean_env):
        clean_env.setenv("AIBENCH_MLIR_LIB_PATH", "/a/lib.so:/b/lib.so")

        assert _fresh().mlir_lib_paths == ["/a/lib.so", "/b/lib.so"]

    def test_mlir_lib_paths_filters_empty_segments(self, clean_env):
        clean_env.setenv("AIBENCH_MLIR_LIB_PATH", "/a::/b:")

        assert _fresh().mlir_lib_paths == ["/a", "/b"]

    def test_mlir_dump_defaults_to_false(self, clean_env):
        assert _fresh().mlir_dump is False

    def test_mlir_dump_true_from_env(self, clean_env):
        clean_env.setenv("AIBENCH_MLIR_DUMP", "1")

        assert _fresh().mlir_dump is True

    def test_mlir_dump_false_from_env(self, clean_env):
        clean_env.setenv("AIBENCH_MLIR_DUMP", "0")

        assert _fresh().mlir_dump is False

    def test_mlir_dump_obj_from_env(self, clean_env):
        clean_env.setenv("AIBENCH_MLIR_DUMP_OBJ", "true")

        assert _fresh().mlir_dump_obj is True


class TestSyclSettings:
    """Tests for SYCL backend settings."""

    def test_sycl_compiler_from_env(self, clean_env):
        clean_env.setenv("AIBENCH_SYCL_COMPILER", "clang++")

        assert _fresh().sycl_compiler == "clang++"

    def test_sycl_target_from_env(self, clean_env):
        clean_env.setenv("AIBENCH_SYCL_TARGET", "bmg-g31")

        assert _fresh().sycl_target == "bmg-g31"

    def test_sycl_include_dirs_split(self, clean_env):
        clean_env.setenv("AIBENCH_SYCL_INCLUDE", "/inc/a:/inc/b")

        assert _fresh().sycl_include_dirs == ["/inc/a", "/inc/b"]

    def test_sycl_include_dirs_filters_empty(self, clean_env):
        clean_env.setenv("AIBENCH_SYCL_INCLUDE", ":/inc/a::")

        assert _fresh().sycl_include_dirs == ["/inc/a"]

    def test_sycl_flag_list_split(self, clean_env):
        clean_env.setenv("AIBENCH_SYCL_FLAGS", "-O3 -g -std=c++20")

        assert _fresh().sycl_flag_list == ["-O3", "-g", "-std=c++20"]


class TestBenchSettings:
    """Tests for benchmark tuning settings."""

    def test_warmup_and_rep_parsed_as_int(self, clean_env):
        clean_env.setenv("AIBENCH_WARMUP", "7")
        clean_env.setenv("AIBENCH_REP", "71")

        s = _fresh()

        assert s.warmup == 7
        assert s.rep == 71

    def test_cpu_min_cache_nuke_mib_parsed_as_int(self, clean_env):
        clean_env.setenv("AIBENCH_CPU_MIN_CACHE_NUKE_MIB", "128")

        assert _fresh().cpu_min_cache_nuke_mib == 128

    def test_invalid_int_raises_validation_error(self, clean_env):
        clean_env.setenv("AIBENCH_WARMUP", "notanint")

        with pytest.raises(ValidationError):
            _fresh()


class TestLogSettings:
    """Tests for the log level setting and its use by setup_logger()."""

    def test_log_defaults_to_none(self, clean_env):
        assert _fresh().log is None

    def test_log_from_env(self, clean_env):
        clean_env.setenv("AIBENCH_LOG", "DEBUG")

        assert _fresh().log == "DEBUG"

    def test_setup_logger_defaults_to_info_when_unset(self, clean_env):
        import logging

        from ai_bench.utils.logger import setup_logger

        logger = setup_logger("cfg_log_default_test")

        assert logger.level == logging.INFO

    def test_setup_logger_uses_level_arg_when_unset(self, clean_env):
        import logging

        from ai_bench.utils.logger import setup_logger

        logger = setup_logger("cfg_log_level_test", level=logging.WARNING)

        assert logger.level == logging.WARNING

    def test_setup_logger_reflects_env(self, clean_env):
        import logging

        from ai_bench.utils.logger import setup_logger

        clean_env.setenv("AIBENCH_LOG", "DEBUG")
        logger = setup_logger("cfg_log_env_test")

        assert logger.level == logging.DEBUG

    def test_setup_logger_settings_override_level_arg(self, clean_env):
        import logging

        from ai_bench.utils.logger import setup_logger

        clean_env.setenv("AIBENCH_LOG", "DEBUG")
        logger = setup_logger("cfg_log_precedence_test", level=logging.WARNING)

        # AIBENCH_LOG takes precedence over the explicit level argument.
        assert logger.level == logging.DEBUG


class TestAibenchEnvVars:
    """Tests for the live AIBENCH_* metadata collector."""

    def test_collects_only_aibench_prefixed(self, clean_env):
        clean_env.setenv("AIBENCH_CARD", "BMG")
        clean_env.setenv("AIBENCH_SYSTEM", "Rig1")
        clean_env.setenv("NOT_AIBENCH", "x")

        result = setting_env_vars()

        assert result.get("AIBENCH_CARD") == "BMG"
        assert result.get("AIBENCH_SYSTEM") == "Rig1"
        assert "NOT_AIBENCH" not in result

    def test_reads_live_without_refresh(self, clean_env):
        assert "AIBENCH_CARD" not in setting_env_vars()

        clean_env.setenv("AIBENCH_CARD", "D770")

        # No cache refresh required: it reads os.environ directly.
        assert setting_env_vars().get("AIBENCH_CARD") == "D770"


class TestConfigureOverrides:
    """Tests for programmatic overrides via settings.configure()."""

    def test_configure_takes_priority_over_env(self, clean_env):
        clean_env.setenv("AIBENCH_SPECS_DIR", "/from/env")
        cfg.configure(specs_dir="/from/configure")

        assert get_settings().specs_dir == "/from/configure"

    def test_configure_persists_across_reads(self, clean_env):
        cfg.configure(warmup=99)

        assert get_settings().warmup == 99
        # The override mutates the cached instance, so repeated reads keep it.
        assert get_settings().warmup == 99

    def test_configure_survives_logging(self, clean_env):
        from ai_bench.utils.logger import setup_logger

        cfg.configure(warmup=99)
        # setup_logger only updates the log setting; it must not drop overrides.
        setup_logger("cfg_persist_test")

        assert get_settings().warmup == 99

    def test_configure_allows_none(self, clean_env):
        clean_env.setenv("AIBENCH_SPECS_DIR", "/from/env")
        cfg.configure(specs_dir=None)

        # None is a valid override and clears the env-derived value.
        assert get_settings().specs_dir is None

    def test_configure_merges_successive_calls(self, clean_env):
        cfg.configure(specs_dir="/a")
        cfg.configure(kernels_dir="/b")

        s = get_settings()

        assert s.specs_dir == "/a"
        assert s.kernels_dir == "/b"

    def test_reset_settings_clears_overrides(self, clean_env):
        cfg.configure(specs_dir="/a")
        assert get_settings().specs_dir == "/a"

        cfg.reset_settings()

        assert get_settings().specs_dir is None


class TestFinderIntegration:
    """Verify finder resolves env values via the settings module."""

    def test_finder_reads_env_through_settings(self, clean_env, tmp_path):
        from ai_bench.utils import finder

        finder.reset_configuration()
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()
        clean_env.setenv("AIBENCH_SPECS_DIR", str(specs_dir))

        assert finder.specs() == specs_dir

        finder.reset_configuration()

    def test_finder_configure_survives_logging(self, clean_env, tmp_path):
        from ai_bench.utils import finder
        from ai_bench.utils.logger import setup_logger

        finder.reset_configuration()
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()
        finder.configure(specs_dir=specs_dir)

        # A logging call must not clear the programmatically configured path.
        setup_logger("cfg_finder_test")

        assert finder.specs() == specs_dir

        finder.reset_configuration()

    def test_reset_configuration_reverts_only_path_settings(self, clean_env):
        from ai_bench.utils import finder

        cfg.configure(specs_dir="/some/specs", log="DEBUG")
        assert get_settings().specs_dir == "/some/specs"

        finder.reset_configuration()

        # Path settings are reverted to their env/default value...
        assert get_settings().specs_dir is None
        # ...while non-path settings are left untouched.
        assert get_settings().log == "DEBUG"
