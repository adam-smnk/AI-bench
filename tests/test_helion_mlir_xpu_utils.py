"""Helion MLIR XPU helpers' host paths."""

from unittest import mock

import pytest
import torch

pytest.importorskip("helion_mlir_backend")

from helion_mlir_xpu_utils import elementwise


def test_block_sizes_search_runs_once_per_shape() -> None:
    """The search over column tiles was a real share of every kernel call."""
    shape = torch.Size([4, 32752])
    with mock.patch.object(
        elementwise, "default_sg_tile", wraps=elementwise.default_sg_tile
    ) as search:
        first = elementwise.block_sizes(shape)
        searches = search.call_count
        assert elementwise.block_sizes(shape) == first
    assert searches > 0
    assert search.call_count == searches
    assert first[0] == 1 and shape[-1] % first[-1] == 0


def test_block_sizes_of_a_shape_without_tiles_raises_every_call() -> None:
    for _ in range(2):
        with pytest.raises(ValueError, match="not a multiple"):
            elementwise.block_sizes((4, 17))
