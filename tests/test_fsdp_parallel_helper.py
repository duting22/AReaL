# SPDX-License-Identifier: Apache-2.0

import pytest

from areal.api import FSDPParallelStrategy
from areal.api.cli_args import FSDPEngineConfig
from areal.engine.fsdp_utils.parallel import ParallelHelper


def test_default_fsdp_size_uses_full_dp_sp_shard():
    strategy = FSDPParallelStrategy(
        data_parallel_size=8,
        context_parallel_size=2,
        tensor_parallel_size=1,
    )

    helper = ParallelHelper.from_parallel_strategy(strategy)

    assert helper.effective_fsdp_size == 16
    assert helper.uses_hsdp is False


def test_custom_fsdp_size_splits_dp_replicate_and_shard():
    strategy = FSDPParallelStrategy(
        data_parallel_size=8,
        context_parallel_size=1,
        tensor_parallel_size=1,
    )

    helper = ParallelHelper.from_parallel_strategy(strategy, fsdp_size=4)

    assert helper.effective_fsdp_size == 4
    assert helper.uses_hsdp is True
    assert helper.fsdp_replicate_size == 2


def test_custom_fsdp_size_can_be_independent_of_sp_size():
    strategy = FSDPParallelStrategy(
        data_parallel_size=9,
        context_parallel_size=4,
        tensor_parallel_size=1,
    )

    helper = ParallelHelper.from_parallel_strategy(strategy, fsdp_size=6)

    assert helper.uses_hsdp is True
    assert helper.effective_fsdp_size == 6
    assert helper.fsdp_replicate_size == 6


def test_custom_fsdp_size_must_divide_dp_sp():
    strategy = FSDPParallelStrategy(
        data_parallel_size=8,
        context_parallel_size=1,
        tensor_parallel_size=1,
    )

    with pytest.raises(ValueError, match="must be divisible by fsdp_size"):
        ParallelHelper.from_parallel_strategy(strategy, fsdp_size=3)


def test_custom_fsdp_size_rejects_expert_parallelism():
    strategy = FSDPParallelStrategy(
        data_parallel_size=8,
        context_parallel_size=1,
        tensor_parallel_size=1,
        expert_parallel_size=2,
    )

    with pytest.raises(ValueError, match="expert parallelism"):
        ParallelHelper.from_parallel_strategy(strategy, fsdp_size=4)


def test_fsdp_engine_config_rejects_invalid_fsdp_size():
    with pytest.raises(ValueError, match="fsdp_size"):
        FSDPEngineConfig(fsdp_size=0)
