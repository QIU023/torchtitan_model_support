"""MiniMax M3 debug model on the K3 text parallelism matrix, logbook kit only.

M3_CELL: comma list of degrees, e.g. "dp=2,tp=2,ep=2,pp=2" (missing keys are 1; "dpr" is the HSDP replicate degree);
sequence parallelism is on under TP.
M3_SEED: the seed checkpoint folder every cell loads (model only). M3_STEPS: steps (20).
Text c4_test, seq 512 (the debug MSA budget is 4 x 16 = 64 keys, so every layer past layer 0 is sparse), 4096 tokens
per step in one-sequence micro-batches per dp rank, 8 // dp pipeline micro-batches, AdamW, seed 42, deterministic.
cell() loads the seed; seed() writes it from a single-GPU build.
"""

import os

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.config import ParallelismConfig

from torchtitan_recipes.data import c4_document_capacity

SEQ_LEN = 512
TOKENS_PER_STEP = 4096


def _degrees(spec: str) -> dict[str, int]:
    degrees = {"dpr": 1, "dp": 1, "tp": 1, "ep": 1, "pp": 1, "vpp": 1}
    for item in filter(None, spec.split(",")):
        key, value = item.split("=")
        degrees[key] = int(value)
    return degrees


def _build(degrees: dict[str, int]):
    from torchtitan_recipes.tests.models.minimax_m3 import minimax_m3_debugmodel

    parallelism = ParallelismConfig(
        data_parallel_replicate_degree=degrees["dpr"],
        data_parallel_shard_degree=degrees["dp"],
        tensor_parallel_degree=degrees["tp"],
        expert_parallel_degree=degrees["ep"],
        pipeline_parallel_degree=degrees["pp"],
        enable_sequence_parallel=degrees["tp"] > 1,
    )
    if degrees["pp"] > 1:
        parallelism.num_pp_microbatches = TOKENS_PER_STEP // SEQ_LEN // (degrees["dp"] * degrees["dpr"])
    if degrees["vpp"] == 2:
        # Two stages per rank is the interleaved schedule's default.
        parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    else:
        assert degrees["vpp"] == 1, degrees
    if os.environ.get("M3_DEFER_FSDP") == "1":
        parallelism.fsdp_defer_gradient_reduction = True
    config = minimax_m3_debugmodel(seq_len=SEQ_LEN, parallelism=parallelism)
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
        max_num_documents=c4_document_capacity(SEQ_LEN),
        shuffle=False,
    )
    config.training.num_tokens_per_train_step = TOKENS_PER_STEP
    config.training.num_tokens_per_microbatch_per_dp_rank = SEQ_LEN
    config.training.steps = int(os.environ.get("M3_STEPS", "20"))
    config.metrics.log_freq = 1
    config.debug.seed = 42
    config.debug.deterministic = True
    if os.environ.get("M3_TYPECHECK") == "1":
        from torchtitan_recipes.tests import _set_spmd_typechecking

        _set_spmd_typechecking(config, typechecking=True)
    return config


def cell():
    config = _build(_degrees(os.environ["M3_CELL"]))
    if os.environ.get("M3_SEED"):
        config.checkpointer = CheckpointManager.Config(
            interval=100000, initial_load_path=os.environ["M3_SEED"], initial_load_model_only=True
        )
    return config


def seed():
    config = _build(_degrees(""))
    config.checkpointer = CheckpointManager.Config(folder=os.environ["M3_SEED_OUT"])
    config.create_seed_checkpoint = True
    return config
