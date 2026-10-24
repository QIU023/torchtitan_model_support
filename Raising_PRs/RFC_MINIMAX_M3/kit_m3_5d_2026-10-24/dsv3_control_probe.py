"""DeepSeek V3 debug model (MoE, untied embeddings) under the same protocol as m3_text18_probe (logbook kit only): a titan control for the
step-1 gradient comparison between dp1 and PP cells. Same env: M3_CELL, M3_SEED, M3_SEED_OUT, M3_STEPS."""

import os

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.config import ParallelismConfig
from torchtitan_recipes.data import c4_document_capacity

from m3_text18_probe import _degrees, SEQ_LEN, TOKENS_PER_STEP


def _build(degrees):
    from torchtitan_recipes.tests.models.deepseek_v3 import deepseek_v3_debugmodel

    parallelism = ParallelismConfig(
        data_parallel_shard_degree=degrees["dp"],
        tensor_parallel_degree=degrees["tp"],
        expert_parallel_degree=degrees["ep"],
        pipeline_parallel_degree=degrees["pp"],
        enable_sequence_parallel=degrees["tp"] > 1,
    )
    if degrees["pp"] > 1:
        parallelism.num_pp_microbatches = TOKENS_PER_STEP // SEQ_LEN // degrees["dp"]
    config = deepseek_v3_debugmodel(seq_len=SEQ_LEN, parallelism=parallelism)
    config.dataloader.max_num_documents = c4_document_capacity(SEQ_LEN)
    config.training.disable_cuda_graphs = True
    config.training.num_tokens_per_train_step = TOKENS_PER_STEP
    config.training.num_tokens_per_microbatch_per_dp_rank = SEQ_LEN
    config.training.steps = int(os.environ.get("M3_STEPS", "20"))
    config.metrics.log_freq = 1
    config.debug.seed = 42
    config.debug.deterministic = True
    return config


def cell():
    config = _build(_degrees(os.environ["M3_CELL"]))
    config.checkpointer = CheckpointManager.Config(
        interval=100000, initial_load_path=os.environ["M3_SEED"], initial_load_model_only=True
    )
    return config


def seed():
    config = _build(_degrees(""))
    config.checkpointer = CheckpointManager.Config(folder=os.environ["M3_SEED_OUT"])
    config.create_seed_checkpoint = True
    return config
