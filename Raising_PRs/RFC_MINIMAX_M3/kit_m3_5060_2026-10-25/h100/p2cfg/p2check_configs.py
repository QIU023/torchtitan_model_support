"""P2 check only (kit): the Qwen3.5 debug recipes with CUDA graphs off; the MoE one on 4 GPUs (tp 2 x pp 2 x ep 2)
instead of the recipe's 8-GPU default."""

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan_recipes.tests.models.qwen3_5 import qwen35_debugmodel, qwen35_debugmodel_moe


def qwen35_dense_nocg():
    config = qwen35_debugmodel()
    config.training.disable_cuda_graphs = True
    return config


def qwen35_moe_nocg_4gpu():
    config = qwen35_debugmodel_moe(
        parallelism=ParallelismConfig(
            pipeline_parallel_degree=2,
            num_pp_microbatches=2,
            expert_parallel_degree=2,
            tensor_parallel_degree=2,
        )
    )
    config.training.disable_cuda_graphs = True
    return config
