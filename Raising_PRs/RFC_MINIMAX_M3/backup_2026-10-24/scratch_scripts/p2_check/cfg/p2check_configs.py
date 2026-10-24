"""P2 check only (scratchpad): the Qwen3.5 debug recipes with CUDA graphs off, since main's multimodal debug run
changes the captured vision input's shape between steps on this box (fails the same way before and after P2)."""

from torchtitan_recipes.tests.models.qwen3_5 import qwen35_debugmodel, qwen35_debugmodel_moe


def qwen35_dense_nocg():
    config = qwen35_debugmodel()
    config.training.disable_cuda_graphs = True
    return config


def qwen35_moe_nocg():
    config = qwen35_debugmodel_moe()
    config.training.disable_cuda_graphs = True
    return config
