# Paths and per-hardware settings of the phase 14 kit, sourced by every script (assignments and exports only).
# HW: h100 | h200 (also B200) | a100 | generic. Every node needs the same layout under $B (each pod has its own copy).
HW=${HW:-h100}
B=${MN_BASE:-$HOME/p14}
V=$B/venv
TT=$B/tt
T=$B/wt_p14
K=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${MN_OUT:-$B/runs}
BRANCH=phase14_multinode_5d
BRANCH_SHA=d6516df37
TORCH_NIGHTLY=2.16.0.dev20261024
TORCHVISION_NIGHTLY=0.30.0.dev20261024
NNODES=${NNODES:-2}
NPROC=${NPROC:-8}
MASTER_PORT=${MASTER_PORT:-29500}
# Node rank and the address of node 0: set by hand, or by the launcher (Kubernetes Indexed Job, PyTorchJob, Slurm).
NODE_RANK=${NODE_RANK:-${JOB_COMPLETION_INDEX:-${PET_NODE_RANK:-${SLURM_NODEID:-0}}}}
MASTER_ADDR=${MASTER_ADDR:-${PET_MASTER_ADDR:-}}

# Tokenizers: K3 needs a tokenizer.json (make_kimi_tokenizer.py, or a Qwen3 one with MN_SPECIAL=qwen); M3 ships one
# (scripts/download_hf_assets.py --repo_id MiniMaxAI/MiniMax-M3 --assets tokenizer --local_dir $B/assets).
ASSETS_k3=${ASSETS_K3:-$B/assets/kimi_k3_tokenizer}
ASSETS_m3=${ASSETS_M3:-$B/assets/MiniMax-M3}

case $HW in
  h100) FLAVOR_k3=k3_16x80g;  FLAVOR_m3=m3_16x80g;  GATE=fused ;;
  # B200 (180 GB) takes the 141 GB downscales and lengths.
  h200|b200) FLAVOR_k3=k3_16x141g; FLAVOR_m3=m3_16x141g; GATE=fused ;;
  # A100: the fused bounded KDA gate needs sm90; no NVLS on the A100 NVSwitch.
  a100) FLAVOR_k3=k3_16x80g;  FLAVOR_m3=m3_16x80g;  GATE=reference; export NCCL_NVLS_ENABLE=0 ;;
  # Any other card (e.g. 8x RTX 5090 on a vast overlay): smoke and communication only; the downscales need 80 GB.
  # M3's full-attention layers move to flex there: the M3 5060 kit needed a shim to keep varlen off FlashAttention on sm120.
  generic) FLAVOR_k3=k3_16x80g_small; FLAVOR_m3=m3_16x80g_small; GATE=reference; export MN_ATTN=${MN_ATTN:-flex} ;;
  *) echo "HW must be h100, h200, b200, a100 or generic, got $HW" >&2; return 1 2>/dev/null || exit 1 ;;
esac

# MN_NET=nebius: a Nebius GPU cluster (pods or VMs); the settings of Nebius's own NCCL test (docs.nebius.com,
# kubernetes/gpu/nccl-test): bootstrap on eth0, data on the mlx5 InfiniBand HCAs.
# MN_NET=together: a Together GPU cluster; its docs give no NCCL settings, so bootstrap on the pod's eth0 and let NCCL
# find the HCAs (run_comm.sh lists the ones it used; set MN_IB_HCA if a RoCE front-end port shows up among them).
case "${MN_NET:-}" in
  nebius)
    MN_IFNAME=${MN_IFNAME:-eth0}
    MN_IB_HCA=${MN_IB_HCA:-mlx5}
    export UCX_NET_DEVICES=${UCX_NET_DEVICES:-eth0} SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1 NCCL_COLLNET_ENABLE=0 ;;
  together)
    MN_IFNAME=${MN_IFNAME:-eth0} ;;
esac

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# Cluster specific, set before the first run (RUNBOOK.md step 3): the interface both nodes reach each other on, and the
# InfiniBand HCAs if any (ibstat). On a vast overlay the interface is eth0 and there is no IB.
[ -n "${MN_IFNAME:-}" ] && export NCCL_SOCKET_IFNAME=$MN_IFNAME GLOO_SOCKET_IFNAME=$MN_IFNAME
[ -n "${MN_IB_HCA:-}" ] && export NCCL_IB_HCA=$MN_IB_HCA
export UV_CACHE_DIR=${UV_CACHE_DIR:-$B/.uvcache}
# The venv's interpreter lives under $B too, so a venv on a shared filesystem works in every pod and after restarts.
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$B/uv-python}
