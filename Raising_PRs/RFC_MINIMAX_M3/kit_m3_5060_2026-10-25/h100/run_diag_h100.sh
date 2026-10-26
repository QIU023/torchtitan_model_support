#!/bin/bash
# The fp32 HF parity localization probes of 10-24 (hf_parity_probes/) on one H100: block selections against the
# reference, the indexer, per-layer hidden states (free and HF-forced), the vision tower, layer-0 attention, the layer-1 MoE.
. "$(dirname "$0")/env_h100.sh"
O=$R/parity_diag; mkdir -p "${O:?}"; P=$O/progress.txt; note() { echo "$(date +%H:%M:%S) $*" >> $P; }
. $V/bin/activate
note "tree $(git -C $TB rev-parse --short HEAD) torch $(python -c 'import torch; print(torch.__version__)')"
for spec in diag_blocks diag_indexer "diag_hidden force" "diag_hidden free" diag_vision diag_attn0 diag_moe; do
  set -- $spec; name=$1${2:+_$2}
  ( cd $TB && M3_FA_OFF=1 PYTHONPATH=$SHIM:$TB CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=$O/cache/ic TRITON_CACHE_DIR=$O/cache/tc \
      timeout 1800 python $K/hf_parity_probes/$1.py ${2:+"$2"} > $O/$name.txt 2>&1 < /dev/null )
  note "$name rc=$?"
done
rm -rf "${O:?}/cache"
note DIAG_DONE
