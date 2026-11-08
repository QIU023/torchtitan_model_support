#!/bin/bash
# Tokenizer for one model, into $B/assets (shared by the nodes). Usage: assets.sh <k3|m3> <env_json_out>
# Writes {"MN_ASSETS": dir, "MN_SPECIAL": model|qwen} to env_json_out; exits non-zero when no tokenizer is usable.
#   m3: MiniMaxAI/MiniMax-M3's tokenizer.json.
#   k3: Kimi K3's tiktoken converted by make_kimi_tokenizer.py; if that fails, Qwen/Qwen3-8B's tokenizer.json
#       (151669 tokens, inside K3's 163840) with Qwen's special tokens.
. "$(dirname "$0")/env.sh"
set -u
MODEL=$1
OUTJ=$2
. $V/bin/activate
cd $T
emit() { printf '{"MN_ASSETS": "%s", "MN_SPECIAL": "%s"}\n' "$1" "$2" > "$OUTJ"; echo "assets $MODEL: $1 ($2)"; exit 0; }
case $MODEL in
  m3)
    D=$B/assets/MiniMax-M3
    [ -f $D/tokenizer.json ] || python scripts/download_hf_assets.py --repo_id MiniMaxAI/MiniMax-M3 \
      --assets tokenizer --local_dir $B/assets
    [ -f $D/tokenizer.json ] && emit $D model ;;
  k3)
    D=$B/assets/kimi_k3_tokenizer
    [ -f $D/tokenizer.json ] || python $K/make_kimi_tokenizer.py $D || rm -f $D/tokenizer.json
    [ -f $D/tokenizer.json ] && emit $D model
    Q=$B/assets/Qwen3-8B
    [ -f $Q/tokenizer.json ] || python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-8B \
      --assets tokenizer --local_dir $B/assets
    [ -f $Q/tokenizer.json ] && emit $Q qwen ;;
esac
echo "assets $MODEL: no usable tokenizer" >&2
exit 1
