"""Write a tokenizer.json for Kimi K3, since TorchTitan loads tokenizers only from tokenizer.json or vocab + merges.

python make_kimi_tokenizer.py <out_dir>

Downloads the tokenizer files of moonshotai/Kimi-K3 at the revision numerical_tests_kimi_k3.py pins. If the repo
already has a tokenizer.json, that is all. Otherwise it loads the repo's tiktoken tokenizer (trust_remote_code),
converts it with transformers' TikTokenConverter, adds the tokenizer's special tokens, and checks that both encode
a sample identically and that every KIMI_K3_SPECIAL_TOKENS string is an added token. A failed check exits non-zero:
then use a Qwen3 tokenizer.json (vocab 151669 < 163840) with MN_SPECIAL=qwen.
"""

import os
import sys

REPO = "moonshotai/Kimi-K3"
REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
SPECIAL = ["<|media_pad|>", "<|media_begin|>", "<|media_end|>", "[PAD]"]
SAMPLE = "Kimi K3 tokenizer check: 1234 apples, naive cafe, 中文测试, <tag> end."


def main(out_dir: str) -> int:
    from huggingface_hub import snapshot_download

    snapshot_download(
        REPO,
        revision=REVISION,
        local_dir=out_dir,
        allow_patterns=["*.json", "*.model", "*.tiktoken", "*.py", "*.txt"],
    )
    target = os.path.join(out_dir, "tokenizer.json")
    if not os.path.exists(target):
        from transformers import AutoTokenizer
        from transformers.convert_slow_tokenizer import TikTokenConverter

        slow = AutoTokenizer.from_pretrained(out_dir, trust_remote_code=True)
        vocab_file = getattr(slow, "vocab_file", None) or os.path.join(out_dir, "tiktoken.model")
        pattern = getattr(slow, "pat_str", None) or getattr(getattr(slow, "model", None), "_pat_str", None)
        if pattern is None:
            print("could not find the tiktoken split pattern on the slow tokenizer", file=sys.stderr)
            return 1
        specials = sorted(slow.get_added_vocab().items(), key=lambda kv: kv[1])
        fast = TikTokenConverter(
            vocab_file=vocab_file,
            pattern=pattern,
            additional_special_tokens=[token for token, _ in specials],
        ).converted()
        fast.save(target)
        slow_ids = slow.encode(SAMPLE, add_special_tokens=False)
    else:
        slow_ids = None

    from tokenizers import Tokenizer

    fast = Tokenizer.from_file(target)
    added = {token.content for token in fast.get_added_tokens_decoder().values()}
    missing = [token for token in SPECIAL if token not in added]
    fast_ids = fast.encode(SAMPLE, add_special_tokens=False).ids
    print(f"vocab {fast.get_vocab_size()} (model vocab 163840), sample ids {fast_ids[:12]}...")
    if missing:
        print(f"special tokens missing from the added tokens: {missing}", file=sys.stderr)
        return 1
    if slow_ids is not None and slow_ids != fast_ids:
        print(f"converted tokenizer differs from the slow one:\n{slow_ids}\n{fast_ids}", file=sys.stderr)
        return 1
    if fast.get_vocab_size() > 163840:
        print("tokenizer vocab exceeds the model vocab 163840", file=sys.stderr)
        return 1
    print(f"ok: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
