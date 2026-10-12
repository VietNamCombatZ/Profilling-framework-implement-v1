from __future__ import annotations
import argparse
from .benchmark import RunConfig, benchmark


def main():
    p = argparse.ArgumentParser(description="Profile small LLM inference on Apple Silicon / CPU.")
    p.add_argument("--model", default="sshleifer/tiny-gpt2", help="Hugging Face model ID or local model directory")
    p.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    p.add_argument("--prompt-tokens", type=int, default=64)
    p.add_argument("--new-tokens", type=int, default=32)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--output-dir", default="results")
    p.add_argument("--trust-remote-code", action="store_true")
    a = p.parse_args()
    if a.prompt_tokens < 1 or a.new_tokens < 1 or a.repeats < 1 or a.warmup < 0:
        p.error("prompt-tokens, new-tokens, and repeats must be >= 1; warmup must be >= 0")
    cfg = RunConfig(
        model=a.model, device=a.device, prompt_tokens=a.prompt_tokens,
        max_new_tokens=a.new_tokens, repeats=a.repeats, warmup=a.warmup,
        seed=a.seed, output_dir=a.output_dir, trust_remote_code=a.trust_remote_code
    )
    benchmark(cfg)


if __name__ == "__main__":
    main()
