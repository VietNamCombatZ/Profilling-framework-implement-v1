"""Small, reproducible LLM inference benchmark for Apple Silicon (MPS) or CPU.

This is an educational MVP, not a replacement for hardware-specific GPU profilers.
It measures wall-clock latency, generated-token throughput, process RSS, and
PyTorch allocator memory where the backend exposes it.
"""
from __future__ import annotations

import gc
import json
import os
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class RunConfig:
    model: str = "sshleifer/tiny-gpt2"
    device: str = "auto"
    prompt_tokens: int = 64
    max_new_tokens: int = 32
    repeats: int = 3
    warmup: int = 1
    seed: int = 1234
    output_dir: str = "results"
    trust_remote_code: bool = False


def choose_device(requested: str) -> str:
    if requested != "auto":
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available in this PyTorch/macOS environment.")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return requested
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sync_device(device: str) -> None:
    # MPS and CUDA are asynchronous: synchronize before reading wall-clock timings.
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def memory_snapshot(device: str) -> dict[str, float | None]:
    proc = psutil.Process(os.getpid())
    rss_mb = proc.memory_info().rss / (1024 ** 2)
    allocated_mb = reserved_mb = None
    if device == "mps":
        try:
            allocated_mb = torch.mps.current_allocated_memory() / (1024 ** 2)
            driver_fn = getattr(torch.mps, "driver_allocated_memory", None)
            if driver_fn:
                reserved_mb = driver_fn() / (1024 ** 2)
        except Exception:
            pass
    elif device == "cuda":
        allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
    return {
        "process_rss_mb": round(rss_mb, 3),
        "torch_allocated_mb": round(allocated_mb, 3) if allocated_mb is not None else None,
        "torch_reserved_or_driver_mb": round(reserved_mb, 3) if reserved_mb is not None else None,
    }


def make_prompt(tokenizer, desired_tokens: int) -> str:
    # Repeat a neutral phrase, then truncate token IDs to the requested prompt length.
    seed_text = (
        "Profiling measures the performance of language model inference. "
        "The experiment records latency, throughput, and memory usage. "
    ) * max(4, desired_tokens // 8 + 2)
    ids = tokenizer(seed_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    ids = ids[:max(1, desired_tokens)]
    return tokenizer.decode(ids, skip_special_tokens=True)


def run_once(model, tokenizer, device: str, prompt: str, max_new_tokens: int) -> dict[str, Any]:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Prefill measurement: one forward pass over the complete prompt, with KV cache.
    sync_device(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        prefill_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    sync_device(device)
    prefill_s = time.perf_counter() - t0
    past_key_values = prefill_out.past_key_values
    next_token = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del prefill_out

    # Decode measurement: autoregressive one-token steps using the KV cache.
    # The first selected token is produced by the prefill logits; time decode calls separately.
    generated = [next_token]
    decode_step_times = []
    current_mask = attention_mask
    with torch.inference_mode():
        for _ in range(max(0, max_new_tokens - 1)):
            if current_mask is not None:
                current_mask = torch.cat(
                    [current_mask, torch.ones((current_mask.shape[0], 1), dtype=current_mask.dtype, device=device)],
                    dim=1,
                )
            sync_device(device)
            step_t0 = time.perf_counter()
            out = model(
                input_ids=next_token,
                attention_mask=current_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            sync_device(device)
            decode_step_times.append(time.perf_counter() - step_t0)
            past_key_values = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            del out

    total_s = prefill_s + sum(decode_step_times)
    n_generated = len(generated)
    decode_s = sum(decode_step_times)
    return {
        "prefill_latency_s": prefill_s,
        "decode_latency_s": decode_s,
        "decode_step_latency_mean_s": statistics.mean(decode_step_times) if decode_step_times else 0.0,
        "decode_step_latency_p50_s": statistics.median(decode_step_times) if decode_step_times else 0.0,
        "end_to_end_model_time_s": total_s,
        "generated_tokens": n_generated,
        "decode_tokens_timed": len(decode_step_times),
        "decode_tokens_per_s": (len(decode_step_times) / decode_s) if decode_s > 0 else 0.0,
        "end_to_end_tokens_per_s": (n_generated / total_s) if total_s > 0 else 0.0,
        "prompt_tokens_actual": int(input_ids.shape[-1]),
        **memory_snapshot(device),
    }


def environment_info(device: str) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "device": device,
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "cuda_available": torch.cuda.is_available(),
    }


def benchmark(config: RunConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    device = choose_device(config.device)
    torch.manual_seed(config.seed)
    if device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    gc.collect()

    print(f"Loading tokenizer/model: {config.model}")
    tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=config.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        trust_remote_code=config.trust_remote_code,
    )
    model.eval().to(device)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = make_prompt(tokenizer, config.prompt_tokens)
    print(f"Device={device}; prompt target={config.prompt_tokens}; new tokens={config.max_new_tokens}")

    for i in range(config.warmup):
        print(f"Warm-up {i + 1}/{config.warmup}")
        _ = run_once(model, tokenizer, device, prompt, config.max_new_tokens)
    sync_device(device)

    rows: list[dict[str, Any]] = []
    for i in range(config.repeats):
        gc.collect()
        row = run_once(model, tokenizer, device, prompt, config.max_new_tokens)
        row.update({
            "run_index": i + 1,
            "model": config.model,
            "device": device,
            "prompt_tokens_requested": config.prompt_tokens,
            "max_new_tokens_requested": config.max_new_tokens,
            "seed": config.seed,
        })
        rows.append(row)
        print(
            f"Run {i + 1}/{config.repeats}: prefill={row['prefill_latency_s']:.4f}s, "
            f"decode={row['decode_latency_s']:.4f}s, "
            f"decode_tps={row['decode_tokens_per_s']:.2f}, RSS={row['process_rss_mb']:.1f}MB"
        )

    metadata = {"config": asdict(config), "environment": environment_info(device)}
    outdir = Path(config.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stem = f"benchmark-{stamp}-p{config.prompt_tokens}-n{config.max_new_tokens}"
    csv_path = outdir / f"{stem}.csv"
    json_path = outdir / f"{stem}.json"
    import pandas as pd
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({"metadata": metadata, "runs": rows}, indent=2), encoding="utf-8")
    print(f"\nSaved:\n  {csv_path}\n  {json_path}")
    return rows, metadata
