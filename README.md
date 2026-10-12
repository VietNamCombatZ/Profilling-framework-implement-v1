# Profiling Framework for AI Model — LLM MVP

A small, reproducible educational framework to profile autoregressive LLM inference on Apple Silicon (MPS) or CPU. It separates a **Prefill forward pass** from **Decode token steps**, records latency/throughput and memory indicators, writes CSV/JSON, and includes a Jupyter Notebook for repeatable experiments.

> Scope note: this MVP measures PyTorch model-level timings. It is not a CUDA kernel profiler and cannot directly validate NVIDIA-specific claims about CUDA kernels or GPU L2 cache. MPS timings are for your Mac environment.

## 1. Requirements

- Apple Silicon Mac with a recent macOS version
- Python 3.10–3.13 (Python 3.12 is a safe default)
- Internet access on first run to download the tiny demo model
- Around 1 GB or more free RAM is recommended for the tiny demo and Python dependencies

## 2. Create environment

```bash
cd profiling_framework_mvp
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

If `python3.12` is not installed, use an available supported version:
```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Do not install packages into Homebrew's managed global Python environment; use the virtual environment above.

## 3. Run the first benchmark

```bash
profile-llm --device auto --prompt-tokens 64 --new-tokens 32 --repeats 3 --warmup 1
```

`auto` prefers MPS when available, then CUDA, then CPU. To force CPU:
```bash
profile-llm --device cpu --prompt-tokens 32 --new-tokens 16 --repeats 2 --warmup 1
```

The default `sshleifer/tiny-gpt2` is a small demonstration model intended to make the pipeline easy to run. It is not representative of the performance of a production LLM. The first run downloads model files from Hugging Face.

## 4. Open the notebook

```bash
jupyter notebook notebooks/llm_profiling_mvp.ipynb
```

Run cells from top to bottom. It uses the same package and defaults to short experiments.

## 5. Metrics and interpretation

- `prefill_latency_s`: wall-clock time for one forward pass over the prompt. This MVP does not include tokenization in that timing.
- `decode_latency_s`: sum of timed model calls after the first next-token selection, using the KV cache.
- `decode_step_latency_mean_s` / `p50_s`: mean and median per timed decode call.
- `decode_tokens_per_s`: number of timed decode calls divided by their combined time. The first generated token comes from Prefill and is not included in this decode-only rate.
- `end_to_end_model_time_s`: prefill plus timed decode calls; excludes model loading and tokenization.
- `end_to_end_tokens_per_s`: generated token count divided by that model time. Since the first token is selected from Prefill, this is an approximate model-time throughput, not user-perceived end-to-end latency.
- `process_rss_mb`: resident memory of the entire Python process, including libraries and model weights.
- `torch_allocated_mb`: PyTorch tensor allocator metric where available (MPS/CUDA); unavailable on CPU.
- `torch_reserved_or_driver_mb`: backend-specific allocator/driver memory indicator where available. It is not guaranteed to equal total physical GPU memory.

The implementation uses greedy argmax decoding to reduce randomness. Report medians and variability across repeats, and record machine/software details.

## 6. Reproducible experiments

Run the following from the project root. Keep the same model, device, warm-up, and repeat count while varying only one factor at a time.

### Experiment A: prompt length
```bash
profile-llm --prompt-tokens 32 --new-tokens 16 --repeats 5 --warmup 1 --output-dir results/prompt32
profile-llm --prompt-tokens 128 --new-tokens 16 --repeats 5 --warmup 1 --output-dir results/prompt128
profile-llm --prompt-tokens 256 --new-tokens 16 --repeats 5 --warmup 1 --output-dir results/prompt256
```

### Experiment B: number of generated tokens
```bash
profile-llm --prompt-tokens 64 --new-tokens 16 --repeats 5 --warmup 1 --output-dir results/new16
profile-llm --prompt-tokens 64 --new-tokens 32 --repeats 5 --warmup 1 --output-dir results/new32
profile-llm --prompt-tokens 64 --new-tokens 64 --repeats 5 --warmup 1 --output-dir results/new64
```

### Experiment C: CPU vs MPS
```bash
profile-llm --device cpu --prompt-tokens 64 --new-tokens 32 --repeats 3 --warmup 1 --output-dir results/cpu
profile-llm --device mps --prompt-tokens 64 --new-tokens 32 --repeats 3 --warmup 1 --output-dir results/mps
```
Only run the MPS command if `torch.backends.mps.is_available()` is true. CPU and MPS results can vary with macOS, thermal state, background tasks, PyTorch version, and memory pressure.

## 7. Research connection

1. **LLM inference phase analysis:** Prefill processes the prompt in a single forward pass; Decode repeatedly processes the latest token while reusing KV cache. The notebook reports these phases separately so the measurements can test how prompt length and generated length affect them.
2. **Systematic characterization of LLM inference on GPUs:** this MVP borrows the experimental style—control one variable, measure latency/throughput/memory, repeat, and interpret bottlenecks. It does not reproduce GPU-specific hardware counters, L2-cache analysis, or CUDA-kernel profiling from GPU studies.
3. **Profiling framework design:** it wraps model loading, warm-up, synchronized timing, repeated runs, metadata capture, and structured output into a reusable CLI/API. This makes experiments easier to reproduce than ad hoc timing cells.
4. **PyTorch Profiler:** a future extension can capture operator-level CPU/MPS traces with `torch.profiler`; the current MVP deliberately keeps the baseline benchmark lightweight to avoid mixing profiler overhead into the timing measurements.

## 8. Limitations

- Default tiny GPT-2 is for pipeline validation, not quality or production performance.
- `process_rss_mb` is process-wide RAM, not pure model memory.
- PyTorch allocator metrics on MPS do not expose every aspect of unified memory pressure.
- Prefill/decode measurements are model-level wall-clock measurements, not hardware kernel measurements.
- Decode implementation is a transparent educational loop. It is not as optimized as `model.generate()` or a production inference engine.
- For scientific reporting, run each configuration more than once, include medians and spread, and state all software/hardware settings.

## 9. Troubleshooting

- **MPS unavailable:** use `--device cpu`; update PyTorch/macOS if needed.
- **Model download error:** check network access and retry. Hugging Face may be temporarily unavailable.
- **Out of memory:** reduce `--prompt-tokens` and `--new-tokens`, close other applications, or use CPU.
- **Unsupported model architecture:** start with the default tiny GPT-2. Some model architectures may need different cache APIs.
- **Slow first run:** the first run includes model download and library initialization; these are outside the reported model timings.
