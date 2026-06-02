# Non-Reasoning LLM Benchmark - Jetson Orin Nano Super 8GB

Throughput, latency, and energy-efficiency benchmarks for eight tiny instruct LLMs (135M–1.2B params) running on a **Jetson Orin Nano Super 8GB** at all four power modes.

Supports two backends from the same GGUF files — **llama.cpp** and **Ollama** — for an apples-to-apples comparison.

Key metric: **output tok/J** = OSL / (avg\_power\_W × RL\_p50\_s)

Published report: [benchmark_report.md](./benchmark_report.md)



## Table of Contents

- [Hardware](#hardware)
- [Models](#models)
- [Metrics](#metrics)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running Benchmarks](#running-benchmarks)
- [Output](#output)
- [Generating Charts and Reports](#generating-charts-and-reports)
- [Flags Reference](#flags-reference)



## Hardware

| Component | Detail |
|-----------|--------|
| Board | Jetson Orin Nano Super 8GB Developer Kit |
| CPU | 6× Arm Cortex-A78AE @ up to 1.728 GHz |
| GPU | NVIDIA Ampere, 1024 CUDA cores, 32 Tensor cores |
| Memory | 8 GB LPDDR5 unified CPU + GPU |
| JetPack | R36.4.7 (Ubuntu 22.04, CUDA 12.6) |



## Models

| Model | Quant | GGUF size |
|-------|-------|----------:|
| SmolLM2-135M-Instruct | Q4\_K\_M | 101 MB |
| SmolLM2-360M-Instruct | Q8\_0 | 369 MB |
| Qwen2.5-0.5B-Instruct | Q4\_K\_M | 469 MB |
| LFM2.5-350M | Q4\_K\_M | 219 MB |
| LFM2.5-1.2B-Instruct | Q4\_K\_M | 698 MB |
| Qwen3-0.6B | Q8\_0 | 610 MB |
| Llama-3.2-1B-Instruct | Q4\_K\_M | 771 MB |
| Gemma3-1B-IT | Q4\_K\_M | 769 MB |
| Gemma3-4B-IT *(OOM at all modes)* | Q4\_K\_M | 2.4 GB |

GGUFs are auto-downloaded from Hugging Face on first run into `~/gguf-models/`.



## Metrics

| Metric | Description |
|--------|-------------|
| **output tok/J** | OSL / (avg\_power\_W × RL\_p50\_s) — primary efficiency metric |
| **TTFT p50** | Time to first token, median over 20 requests (ms) |
| **ITL p50** | Inter-token latency, median (ms) |
| **Tok/s** | Output token throughput per user |
| **Power (W)** | `VDD_CPU_GPU_CV` rail average over the aiperf run window |

**Sweep:** 4 prompt lengths × 3 gen lengths × 20 requests = 240 measurements per model per power mode.

| Prompt tokens | 128 | 512 | 1024 | 2048 |
|---------------|-----|-----|------|------|
| **Gen tokens** | 64, 128, 256 | 64, 128, 256 | 64, 128, 256 | 64, 128, 256 |

**Concurrency:** 1 user, 1 request at a time (`--parallel 1`, `--concurrency 1`).



## Prerequisites

- Jetson Orin Nano Super running JetPack R36.x (CUDA 12.x)
- `sudo` access (for `tegrastats`, `nvpmodel`, `jetson_clocks`)
- `tmux` (the script auto-relaunches itself inside a tmux session)

### llama.cpp (required for `--backend llamacpp`)

Build `llama-server` with CUDA for SM\_87 (Jetson Orin Ampere):

```bash
git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
cd ~/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build --config Release -j$(nproc) --target llama-server
```

Binary expected at `~/llama.cpp/build/bin/llama-server`. Override with:
```bash
export LLAMACPP_BIN=/path/to/llama-server
```

### Ollama (required for `--backend ollama`)

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Ollama imports the same local GGUF files — no separate model download.

### aiperf (load generator, required)

```bash
python3 -m venv ~/aiperf-env
source ~/aiperf-env/bin/activate
pip install aiperf
```

The script tries `~/venv` then `~/aiperf-env`.

### HuggingFace CLI (for model auto-download)

```bash
pip install huggingface_hub
huggingface-cli login   # required for gated models (Llama-3.2, Gemma3)
```



## Running Benchmarks

The script **auto-launches inside tmux** on every invocation:

```bash
bash bench-non-reasoning.sh [flags]
# → Launched in tmux session 'non-reasoning-bench'
# → Attach with:  tmux attach -t non-reasoning-bench
```

Detach: `Ctrl+B D` · Reattach: `tmux attach -t non-reasoning-bench`

### llama.cpp backend (default)

```bash
bash bench-non-reasoning.sh --power-mode 1   # 25W — recommended sweet spot
bash bench-non-reasoning.sh --power-mode 0   # 15W
bash bench-non-reasoning.sh --power-mode 2   # MAXN
bash bench-non-reasoning.sh --power-mode 3   # 7W
```

### Ollama backend

```bash
bash bench-non-reasoning.sh --backend ollama --power-mode 1
```

### Both backends in one run

```bash
bash bench-non-reasoning.sh --backend both --power-mode 1
```

Runs llama.cpp first, then Ollama. Results land in separate `llamacpp/` and `ollama/` subdirs inside the same artifact dir.

### Quick test — single model

```bash
bash bench-non-reasoning.sh --only smollm2-135m --reqs 2
```

### Resume an interrupted run

```bash
bash bench-non-reasoning.sh --resume artifacts/blog-all-20260602-0139-25w --power-mode 1
```

Already-completed combos (existing `profile_export_aiperf.json`) are skipped automatically.



## Output

```
artifacts/blog-all-YYYYMMDD-HHMM-<mode>/
├── tegrastats.log                    # raw power + thermal (500 ms interval)
├── model_timing.log                  # per-model start/end epochs
├── report.md                         # auto-generated results table
├── llamacpp/                         # llama.cpp results
│   ├── <model>-server.log
│   └── <model>/gen<G>/ctx<P>/
│       ├── profile_export_aiperf.json
│       └── profile_export_aiperf_timeslices.json
└── ollama/                           # Ollama results
    └── <model>/gen<G>/ctx<P>/
        └── profile_export_aiperf.json
```



## Flags Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--backend <b>` | `llamacpp` | `llamacpp` \| `ollama` \| `both` |
| `--power-mode N` | `0` | 0=15W · 1=25W · 2=MAXN · 3=7W |
| `--reqs N` | `20` | Requests per benchmark combo |
| `--only <name>` | — | Single model filter (substring match) |
| `--skip-smoke` | — | Skip smoke test before each model |
| `--dry-run` | — | Print plan without benchmarking |
| `--resume <dir>` | — | Resume from an existing artifact dir |

> **CMA note:** At 7W, Jetson CMA address space fragments after sequential model loads. The script calls `/proc/sys/vm/compact_memory` between models; a reboot may still be needed before running 7W after other modes. nvpmodel itself requires a reboot when switching to or from 7W — the script detects this and exits with instructions rather than benchmarking at the wrong mode.
