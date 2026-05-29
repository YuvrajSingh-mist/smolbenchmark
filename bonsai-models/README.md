# Bonsai Jetson Benchmark

Throughput and energy-efficiency benchmarks for the Bonsai and Ternary-Bonsai model families running on a Jetson Orin Nano Super 8GB. Measures tok/s, TTFT, ITL, and tok/J (tokens per joule) across all six model variants at four nvpmodel power envelopes.

---

## Table of Contents

- [Hardware](#hardware)
- [Models](#models)
- [Metrics](#metrics)
- [Installation & Setup](#installation--setup)
- [Running Benchmarks](#running-benchmarks)
- [Output](#output)
- [Generating Reports](#generating-reports)

---

## Hardware

| | |
|---|---|
| Board | Jetson Orin Nano Super 8GB |
| Memory | 8 GB LPDDR5 unified (CPU + GPU) |
| Storage | eMMC |
| JetPack | 6.x (CUDA 12.6, SM_87) |

---

## Models

| Model | Quant | Size |
|---|---|---|
| Bonsai-1.7B | Q1_0 | ~237 MB |
| Bonsai-4B | Q1_0 | ~540 MB |
| Bonsai-8B | Q1_0 | ~1.1 GB |
| Ternary-Bonsai-1.7B | Q2_0 | ~300 MB |
| Ternary-Bonsai-4B | Q2_0 | ~700 MB |
| Ternary-Bonsai-8B | Q2_0 | ~1.4 GB |

All models use size-matched Qwen3 tokenizers (1.7B→Qwen3-1.7B, 4B→Qwen3-4B, 8B→Qwen3-8B).

---

## Metrics

| Metric | Description |
|---|---|
| **TTFT** | Time to first token (ms) |
| **ITL** | Inter-token latency (ms) |
| **Tok/s** | Output token throughput per user |
| **Tok/J** | Tokens per joule — tok/s ÷ VDD_CPU_GPU_CV (W). Primary efficiency metric. |

Sweep: 4 prompt lengths × 3 gen lengths × 10 requests = 12 runs per model, 72 per power mode.

| Prompt tokens | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|
| **Gen tokens** | 128, 256, 512 | 128, 256, 512 | 128, 256, 512 | 128, 256, 512 |

---

## Installation & Setup

### Prerequisites

- Jetson Orin running JetPack 6.x
- `aiperf` installed in `~/venv` (included in Bonsai-demo setup)

### 1. Install Bonsai-demo (models + deps)

```bash
git clone https://github.com/PrismML-Eng/Bonsai-demo.git
cd Bonsai-demo

# Optional: choose model size to pre-download (8B default)
export BONSAI_MODEL=8B

# Installs deps and downloads models
./setup.sh
```

### 2. Build llama-server with CUDA for Jetson

`setup.sh` downloads a CPU-only binary on arm64 — there is no pre-built CUDA release for aarch64. Build from source instead:

```bash
cd Bonsai-demo
bash scripts/build_cuda_linux.sh --archs 87 --output cuda
```

`--archs 87` targets SM_87 (Jetson Orin Ampere) specifically. Without it the script compiles a fat binary for desktop GPU architectures (80/86/89/90) that falls back to slow PTX JIT on Orin. The build takes ~20-30 minutes.

The binary will be at `Bonsai-demo/bin/cuda/llama-server`.

### 3. Clone this benchmark repo

```bash
git clone <this-repo>
cd benchmark-jetson/bonsai-models
```

---

## Running Benchmarks

All commands assume you are in `benchmark-jetson/bonsai-models/`.

### Full run (default: MAXN_SUPER)

```bash
bash benchmark_all_bonsai.sh --reqs 20
```

### Specific power mode

```bash
# 0 = 15W  |  1 = 25W  |  2 = MAXN_SUPER  |  3 = 7W
bash benchmark_all_bonsai.sh --power-mode 0 --reqs 20   # 15W
bash benchmark_all_bonsai.sh --power-mode 1 --reqs 20   # 25W
bash benchmark_all_bonsai.sh --power-mode 2 --reqs 20   # MAXN_SUPER
bash benchmark_all_bonsai.sh --power-mode 3 --reqs 20   # 7W
```

### Single model (quick test)

```bash
bash benchmark_all_bonsai.sh --only bonsai-1.7b --reqs 5
```

### Resume an interrupted run

Always resume inside tmux so the session survives terminal disconnects:

```bash
tmux new-session -d -s bonsai-bench && \
tmux send-keys -t bonsai-bench "cd ~/Desktop/benchmark-jetson/bonsai-models && \
bash benchmark_all_bonsai.sh --resume artifacts/bonsai-all-YYYYMMDD-HHMM --power-mode 3 --reqs 20" Enter && \
tmux attach -t bonsai-bench
```

Replace `bonsai-all-YYYYMMDD-HHMM` with the actual artifact folder name. The script detects already-completed combos and skips them automatically.

Detach: `Ctrl+B D` — reattach: `tmux attach -t bonsai-bench`

### Flags reference

| Flag | Default | Description |
|---|---|---|
| `--power-mode N` | `2` | nvpmodel mode (0=15W, 1=25W, 2=MAXN, 3=7W) |
| `--reqs N` | `10` | Requests per benchmark run |
| `--only <name>` | — | Run a single model (partial match) |
| `--resume <dir>` | — | Resume from existing artifact dir, skip completed combos |
| `--skip-download` | — | Skip HuggingFace model download check |
| `--skip-smoke` | — | Skip smoke test before each model |
| `--no-lock-clocks` | — | Skip `jetson_clocks` (allow DVFS scaling) |
| `--dry-run` | — | Print what would run, no benchmark |

> **Memory note:** On 8GB Orin, load the 8B model first after a fresh reboot. Physical memory fragments after ~30 min of activity, making the 1 GB contiguous allocation for 8B models fail. The script handles this by ordering 8B first.

---

## Output

Each run creates a timestamped directory under `artifacts/`:

```
artifacts/bonsai-all-YYYYMMDD-HHMM/
├── tegrastats.log               # raw power + thermal samples (500ms interval)
├── model_timing.log             # per-model start/end epochs for power windowing
├── <ModelName>-server.log       # llama-server stdout per model
└── <ModelName>/
    └── gen<G>/
        └── ctx<P>/
            ├── profile_export_aiperf.json
            └── profile_export_aiperf_timeslices.json
```

---

## Generating Reports

### Per-run report

```bash
python3 gen_report.py --artifact bonsai-all-20260527-0200 --label "25W"
# writes artifacts/bonsai-all-20260527-0200/report.md
```

### Combined charts (multi-run comparison)

Edit the `RUNS` dict in `gen_combined_charts.py` to include your artifact dirs, then:

```bash
source ~/venv/bin/activate
python3 gen_combined_charts.py
# writes artifacts/charts/*.png
```
