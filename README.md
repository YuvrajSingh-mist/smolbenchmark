# smolbenchmark — Tiny LLM Benchmarks on Jetson Orin Nano Super 8GB

End-to-end throughput, latency, and energy-efficiency benchmarks for small open-weight LLMs on a **$250 NVIDIA Jetson Orin Nano Super 8GB** edge device.

All benchmarks use [aiperf](https://github.com/NVIDIA/aiperf) for load generation and `tegrastats` for power telemetry. Key metric throughout: **output tok/J** (output tokens per joule of GPU+CPU rail energy).

---

## Repository layout

```
smolbenchmark/
├── non-reasoning-models/   # Tiny instruct LLMs (llama.cpp + Ollama backends)
│   ├── bench-non-reasoning.sh      # main benchmark script
│   ├── generate_combined_charts.py # chart generation
│   ├── generate_report.py          # per-run report
│   ├── benchmark_report.md         # published benchmark report
│   └── artifacts/charts/           # generated PNG charts
│
└── bonsai-models/          # Bonsai / Ternary-Bonsai model family
    ├── benchmark_all_bonsai.sh
    └── ...
```

---

## Platform

| Component | Detail |
|-----------|--------|
| Board | Jetson Orin Nano Super 8GB Developer Kit |
| CPU | 6× Arm Cortex-A78AE @ up to 1.728 GHz |
| GPU | NVIDIA Ampere, 1024 CUDA cores, 32 Tensor cores |
| Memory | 8 GB LPDDR5 unified CPU + GPU |
| JetPack | R36.4.7 (Ubuntu 22.04, CUDA 12.6) |

**Power modes** (set via `nvpmodel`):

| Mode | ID | GPU clock | Power draw |
|------|----|-----------|------------|
| 7W   | 3  | ~408 MHz  | ~2 W under load |
| 15W  | 0  | ~612 MHz  | ~5 W under load |
| 25W  | 1  | ~820 MHz  | ~7 W under load |
| MAXN | 2  | 1020 MHz  | ~10 W under load |

---

## Benchmarks

### [non-reasoning-models](./non-reasoning-models/)

Eight tiny instruct LLMs (135M–1.2B params, Q4\_K\_M / Q8\_0) benchmarked across all four power modes with two backends:

- **llama.cpp** CUDA (`-ngl 99`, all layers on GPU)
- **Ollama** (same GGUF, apples-to-apples comparison)

Sweep: 4 prompt lengths × 3 gen lengths × 20 requests = 240 measurements per model.

→ See [non-reasoning-models/README.MD](./non-reasoning-models/README.MD)

### [bonsai-models](./bonsai-models/)

Bonsai and Ternary-Bonsai model families (1.7B / 4B / 8B) with extreme quantization (Q1\_0 / Q2\_0).

→ See [bonsai-models/README.md](./bonsai-models/README.md)

---

## Raw data

All per-cell JSON exports are on Hugging Face:

| Dataset | Mode |
|---------|------|
| [YuvrajSingh9886/jetson-non-reasoning-benchmark-7w](https://huggingface.co/datasets/YuvrajSingh9886/jetson-non-reasoning-benchmark-7w) | 7W |
| [YuvrajSingh9886/jetson-non-reasoning-benchmark-15w](https://huggingface.co/datasets/YuvrajSingh9886/jetson-non-reasoning-benchmark-15w) | 15W |
| [YuvrajSingh9886/jetson-non-reasoning-benchmark-25w](https://huggingface.co/datasets/YuvrajSingh9886/jetson-non-reasoning-benchmark-25w) | 25W |
| [YuvrajSingh9886/jetson-non-reasoning-benchmark-maxn](https://huggingface.co/datasets/YuvrajSingh9886/jetson-non-reasoning-benchmark-maxn) | MAXN |
