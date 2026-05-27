#!/usr/bin/env python3
"""Generate unified Bonsai benchmark report from aiperf + tegrastats data."""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

_p = argparse.ArgumentParser()
_p.add_argument("--artifact", default="bonsai-all-20260525-0243",
                help="Artifact dir name under ~/Desktop/benchmark-jetson/bonsai-models/artifacts/")
_p.add_argument("--output", default=None,
                help="Output .md path (default: artifacts/report.md)")
_p.add_argument("--label", default="",
                help="Optional label appended to report header (e.g. 'MAXN_SUPER')")
_args = _p.parse_args()

ARTIFACTS = Path.home() / "Desktop/benchmark-jetson/bonsai-models/artifacts"
OUTPUT = Path(_args.output) if _args.output else ARTIFACTS / "report.md"

# Map: display_name -> (artifact_dir, quant, max_ctx_size)
ART = _args.artifact
MODEL_INFO = {
    "Bonsai-1.7B":        (ART, "Q1_0",  2560),
    "Bonsai-4B":          (ART, "Q1_0",  2560),
    "Bonsai-8B":          (ART, "Q1_0",  1536),
    "Ternary-Bonsai-1.7B":(ART, "Q2_0",  2560),
    "Ternary-Bonsai-4B":  (ART, "Q2_0",  2560),
    "Ternary-Bonsai-8B":  (None,         "Q2_0",  1536),
}

PROMPT_LENGTHS = [256, 512, 1024, 2048]
GEN_LENGTHS    = [128, 256, 512]


def parse_tegrastats(log_path):
    """Return list of (epoch_float, power_mw, tj_c) tuples."""
    records = []
    with open(log_path) as f:
        for line in f:
            m = re.match(r'(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})', line)
            if not m:
                continue
            dt = datetime.strptime(m.group(1), '%m-%d-%Y %H:%M:%S')
            epoch = dt.timestamp()
            pw = re.search(r'VDD_CPU_GPU_CV (\d+)mW/', line)
            tj = re.search(r'tj@([\d.]+)C', line)
            if pw and tj:
                records.append((epoch, int(pw.group(1)), float(tj.group(1))))
    return records


def avg_power_in_window(tegra_records, t_start, t_end):
    """Return (avg_watts, avg_tj, n_samples) for tegrastats in [t_start, t_end]."""
    samples = [(mw, tj) for (ep, mw, tj) in tegra_records if t_start <= ep <= t_end]
    if not samples:
        return None, None, 0
    avg_w  = sum(mw for mw, _ in samples) / len(samples) / 1000.0
    avg_tj = sum(tj for _, tj in samples) / len(samples)
    return avg_w, avg_tj, len(samples)


def iso_to_epoch(s):
    """Parse ISO timestamp like 2026-05-24T21:14:44.976587 to epoch float."""
    return datetime.fromisoformat(s).timestamp()


def load_combo(model_dir, gen, ctx):
    """Load aiperf JSON for one (gen, ctx) combo. Returns dict or None."""
    p = model_dir / f"gen{gen}" / f"ctx{ctx}" / "profile_export_aiperf.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def metric_avg(d, key):
    v = d.get(key)
    if v and isinstance(v, dict):
        return v.get("avg")
    return v


# Pre-load tegrastats for each artifact dir (lazy cache)
_tegra_cache = {}


def get_tegra(artifact_dir_name):
    if artifact_dir_name not in _tegra_cache:
        p = ARTIFACTS / artifact_dir_name / "tegrastats.log"
        if p.exists():
            _tegra_cache[artifact_dir_name] = parse_tegrastats(p)
        else:
            _tegra_cache[artifact_dir_name] = []
    return _tegra_cache[artifact_dir_name]


# ── collect rows ──────────────────────────────────────────────────────────────

rows = []   # (model, quant, ctx, gen, ttft, itl, tok_s, power_w, tok_j, tj, note)
skipped_models = []

for model_name, (art_dir, quant, max_ctx) in MODEL_INFO.items():
    if art_dir is None:
        skipped_models.append((model_name, quant, "NvMap ENOMEM — CUDA alloc failed at model load"))
        for gen in GEN_LENGTHS:
            max_prompt = max_ctx - gen
            for ctx in PROMPT_LENGTHS:
                if ctx > max_prompt:
                    continue
                rows.append((model_name, quant, ctx, gen, *([None]*32), "OOM"))
        continue

    model_dir  = ARTIFACTS / art_dir / model_name
    tegra_recs = get_tegra(art_dir)

    for gen in GEN_LENGTHS:
        max_prompt = max_ctx - gen
        for ctx in PROMPT_LENGTHS:
            if ctx > max_prompt:
                # skip — model ctx window too small
                continue

            combo = load_combo(model_dir, gen, ctx)
            if combo is None:
                rows.append((model_name, quant, ctx, gen, *([None]*32), "OOM"))
                continue

            def pct(key, p):
                return combo.get(key, {}).get(p)

            ttft_avg  = pct("time_to_first_token",        "avg")
            ttft_p50  = pct("time_to_first_token",        "p50")
            ttft_p90  = pct("time_to_first_token",        "p90")
            ttft_p99  = pct("time_to_first_token",        "p99")

            t2t_avg   = pct("time_to_second_token",       "avg")
            t2t_p50   = pct("time_to_second_token",       "p50")
            t2t_p90   = pct("time_to_second_token",       "p90")
            t2t_p99   = pct("time_to_second_token",       "p99")

            itl_avg   = pct("inter_token_latency",        "avg")
            itl_p50   = pct("inter_token_latency",        "p50")
            itl_p90   = pct("inter_token_latency",        "p90")
            itl_p99   = pct("inter_token_latency",        "p99")

            tok_s     = pct("output_token_throughput",    "avg")   # avg only in JSON
            req_tput  = pct("request_throughput",         "avg")   # avg only in JSON

            e2e_avg   = pct("e2e_output_token_throughput","avg")
            e2e_p50   = pct("e2e_output_token_throughput","p50")
            e2e_p90   = pct("e2e_output_token_throughput","p90")
            e2e_p99   = pct("e2e_output_token_throughput","p99")

            req_lat_avg = pct("request_latency",          "avg")
            req_lat_p50 = pct("request_latency",          "p50")
            req_lat_p90 = pct("request_latency",          "p90")
            req_lat_p99 = pct("request_latency",          "p99")

            pre_avg   = pct("prefill_throughput_per_user","avg")
            pre_p50   = pct("prefill_throughput_per_user","p50")
            pre_p90   = pct("prefill_throughput_per_user","p90")
            pre_p99   = pct("prefill_throughput_per_user","p99")

            isl       = pct("input_sequence_length",      "avg")
            osl       = pct("output_sequence_length",     "avg")
            osl_mm    = pct("osl_mismatch_diff_pct",      "avg")

            t_start = iso_to_epoch(combo["start_time"])
            t_end   = iso_to_epoch(combo["end_time"])
            power_w, avg_tj, n = avg_power_in_window(tegra_recs, t_start, t_end)
            tok_j = tok_s / power_w if (tok_s and power_w) else None

            rows.append((model_name, quant, ctx, gen,
                         isl, osl, osl_mm,
                         ttft_avg, ttft_p50, ttft_p90, ttft_p99,
                         t2t_avg,  t2t_p50,  t2t_p90,  t2t_p99,
                         itl_avg,  itl_p50,  itl_p90,  itl_p99,
                         tok_s, req_tput,
                         e2e_avg,  e2e_p50,  e2e_p90,  e2e_p99,
                         req_lat_avg, req_lat_p50, req_lat_p90, req_lat_p99,
                         pre_avg,  pre_p50,  pre_p90,  pre_p99,
                         power_w, tok_j, avg_tj, "OK"))


# ── summary stats ─────────────────────────────────────────────────────────────

best_tok_j = {}   # model_name -> best row
# row layout: (model,quant,ctx,gen, isl,osl,osl_mm,
#   ttft_avg,p50,p90,p99, t2t_avg,p50,p90,p99, itl_avg,p50,p90,p99,
#   tok_s,req_tput, e2e_avg,p50,p90,p99, req_lat_avg,p50,p90,p99,
#   pre_avg,p50,p90,p99, power_w,tok_j,avg_tj, note)
TOK_J_IDX = 34    # index of tok_j  (0-3: model/quant/ctx/gen, 4-32: 29 metrics, 33: power_w, 34: tok_j)
for row in rows:
    note = row[-1]
    tok_j = row[TOK_J_IDX]
    if note != "OK" or tok_j is None:
        continue
    m = row[0]
    if m not in best_tok_j or tok_j > best_tok_j[m][TOK_J_IDX]:
        best_tok_j[m] = row

model_power = {}  # model_name -> (avg_power, peak_tj)
for m_name, (art_dir, quant, max_ctx) in MODEL_INFO.items():
    if art_dir is None:
        continue
    tegra_recs = get_tegra(art_dir)
    # timing window for this model
    timing_log = ARTIFACTS / art_dir / "model_timing.log"
    t_start = t_end = None
    if timing_log.exists():
        with open(timing_log) as f:
            for line in f:
                if f"MODEL_START:{m_name}:" in line:
                    t_start = float(line.strip().split(":")[-1])
                elif f"MODEL_END:{m_name}:" in line:
                    t_end = float(line.strip().split(":")[-1])
    if t_start and t_end:
        samps = [(mw, tj) for (ep, mw, tj) in tegra_recs if t_start <= ep <= t_end]
        if samps:
            avg_pw = sum(mw for mw, _ in samps) / len(samps) / 1000.0
            peak_tj = max(tj for _, tj in samps)
            model_power[m_name] = (avg_pw, peak_tj)


# ── write report.md ───────────────────────────────────────────────────────────

def fmt(v, decimals=1, suffix=""):
    if v is None:
        return "—"
    return f"{v:.{decimals}f}{suffix}"


now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

lines = []
_title_suffix = f" — {_args.label}" if _args.label else ""
lines.append(f"# Bonsai All-Model Benchmark — Jetson Orin Nano Super 8GB{_title_suffix}")
lines.append(f"")
lines.append(f"**Date:** {now_str}  ")
lines.append(f"**Backend:** llama.cpp (build-jetson) / CUDA / `-ngl 99`  ")
lines.append(f"**Platform:** NVIDIA Jetson Orin Nano Super 8GB (6-core Cortex-A78AE + Ampere GPU)  ")
lines.append(f"**Sweep:** prompt ∈ {{256, 512, 1024, 2048}} tok × gen ∈ {{128, 256, 512}} tok × 20 reqs/combo  ")
lines.append(f"**Key metric:** tok/J = output tokens per second ÷ VDD_CPU_GPU_CV power (watts)  ")
lines.append(f"")

# Skipped models note
if skipped_models:
    lines.append(f"## Skipped / OOM Models")
    lines.append(f"")
    for sn, sq, reason in skipped_models:
        lines.append(f"- **{sn}** ({sq}): {reason}")
    lines.append(f"")

lines.append(f"## Full Results Table")
lines.append(f"")
lines.append(f"Cells marked `—` = OOM (server crashed or skipped due to context window limit).")
lines.append(f"Power = VDD_CPU_GPU_CV average over aiperf run window (CPU+GPU+CV rail, milliwatts → watts).")
lines.append(f"")
lines.append(
    "| Model | Quant | ISL (tok) | OSL (tok) | OSL mismatch (%) |"
    " TTFT avg (ms) | TTFT p50 | TTFT p90 | TTFT p99 |"
    " T2T avg (ms) | T2T p50 | T2T p90 | T2T p99 |"
    " ITL avg (ms) | ITL p50 | ITL p90 | ITL p99 |"
    " Tok/s (server) | Req/s |"
    " E2E tok/s avg | E2E p50 | E2E p90 | E2E p99 |"
    " Req lat avg (ms) | Req lat p50 | Req lat p90 | Req lat p99 |"
    " Prefill tok/s avg | Prefill p50 | Prefill p90 | Prefill p99 |"
    " Power (W) | **Tok/J** |"
)
_sep = "|-------|-------|:---:|:---:|---:|" + ("---:|" * 28)
lines.append(_sep)

for row in rows:
    (m, q, ctx, gen,
     isl, osl, osl_mm,
     ttft_avg, ttft_p50, ttft_p90, ttft_p99,
     t2t_avg,  t2t_p50,  t2t_p90,  t2t_p99,
     itl_avg,  itl_p50,  itl_p90,  itl_p99,
     tok_s, req_tput,
     e2e_avg,  e2e_p50,  e2e_p90,  e2e_p99,
     rl_avg,   rl_p50,   rl_p90,   rl_p99,
     pre_avg,  pre_p50,  pre_p90,  pre_p99,
     pw, tok_j, tj, note) = row
    if note == "OK":
        lines.append(
            f"| {m} | {q} | {fmt(isl,1)} | {fmt(osl,1)} | {fmt(osl_mm,2)} |"
            f" {fmt(ttft_avg,1)} | {fmt(ttft_p50,1)} | {fmt(ttft_p90,1)} | {fmt(ttft_p99,1)} |"
            f" {fmt(t2t_avg,1)} | {fmt(t2t_p50,1)} | {fmt(t2t_p90,1)} | {fmt(t2t_p99,1)} |"
            f" {fmt(itl_avg,2)} | {fmt(itl_p50,2)} | {fmt(itl_p90,2)} | {fmt(itl_p99,2)} |"
            f" {fmt(tok_s,2)} | {fmt(req_tput,3)} |"
            f" {fmt(e2e_avg,2)} | {fmt(e2e_p50,2)} | {fmt(e2e_p90,2)} | {fmt(e2e_p99,2)} |"
            f" {fmt(rl_avg,1)} | {fmt(rl_p50,1)} | {fmt(rl_p90,1)} | {fmt(rl_p99,1)} |"
            f" {fmt(pre_avg,1)} | {fmt(pre_p50,1)} | {fmt(pre_p90,1)} | {fmt(pre_p99,1)} |"
            f" {fmt(pw,2)} | **{fmt(tok_j,3)}** |"
        )
    else:
        dash = " — |" * 30
        lines.append(f"| {m} | {q} | {ctx} | {gen} |{dash}")

lines.append(f"")
lines.append(f"## Per-Model Best Tok/J (optimal configuration)")
lines.append(f"")
lines.append("| Model | Quant | Best Tok/J | at Prompt | at Gen | Tok/s | Power (W) |")
lines.append("|-------|-------|---:|:---:|:---:|---:|---:|")

for m_name in MODEL_INFO:
    quant = MODEL_INFO[m_name][1]
    if m_name in best_tok_j:
        r = best_tok_j[m_name]
        bctx, bgen, btok_s, bpw, btok_j = r[2], r[3], r[19], r[33], r[34]
        lines.append(f"| {m_name} | {quant} | **{fmt(btok_j,3)}** | {bctx} | {bgen} | {fmt(btok_s,2)} | {fmt(bpw,2)} |")
    else:
        lines.append(f"| {m_name} | {quant} | — | — | — | — | — |")

lines.append(f"")
lines.append(f"## Thermal & Power Summary (per-model window)")
lines.append(f"")
lines.append("| Model | Quant | Avg Power (W) | Peak TJ (°C) | Status |")
lines.append("|-------|-------|---:|---:|:---:|")

for m_name in MODEL_INFO:
    quant = MODEL_INFO[m_name][1]
    art_dir = MODEL_INFO[m_name][0]
    if art_dir is None:
        lines.append(f"| {m_name} | {quant} | — | — | OOM / Skipped |")
    elif m_name in model_power:
        avg_pw, peak_tj = model_power[m_name]
        lines.append(f"| {m_name} | {quant} | {fmt(avg_pw,2)} | {fmt(peak_tj,1)} | OK |")
    else:
        lines.append(f"| {m_name} | {quant} | — | — | Partial / No timing |")

lines.append(f"")
lines.append(f"## Notes")
lines.append(f"")
lines.append(f"- **Bonsai-8B**: 9/9 valid combos complete. ctx=2048 excluded by design (ctx cap 1536 = 1024 prompt + 512 gen max).")
lines.append(f"- **Ternary-Bonsai-8B**: Skipped — JetPack R36.4.7 NvMap regression bug. "
             f"cudaMalloc fails for contiguous allocations ≥1.9 GB even on fresh boot. "
             f"Same bug blocks all large Q4 models (Gemma 3 4B, Llama 3.1 8B) on this JetPack version. "
             f"Fix: upgrade to JetPack 6.2.2 (L4T 36.5) via clean flash.")
lines.append(f"- **Context window**: `--no-cache-prompt -c 2560` for 1.7B/4B models, `-c 1536` for 8B.")
lines.append(f"- **Power measurement**: `VDD_CPU_GPU_CV` rail (CPU + GPU + CV) from `tegrastats`, "
             f"1 Hz sampling, averaged over each aiperf run window.")
lines.append(f"- **Concurrency**: 1 (single-user; tok/J reflects single-inference energy cost).")

with open(OUTPUT, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"Report written to {OUTPUT}")
print(f"Total rows: {len(rows)}")
ok_rows = [r for r in rows if r[-1] == "OK"]
oom_rows = [r for r in rows if r[-1] == "OOM"]
print(f"  OK:  {len(ok_rows)}")
print(f"  OOM: {len(oom_rows)}")
