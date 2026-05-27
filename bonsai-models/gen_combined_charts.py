#!/usr/bin/env python3
"""Generate combined 15W + MAXN charts for the Bonsai benchmark report."""

import json, re
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

ARTIFACTS = Path.home() / "Desktop/benchmark-jetson/bonsai-models/artifacts"
OUT_DIR   = ARTIFACTS / "charts"
OUT_DIR.mkdir(exist_ok=True)

RUNS = {
    "15W":   "bonsai-all-20260525-0243",
    "MAXN":  "bonsai-all-20260526-0239",
}
MODEL_INFO = {
    "Bonsai-1.7B":         ("Q1_0", 2560),
    "Bonsai-4B":           ("Q1_0", 2560),
    "Bonsai-8B":           ("Q1_0", 1536),
    "Ternary-Bonsai-1.7B": ("Q2_0", 2560),
    "Ternary-Bonsai-4B":   ("Q2_0", 2560),
}
MODELS         = list(MODEL_INFO.keys())
PROMPT_LENGTHS = [256, 512, 1024, 2048]
GEN_LENGTHS    = [128, 256, 512]

# ── tegrastats ─────────────────────────────────────────────────────────────────
_tcache = {}
def get_tegra(art):
    if art not in _tcache:
        records = []
        p = ARTIFACTS / art / "tegrastats.log"
        if p.exists():
            with open(p) as f:
                for line in f:
                    m = re.match(r'(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})', line)
                    if not m: continue
                    ep = datetime.strptime(m.group(1), '%m-%d-%Y %H:%M:%S').timestamp()
                    pw = re.search(r'VDD_CPU_GPU_CV (\d+)mW/', line)
                    tj = re.search(r'tj@([\d.]+)C', line)
                    if pw and tj:
                        records.append((ep, int(pw.group(1)), float(tj.group(1))))
        _tcache[art] = records
    return _tcache[art]

# ── build combined dataframe ───────────────────────────────────────────────────
rows = []
for mode, art in RUNS.items():
    tegra = get_tegra(art)
    for model, (quant, max_ctx) in MODEL_INFO.items():
        for gen in GEN_LENGTHS:
            for ctx in PROMPT_LENGTHS:
                if ctx > max_ctx - gen: continue
                p = ARTIFACTS / art / model / f"gen{gen}" / f"ctx{ctx}" / "profile_export_aiperf.json"
                if not p.exists(): continue
                d = json.loads(p.read_text())
                def pct(k, v): return (d.get(k) or {}).get(v)
                tok_s   = pct("output_token_throughput",     "avg")
                ttft    = pct("time_to_first_token",         "avg")
                ttft_p99= pct("time_to_first_token",         "p99")
                ttft_p90= pct("time_to_first_token",         "p90")
                itl     = pct("inter_token_latency",         "avg")
                itl_p99 = pct("inter_token_latency",         "p99")
                prefill = pct("prefill_throughput_per_user", "avg")
                rl_avg  = pct("request_latency",             "avg")
                rl_p99  = pct("request_latency",             "p99")
                e2e     = pct("e2e_output_token_throughput", "avg")
                t0 = datetime.fromisoformat(d["start_time"]).timestamp()
                t1 = datetime.fromisoformat(d["end_time"]).timestamp()
                samps = [(mw, tj) for (ep, mw, tj) in tegra if t0 <= ep <= t1]
                if not samps: continue
                power_w = sum(mw for mw, _ in samps) / len(samps) / 1000
                tok_j   = tok_s / power_w if power_w else None
                rows.append(dict(
                    mode=mode, model=model, quant=quant, prompt=ctx, gen=gen,
                    tok_s=tok_s, ttft=ttft, ttft_p90=ttft_p90, ttft_p99=ttft_p99,
                    itl=itl, itl_p99=itl_p99,
                    prefill=prefill, rl_avg=rl_avg, rl_p99=rl_p99,
                    e2e=e2e, power_w=power_w, tok_j=tok_j,
                ))

df = pd.DataFrame(rows)

# ── style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", font_scale=1.1)
MODEL_PAL  = dict(zip(MODELS, sns.color_palette("tab10", len(MODELS))))
MODE_STYLE = {"15W": ("-", "o"), "MAXN": ("--", "s")}
MODE_PAL   = {"15W": "#4c72b0", "MAXN": "#dd8452"}
WIDTH = 0.35

def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")

def mode_legend(ax):
    handles = [mlines.Line2D([],[],color="grey",ls=ls,marker=mk,label=f"{m} mode")
               for m,(ls,mk) in MODE_STYLE.items()]
    return handles

def model_legend(ax):
    return [mlines.Line2D([],[],color=MODEL_PAL[m],ls="-",lw=3,label=m) for m in MODELS]

# ── 1. Tok/s vs Prompt — faceted by model, 15W vs MAXN ────────────────────────
for gen_val in [128, 256, 512]:
    fig, axes = plt.subplots(1, len(MODELS), figsize=(22, 4), sharey=False)
    fig.suptitle(f"Output Tok/s vs Prompt — 15W vs MAXN_SUPER (gen={gen_val})",
                 fontsize=13, fontweight="bold")
    for ax, model in zip(axes, MODELS):
        for mode, (ls, mk) in MODE_STYLE.items():
            sub = df[(df.model==model)&(df.gen==gen_val)&(df["mode"]==mode)].sort_values("prompt")
            ax.plot(sub.prompt, sub.tok_s, marker=mk, ls=ls, lw=2,
                    color=MODEL_PAL[model], alpha=0.95 if mode=="MAXN" else 0.55,
                    label=mode)
        ax.set_title(model, fontsize=9)
        ax.set_xlabel("Prompt (tok)")
        ax.set_ylabel("Tok/s" if ax is axes[0] else "")
        ax.set_xticks([256,512,1024,2048])
        ax.legend(fontsize=8)
    plt.tight_layout()
    save(fig, f"1_tok_s_vs_prompt_gen{gen_val}.png")

# ── 2. Tok/J vs Prompt — faceted by model, 15W vs MAXN ────────────────────────
fig, axes = plt.subplots(1, len(MODELS), figsize=(22, 4), sharey=False)
fig.suptitle("Energy Efficiency (Tok/J) vs Prompt — 15W vs MAXN_SUPER (gen=512)",
             fontsize=13, fontweight="bold")
for ax, model in zip(axes, MODELS):
    for mode, (ls, mk) in MODE_STYLE.items():
        sub = df[(df.model==model)&(df.gen==512)&(df["mode"]==mode)].sort_values("prompt")
        ax.plot(sub.prompt, sub.tok_j, marker=mk, ls=ls, lw=2,
                color=MODEL_PAL[model], alpha=0.95 if mode=="MAXN" else 0.55,
                label=mode)
    ax.set_title(model, fontsize=9)
    ax.set_xlabel("Prompt (tok)")
    ax.set_ylabel("Tok/J" if ax is axes[0] else "")
    ax.set_xticks([256,512,1024,2048])
    ax.legend(fontsize=8)
plt.tight_layout()
save(fig, "2_tok_j_vs_prompt.png")

# ── 3. Best Tok/J grouped bar — 15W vs MAXN ───────────────────────────────────
best = df.groupby(["model","mode"])["tok_j"].max().unstack("mode")
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(MODELS))
b1 = ax.bar(x-WIDTH/2, [best.loc[m,"15W"]  for m in MODELS], WIDTH,
            label="15W",  color=MODE_PAL["15W"],  edgecolor="white")
b2 = ax.bar(x+WIDTH/2, [best.loc[m,"MAXN"] for m in MODELS], WIDTH,
            label="MAXN", color=MODE_PAL["MAXN"], edgecolor="white")
for bars, md in [(b1,"15W"),(b2,"MAXN")]:
    for bar, m in zip(bars, MODELS):
        v = best.loc[m, md]
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold" if md=="MAXN" else "normal")
ax.set_xticks(x)
ax.set_xticklabels([m.replace("Ternary-Bonsai-","T-B-").replace("Bonsai-","B-") for m in MODELS],
                   rotation=12, ha="right")
ax.set_title("Best Tok/J per Model — 15W vs MAXN_SUPER (ctx=256, gen=512)", fontweight="bold")
ax.set_ylabel("Tok/J")
ax.legend(fontsize=9)
plt.tight_layout()
save(fig, "3_best_tok_j_bar.png")

# ── 4. Avg Power grouped bar ───────────────────────────────────────────────────
avg_pw = df.groupby(["model","mode"])["power_w"].mean().unstack("mode")
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(MODELS))
b1 = ax.bar(x-WIDTH/2, [avg_pw.loc[m,"15W"]  for m in MODELS], WIDTH,
            label="15W",  color=MODE_PAL["15W"],  edgecolor="white")
b2 = ax.bar(x+WIDTH/2, [avg_pw.loc[m,"MAXN"] for m in MODELS], WIDTH,
            label="MAXN", color=MODE_PAL["MAXN"], edgecolor="white")
for bars, md in [(b1,"15W"),(b2,"MAXN")]:
    for bar, m in zip(bars, MODELS):
        v = avg_pw.loc[m, md]
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                f"{v:.2f}W", ha="center", va="bottom", fontsize=9,
                fontweight="bold" if md=="MAXN" else "normal")
ax.set_xticks(x)
ax.set_xticklabels([m.replace("Ternary-Bonsai-","T-B-").replace("Bonsai-","B-") for m in MODELS],
                   rotation=12, ha="right")
ax.set_title("Average Power Draw — 15W vs MAXN_SUPER (VDD_CPU_GPU_CV)", fontweight="bold")
ax.set_ylabel("Power (W)")
ax.legend(fontsize=9)
plt.tight_layout()
save(fig, "4_avg_power_bar.png")

# ── 5. TTFT vs Prompt — all models, solid=15W dashed=MAXN, gen=128 ─────────────
fig, ax = plt.subplots(figsize=(10, 5))
for model in MODELS:
    for mode, (ls, mk) in MODE_STYLE.items():
        sub = df[(df.model==model)&(df.gen==128)&(df["mode"]==mode)].sort_values("prompt")
        lbl = f"{model}" if mode=="15W" else None
        ax.plot(sub.prompt, sub.ttft, marker=mk, ls=ls, lw=2,
                color=MODEL_PAL[model], alpha=0.95 if mode=="MAXN" else 0.6, label=lbl)
ax.set_title("TTFT (avg) vs Prompt — 15W (solid) vs MAXN (dashed), gen=128", fontweight="bold")
ax.set_xlabel("Prompt (tok)")
ax.set_ylabel("TTFT avg (ms)")
ax.set_xticks([256,512,1024,2048])
model_h = model_legend(ax)
mode_h  = mode_legend(ax)
ax.legend(handles=model_h+mode_h, fontsize=8, ncol=2)
plt.tight_layout()
save(fig, "5_ttft_vs_prompt.png")

# ── 6. Speedup ratio bar ───────────────────────────────────────────────────────
toks15 = df[df["mode"]=="15W"].groupby("model")["tok_s"].mean()
toksmx = df[df["mode"]=="MAXN"].groupby("model")["tok_s"].mean()
speedup = (toksmx / toks15).reindex(MODELS)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar([m.replace("Ternary-Bonsai-","T-B-").replace("Bonsai-","B-") for m in MODELS],
              speedup.values, color=MODE_PAL["MAXN"], edgecolor="white", linewidth=1.2)
for bar, val in zip(bars, speedup.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f"{val:.2f}×", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.axhline(1.0, color="grey", ls="--", lw=1, alpha=0.7)
ax.set_ylim(0, speedup.max()*1.2)
ax.set_ylabel("MAXN / 15W throughput ratio")
ax.set_title("Throughput Speedup — MAXN_SUPER vs 15W (avg over all combos)", fontweight="bold")
plt.tight_layout()
save(fig, "6_speedup_ratio.png")

# ── 7. TTFT spread (avg vs p99) — both modes, gen=128, per model ───────────────
fig, axes = plt.subplots(1, len(MODELS), figsize=(22, 4), sharey=False)
fig.suptitle("TTFT avg vs p99 spread — 15W (solid) vs MAXN (dashed), gen=128",
             fontsize=13, fontweight="bold")
for ax, model in zip(axes, MODELS):
    for mode, (ls, mk) in MODE_STYLE.items():
        sub = df[(df.model==model)&(df.gen==128)&(df["mode"]==mode)].sort_values("prompt")
        c = MODEL_PAL[model]
        a = 0.9 if mode=="MAXN" else 0.55
        ax.plot(sub.prompt, sub.ttft,     marker=mk, ls=ls, lw=2, color=c, alpha=a, label=f"avg {mode}")
        ax.plot(sub.prompt, sub.ttft_p99, marker=mk, ls=ls, lw=1, color=c, alpha=a*0.6)
        ax.fill_between(sub.prompt, sub.ttft, sub.ttft_p99, alpha=0.08, color=c)
    ax.set_title(model, fontsize=9)
    ax.set_xlabel("Prompt (tok)")
    ax.set_ylabel("TTFT (ms)" if ax is axes[0] else "")
    ax.set_xticks([256,512,1024,2048])
    ax.legend(fontsize=7)
plt.tight_layout()
save(fig, "7_ttft_spread.png")

# ── 8. Tok/J heatmap — 2 rows (15W / MAXN), one panel per model ───────────────
fig, axes = plt.subplots(2, len(MODELS), figsize=(22, 7))
for row, mode in enumerate(["15W","MAXN"]):
    for col, model in enumerate(MODELS):
        ax = axes[row][col]
        pivot = df[(df.model==model)&(df["mode"]==mode)].pivot_table(
            index="gen", columns="prompt", values="tok_j", aggfunc="mean")
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".3f", cmap="YlGnBu",
                    linewidths=0.5, cbar=False, vmin=df["tok_j"].min(), vmax=df["tok_j"].max())
        ax.set_title(f"{model}\n({mode})", fontsize=8)
        ax.set_xlabel("Prompt (tok)" if row==1 else "")
        ax.set_ylabel("Gen (tok)" if col==0 else "")
fig.suptitle("Tok/J Heatmap — 15W (top) vs MAXN_SUPER (bottom)", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "8_tok_j_heatmap.png")

# ── 9. ITL grouped bar — 15W vs MAXN (ctx=256, gen=256) ───────────────────────
sub15 = df[(df["mode"]=="15W")&(df.prompt==256)&(df.gen==256)].set_index("model")
submx = df[(df["mode"]=="MAXN")&(df.prompt==256)&(df.gen==256)].set_index("model")
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(MODELS))
b1 = ax.bar(x-WIDTH/2, [sub15.loc[m,"itl"]  for m in MODELS], WIDTH, label="15W",  color=MODE_PAL["15W"],  edgecolor="white")
b2 = ax.bar(x+WIDTH/2, [submx.loc[m,"itl"]  for m in MODELS], WIDTH, label="MAXN", color=MODE_PAL["MAXN"], edgecolor="white")
for bars, md, subdf in [(b1,"15W",sub15),(b2,"MAXN",submx)]:
    for bar, m in zip(bars, MODELS):
        v = subdf.loc[m,"itl"]
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([m.replace("Ternary-Bonsai-","T-B-").replace("Bonsai-","B-") for m in MODELS], rotation=12, ha="right")
ax.set_title("Inter-Token Latency — 15W vs MAXN (ctx=256, gen=256)", fontweight="bold")
ax.set_ylabel("ITL avg (ms)")
ax.legend(fontsize=9)
plt.tight_layout()
save(fig, "9_itl_compare.png")

# ── 10. Prefill tok/s — 15W vs MAXN (gen=128, avg over prompts) ───────────────
pf15 = df[(df["mode"]=="15W")&(df.gen==128)].groupby("model")["prefill"].mean()
pfmx = df[(df["mode"]=="MAXN")&(df.gen==128)].groupby("model")["prefill"].mean()
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(MODELS))
b1 = ax.bar(x-WIDTH/2, [pf15[m] for m in MODELS], WIDTH, label="15W",  color=MODE_PAL["15W"],  edgecolor="white")
b2 = ax.bar(x+WIDTH/2, [pfmx[m] for m in MODELS], WIDTH, label="MAXN", color=MODE_PAL["MAXN"], edgecolor="white")
for bars, ser in [(b1,pf15),(b2,pfmx)]:
    for bar, m in zip(bars, MODELS):
        v = ser[m]
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                f"{v:.0f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([m.replace("Ternary-Bonsai-","T-B-").replace("Bonsai-","B-") for m in MODELS], rotation=12, ha="right")
ax.set_title("Prefill Throughput — 15W vs MAXN_SUPER (gen=128, avg over prompts)", fontweight="bold")
ax.set_ylabel("Prefill tok/s")
ax.legend(fontsize=9)
plt.tight_layout()
save(fig, "10_prefill_compare.png")

# ── 11. Tok/s vs Power scatter — both modes, all combos ───────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
for model in MODELS:
    for mode, (ls, mk) in MODE_STYLE.items():
        sub = df[(df.model==model)&(df["mode"]==mode)]
        ax.scatter(sub.power_w, sub.tok_s, color=MODEL_PAL[model], marker=mk,
                   s=70, alpha=0.75)
    cx = df[df.model==model]["power_w"].mean()
    cy = df[df.model==model]["tok_s"].mean()
    ax.annotate(model.replace("Ternary-Bonsai-","T-B-").replace("Bonsai-","B-"),
                xy=(cx, cy), fontsize=8, ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc=MODEL_PAL[model], alpha=0.25))
mode_h = [mlines.Line2D([],[],color="grey",ls="",marker=mk,label=f"{m}",ms=8)
          for m,(ls,mk) in MODE_STYLE.items()]
model_h= [mlines.Line2D([],[],color=MODEL_PAL[m],ls="-",lw=3,label=m) for m in MODELS]
ax.legend(handles=model_h+mode_h, fontsize=8, ncol=2)
ax.set_title("Tok/s vs Power — All Combos, Both Modes (○=15W  □=MAXN)", fontweight="bold")
ax.set_xlabel("Power (W)")
ax.set_ylabel("Tok/s")
plt.tight_layout()
save(fig, "11_tok_s_vs_power_scatter.png")

# ── 12. Request latency p99 — 15W vs MAXN (ctx=256, gen=128) ──────────────────
sub15 = df[(df["mode"]=="15W")&(df.prompt==256)&(df.gen==128)].set_index("model")
submx = df[(df["mode"]=="MAXN")&(df.prompt==256)&(df.gen==128)].set_index("model")
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(MODELS))
b1 = ax.bar(x-WIDTH/2, [sub15.loc[m,"rl_p99"] for m in MODELS], WIDTH, label="15W",  color=MODE_PAL["15W"],  edgecolor="white")
b2 = ax.bar(x+WIDTH/2, [submx.loc[m,"rl_p99"] for m in MODELS], WIDTH, label="MAXN", color=MODE_PAL["MAXN"], edgecolor="white")
for bars, subdf in [(b1,sub15),(b2,submx)]:
    for bar, m in zip(bars, MODELS):
        v = subdf.loc[m,"rl_p99"]
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,
                f"{v:.0f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([m.replace("Ternary-Bonsai-","T-B-").replace("Bonsai-","B-") for m in MODELS], rotation=12, ha="right")
ax.set_title("Request Latency p99 — 15W vs MAXN (ctx=256, gen=128)", fontweight="bold")
ax.set_ylabel("Request Latency p99 (ms)")
ax.legend(fontsize=9)
plt.tight_layout()
save(fig, "12_request_latency_p99.png")

print(f"\nAll {12 + 2} charts saved to {OUT_DIR}")
