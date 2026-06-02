#!/bin/bash
# bench_all.sh — Blog benchmark: tiny LLMs on Jetson Orin Nano Super 8GB
#
# Per model: ONE server launch → smoke (32/256/512 tok) → if PASS, aiperf sweep.
# Sweeps: prompt in {128,512,1024,2048} x gen in {64,128,256}
# Key metric: output tok/J (output tokens per joule), from aiperf + tegrastats.
#
# Usage:
#   bash bench_all.sh                   # full run, all models
#   bash bench_all.sh --reqs 5          # 5 requests per run
#   bash bench_all.sh --only smollm2    # single model (substring match)
#   bash bench_all.sh --skip-smoke      # skip smoke, bench directly
#   bash bench_all.sh --dry-run         # check files, no bench
#   bash bench_all.sh --resume DIR      # resume a previous run

set -e

# ── Config ────────────────────────────────────────────────────────────────────
REQS=20
ONLY_MODEL=""
SKIP_SMOKE=0
DRY_RUN=0
RESUME_DIR=""
POWER_MODE=0          # nvpmodel ID: 0=15W  1=25W  2=MAXN_SUPER  3=7W
POWER_MODE_NAME="15w" # used in artifact dir suffix
CONCURRENCY=1
SLICE_DURATION=30
RANDOM_SEED=42
REQUEST_TIMEOUT=180
COOLDOWN_COMBO=10
COOLDOWN_MODEL=30
SERVER_STARTUP_TIMEOUT=300

PROMPT_LENGTHS=(128 512 1024 2048)
GEN_LENGTHS=(64 128 256)
# ctx_size: max_prompt(2048) + max_gen(256) = 2304, padded to 2560
CONTEXT_SIZE=2560

SERVER_BIN="$HOME/llama.cpp/build/bin/llama-server"

GGUF_DIR="$HOME/gguf-models"
TEGRA_PIDFILE="/tmp/blog_bench_tegrastats.pid"
SERVER_PIDFILE="/tmp/blog_bench_server.pid"
REPORT_PY="/tmp/blog_report.py"

# ── Model table: name|quant|gguf_path|tokenizer|ctx_size ─────────────────────
declare -a MODELS=(
    "smollm2-135m|Q4_K_M|$GGUF_DIR/smollm2-135mq4_k_m.gguf|HuggingFaceTB/SmolLM2-135M-Instruct|2560"
    "smollm2-360m|Q8_0|$GGUF_DIR/smollm2-360mq8_0|HuggingFaceTB/SmolLM2-360M-Instruct|2560"
    "qwen2.5-0.5b|Q4_K_M|$GGUF_DIR/qwen2-5-0-5bq4_k_m|Qwen/Qwen2.5-0.5B-Instruct|2560"
    "qwen3-0.6b|Q8_0|$GGUF_DIR/qwen3-0-6bq8_0|Qwen/Qwen3-0.6B|2560"
    "llama3.2-1b|Q4_K_M|$GGUF_DIR/llama3-2-1bq4_k_m.gguf|meta-llama/Llama-3.2-1B-Instruct|2560"
    "gemma3-1b|Q4_K_M|$GGUF_DIR/gemma3-1b-q4_k_m.gguf|google/gemma-3-1b-it|2560"
    "gemma3-4b|Q4_K_M|$GGUF_DIR/gemma3-4b-q4_k_m.gguf|google/gemma-3-4b-it|2560"
    "lfm2.5-350m|Q4_K_M|$GGUF_DIR/lfm2.5-350m-q4_k_m.gguf|LiquidAI/LFM2.5-350M|2560"
    "lfm2.5-1.2b|Q4_K_M|$GGUF_DIR/lfm2.5-1.2b-q4_k_m.gguf|LiquidAI/LFM2.5-1.2B-Instruct|2560"
)

# ── GGUF download sources: local_filename -> "hf_repo hf_filename" ────────────
declare -A GGUF_SOURCES=(
    ["smollm2-135mq4_k_m.gguf"]="bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q4_K_M.gguf"
    ["smollm2-360mq8_0"]="bartowski/SmolLM2-360M-Instruct-GGUF SmolLM2-360M-Instruct-Q8_0.gguf"
    ["qwen2-5-0-5bq4_k_m"]="Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf"
    ["qwen3-0-6bq8_0"]="Qwen/Qwen3-0.6B-GGUF qwen3-0.6b-q8_0.gguf"
    ["llama3-2-1bq4_k_m.gguf"]="bartowski/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    ["gemma3-1b-q4_k_m.gguf"]="lmstudio-community/gemma-3-1b-it-GGUF gemma-3-1b-it-Q4_K_M.gguf"
    ["gemma3-4b-q4_k_m.gguf"]="lmstudio-community/gemma-3-4b-it-GGUF gemma-3-4b-it-Q4_K_M.gguf"
    ["lfm2.5-350m-q4_k_m.gguf"]="LiquidAI/LFM2.5-350M-GGUF LFM2.5-350M-Q4_K_M.gguf"
    ["lfm2.5-1.2b-q4_k_m.gguf"]="LiquidAI/LFM2.5-1.2B-Instruct-GGUF LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
)

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --reqs)        REQS="$2";         shift 2 ;;
        --only)        ONLY_MODEL="$2";   shift 2 ;;
        --skip-smoke)  SKIP_SMOKE=1;      shift ;;
        --dry-run)     DRY_RUN=1;         shift ;;
        --resume)      RESUME_DIR="$2";   shift 2 ;;
        --maxn)        POWER_MODE=2; POWER_MODE_NAME="maxn"; shift ;;
        --power-mode)  POWER_MODE="$2"
                       case "$POWER_MODE" in
                           0) POWER_MODE_NAME="15w"   ;;
                           1) POWER_MODE_NAME="25w"   ;;
                           2) POWER_MODE_NAME="maxn"  ;;
                           3) POWER_MODE_NAME="7w"    ;;
                           *) POWER_MODE_NAME="pwr${POWER_MODE}" ;;
                       esac
                       shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Resolve artifact dir ──────────────────────────────────────────────────────
if [ -n "$RESUME_DIR" ]; then
    [ ! -d "$RESUME_DIR" ] && echo "ERROR: --resume dir not found: $RESUME_DIR" && exit 1
    BASE_ARTIFACT="$RESUME_DIR"
    echo "  [RESUME] Reusing artifact dir: $BASE_ARTIFACT"
else
    BASE_ARTIFACT="$HOME/Desktop/benchmark/smolbenchmark/non-reasoning-models/artifacts/blog-all-$(date +%Y%m%d-%H%M)-${POWER_MODE_NAME}"
fi

TEGRA_LOG="$BASE_ARTIFACT/tegrastats.log"
TIMING_LOG="$BASE_ARTIFACT/model_timing.log"

# ── Helpers ───────────────────────────────────────────────────────────────────
banner() { echo ""; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; echo "  $*"; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }
log()    { echo "  [$(date +%H:%M:%S)] $*"; }

# ── Initial cleanup ───────────────────────────────────────────────────────────
banner "Cleanup: killing stale processes before start"
pkill -f "llama-server.*8080" 2>/dev/null && log "killed llama-server" || log "no llama-server found"
sudo pkill -f "tegrastats" 2>/dev/null && log "killed tegrastats" || log "no tegrastats found"
rm -f "$TEGRA_PIDFILE" "$SERVER_PIDFILE"
sleep 2
echo 1 | sudo tee /proc/sys/vm/compact_memory > /dev/null 2>&1 || true
CMA_FREE_KB=$(awk '/CmaFree/{print $2}' /proc/meminfo)
log "CMA free: $(( CMA_FREE_KB / 1024 )) MiB / $(( $(awk '/CmaTotal/{print $2}' /proc/meminfo) / 1024 )) MiB total"

kill_server() {
    if [ -f "$SERVER_PIDFILE" ]; then
        kill "$(cat $SERVER_PIDFILE)" 2>/dev/null || true
        rm -f "$SERVER_PIDFILE"
    fi
    pkill -f "llama-server.*8080" 2>/dev/null || true
    # Jetson unified memory is not released immediately on process exit — drain it
    sleep 12
    echo 1 | sudo tee /proc/sys/vm/compact_memory > /dev/null 2>&1 || true
    sleep 2
}

ensure_server_alive() {
    local model_path="$1" ctx_size="$2" srv_log="$3"
    local code
    code=$(curl -s http://localhost:8080/v1/models --max-time 3 -o /dev/null -w "%{http_code}" 2>/dev/null; true)
    [ "$code" = "200" ] && return 0

    log "  [!] Server not responding (HTTP $code) — restarting..."
    kill_server
    echo 1 | sudo tee /proc/sys/vm/compact_memory > /dev/null 2>&1 || true
    sleep 3
    "$SERVER_BIN" \
        -m "$model_path" \
        --host 0.0.0.0 --port 8080 \
        -ngl 99 --parallel 1 -c "$ctx_size" \
        --no-cache-prompt --cache-ram 0 \
        >> "$srv_log" 2>&1 &
    echo $! > "$SERVER_PIDFILE"
    log "  [RESTART] PID $(cat $SERVER_PIDFILE) — waiting for HTTP 200..."
    local elapsed=0
    while [ "$elapsed" -lt "$SERVER_STARTUP_TIMEOUT" ]; do
        sleep 2; elapsed=$((elapsed + 2))
        if ! kill -0 "$(cat $SERVER_PIDFILE 2>/dev/null)" 2>/dev/null; then
            log "  [RESTART FAIL] Server died during restart"; return 1
        fi
        code=$(curl -s http://localhost:8080/v1/models --max-time 3 -o /dev/null -w "%{http_code}" 2>/dev/null; true)
        [ "$code" = "200" ] && log "  [RESTART OK] t=${elapsed}s" && return 0
    done
    log "  [RESTART FAIL] Timed out after ${SERVER_STARTUP_TIMEOUT}s"; return 1
}

stop_tegrastats() {
    [ -f "$TEGRA_PIDFILE" ] && sudo kill "$(cat $TEGRA_PIDFILE)" 2>/dev/null || true
    rm -f "$TEGRA_PIDFILE"
    sudo pkill -f "tegrastats" 2>/dev/null || true
}

# ── Parse model table ─────────────────────────────────────────────────────────
declare -a MODEL_NAMES MODEL_QUANTS MODEL_PATHS MODEL_TOKENIZERS MODEL_CTX_SIZES

for entry in "${MODELS[@]}"; do
    IFS='|' read -r n q p t c <<< "$entry"
    MODEL_NAMES+=("$n")
    MODEL_QUANTS+=("$q")
    MODEL_PATHS+=("$p")
    MODEL_TOKENIZERS+=("$t")
    MODEL_CTX_SIZES+=("${c:-$CONTEXT_SIZE}")
done

banner "Blog LLM Benchmark  |  Jetson Orin Nano Super 8GB"
echo "  Date      : $(date --iso-8601=seconds)"
echo "  Requests  : $REQS per run"
echo "  Prompts   : ${PROMPT_LENGTHS[*]}"
echo "  Gen lens  : ${GEN_LENGTHS[*]}"
echo "  Artifacts : $BASE_ARTIFACT"
echo "  Server    : $SERVER_BIN"
echo "  Power     : mode $POWER_MODE ($POWER_MODE_NAME)"
[ -n "$ONLY_MODEL" ] && echo "  Filter    : $ONLY_MODEL"
[ "$DRY_RUN" = 1 ]   && echo "  Mode      : DRY RUN"
[ -n "$RESUME_DIR" ]  && echo "  Mode      : RESUME"

mkdir -p "$BASE_ARTIFACT"

# ── Auto-download missing GGUFs ──────────────────────────────────────────────
banner "Pre-flight: checking / downloading missing GGUFs"
mkdir -p "$GGUF_DIR"
for local_name in "${!GGUF_SOURCES[@]}"; do
    local_path="$GGUF_DIR/$local_name"
    if [ -f "$local_path" ]; then
        log "  [OK]   $local_name"
        continue
    fi
    read -r hf_repo hf_file <<< "${GGUF_SOURCES[$local_name]}"
    log "  [DL]   $local_name  ←  $hf_repo / $hf_file"
    tmp_path="$GGUF_DIR/${hf_file}"
    if hf download "$hf_repo" "$hf_file" --local-dir "$GGUF_DIR" 2>&1 | tail -3; then
        [ -f "$tmp_path" ] && mv "$tmp_path" "$local_path" && log "  [DONE] $local_name"
    else
        log "  [FAIL] Could not download $local_name — model will be skipped"
    fi
done

# ── Lock clocks + power mode ──────────────────────────────────────────────────
if [ "$DRY_RUN" = 0 ]; then
    banner "Power mode + clock lock  ($POWER_MODE_NAME / nvpmodel -m $POWER_MODE)"
    sudo nvpmodel -m "$POWER_MODE" 2>/dev/null \
        && log "nvpmodel -m $POWER_MODE ($POWER_MODE_NAME) OK" \
        || log "nvpmodel not available"
    sudo jetson_clocks        2>/dev/null && log "jetson_clocks OK"       || log "jetson_clocks not available"
    sudo jetson_clocks --fan  2>/dev/null && log "jetson_clocks --fan OK" || log "fan control not available"
    # Confirm active mode
    ACTIVE_MODE=$(sudo nvpmodel -q 2>/dev/null | head -1 || echo "unknown")
    log "Active power mode: $ACTIVE_MODE"
fi

# ── Activate aiperf venv ──────────────────────────────────────────────────────
source "$HOME/venv/bin/activate" 2>/dev/null || \
source "$HOME/aiperf-env/bin/activate" 2>/dev/null || \
{ echo "ERROR: no aiperf venv found (~venv or ~aiperf-env)"; exit 1; }

# ── Start tegrastats ──────────────────────────────────────────────────────────
if [ "$DRY_RUN" = 0 ]; then
    banner "Start tegrastats"
    stop_tegrastats
    sudo tegrastats --interval 500 --logfile "$TEGRA_LOG" &
    echo $! > "$TEGRA_PIDFILE"
    log "tegrastats PID $(cat $TEGRA_PIDFILE)"
fi

# ── Model loop — smoke then benchmark on the same server ─────────────────────
banner "Model loop (smoke + benchmark, one server launch per model)"

declare -a SKIPPED_MODELS SMOKE_PASSED_MODELS BENCH_MODELS

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    MODEL_QUANT="${MODEL_QUANTS[$i]}"
    MODEL_PATH="${MODEL_PATHS[$i]}"
    MODEL_TOKENIZER="${MODEL_TOKENIZERS[$i]}"
    MODEL_CTX_SIZE="${MODEL_CTX_SIZES[$i]}"
    MAX_PROMPT=$(( MODEL_CTX_SIZE - ${GEN_LENGTHS[-1]} ))

    if [ -n "$ONLY_MODEL" ] && [[ "${MODEL_NAME,,}" != *"${ONLY_MODEL,,}"* ]]; then
        continue
    fi

    if [ "$DRY_RUN" = 1 ]; then
        log "[DRY RUN] $MODEL_NAME  path=$([ -f "$MODEL_PATH" ] && echo OK || echo MISSING)"
        continue
    fi

    # Resume: skip if all combos already done
    MISSING_COMBOS=0
    for G in "${GEN_LENGTHS[@]}"; do
        for P in "${PROMPT_LENGTHS[@]}"; do
            [ ! -f "$BASE_ARTIFACT/${MODEL_NAME}/gen${G}/ctx${P}/profile_export_aiperf.json" ] && \
                MISSING_COMBOS=$((MISSING_COMBOS + 1))
        done
    done
    if [ "$MISSING_COMBOS" = 0 ]; then
        log "  [RESUME SKIP] $MODEL_NAME — all combos already complete"
        BENCH_MODELS+=("$i")
        continue
    fi

    echo ""
    echo "  ┌─────────────────────────────────────────────────────┐"
    printf "  │  Model : %-43s│\n" "$MODEL_NAME ($MODEL_QUANT)"
    printf "  │  Tok   : %-43s│\n" "$MODEL_TOKENIZER"
    printf "  │  Ctx   : %-43s│\n" "-c $MODEL_CTX_SIZE  (max prompt: $MAX_PROMPT tok)"
    CMA_FREE=$(awk '/CmaFree/{print $2}' /proc/meminfo 2>/dev/null || echo 0)
    printf "  │  CMA   : %-43s│\n" "$(( CMA_FREE / 1024 )) MiB free"
    echo "  └─────────────────────────────────────────────────────┘"

    if [ ! -f "$MODEL_PATH" ]; then
        log "  [SKIP] GGUF not found: $MODEL_PATH"
        SKIPPED_MODELS+=("$MODEL_NAME (file not found)")
        continue
    fi

    # ── Start server ──────────────────────────────────────────────────────────
    SERVER_LOG="$BASE_ARTIFACT/${MODEL_NAME}-server.log"
    log "Launching llama-server..."
    "$SERVER_BIN" \
        -m "$MODEL_PATH" \
        --host 0.0.0.0 --port 8080 \
        -ngl 99 --parallel 1 -c "$MODEL_CTX_SIZE" \
        --no-cache-prompt --cache-ram 0 \
        > "$SERVER_LOG" 2>&1 &
    echo $! > "$SERVER_PIDFILE"
    log "  PID: $(cat $SERVER_PIDFILE)  log: $SERVER_LOG"

    # ── Wait for HTTP 200 ─────────────────────────────────────────────────────
    log "Waiting for HTTP 200 on /v1/models..."
    READY=0; ELAPSED=0
    while [ "$ELAPSED" -lt "$SERVER_STARTUP_TIMEOUT" ]; do
        sleep 2; ELAPSED=$((ELAPSED + 2))
        if ! kill -0 "$(cat $SERVER_PIDFILE 2>/dev/null)" 2>/dev/null; then
            log "  [!] Server died at t=${ELAPSED}s (OOM?)"
            grep -E "error|OOM|failed|CUDA" "$SERVER_LOG" | tail -5 | sed 's/^/      /'
            kill_server
            break
        fi
        CODE=$(curl -s http://localhost:8080/v1/models --max-time 3 -o /dev/null -w "%{http_code}" 2>/dev/null; true)
        if   [ "$CODE" = "200" ]; then READY=1; log "  [OK] HTTP 200 at t=${ELAPSED}s"; break
        elif [ "$CODE" = "503" ]; then log "  t=${ELAPSED}s: loading weights..."
        else                           log "  t=${ELAPSED}s: HTTP $CODE"; fi
    done

    if [ "$READY" = 0 ]; then
        log "  [FAIL] $MODEL_NAME — server did not start"
        SKIPPED_MODELS+=("$MODEL_NAME (server failed to start)")
        kill_server
        continue
    fi

    # ── Smoke test: 3 graded prompts ─────────────────────────────────────────
    if [ "$SKIP_SMOKE" = 1 ]; then
        log "  [SMOKE SKIP] --skip-smoke set"
        SMOKE_PASSED_MODELS+=("$MODEL_NAME (smoke skipped)")
    else
        log "  ── Smoke ────────────────────────────────────────────"
        SMOKE_PASS=1
        for tok_count in 32 256 512; do
            msg=$(python3 -c "print('The quick brown fox jumped over the lazy dog. ' * $((tok_count / 8 + 1)))")
            log "  Sending ~${tok_count}-tok prompt..."
            set +e
            resp=$(curl -s http://localhost:8080/v1/chat/completions \
                -H "Content-Type: application/json" \
                -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"$msg\"}],\"max_tokens\":32}" \
                --max-time 120 2>/dev/null)
            CURL_EXIT=$?
            set -e
            if ! kill -0 "$(cat $SERVER_PIDFILE 2>/dev/null)" 2>/dev/null; then
                log "  [!] Server died after ~${tok_count}-tok prompt (OOM?)"
                grep -E "error|OOM|failed|CUDA|memory" "$SERVER_LOG" | tail -5 | sed 's/^/      /'
                SMOKE_PASS=0; break
            fi
            if [ "$CURL_EXIT" != 0 ]; then
                log "  [!] curl failed (exit $CURL_EXIT) at ~${tok_count} tok"
                SMOKE_PASS=0; break
            fi
            set +e
            echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('choices',[{}])[0].get('message')" 2>/dev/null
            PY_EXIT=$?
            set -e
            if [ "$PY_EXIT" = 0 ]; then
                log "  [OK] ~${tok_count} tok"
            else
                log "  [!] ~${tok_count} tok — bad response: ${resp:0:120}"
                SMOKE_PASS=0; break
            fi
        done
        log "  ─────────────────────────────────────────────────────"

        if [ "$SMOKE_PASS" = 0 ]; then
            log "  [SMOKE FAIL] $MODEL_NAME — skipping benchmark"
            SKIPPED_MODELS+=("$MODEL_NAME (smoke failed)")
            kill_server
            continue
        fi
        log "  [SMOKE PASS] $MODEL_NAME — starting aiperf sweep"
        SMOKE_PASSED_MODELS+=("$MODEL_NAME")
    fi
    BENCH_MODELS+=("$i")

    # ── Record model start time for power windowing ───────────────────────────
    echo "MODEL_START:${MODEL_NAME}:${MODEL_QUANT}:$(date +%s)" >> "$TIMING_LOG"

    # ── aiperf sweep on the same server ──────────────────────────────────────
    RUN_NUM=0
    TOTAL_RUNS=$(( ${#GEN_LENGTHS[@]} * ${#PROMPT_LENGTHS[@]} ))

    for GEN in "${GEN_LENGTHS[@]}"; do
        for CTX in "${PROMPT_LENGTHS[@]}"; do
            RUN_NUM=$((RUN_NUM + 1))
            ARTIFACT_DIR="$BASE_ARTIFACT/${MODEL_NAME}/gen${GEN}/ctx${CTX}"
            mkdir -p "$ARTIFACT_DIR"

            if [ -f "$ARTIFACT_DIR/profile_export_aiperf.json" ]; then
                log "  [$RUN_NUM/$TOTAL_RUNS] prompt=$CTX gen=$GEN  [RESUME SKIP]"
                continue
            fi

            log "  [$RUN_NUM/$TOTAL_RUNS] prompt=$CTX  gen=$GEN  reqs=$REQS"

            if ! ensure_server_alive "$MODEL_PATH" "$MODEL_CTX_SIZE" "$SERVER_LOG"; then
                log "  [ABORT] Cannot recover server — skipping remaining combos"
                SKIPPED_MODELS+=("$MODEL_NAME (server unrecoverable at gen=$GEN ctx=$CTX)")
                break 2
            fi

            aiperf profile \
                --model                          "$MODEL_NAME" \
                --streaming \
                --endpoint-type                  'chat' \
                --url                            'http://localhost:8080' \
                --tokenizer                      "$MODEL_TOKENIZER" \
                --synthetic-input-tokens-mean    "$CTX" \
                --synthetic-input-tokens-stddev  0 \
                --output-tokens-mean             "$GEN" \
                --request-count                  "$REQS" \
                --concurrency                    "$CONCURRENCY" \
                --slice-duration                 "$SLICE_DURATION" \
                --random-seed                    "$RANDOM_SEED" \
                --request-timeout-seconds        "$REQUEST_TIMEOUT" \
                --artifact-dir                   "$ARTIFACT_DIR" \
                || log "  aiperf run failed (ctx=$CTX gen=$GEN)"

            [ "$RUN_NUM" -lt "$TOTAL_RUNS" ] && { log "  Cooldown ${COOLDOWN_COMBO}s..."; sleep "$COOLDOWN_COMBO"; }
        done
    done

    echo "MODEL_END:${MODEL_NAME}:${MODEL_QUANT}:$(date +%s)" >> "$TIMING_LOG"
    kill_server
    log "Model done. Cooling ${COOLDOWN_MODEL}s before next model..."
    sleep "$COOLDOWN_MODEL"
done

deactivate 2>/dev/null || true

# ── Stop tegrastats ───────────────────────────────────────────────────────────
if [ "$DRY_RUN" = 0 ]; then
    banner "Stop tegrastats"
    stop_tegrastats
    log "tegrastats stopped"
fi

[ "$DRY_RUN" = 1 ] && banner "Dry run complete" && exit 0

# ── Summary ───────────────────────────────────────────────────────────────────
banner "Results"
echo "  PASSED  (${#SMOKE_PASSED_MODELS[@]})"
for m in "${SMOKE_PASSED_MODELS[@]}"; do echo "    [PASS] $m"; done
echo ""
echo "  FAILED / SKIPPED  (${#SKIPPED_MODELS[@]})"
for m in "${SKIPPED_MODELS[@]}"; do echo "    [FAIL] $m"; done
echo ""

[ "${#BENCH_MODELS[@]}" = 0 ] && echo "  No models benchmarked." && exit 1

# ── Generate report.md ────────────────────────────────────────────────────────
banner "Generating report.md"

cat > "$REPORT_PY" << 'PYEOF'
import re, sys, json, os, glob
from datetime import datetime

base_dir    = sys.argv[1]
tegra_log   = sys.argv[2]
timing_log  = sys.argv[3]
skipped_arg = sys.argv[4] if len(sys.argv) > 4 else ""
report_path = os.path.join(base_dir, "report.md")

skipped = [s for s in skipped_arg.split("||") if s] if skipped_arg else []

# ── Parse tegrastats ──────────────────────────────────────────────────────────
samples = []
try:
    for line in open(tegra_log):
        ts_m = re.match(r'(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})', line)
        if not ts_m:
            continue
        try:
            ts = datetime.strptime(ts_m.group(1), "%m-%d-%Y %H:%M:%S").timestamp()
        except:
            continue
        pw_m  = re.search(r'VDD_CPU_GPU_CV (\d+)mW', line)
        cpu_m = re.search(r'cpu@([\d.]+)C', line)
        gpu_m = re.search(r'gpu@([\d.]+)C', line)
        tj_m  = re.search(r'tj@([\d.]+)C',  line)
        if pw_m:
            samples.append((ts, int(pw_m.group(1))/1000.0,
                float(cpu_m.group(1)) if cpu_m else None,
                float(gpu_m.group(1)) if gpu_m else None,
                float(tj_m.group(1))  if tj_m  else None))
except FileNotFoundError:
    pass

def power_in_window(t0, t1):
    w = [s for s in samples if t0 <= s[0] <= t1]
    if not w:
        return None, None, None, None
    avg_pw = sum(s[1] for s in w) / len(w)
    cpu_t  = [s[2] for s in w if s[2] is not None]
    gpu_t  = [s[3] for s in w if s[3] is not None]
    tj_t   = [s[4] for s in w if s[4] is not None]
    return (avg_pw,
            sum(cpu_t)/len(cpu_t) if cpu_t else None,
            sum(gpu_t)/len(gpu_t) if gpu_t else None,
            max(tj_t)             if tj_t  else None)

# ── Parse model timing + quant ────────────────────────────────────────────────
model_windows = {}
model_quants  = {}
try:
    for line in open(timing_log):
        line = line.strip()
        if line.startswith("MODEL_START:"):
            # FORMAT: MODEL_START:name:quant:ts
            parts = line.split(":", 3)
            if len(parts) == 4:
                _, name, quant, ts = parts
                model_windows.setdefault(name, {})["start"] = float(ts)
                model_quants[name] = quant
        elif line.startswith("MODEL_END:"):
            parts = line.split(":", 3)
            if len(parts) == 4:
                _, name, _, ts = parts
                model_windows.setdefault(name, {})["end"] = float(ts)
except FileNotFoundError:
    pass

# ── Discover result files ─────────────────────────────────────────────────────
results = []
for json_path in sorted(glob.glob(f"{base_dir}/**/profile_export_aiperf.json", recursive=True)):
    rel   = os.path.relpath(json_path, base_dir)
    parts = rel.split(os.sep)
    if len(parts) < 4:
        continue
    model_name = parts[0]
    gen = int(re.sub(r'\D', '', parts[1]))
    ctx = int(re.sub(r'\D', '', parts[2]))
    try:
        d = json.load(open(json_path))
    except:
        continue
    def g(key, stat="avg"): return (d.get(key, {}) or {}).get(stat)
    win  = model_windows.get(model_name, {})
    avg_pw, avg_cpu, avg_gpu, peak_tj = power_in_window(
        win.get("start", 0), win.get("end", 9e18))
    tps = g("output_token_throughput_per_user")
    tok_j = (tps / avg_pw) if (tps and avg_pw) else None
    quant = model_quants.get(model_name, "?")
    results.append({
        "model": model_name, "quant": quant,
        "prompt": ctx, "gen": gen,
        # ISL / OSL
        "isl":         g("input_sequence_length"),
        "osl":         g("output_sequence_length"),
        "osl_mis":     g("osl_mismatch_diff_pct"),
        # TTFT
        "ttft_avg":    g("time_to_first_token"),
        "ttft_p50":    g("time_to_first_token",  "p50"),
        "ttft_p90":    g("time_to_first_token",  "p90"),
        "ttft_p99":    g("time_to_first_token",  "p99"),
        # T2T (time to second token)
        "t2t_avg":     g("time_to_second_token"),
        "t2t_p50":     g("time_to_second_token", "p50"),
        "t2t_p90":     g("time_to_second_token", "p90"),
        "t2t_p99":     g("time_to_second_token", "p99"),
        # ITL
        "itl_avg":     g("inter_token_latency"),
        "itl_p50":     g("inter_token_latency",  "p50"),
        "itl_p90":     g("inter_token_latency",  "p90"),
        "itl_p99":     g("inter_token_latency",  "p99"),
        # Throughput
        "tps":         tps,
        "req_s":       g("request_throughput"),
        # E2E tok/s
        "e2e_avg":     g("e2e_output_token_throughput"),
        "e2e_p50":     g("e2e_output_token_throughput", "p50"),
        "e2e_p90":     g("e2e_output_token_throughput", "p90"),
        "e2e_p99":     g("e2e_output_token_throughput", "p99"),
        # Request latency
        "rl_avg":      g("request_latency"),
        "rl_p50":      g("request_latency", "p50"),
        "rl_p90":      g("request_latency", "p90"),
        "rl_p99":      g("request_latency", "p99"),
        # Prefill throughput
        "pre_avg":     g("prefill_throughput_per_user"),
        "pre_p50":     g("prefill_throughput_per_user", "p50"),
        "pre_p90":     g("prefill_throughput_per_user", "p90"),
        "pre_p99":     g("prefill_throughput_per_user", "p99"),
        # Power / efficiency
        "power_w": avg_pw, "avg_cpu_c": avg_cpu, "avg_gpu_c": avg_gpu,
        "peak_tj_c": peak_tj, "tok_j": tok_j,
    })
results.sort(key=lambda r: (r["model"], r["gen"], r["prompt"]))

# ── Per-model thermal summary ─────────────────────────────────────────────────
thermal = {}
for name, win in model_windows.items():
    avg_pw, avg_cpu, avg_gpu, peak_tj = power_in_window(
        win.get("start", 0), win.get("end", 9e18))
    thermal[name] = {"avg_pw": avg_pw, "avg_cpu": avg_cpu,
                     "avg_gpu": avg_gpu, "peak_tj": peak_tj,
                     "throttled": peak_tj is not None and peak_tj > 85}

# ── Best tok/J per model ──────────────────────────────────────────────────────
best_tokj = {}
for r in results:
    m = r["model"]
    if r["tok_j"] is not None:
        if m not in best_tokj or r["tok_j"] > best_tokj[m]["tok_j"]:
            best_tokj[m] = r

# ── Write report.md ───────────────────────────────────────────────────────────
lines = []
def L(s=""): lines.append(s)
def fmt(v, fs, fallback="—"): return format(v, fs) if v is not None else fallback

L("# Tiny LLM Benchmark — Jetson Orin Nano Super 8GB")
L()
L(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
L(f"**Backend:** llama.cpp CUDA (`-ngl 99`)  **Concurrency:** 1  ")
L(f"**Sweep:** prompt in {{128,512,1024,2048}} tok  ×  gen in {{64,128,256}} tok")
L(f"**Artifacts:** `{base_dir}`")
L()

if skipped:
    L("## Skipped / Failed Models")
    L()
    for s in skipped:
        L(f"- {s}")
    L()

L("## Full Results")
L()
L("> Cells marked `—` = OOM (server crashed or skipped). "
  "Power = VDD\\_CPU\\_GPU\\_CV average over aiperf run window (CPU+GPU+CV rail, milliwatts → watts).")
L()
H = ("| Model | Quant | ISL (tok) | OSL (tok) | OSL mismatch (%) "
     "| TTFT avg (ms) | TTFT p50 | TTFT p90 | TTFT p99 "
     "| T2T avg (ms) | T2T p50 | T2T p90 | T2T p99 "
     "| ITL avg (ms) | ITL p50 | ITL p90 | ITL p99 "
     "| Tok/s (server) | Req/s "
     "| E2E tok/s avg | E2E p50 | E2E p90 | E2E p99 "
     "| Req lat avg (ms) | Req lat p50 | Req lat p90 | Req lat p99 "
     "| Prefill tok/s avg | Prefill p50 | Prefill p90 | Prefill p99 "
     "| Power (W) | **Output Tok/J** |")
S = ("|-------|:-----:|----------:|----------:|-----------------:"
     "|--------------:|---------:|---------:|---------:"
     "|-------------:|--------:|--------:|--------:"
     "|--------------:|--------:|--------:|--------:"
     "|---------------:|------:"
     "|--------------:|--------:|--------:|--------:"
     "|-----------------:|------------:|------------:|------------:"
     "|------------------:|------------:|------------:|------------:"
     "|----------:|----------:|")
L(H); L(S)
for r in results:
    L(f"| {r['model']} | {r['quant']} "
      f"| {fmt(r['isl'],'.0f')} | {fmt(r['osl'],'.1f')} | {fmt(r['osl_mis'],'.2f')} "
      f"| {fmt(r['ttft_avg'],'.1f')} | {fmt(r['ttft_p50'],'.1f')} | {fmt(r['ttft_p90'],'.1f')} | {fmt(r['ttft_p99'],'.1f')} "
      f"| {fmt(r['t2t_avg'],'.2f')} | {fmt(r['t2t_p50'],'.2f')} | {fmt(r['t2t_p90'],'.2f')} | {fmt(r['t2t_p99'],'.2f')} "
      f"| {fmt(r['itl_avg'],'.2f')} | {fmt(r['itl_p50'],'.2f')} | {fmt(r['itl_p90'],'.2f')} | {fmt(r['itl_p99'],'.2f')} "
      f"| {fmt(r['tps'],'.2f')} | {fmt(r['req_s'],'.3f')} "
      f"| {fmt(r['e2e_avg'],'.2f')} | {fmt(r['e2e_p50'],'.2f')} | {fmt(r['e2e_p90'],'.2f')} | {fmt(r['e2e_p99'],'.2f')} "
      f"| {fmt(r['rl_avg'],'.1f')} | {fmt(r['rl_p50'],'.1f')} | {fmt(r['rl_p90'],'.1f')} | {fmt(r['rl_p99'],'.1f')} "
      f"| {fmt(r['pre_avg'],'.1f')} | {fmt(r['pre_p50'],'.1f')} | {fmt(r['pre_p90'],'.1f')} | {fmt(r['pre_p99'],'.1f')} "
      f"| {fmt(r['power_w'],'.2f')} | **{fmt(r['tok_j'],'.3f')}** |")  # output tok/J

L()
L("## Best Output Tok/J per Model")
L()
L("| Model | Quant | Best Output Tok/J | ISL (tok) | OSL (tok) | Output Tok/s | Power (W) |")
L("|-------|:-----:|------------------:|----------:|----------:|-------------:|----------:|")
for name in sorted(best_tokj.keys()):
    b = best_tokj[name]
    L(f"| {b['model']} | {b['quant']} | **{fmt(b['tok_j'],'.3f')}** "
      f"| {fmt(b['isl'],'.0f')} | {fmt(b['osl'],'.1f')} "
      f"| {fmt(b['tps'],'.2f')} | {fmt(b['power_w'],'.2f')} |")

L()
L("## Thermal Summary")
L()
L("| Model | Avg Power (W) | Avg CPU (°C) | Avg GPU (°C) | Peak TJ (°C) | Throttled |")
L("|-------|-------------:|-------------:|-------------:|-------------:|:---------:|")
for name in sorted(thermal.keys()):
    t = thermal[name]
    L(f"| {name} | {fmt(t['avg_pw'],'.2f')} | {fmt(t['avg_cpu'],'.1f')} "
      f"| {fmt(t['avg_gpu'],'.1f')} | {fmt(t['peak_tj'],'.1f')} "
      f"| {'YES' if t['throttled'] else 'No'} |")

L()
L("---")
L(f"*Generated by `bench_all.sh` on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

with open(report_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\n  Report -> {report_path}")
print(f"  {len(results)} rows  |  {len(thermal)} models  |  {len(skipped)} skipped")
PYEOF

SKIPPED_STR=""
for s in "${SKIPPED_MODELS[@]}"; do SKIPPED_STR="${SKIPPED_STR}${s}||"; done

python3 "$REPORT_PY" "$BASE_ARTIFACT" "$TEGRA_LOG" "$TIMING_LOG" "$SKIPPED_STR"

banner "Done"
echo "  Artifacts : $BASE_ARTIFACT"
echo "  Report    : $BASE_ARTIFACT/report.md"
echo ""
echo "  Dashboard (optional):"
echo "    source ~/venv/bin/activate"
echo "    AIPERF_DASHBOARD_HOST=0.0.0.0 aiperf plot $BASE_ARTIFACT --dashboard --port 8050"
echo ""
