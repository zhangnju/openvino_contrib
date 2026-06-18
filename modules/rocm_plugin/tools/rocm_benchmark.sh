#!/usr/bin/env bash
# ROCm Plugin Regression Benchmark
# Usage: rocm_benchmark.sh [--quick] [--model yolo|bert|all]
#   --quick  : shorter run time (30s yolo throughput instead of 120s)
#   --model  : which models to test (default: all)
#
# Exit code: 0 = all pass, 1 = one or more failed

set -uo pipefail

BENCH=/home/openvino/bin/intel64/Release/benchmark_app
HG_CFG=/tmp/rocm_hg.json
YOLO_MODEL=/home/yolo26x.onnx
BERT_MODEL=/tmp/bert_model/bertsquad-12.onnx

# Thresholds (FAIL if below)
YOLO_THROUGHPUT_MIN=250   # FPS, nireq=8, hipGraph+fused_tuning
YOLO_LATENCY_MIN=180      # FPS, nireq=1
BERT_MIN=195              # FPS, nireq=1  (after FC+GELU MIGraphX fusion)

QUICK=0
RUN_YOLO=1
RUN_BERT=1

for arg in "$@"; do
  case $arg in
    --quick)      QUICK=1 ;;
    --model=yolo) RUN_BERT=0 ;;
    --model=bert) RUN_YOLO=0 ;;
    --model=all)  RUN_YOLO=1; RUN_BERT=1 ;;
  esac
done

YOLO_TPUT_T=120; [ $QUICK -eq 1 ] && YOLO_TPUT_T=30

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
PASS=0; FAIL=0
declare -a RESULTS

stamp=$(date '+%Y-%m-%d %H:%M:%S')
echo "========================================================"
echo "  ROCm Plugin Regression Benchmark  —  $stamp"
echo "========================================================"

ensure_hip_graph_cfg() {
  [ -f "$HG_CFG" ] || echo '{"ROCM": {"ROCM_USE_HIP_GRAPH": "YES"}}' > "$HG_CFG"
}

# $1=label  $2=fps  $3=median  $4=threshold
record_result() {
  local label="$1" fps="$2" median="$3" threshold="$4"
  local fps_int="${fps%.*}"
  if [ -z "$fps" ]; then
    echo -e "  ${RED}FAIL${NC}  no output"
    RESULTS+=("FAIL | $label | fps=N/A | min=$threshold")
    ((FAIL++)) || true
  elif [ "$fps_int" -ge "$threshold" ] 2>/dev/null; then
    echo -e "  ${GREEN}PASS${NC}  Throughput: ${fps} FPS (>= ${threshold})  Median: ${median} ms"
    RESULTS+=("PASS | $label | fps=$fps | min=$threshold")
    ((PASS++)) || true
  else
    echo -e "  ${RED}FAIL${NC}  Throughput: ${fps} FPS (< ${threshold})  Median: ${median} ms"
    RESULTS+=("FAIL | $label | fps=$fps | min=$threshold")
    ((FAIL++)) || true
  fi
}

# ── yolo26x ──────────────────────────────────────────────────────────────────
if [ $RUN_YOLO -eq 1 ]; then
  echo ""
  echo "════ yolo26x ════"
  if [ ! -f "$YOLO_MODEL" ]; then
    echo -e "  ${YELLOW}SKIP${NC} — model not found: $YOLO_MODEL"
    RESULTS+=("SKIP | yolo26x throughput | model missing")
    RESULTS+=("SKIP | yolo26x latency    | model missing")
  else
    ensure_hip_graph_cfg

    # 1. Throughput mode: nireq=8, hipGraph, fused tuning
    echo ""
    echo "--- yolo26x | throughput | nireq=8 hipGraph fused_tuning (${YOLO_TPUT_T}s) ---"
    out=$(ROCMLIR_ENABLE_TUNING_FUSED=1 $BENCH \
          -m "$YOLO_MODEL" -d ROCM.0 \
          -t "$YOLO_TPUT_T" -nireq 8 \
          -load_config "$HG_CFG" 2>&1)
    fps=$(echo "$out" | grep -oP 'Throughput:\s+\K[0-9.]+' | tail -1)
    med=$(echo "$out" | grep -oP 'Median:\s+\K[0-9.]+' | tail -1)
    [ -z "$fps" ] && { echo "$out" | tail -5; }
    record_result "yolo26x throughput (nireq=8 hipGraph ${YOLO_TPUT_T}s)" "$fps" "$med" "$YOLO_THROUGHPUT_MIN"

    # 2. Latency mode: nireq=1, no hipGraph
    echo ""
    echo "--- yolo26x | latency | nireq=1 no-hipGraph (15s) ---"
    out=$(ROCMLIR_ENABLE_TUNING_FUSED=1 $BENCH \
          -m "$YOLO_MODEL" -d ROCM.0 \
          -t 15 -nireq 1 2>&1)
    fps=$(echo "$out" | grep -oP 'Throughput:\s+\K[0-9.]+' | tail -1)
    med=$(echo "$out" | grep -oP 'Median:\s+\K[0-9.]+' | tail -1)
    record_result "yolo26x latency (nireq=1 15s)" "$fps" "$med" "$YOLO_LATENCY_MIN"
  fi
fi

# ── bertsquad-12 ─────────────────────────────────────────────────────────────
if [ $RUN_BERT -eq 1 ]; then
  echo ""
  echo "════ bertsquad-12 (FP16) ════"
  if [ ! -f "$BERT_MODEL" ]; then
    echo -e "  ${YELLOW}SKIP${NC} — model not found: $BERT_MODEL"
    RESULTS+=("SKIP | bertsquad-12 | model missing")
  else
    echo ""
    echo "--- bertsquad-12 | nireq=1 (15s) ---"
    out=$($BENCH \
          -m "$BERT_MODEL" -d ROCM.0 \
          -t 15 -nireq 1 \
          -shape "unique_ids_raw_output___9:0[1],segment_ids:0[1,256],input_mask:0[1,256],input_ids:0[1,256]" \
          2>&1)
    fps=$(echo "$out" | grep -oP 'Throughput:\s+\K[0-9.]+' | tail -1)
    med=$(echo "$out" | grep -oP 'Median:\s+\K[0-9.]+' | tail -1)
    [ -z "$fps" ] && { echo "$out" | tail -5; }
    record_result "bertsquad-12 (nireq=1 15s)" "$fps" "$med" "$BERT_MIN"
  fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  Summary"
echo "========================================================"
for r in "${RESULTS[@]}"; do
  status="${r%%|*}"
  rest="${r#*|}"
  if [[ $status == *PASS* ]]; then
    echo -e "  ${GREEN}PASS${NC} |$rest"
  elif [[ $status == *FAIL* ]]; then
    echo -e "  ${RED}FAIL${NC} |$rest"
  else
    echo -e "  ${YELLOW}SKIP${NC} |$rest"
  fi
done
echo ""
echo "  Passed: $PASS   Failed: $FAIL"
echo "========================================================"
echo ""
echo "Thresholds:"
echo "  yolo26x throughput (nireq=8 hipGraph) : >= $YOLO_THROUGHPUT_MIN FPS"
echo "  yolo26x latency    (nireq=1)          : >= $YOLO_LATENCY_MIN FPS"
echo "  bertsquad-12       (nireq=1)          : >= $BERT_MIN FPS"
echo ""
echo "Options: --quick (30s yolo)  --model=yolo|bert|all"

[ $FAIL -gt 0 ] && exit 1
exit 0
