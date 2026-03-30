#!/usr/bin/env bash
# 一键消融（仅 DRB）：扫描 MAX_AGENT_TOOL_LOOPS（ATL）与 MAX_WEB_RESEARCH_LOOPS（WRL，对应 run_research_concurrent 的 --max_loops）
# DRB 跑完后由 run_research_concurrent 自动执行 process_drb（可用 --no-process-drb 关闭）。
# 默认 SKIP_RACE=1：不跑 RACE（无需外网评测）；需要联网评测时：RUN_RACE=1 或 SKIP_RACE=0
#
# 用法（在仓库根目录）:
#   bash benchmarks/run_loop_ablation.sh
#
# —— 只在一部分超参数上实验 ——
# 1) 只扫 ATL 的几个取值（WRL 固定）:
#    VALUES="1 3 5" MODE=sweep_agent BASELINE_WRL=3 bash benchmarks/run_loop_ablation.sh
# 2) 只扫 WRL 的几个取值（ATL 固定）:
#    VALUES="2 4" MODE=sweep_web BASELINE_ATL=3 bash benchmarks/run_loop_ablation.sh
# 3) 子网格（只对 VALUES 里的数做笛卡尔积，例如 3×3 而非 5×5）:
#    VALUES="1 3 5" MODE=grid bash benchmarks/run_loop_ablation.sh
# 4) 任意 (ATL,WRL) 组合（非完整网格）:
#    MODE=pairs CUSTOM_PAIRS="1:3 2:3 3:5 5:1" bash benchmarks/run_loop_ablation.sh
#    绘图时同上 MODE=pairs（SKIP_RUN=1 重绘时也要 MODE=pairs）
#
# 仅汇总/出图（已有输出目录时）:
#   SKIP_RUN=1 bash benchmarks/run_loop_ablation.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/benchmarks"

# ---------- 可改参数 ----------
MODE="${MODE:-sweep_agent}"          # sweep_agent | sweep_web | grid | pairs
VALUES="${VALUES:-5}"        # 空格分隔；可改成子集如 "2 4"
CUSTOM_PAIRS="${CUSTOM_PAIRS:-}"     # MODE=pairs 时用，如 "1:3 2:4"
BASELINE_ATL="${BASELINE_ATL:-3}"    # 扫 WRL 时固定的 ATL
BASELINE_WRL="${BASELINE_WRL:-3}"    # 扫 ATL 时固定的 WRL

PROVIDER="${PROVIDER:-openai}"
MODEL="${MODEL:-${LLM_MODEL:-o3-mini}}"
MAX_CONCURRENT="${MAX_CONCURRENT:-5}"
LIMIT="${LIMIT:-20}"                   # 例如 LIMIT=5 做快速试跑
TASK_IDS="${TASK_IDS:-}"             # 例如 TASK_IDS=1,2,3

SKIP_RUN="${SKIP_RUN:-0}"
# 默认跳过 RACE（离线/无外网）；要跑评测：RUN_RACE=1
SKIP_RACE="${SKIP_RACE:-1}"
RUN_RACE="${RUN_RACE:-0}"
if [[ "$RUN_RACE" == "1" ]]; then
  SKIP_RACE=0
fi
SKIP_PLOT="${SKIP_PLOT:-0}"

ABLATION_ROOT="${ABLATION_ROOT:-$ROOT/benchmarks/ablation_loop_runs}"
MANIFEST="${MANIFEST:-$ABLATION_ROOT/manifest.jsonl}"

mkdir -p "$ABLATION_ROOT"

if [[ "$SKIP_RUN" == "1" ]]; then
  echo "SKIP_RUN=1：不跑 benchmark/RACE，仅根据已有 manifest 绘图。"
  if [[ ! -s "$MANIFEST" ]]; then
    echo "错误: $MANIFEST 不存在或为空。请先完整跑一轮或指定 MANIFEST=..." >&2
    exit 1
  fi
  python -u "$ROOT/benchmarks/plot_loop_ablation.py" plot \
    --manifest "$MANIFEST" \
    --ablation-root "$ABLATION_ROOT" \
    --mode "$MODE" \
    --baseline-atl "$BASELINE_ATL" \
    --baseline-wrl "$BASELINE_WRL"
  exit 0
fi

run_one() {
  local atl="$1"
  local wrl="$2"
  local tag="atl${atl}_wrl${wrl}"
  local out="$ABLATION_ROOT/$tag"

  mkdir -p "$out"

  export MAX_AGENT_TOOL_LOOPS="$atl"

  # 每组固定记录：与每条 JSON 内 metadata.loop_limits 互证
  python - "$out" "$atl" "$wrl" "$MODEL" "$PROVIDER" <<'PY'
import json, sys, datetime
out, atl, wrl, model, provider = sys.argv[1:6]
cfg = {
    "tag": f"atl{atl}_wrl{wrl}",
    "max_agent_tool_loops": int(atl),
    "max_web_research_loops": int(wrl),
    "model": model,
    "provider": provider,
    "benchmark": "drb",
    "written_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "note": "run_research_concurrent 将本组上限写入 LangGraph config.configurable（max_web_research_loops / max_agent_tool_loops），不依赖全局 os.environ，避免被其它路径污染。shell export 的 MAX_AGENT_TOOL_LOOPS 仅在任务入口被读取一次并写入 configurable。",
}
open(f"{out}/run_config.json", "w", encoding="utf-8").write(
    json.dumps(cfg, ensure_ascii=False, indent=2) + "\n"
)
PY

  local extra=()
  [[ -n "$LIMIT" ]] && extra+=(--limit "$LIMIT")
  [[ -n "$TASK_IDS" ]] && extra+=(--task_ids "$TASK_IDS")

  local model_name="ablation_${tag}"
  echo "========== Run: MAX_AGENT_TOOL_LOOPS=$atl  --max_loops=$wrl  -> $out =========="
  python -u run_research_concurrent.py \
    --benchmark drb \
    --output_dir "$out" \
    --provider "$PROVIDER" \
    --model "$MODEL" \
    --max_concurrent "$MAX_CONCURRENT" \
    --max_loops "$wrl" \
    --drb-jsonl-name "$model_name" \
    "${extra[@]}"

  if [[ "$SKIP_RACE" == "1" ]]; then
    python -u "$ROOT/benchmarks/plot_loop_ablation.py" summarize \
      --output-dir "$out" \
      --tag "$tag" \
      --atl "$atl" \
      --wrl "$wrl" \
      --model-name "$model_name" \
      --race-overall "" \
      --manifest "$MANIFEST"
    return 0
  fi

  local race_out="$out/race_eval"
  mkdir -p "$race_out"
  echo "---------- RACE eval -> $race_out ----------"
  ( cd deep_research_bench && python -u deepresearch_bench_race.py "$model_name" --output_dir "$race_out" )

  local overall=""
  if [[ -f "$race_out/race_result.txt" ]]; then
    overall="$(grep -E '^Overall Score:' "$race_out/race_result.txt" | tail -1 | awk '{print $3}' || true)"
  fi

  python -u "$ROOT/benchmarks/plot_loop_ablation.py" summarize \
    --output-dir "$out" \
    --tag "$tag" \
    --atl "$atl" \
    --wrl "$wrl" \
    --model-name "$model_name" \
    --race-overall "${overall:-}" \
    --manifest "$MANIFEST"
}

if [[ "$SKIP_RUN" == "0" ]]; then
  : >"$MANIFEST"
fi

case "$MODE" in
  sweep_agent)
    for atl in $VALUES; do
      run_one "$atl" "$BASELINE_WRL"
    done
    ;;
  sweep_web)
    for wrl in $VALUES; do
      run_one "$BASELINE_ATL" "$wrl"
    done
    ;;
  grid)
    for atl in $VALUES; do
      for wrl in $VALUES; do
        run_one "$atl" "$wrl"
      done
    done
    ;;
  pairs)
    if [[ -z "${CUSTOM_PAIRS// /}" ]]; then
      echo "MODE=pairs 需要 CUSTOM_PAIRS，例如: CUSTOM_PAIRS=\"1:3 2:3 3:5\"" >&2
      exit 1
    fi
    for pair in $CUSTOM_PAIRS; do
      if [[ "$pair" != *:* ]]; then
        echo "无效项 '$pair'，应为 atl:wrl（如 2:5）" >&2
        exit 1
      fi
      atl="${pair%%:*}"
      wrl="${pair#*:}"
      run_one "$atl" "$wrl"
    done
    ;;
  *)
    echo "Unknown MODE=$MODE (use sweep_agent | sweep_web | grid | pairs)" >&2
    exit 1
    ;;
esac

if [[ "$SKIP_PLOT" == "1" ]]; then
  echo "SKIP_PLOT=1，跳过绘图。"
  exit 0
fi

echo "---------- 绘图 ----------"
  python -u "$ROOT/benchmarks/plot_loop_ablation.py" plot \
  --manifest "$MANIFEST" \
  --ablation-root "$ABLATION_ROOT" \
  --mode "$MODE" \
  --baseline-atl "$BASELINE_ATL" \
  --baseline-wrl "$BASELINE_WRL"

echo "完成。清单: $MANIFEST  汇总表: $ABLATION_ROOT/ablation_summary.tsv  图: $ABLATION_ROOT/*.png"
if [[ "$SKIP_RACE" == "1" ]]; then
  echo "（已跳过 RACE；联网评测请设 RUN_RACE=1 或 SKIP_RACE=0 后重跑本脚本或单独跑 deepresearch_bench_race.py）"
fi
