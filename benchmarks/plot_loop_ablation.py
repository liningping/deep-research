#!/usr/bin/env python3
"""
消融结果汇总与双 Y 轴图：主 Y = RACE Overall Score，副 Y = 平均每条约估 token。

说明：当前 benchmark 落盘 JSON 不含真实 API token；默认用 tiktoken(cl100k_base)
对 article 编码长度求平均，不可用时退回 len(text)//4。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _try_tiktoken_count(text: str) -> Optional[int]:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return None


_DRB_REPORT_JSON = re.compile(r"^\d+\.json$")


def avg_estimated_tokens_for_run(output_dir: Path) -> Tuple[float, int]:
    """Return (mean_tokens_per_task, num_report_json_files)."""
    files = sorted(output_dir.glob("*.json"))
    if not files:
        return float("nan"), 0
    counts: List[int] = []
    for p in files:
        if not _DRB_REPORT_JSON.match(p.name):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        article = data.get("article") or ""
        if not isinstance(article, str):
            article = str(article)
        n = _try_tiktoken_count(article)
        if n is None:
            n = max(len(article) // 4, 0)
        counts.append(n)
    if not counts:
        return float("nan"), 0
    return sum(counts) / len(counts), len(counts)


def cmd_summarize(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    avg_tok, n_json = avg_estimated_tokens_for_run(out)
    race_val: Optional[float] = None
    if args.race_overall and str(args.race_overall).strip() not in ("", "nan"):
        try:
            race_val = float(str(args.race_overall).strip())
        except ValueError:
            race_val = None

    row = {
        "tag": args.tag,
        "max_agent_tool_loops": int(args.atl),
        "max_web_research_loops": int(args.wrl),
        "output_dir": str(out.resolve()),
        "model_name": args.model_name,
        "race_overall": race_val,
        "avg_est_tokens": avg_tok if not math.isnan(avg_tok) else None,
        "n_json": n_json,
    }
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_summary_tsv(rows: List[Dict[str, Any]], out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "ablation_summary.tsv"
    cols = (
        "tag",
        "max_agent_tool_loops",
        "max_web_research_loops",
        "race_overall",
        "avg_est_tokens",
        "n_json",
        "model_name",
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            line = []
            for c in cols:
                v = r.get(c)
                line.append("" if v is None else str(v))
            f.write("\t".join(line) + "\n")
    print(f"已保存 {path}")


def cmd_plot(args: argparse.Namespace) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("需要 matplotlib: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    rows = _load_manifest(Path(args.manifest))
    if not rows:
        print("manifest 为空或不存在，无法绘图。", file=sys.stderr)
        sys.exit(1)

    mode = args.mode
    base_atl = int(args.baseline_atl)
    base_wrl = int(args.baseline_wrl)
    out_root = Path(args.ablation_root)
    _write_summary_tsv(rows, out_root)

    def series_for_sweep_agent() -> Tuple[List[int], List[float], List[float]]:
        xs: List[int] = []
        y_race: List[float] = []
        y_tok: List[float] = []
        for r in sorted(rows, key=lambda x: x["max_agent_tool_loops"]):
            if r.get("max_web_research_loops") != base_wrl:
                continue
            xs.append(int(r["max_agent_tool_loops"]))
            ro = r.get("race_overall")
            y_race.append(float(ro) if ro is not None else float("nan"))
            t = r.get("avg_est_tokens")
            y_tok.append(float(t) if t is not None else float("nan"))
        return xs, y_race, y_tok

    def series_for_sweep_web() -> Tuple[List[int], List[float], List[float]]:
        xs: List[int] = []
        y_race: List[float] = []
        y_tok: List[float] = []
        for r in sorted(rows, key=lambda x: x["max_web_research_loops"]):
            if r.get("max_agent_tool_loops") != base_atl:
                continue
            xs.append(int(r["max_web_research_loops"]))
            ro = r.get("race_overall")
            y_race.append(float(ro) if ro is not None else float("nan"))
            t = r.get("avg_est_tokens")
            y_tok.append(float(t) if t is not None else float("nan"))
        return xs, y_race, y_tok

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    if mode == "sweep_agent":
        xs, y_race, y_tok = series_for_sweep_agent()
        ax1.set_xlabel("MAX_AGENT_TOOL_LOOPS（子研究智能体工具轮次上限）")
        title = f"敏感性：扫 ATL，固定 MAX_WEB_RESEARCH_LOOPS={base_wrl}"
    elif mode == "sweep_web":
        xs, y_race, y_tok = series_for_sweep_web()
        ax1.set_xlabel("MAX_WEB_RESEARCH_LOOPS（督导研究迭代上限，对应 --max_loops）")
        title = f"敏感性：扫 WRL，固定 MAX_AGENT_TOOL_LOOPS={base_atl}"
    elif mode == "grid":
        atl_vals = sorted({int(r["max_agent_tool_loops"]) for r in rows})
        wrl_vals = sorted({int(r["max_web_research_loops"]) for r in rows})
        has_race = any(r.get("race_overall") is not None for r in rows)
        mat = [[float("nan") for _ in wrl_vals] for _ in atl_vals]
        if has_race:
            for r in rows:
                if r.get("race_overall") is None:
                    continue
                i = atl_vals.index(int(r["max_agent_tool_loops"]))
                j = wrl_vals.index(int(r["max_web_research_loops"]))
                mat[i][j] = float(r["race_overall"])
            cmap, cbar_label, png_name, gtitle = (
                "viridis",
                "RACE Overall",
                "ablation_grid_race.png",
                "网格消融：RACE Overall（颜色）",
            )
        else:
            for r in rows:
                i = atl_vals.index(int(r["max_agent_tool_loops"]))
                j = wrl_vals.index(int(r["max_web_research_loops"]))
                t = r.get("avg_est_tokens")
                mat[i][j] = float(t) if t is not None else float("nan")
            cmap, cbar_label, png_name, gtitle = (
                "plasma",
                "Avg est. tokens / task",
                "ablation_grid_tokens.png",
                "网格消融：约估 token（未跑 RACE）",
            )
        im = ax1.imshow(mat, aspect="auto", cmap=cmap)
        ax1.set_xticks(range(len(wrl_vals)), labels=[str(x) for x in wrl_vals])
        ax1.set_yticks(range(len(atl_vals)), labels=[str(x) for x in atl_vals])
        ax1.set_xlabel("MAX_WEB_RESEARCH_LOOPS")
        ax1.set_ylabel("MAX_AGENT_TOOL_LOOPS")
        plt.colorbar(im, ax=ax1, label=cbar_label)
        ax1.set_title(gtitle)
        heatmap_path = out_root / png_name
        fig.tight_layout()
        fig.savefig(heatmap_path, dpi=150)
        plt.close(fig)
        print(f"已保存 {heatmap_path}")
        return
    elif mode == "pairs":
        plt.close(fig)
        figp, axp = plt.subplots(figsize=(7, 6))
        ats = [int(r["max_agent_tool_loops"]) for r in rows]
        wrs = [int(r["max_web_research_loops"]) for r in rows]
        tokf: List[float] = []
        for r in rows:
            t = r.get("avg_est_tokens")
            tokf.append(float(t) if t is not None else float("nan"))
        sc = axp.scatter(
            ats,
            wrs,
            c=tokf,
            cmap="plasma",
            s=140,
            edgecolors="black",
            linewidths=0.6,
        )
        plt.colorbar(sc, ax=axp, label="Avg est. tokens / task")
        axp.set_xlabel("MAX_AGENT_TOOL_LOOPS")
        axp.set_ylabel("MAX_WEB_RESEARCH_LOOPS")
        axp.set_title("自定义 (ATL, WRL) 组合 — 颜色 = 约估 token / 任务")
        axp.set_xticks(sorted(set(ats)))
        axp.set_yticks(sorted(set(wrs)))
        axp.grid(True, alpha=0.3)
        pair_path = out_root / "ablation_pairs.png"
        figp.tight_layout()
        figp.savefig(pair_path, dpi=150)
        plt.close(figp)
        print(f"已保存 {pair_path}")
        return
    else:
        print(f"未知 mode={mode}", file=sys.stderr)
        sys.exit(1)

    if not xs:
        print("无可用数据点（检查 manifest 与 baseline 固定值）", file=sys.stderr)
        sys.exit(1)

    has_race = any(not math.isnan(v) for v in y_race)
    color_race = "#1f77b4"
    color_tok = "#ff7f0e"

    if has_race:
        ax1.plot(xs, y_race, "o-", color=color_race, label="RACE Overall")
        ax1.set_ylabel("RACE Overall Score", color=color_race)
        ax1.tick_params(axis="y", labelcolor=color_race)
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        ax2 = ax1.twinx()
        ax2.plot(xs, y_tok, "s--", color=color_tok, label="Avg est. tokens / task")
        ax2.set_ylabel("平均每任务约估 token（输出正文）", color=color_tok)
        ax2.tick_params(axis="y", labelcolor=color_tok)
        lines1, lab1 = ax1.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, lab1 + lab2, loc="best")
    else:
        ax1.plot(xs, y_tok, "s-", color=color_tok, label="Avg est. tokens / task")
        ax1.set_ylabel("平均每任务约估 token（输出正文）")
        ax1.set_title(title + "（未跑 RACE，仅 token）")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best")

    fig.tight_layout()
    safe = re.sub(r"[^\w\-]+", "_", mode)
    png = out_root / f"ablation_{safe}.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"已保存 {png}")


def main() -> None:
    p = argparse.ArgumentParser(description="消融汇总 / 双轴图")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("summarize", help="追加一行 manifest（由 shell 调用）")
    s.add_argument("--output-dir", required=True)
    s.add_argument("--tag", required=True)
    s.add_argument("--atl", type=int, required=True)
    s.add_argument("--wrl", type=int, required=True)
    s.add_argument("--model-name", required=True)
    s.add_argument("--race-overall", default="")
    s.add_argument("--manifest", required=True)
    s.set_defaults(func=cmd_summarize)

    g = sub.add_parser("plot", help="读取 manifest 绘图")
    g.add_argument("--manifest", required=True)
    g.add_argument("--ablation-root", required=True)
    g.add_argument("--mode", required=True)
    g.add_argument("--baseline-atl", type=int, default=3)
    g.add_argument("--baseline-wrl", type=int, default=3)
    g.set_defaults(func=cmd_plot)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
