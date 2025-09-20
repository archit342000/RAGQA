"""Aggregate evaluation runs into tables and an HTML report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from .metrics import paired_bootstrap_ci, win_loss_tie


def load_run(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    data["_path"] = str(path)
    return data


def items_dataframe(run: Dict[str, object]) -> pd.DataFrame:
    df = pd.DataFrame(run.get("items", []))
    df["engine"] = run["engine"]
    df["path"] = run.get("_path")
    df["tags"] = df["tags"].apply(lambda tags: tags if isinstance(tags, list) else [])
    return df


def summarise_run(run: Dict[str, object]) -> Dict[str, object]:
    row: Dict[str, object] = {"engine": run["engine"], "path": run.get("_path")}
    row.update(run.get("aggregates", {}))
    latency = run.get("latency", {})
    for stage, stats in latency.items():
        for stat_name, value in stats.items():
            row[f"latency_{stage}_{stat_name}"] = value
    return row


def per_tag_table(df: pd.DataFrame, post_metric: str, pre_metric: str, tags: List[str]) -> pd.DataFrame:
    exploded = df.explode("tags")
    exploded["tags"] = exploded["tags"].fillna("untagged")
    groups = exploded.groupby(["engine", "tags"])
    summary = groups[[pre_metric, post_metric, "mrr_at_10", "ndcg_at_10", "context_precision"]].mean().reset_index()
    summary.rename(columns={pre_metric: f"mean_{pre_metric}", post_metric: f"mean_{post_metric}"}, inplace=True)
    return summary


def paired_comparisons(engine_frames: Dict[str, pd.DataFrame], metric: str) -> pd.DataFrame:
    rows = []
    engines = list(engine_frames.keys())
    for i in range(len(engines)):
        for j in range(i + 1, len(engines)):
            a = engines[i]
            b = engines[j]
            merged = pd.merge(engine_frames[a][["id", metric]], engine_frames[b][["id", metric]], on="id", suffixes=(f"_{a}", f"_{b}"))
            metrics_a = merged[f"{metric}_{a}"].to_numpy()
            metrics_b = merged[f"{metric}_{b}"].to_numpy()
            stats = paired_bootstrap_ci(metrics_a, metrics_b)
            wlt = win_loss_tie(metrics_a, metrics_b)
            row = {
                "engine_a": a,
                "engine_b": b,
                "metric": metric,
                "diff_mean": stats["diff_mean"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "p_value": stats["p_value"],
                **wlt,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def render_html(template_dir: Path, context: Dict[str, object], out_path: Path) -> None:
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    tmpl = env.get_template("report.html.j2")
    out_path.write_text(tmpl.render(**context), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate evaluation runs")
    parser.add_argument("--runs", nargs="+", help="Paths or glob patterns to run JSON files")
    parser.add_argument("--out", type=str, default="report.html", help="Destination HTML report path")
    return parser.parse_args()


def expand_runs(patterns: List[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        globbed = list(Path().glob(pattern))
        if globbed:
            paths.extend(globbed)
        else:
            candidate = Path(pattern)
            if candidate.exists():
                paths.append(candidate)
    if not paths:
        raise ValueError("No evaluation runs found for given patterns")
    return sorted({path.resolve() for path in paths})


def main() -> None:
    args = parse_args()
    run_paths = expand_runs(args.runs)
    runs = [load_run(path) for path in run_paths]

    item_frames = [items_dataframe(run) for run in runs]
    concatenated = pd.concat(item_frames, ignore_index=True)
    post_metric = "post_hit_at_k"
    pre_metric = "pre_hit_at_k"

    summary_rows = [summarise_run(run) for run in runs]
    summary_df = pd.DataFrame(summary_rows)

    per_tag_df = per_tag_table(concatenated, post_metric, pre_metric, [])

    engine_frames = {engine: df for engine, df in concatenated.groupby("engine")}
    comparisons_df = paired_comparisons(engine_frames, metric=post_metric)

    out_path = Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "summary.csv"
    per_tag_csv = out_dir / "per_tag.csv"
    comparisons_csv = out_dir / "comparisons.csv"

    summary_df.to_csv(summary_csv, index=False)
    per_tag_df.to_csv(per_tag_csv, index=False)
    comparisons_df.to_csv(comparisons_csv, index=False)

    context = {
        "generated_at": datetime.utcnow().isoformat(),
        "summary": summary_df.to_dict(orient="records"),
        "per_tag": per_tag_df.to_dict(orient="records"),
        "comparisons": comparisons_df.to_dict(orient="records"),
    }
    template_dir = Path("templates")
    render_html(template_dir, context, out_path)


if __name__ == "__main__":  # pragma: no cover
    main()
