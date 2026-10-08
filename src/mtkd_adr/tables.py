"""Turn result JSON files into a long CSV and LaTeX tables (no jinja2 needed)."""
from __future__ import annotations

import argparse
import json
import os
from typing import List

import pandas as pd

METRICS = ["accuracy", "attack_recall", "benign_recall", "attack_precision", "macro_f1", "perturbed_fraction"]


def flatten(results: dict) -> pd.DataFrame:
    """One row per (dataset, arch, model, threat_model, attack, epsilon)."""
    dataset = results["_meta"]["dataset"]
    rows = []
    for arch, arch_res in results.items():
        if arch == "_meta":
            continue
        for group in ("models", "teachers"):
            for model_name, res in arch_res[group].items():
                base = {"dataset": dataset, "arch": arch, "model": model_name}
                clean = res["clean"]
                rows.append({**base, "threat_model": "none", "attack": "clean", "epsilon": 0.0,
                             **{m: clean.get(m) for m in METRICS}})
                for threat, by_attack in res["attacks"].items():
                    for atk, by_eps in by_attack.items():
                        for eps, m in by_eps.items():
                            rows.append({**base, "threat_model": threat, "attack": atk,
                                         "epsilon": float(eps), **{k: m.get(k) for k in METRICS}})
    return pd.DataFrame(rows)


def _latex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def to_latex(table: pd.DataFrame, caption: str, label: str) -> str:
    cols = list(table.columns)
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             "Model & " + " & ".join(_latex_escape(c) for c in cols) + r" \\", r"\midrule"]
    for idx, row in table.iterrows():
        cells = ["--" if pd.isna(v) else f"{v:.3f}" for v in row.values]
        lines.append(_latex_escape(idx) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{{_latex_escape(caption)}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)


def make_tables(result_paths: List[str], out_dir: str, metric: str = "attack_recall") -> None:
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    for p in result_paths:
        with open(p) as f:
            frames.append(flatten(json.load(f)))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(out_dir, "results_long.csv"), index=False)

    tex_parts = []
    for (dataset, arch), g in df.groupby(["dataset", "arch"]):
        g = g.copy()
        g["column"] = g.apply(
            lambda r: "clean" if r["attack"] == "clean" else f"{r['attack']} e={r['epsilon']:g}", axis=1)
        for threat in [t for t in g["threat_model"].unique() if t != "none"]:
            sub = g[(g["threat_model"] == threat) | (g["attack"] == "clean")]
            order = list(dict.fromkeys(sub.sort_values(["attack", "epsilon"])["column"]))
            order = ["clean"] + [c for c in order if c != "clean"]
            pivot = sub.pivot_table(index="model", columns="column", values=metric, aggfunc="first")[order]
            name = f"{dataset}_{arch}_{threat}_{metric}"
            pivot.to_csv(os.path.join(out_dir, f"{name}.csv"))
            tex_parts.append(to_latex(pivot, f"{dataset} {arch}, {threat} threat model, {metric}",
                                      f"tab:{name}"))
    with open(os.path.join(out_dir, f"tables_{metric}.tex"), "w") as f:
        f.write("\n\n".join(tex_parts) + "\n")
    print(f"Wrote results_long.csv, per table CSVs and tables_{metric}.tex to {out_dir}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Build CSV and LaTeX tables from result JSON files")
    ap.add_argument("results", nargs="+", help="one or more *_results.json files")
    ap.add_argument("--out-dir", default="outputs/tables")
    ap.add_argument("--metric", default="attack_recall", choices=METRICS)
    args = ap.parse_args(argv)
    make_tables(args.results, args.out_dir, args.metric)


if __name__ == "__main__":
    main()
