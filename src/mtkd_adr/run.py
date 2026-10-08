"""Command line entry point: train and evaluate on one dataset."""
from __future__ import annotations

import argparse
import json
import os
import platform
import time

import foolbox
import torch

from .config import load_config
from .data.preprocess import prepare_data
from .pipeline import run_architecture
from .utils import NumpyEncoder, get_device, set_seed


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="MTKD ADR: robust intrusion detection experiments")
    ap.add_argument("--config", required=True, help="dataset YAML, e.g. configs/cicids2017.yaml")
    ap.add_argument("--raw-dir", help="folder with the raw dataset CSVs (searched recursively)")
    ap.add_argument("--arch", help="comma separated subset of: cnn,lstm,clstm")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--sample-size", type=int, help="max total rows kept after subsampling")
    ap.add_argument("--sample-mode", choices=["stratified", "balanced"])
    ap.add_argument("--rows-per-file", type=int, help="cap rows read from each CSV file")
    ap.add_argument("--out-dir", help="where checkpoints and results are written")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--eval-only", action="store_true", help="load saved checkpoints and only evaluate")
    ap.add_argument("--force", action="store_true", help="retrain even if checkpoints exist")
    return ap


def overrides_from_args(args) -> dict:
    o = {"data": {}, "train": {}}
    if args.raw_dir is not None:
        o["data"]["raw_dir"] = args.raw_dir
    if args.sample_size is not None:
        o["data"]["sample_size"] = args.sample_size
    if args.sample_mode is not None:
        o["data"]["sample_mode"] = args.sample_mode
    if args.rows_per_file is not None:
        o["data"]["rows_per_file"] = args.rows_per_file
    if args.epochs is not None:
        o["train"]["epochs"] = args.epochs
    if args.arch is not None:
        o["archs"] = [a.strip() for a in args.arch.split(",") if a.strip()]
    if args.out_dir is not None:
        o["out_dir"] = args.out_dir
    if args.seed is not None:
        o["seed"] = args.seed
    return o


def main(argv=None) -> str:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, overrides_from_args(args))
    device = get_device()
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    print(f"Device: {device} | seed: {cfg.seed} | archs: {cfg.archs}")

    t0 = time.time()
    data = prepare_data(cfg.data, cfg.seed)
    print(f"Preprocessing took {time.time() - t0:.1f}s")

    all_results = {}
    for arch in cfg.archs:
        t1 = time.time()
        set_seed(cfg.seed)  # every architecture starts from the same seed
        all_results[arch] = run_architecture(arch, data, cfg, device, args.eval_only, args.force)
        print(f"[{arch}] total time: {(time.time() - t1) / 60:.1f} min")

    out = {
        "_meta": {
            "dataset": cfg.data.name,
            "N_attack": data.n_attack,
            "N_benign": data.n_benign,
            "N_total": data.n_attack + data.n_benign,
            "sample_mode": cfg.data.sample_mode,
            "sample_size_requested": cfg.data.sample_size,
            "fingerprint": data.fingerprint,
            "epochs": cfg.train.epochs,
            "mape_threshold": cfg.attack.mape_threshold,
            "perturbation_constrained_to": [data.feature_names[i] for i in data.feature_indices],
            "n_features": data.n_features,
            "seed": cfg.seed,
            "device": str(device) + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
            "torch": torch.__version__,
            "foolbox": foolbox.__version__,
            "python": platform.python_version(),
            "config": cfg.to_dict(),
        },
        **all_results,
    }
    out_path = os.path.join(cfg.out_dir, f"{cfg.data.name}_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    print(f"\nTotal wall time: {(time.time() - t0) / 60:.1f} min")
    print(f"Results written to {out_path}")
    return out_path


if __name__ == "__main__":
    main()
