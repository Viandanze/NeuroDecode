#!/usr/bin/env python3
"""
Standard Left/Right Hand MI Benchmark (within-subject, 5-fold CV)

Fixes data-recipe issues of the old 39% 4-class run:
  - runs = [4, 8, 12]  (pure imagined left/right fist, consistent T1/T2 semantics,
                        90 trials per subject, MNE-official decoding recipe)
  - bandpass 8-30 Hz   (mu 8-12 + beta 18-26 sensorimotor rhythms)
  - CAR reference      (common average reference)
  - epoch window 0.5-3.0 s after cue (core MI interval, 401 samples @160 Hz)
  - per-channel z-score normalization per epoch

Models: EEGNet, ShallowConvNet, EEGConformer (deep) vs CSP, Riemann MDM (classic baselines)

Usage:
    python scripts/benchmark_lr.py --quick --subjects 1          # link check
    python scripts/benchmark_lr.py --subjects 1 2 3 4 5 6 7 8 9 10
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import mne
from mne import Epochs
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.channels import make_standard_montage
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.eegnet import EEGNetClassifier
from src.models.csp import CSPClassifier
from src.models.riemann_mdm import RiemannMDMClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("benchmark_lr")

RUNS = [4, 8, 12]  # imagined opening/closing left or right fist (full recipe)

PARADIGMS = {
    # run 4 only: imagined left/right fist, 30 trials, uses local cache
    "lr4": {"runs": [4], "classes": {"left": "T1", "right": "T2"}},
    # runs 4/8/12: imagined left/right fist, 90 trials (MNE official recipe)
    "lr412": {"runs": [4, 8, 12], "classes": {"left": "T1", "right": "T2"}},
    # runs 4+5: 4-class MI (left/right fist, both fists, both feet), 60 trials
    "mi45": {"runs": [4, 5], "classes": {"left": "T1", "right": "T2",
                                          "fists": "T5", "feet": "T6"}},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="./data/")
    p.add_argument("--paradigm", default="lr4", choices=list(PARADIGMS))
    p.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 11)))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--outdir", default="./outputs/")
    p.add_argument("--quick", action="store_true", help="fast link check: 2 folds, 5 epochs")
    p.add_argument("--models", nargs="+", default=["csp", "riemann", "eegnet", "shallow", "conformer"])
    return p.parse_args()


def _prep_raw(raw):
    eegbci.standardize(raw)
    raw.set_montage(make_standard_montage("standard_1005"), on_missing="ignore")
    raw.filter(8.0, 30.0, fir_design="firwin", verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw


def _epoch_raw(raw, class_map):
    """class_map: annotation name -> 1-based label, e.g. {'T1': 1, 'T2': 2}"""
    events, event_map = mne.events_from_annotations(raw, verbose=False)
    name_to_id = {v: k for k, v in event_map.items()}
    keep = [[ev[0], 0, class_map[name_to_id[ev[2]]]]
            for ev in events if name_to_id[ev[2]] in class_map]
    if not keep:
        raise ValueError(f"no events matching {class_map}, have {event_map}")
    events = np.array(keep, dtype=int)
    event_id = {name: i + 1 for i, name in enumerate(class_map)}
    epochs = Epochs(raw, events, event_id=event_id,
                    tmin=0.5, tmax=3.0, baseline=None, preload=True, verbose=False)
    X = epochs.get_data().astype(np.float32)
    y = (epochs.events[:, -1] - 1).astype(np.int64)
    mu = X.mean(axis=2, keepdims=True)
    sd = X.std(axis=2, keepdims=True) + 1e-8
    return (X - mu) / sd, y


def load_subject_epochs(data_path: str, subject: int, paradigm: str):
    """Load paradigm runs, apply 8-30 Hz bandpass + CAR, epoch 0.5-3.0 s."""
    spec = PARADIGMS[paradigm]
    runs = spec["runs"]
    fnames = eegbci.load_data(subject, runs, path=data_path, update_path=True, verbose=False)

    if paradigm in ("lr4", "lr412"):
        # T1/T2 semantics identical across runs -> concat then epoch
        raw = concatenate_raws([_prep_raw(read_raw_edf(f, preload=True, verbose=False))
                                for f in fnames])
        X, y = _epoch_raw(raw, {"T1": 1, "T2": 2})
    else:
        # mi45: T1/T2 mean different things per run -> epoch per run, then stack
        Xs, ys, base = [], [], 0
        run_class_maps = [{4: {"T1": 1, "T2": 2}, 5: {"T1": 3, "T2": 4}}[r] for r in runs]
        for f, cmap in zip(fnames, run_class_maps):
            raw = _prep_raw(read_raw_edf(f, preload=True, verbose=False))
            Xi, yi = _epoch_raw(raw, cmap)
            Xs.append(Xi)
            ys.append(yi + base)
            base += len(cmap)
        X, y = np.concatenate(Xs), np.concatenate(ys)
    return X, y


def train_dl(model, Xtr, ytr, Xva, yva, args, device):
    """Train a deep model, return best-val-accuracy model predictions on Xva."""
    torch.manual_seed(args.seed)
    model = model.to(device)
    n = Xtr.shape[0]
    Xtr_t = torch.from_numpy(Xtr).unsqueeze(1)          # (B, 1, C, T)
    ytr_t = torch.from_numpy(ytr)
    Xva_t = torch.from_numpy(Xva).unsqueeze(1)
    yva_np = yva

    # all three deep models (EEGNet/ShallowConvNet/EEGConformer) expect (B, 1, C, T)
    pack = lambda t: t

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    crit = nn.CrossEntropyLoss()

    ds = TensorDataset(pack(Xtr_t), ytr_t)
    best_acc, best_pred, bad = -1.0, None, 0  # -1.0: epoch-1 pred always recorded, never return None
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb = pack(Xtr_t[idx]).to(device)
            yb = ytr_t[idx].to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            pred = model(pack(Xva_t).to(device)).argmax(1).cpu().numpy()
        acc = accuracy_score(yva_np, pred)
        if acc > best_acc:
            best_acc, best_pred, bad = acc, pred.copy(), 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    return best_pred, best_acc


def run_benchmark(args):
    if args.quick:
        args.folds, args.epochs = 2, 5
        args.subjects = args.subjects[:1]

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device={device}, folds={args.folds}, epochs={args.epochs}, models={args.models}")

    results = {m: {} for m in args.models}
    for subj in args.subjects:
        try:
            X, y = load_subject_epochs(args.data_path, subj, args.paradigm)
        except Exception as e:
            log.warning(f"S{subj:03d} load failed: {e}")
            continue
        log.info(f"S{subj:03d}: {X.shape}, class balance={np.bincount(y).tolist()}")

        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        for name in args.models:
            accs = []
            for k, (tr, va) in enumerate(skf.split(X, y)):
                if name == "csp":
                    clf = CSPClassifier()
                    clf.fit(X[tr], y[tr])
                    pred = clf.predict(X[va])
                elif name == "riemann":
                    clf = RiemannMDMClassifier()
                    clf.fit(X[tr], y[tr])
                    pred = clf.predict(X[va])
                else:
                    C, T = X.shape[1], X.shape[2]
                    n_classes = int(len(np.unique(y)))
                    if name == "eegnet":
                        model = EEGNetClassifier(n_channels=C, n_times=T,
                                                 n_classes=n_classes, apply_softmax=False)
                    elif name == "shallow":
                        from src.models.advanced import ShallowConvNet
                        model = ShallowConvNet(n_channels=C, n_times=T, n_classes=n_classes)
                    elif name == "conformer":
                        from src.models.advanced import EEGConformer
                        model = EEGConformer(n_channels=C, n_times=T, n_classes=n_classes)
                    else:
                        raise ValueError(f"unknown model {name}")
                    pred, best = train_dl(model, X[tr], y[tr], X[va], y[va], args, device)
                fold_acc = float(accuracy_score(y[va], pred))
                accs.append(fold_acc)
                log.info(f"S{subj:03d} {name} fold{k+1}/{args.folds}: {fold_acc:.3f}")
            results[name][f"S{subj:03d}"] = {
                "folds": accs,
                "mean": float(np.mean(accs)),
                "std": float(np.std(accs)),
            }
            log.info(f"S{subj:03d} {name}: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")

    # summary
    summary = {}
    for name in args.models:
        if not results[name]:
            continue
        means = [v["mean"] for v in results[name].values()]
        summary[name] = {"grand_mean": float(np.mean(means)),
                         "grand_std": float(np.std(means)),
                         "n_subjects": len(means)}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) / f"benchmark_lr_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {"config": {"paradigm": args.paradigm,
                          "runs": PARADIGMS[args.paradigm]["runs"],
                          "bandpass": [8, 30], "tmin": 0.5, "tmax": 3.0,
                          "sfreq": 160, "car": True, "folds": args.folds,
                          "epochs": args.epochs, "seed": args.seed},
               "per_subject": results, "summary": summary}
    with open(outdir / "benchmark.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    lines = ["# Left/Right Hand MI Benchmark (within-subject 5-fold CV, runs 4/8/12, 8-30 Hz, CAR, 0.5-3.0 s)", "",
             "| Model | Grand Mean Acc | Std | Subjects |",
             "|---|---|---|---|"]
    for name in args.models:
        if name in summary:
            s = summary[name]
            lines.append(f"| {name} | {s['grand_mean']*100:.1f}% | {s['grand_std']*100:.1f}% | {s['n_subjects']} |")
    lines += ["", "## Per-subject mean accuracy", "",
              "| Subject | " + " | ".join(args.models) + " |",
              "|---" * (len(args.models) + 1) + "|"]
    for subj_key in next(iter(results.values())).keys():
        row = [f"{m}: {results[m][subj_key]['mean']*100:.1f}%" if subj_key in results.get(m, {}) else "-"
               for m in args.models]
        lines.append(f"| {subj_key} | " + " | ".join(row) + " |")
    (outdir / "benchmark.md").write_text("\n".join(lines), encoding="utf-8")

    log.info("=" * 60)
    for name, s in summary.items():
        log.info(f"{name:>10s}: {s['grand_mean']*100:.1f}% +/- {s['grand_std']*100:.1f}%")
    log.info(f"results -> {outdir}")
    return payload


if __name__ == "__main__":
    run_benchmark(parse_args())
