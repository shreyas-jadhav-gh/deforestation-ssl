"""
Fine-tune a U-Net (ResNet-18 encoder, all 13 Sentinel-2 bands) for forest-loss segmentation,
with different encoder initialisations and label fractions, and evaluate on:
    amazon_val   (model selection)
    amazon_test  (same region, unseen area)
    congo_test   (cross-region generalisation)

Encoder initialisations (--inits):
    scratch       random initialisation (the supervised baseline)
    ssl4eo        MoCo on SSL4EO-S12 (public weights, 13 bands, the same ones TorchGeo uses)
    imagenet      supervised ImageNet weights (first layer adapted to 13 bands by the library)
    NAME=PATH     your own encoder checkpoint, e.g. moco_both=/kaggle/working/.../encoder.pt
                  (PATH may be find:pretrain/moco_both/encoder.pt to search the results folder and
                  the inputs, e.g. an earlier notebook version's output attached on Kaggle)

Input (--input): post = the 2024 image only (default); both = 2024 and 2019 images, a
Siamese change-detection U-Net with a shared encoder (see SiameseDiffUnet).

Examples (from the repository root):
    python src/train_seg.py --inits scratch ssl4eo --fracs 0.01 0.05 0.1 0.25 1.0 --seeds 0 1 2
    python src/train_seg.py --inits scratch --fracs 0.05 --seeds 0 --steps 30 --eval-every 15 \
        --batch-size 4 --max-patches 24 --tag smoke          # quick test

Model selection uses the best IoU over all probability thresholds on Amazon validation; that
threshold is then fixed and applied to the test sets (columns ending in _t). Columns without
_t use a threshold of 0.5, and *_ap is the threshold-free average precision.

Every finished run is appended to results/seg_results.csv (or --results-file). Runs already in
that file (same tag, init, learning rate, fraction, seed and steps) are skipped, so an
interrupted job can be resumed. Two processes can run side by side on Kaggle's two GPUs with --device cuda:0 / cuda:1,
as long as each writes its own --results-file.
"""

import argparse
import datetime
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

SSL4EO_URL = ("https://hf.co/torchgeo/resnet18_sentinel2_all_moco/resolve/"
              "5b8cddc9a14f3844350b7f40b85bcd32aed75918/resnet18_sentinel2_all_moco-59bfdff9.pth")
EVAL_SETS = [("amazon_val", "amazon", "val"),
             ("amazon_test", "amazon", "test"),
             ("congo_test", "congo", "test")]
SELECT_SET = "amazon_val"


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
def load_encoder(encoder, state):
    """Load ResNet-18 weights into the U-Net encoder, failing loudly if anything is missing."""
    state = {k.removeprefix("module.").removeprefix("encoder."): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if not k.startswith("fc.")}
    # call nn.Module's version directly: smp's override does not return the key report
    report = nn.Module.load_state_dict(encoder, state, strict=False)
    missing = [k for k in report.missing_keys if not k.endswith("num_batches_tracked")]
    if missing or report.unexpected_keys:
        raise RuntimeError(f"encoder weights do not match: missing {missing[:5]}, "
                           f"unexpected {report.unexpected_keys[:5]}")
    print(f"  loaded {len(state)} encoder tensors")


def build_model(init):
    name, _, path = init.partition("=")
    if name == "imagenet":
        return smp.Unet("resnet18", encoder_weights="imagenet", in_channels=len(C.BANDS), classes=2)
    model = smp.Unet("resnet18", encoder_weights=None, in_channels=len(C.BANDS), classes=2)
    if name == "scratch":
        return model
    if name == "ssl4eo":
        state = torch.hub.load_state_dict_from_url(SSL4EO_URL, map_location="cpu", progress=False,
                                                   check_hash=True, weights_only=True)
    elif path:
        if path.startswith("find:"):   # e.g. find:pretrain/moco_both/encoder.pt
            path = str(C.find_path(path[len("find:"):]))
            print(f"  using encoder {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        state = ckpt.get("encoder", ckpt)
    else:
        raise ValueError(f"unknown init '{init}' (use scratch, ssl4eo, imagenet or NAME=PATH)")
    load_encoder(model.encoder, state)
    return model


class SiameseDiffUnet(nn.Module):
    """Change detection from two dates (input: 2024 bands then 2019 bands, 26 channels).
    One shared encoder looks at each date; the U-Net decoder sees the difference of their
    features at every scale (the 'FC-Siam-diff' idea, Daudt et al. 2018). Because the encoder is
    the ordinary 13-band one, every pretrained initialisation can be used unchanged."""

    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        params = inspect.signature(unet.decoder.forward).parameters.values()
        self.star_args = any(p.kind == p.VAR_POSITIONAL for p in params)   # differs between smp versions

    def forward(self, x):
        n = len(C.BANDS)
        both = self.unet.encoder(torch.cat([x[:, :n], x[:, n:]], 0))      # one pass for both dates
        feats = [f[: len(x)] - f[len(x):] for f in both]                    # 2024 minus 2019
        out = self.unet.decoder(*feats) if self.star_args else self.unet.decoder(feats)
        return self.unet.segmentation_head(out)


# -----------------------------------------------------------------------------
# Training pieces
# -----------------------------------------------------------------------------
def batch_input(data, idx):
    """uint16 model input for the patches `idx`: 2024 bands, plus 2019 bands if they were loaded."""
    if "pre" in data:
        return np.concatenate([data["post"][idx], data["pre"][idx]], axis=1)
    return data["post"][idx]


def to_input(images_uint16, device):
    """uint16 patches (B, C, H, W) -> float tensor on device, divided by 10000."""
    x = torch.from_numpy(images_uint16.astype(np.float32))
    return x.to(device, non_blocking=True) / C.SCALE


def augment(x, y, rng):
    """Random rotation by a multiple of 90 degrees plus optional flip (8 symmetries), per patch."""
    xs, ys = [], []
    for i, k in enumerate(rng.integers(0, 8, size=x.shape[0])):
        xi, yi = x[i], y[i]
        if k >= 4:
            xi, yi = xi.flip(-1), yi.flip(-1)
        if k % 4:
            xi, yi = torch.rot90(xi, int(k % 4), (-2, -1)), torch.rot90(yi, int(k % 4), (-2, -1))
        xs.append(xi)
        ys.append(yi)
    return torch.stack(xs), torch.stack(ys)


def seg_loss(logits, y):
    """Cross-entropy + Dice for the 'cleared' class; ignore pixels (255) are left out of both."""
    logits = logits.float()
    valid = y != C.IGNORE
    if not valid.any():
        return logits.sum() * 0.0
    ce = F.cross_entropy(logits, y, ignore_index=C.IGNORE)
    p = logits.softmax(1)[:, 1][valid]
    t = (y[valid] == 1).float()
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
    return ce + dice


@torch.no_grad()
def evaluate(model, data, device, batch_size, amp):
    """Returns per-patch pixel counts (tp, fp, fn, tn) at probability 0.5, and a histogram of
    predicted 'cleared' probabilities (row 0: forest pixels, row 1: cleared pixels) for
    threshold-free metrics."""
    model.eval()
    labels = data["label"]
    counts = np.zeros((len(labels), 4), dtype=np.int64)
    hist = torch.zeros(2, C.NBINS, dtype=torch.int64, device=device)
    for s in range(0, len(labels), batch_size):
        x = to_input(batch_input(data, slice(s, s + batch_size)), device)
        y = torch.from_numpy(labels[s:s + batch_size].astype(np.int64)).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
        prob = logits.float().softmax(1)[:, 1]
        valid = y != C.IGNORE
        true, false = (y == 1) & valid, (y == 0) & valid
        bins = (prob * C.NBINS).long().clamp_(0, C.NBINS - 1)
        hist[0] += torch.bincount(bins[false], minlength=C.NBINS)
        hist[1] += torch.bincount(bins[true], minlength=C.NBINS)
        pos = prob >= 0.5
        c = torch.stack([(pos & true).sum((1, 2)), (pos & false).sum((1, 2)),
                         (~pos & true).sum((1, 2)), (~pos & false).sum((1, 2))], dim=1)
        counts[s:s + len(x)] = c.cpu().numpy()
    model.train()
    return counts, hist.cpu().numpy()


def summarise(hist, k_val):
    """Metrics at 0.5, at the validation-tuned threshold k_val, and average precision."""
    out = {}
    for k, v in C.metrics_at_threshold(hist, C.NBINS // 2).items():
        out[k] = v
    for k, v in C.metrics_at_threshold(hist, k_val).items():
        out[f"{k}_t"] = v
    out["ap"] = C.average_precision(hist)
    return out


def lr_factor(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


# -----------------------------------------------------------------------------
# One run
# -----------------------------------------------------------------------------
def run_one(init, lr, frac, seed, train, evals, args, device, out):
    init_name = init.partition("=")[0]
    run = f"{args.tag}_{init_name}{'_cd' if args.input == 'both' else ''}_lr{lr:g}_f{frac:g}_s{seed}"
    amp = device.type == "cuda" and not args.no_amp
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    subset = C.nested_subset(len(train["post"]), frac, seed)
    lab = train["label"][subset]
    cleared_share = (lab == 1).sum() / max(1, (lab != C.IGNORE).sum())
    print(f"\n=== {run}: {len(subset)} training patches "
          f"({cleared_share:.1%} of valid pixels cleared) ===")

    model = build_model(init)
    if args.input == "both":
        model = SiameseDiffUnet(model)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(s, args.steps, args.warmup))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    stream = C.batch_stream(subset, args.batch_size, rng)

    best = {"iou": -1.0, "step": 0, "state": None}
    curve, losses = [], []
    t0 = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    for step in range(1, args.steps + 1):
        idx = next(stream)
        x = to_input(batch_input(train, idx), device)
        y = torch.from_numpy(train["label"][idx].astype(np.int64)).to(device)
        x, y = augment(x, y, rng)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
        loss = seg_loss(logits, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        losses.append(loss.item())

        if step % 50 == 0:
            rate = step * args.batch_size / (time.time() - t0)
            print(f"  step {step}/{args.steps}  loss {np.mean(losses[-50:]):.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {rate:.0f} img/s")
        if step % args.eval_every == 0 or step == args.steps:
            # model selection: best IoU on Amazon validation over all thresholds
            counts, hist = evaluate(model, evals[SELECT_SET], device, args.eval_batch_size, amp)
            k = C.best_threshold(hist)
            m = C.metrics_at_threshold(hist, k)
            m05 = C.metrics_at_threshold(hist, C.NBINS // 2)
            ap = C.average_precision(hist)
            curve.append({"step": step, "train_loss": float(np.mean(losses[-args.eval_every:])),
                          "val_iou_t": m["iou"], "val_threshold": k / C.NBINS,
                          "val_iou": m05["iou"], "val_ap": ap})
            better = m["iou"] > best["iou"] or best["state"] is None
            if better:
                best = {"iou": m["iou"], "step": step, "k": k,
                        "state": {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}}
            print(f"  [eval] step {step}: val IoU {m['iou']:.4f} at threshold {k / C.NBINS:.3f} "
                  f"(IoU {m05['iou']:.4f} at 0.5)  AP {ap:.4f}{'  (best so far)' if better else ''}")
    train_min = (time.time() - t0) / 60

    # final evaluation of the best checkpoint on every set, using the threshold tuned on
    # Amazon validation (fixed before looking at the test sets) and at 0.5
    model.load_state_dict(best["state"])
    row = {"tag": args.tag, "init": init_name, "input": args.input, "lr": lr, "frac": frac, "seed": seed,
           "steps": args.steps, "n_train": len(subset), "best_step": best["step"],
           "threshold": best["k"] / C.NBINS}
    per_patch, hists = [], {}
    for set_name, _, _ in EVAL_SETS:
        counts, hist = evaluate(model, evals[set_name], device, args.eval_batch_size, amp)
        hists[set_name] = hist
        for metric, value in summarise(hist, best["k"]).items():
            row[f"{set_name}_{metric}"] = round(float(value), 5)
        per_patch.append(pd.DataFrame(counts, columns=["tp", "fp", "fn", "tn"])
                         .assign(patch_id=evals[set_name]["ids"], set=set_name))
    row.update({"train_min": round(train_min, 2), "batch_size": args.batch_size,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                "peak_gpu_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
                if device.type == "cuda" else 0,
                "finished": datetime.datetime.now().isoformat(timespec="seconds")})
    print(f"  done in {train_min:.1f} min (best step {best['step']}, threshold {row['threshold']:.3f}): "
          + "  ".join(f"{s} IoU {row[f'{s}_iou_t']:.4f} AP {row[f'{s}_ap']:.4f}"
                      for s, _, _ in EVAL_SETS))

    # save results
    csv = out / args.results_file
    rows = pd.DataFrame([row])
    if csv.exists():   # rewrite the whole file so rows with different columns stay aligned
        rows = pd.concat([pd.read_csv(csv), rows], ignore_index=True)
    tmp = csv.with_name(csv.name + ".tmp")
    rows.to_csv(tmp, index=False)
    os.replace(tmp, csv)
    (out / "per_patch").mkdir(exist_ok=True)
    pd.concat(per_patch).to_csv(out / "per_patch" / f"{run}.csv", index=False)
    (out / "curves").mkdir(exist_ok=True)
    (out / "curves" / f"{run}.json").write_text(json.dumps(curve))
    (out / "hists").mkdir(exist_ok=True)
    np.savez_compressed(out / "hists" / f"{run}.npz", **hists)
    if seed in args.save_seeds:
        (out / "checkpoints").mkdir(exist_ok=True)
        torch.save({k: v.half() if v.is_floating_point() else v for k, v in best["state"].items()},
                   out / "checkpoints" / f"{run}.pt")
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inits", nargs="+", default=["scratch"])
    p.add_argument("--fracs", nargs="+", type=float, default=[1.0])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--lrs", nargs="+", type=float, default=[5e-4],
                   help="learning rate(s); several values = a learning-rate search")
    p.add_argument("--lr-from-tag", default=None,
                   help="use, for each init, the learning rate with the best mean Amazon-validation "
                        "IoU among earlier runs with this tag in the results file (ignores --lrs)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=150)
    p.add_argument("--tag", default="main", help="label for this group of runs")
    p.add_argument("--results-file", default="seg_results.csv",
                   help="results CSV name; use a different one for each process running in parallel")
    p.add_argument("--save-seeds", nargs="*", type=int, default=[0],
                   help="save the best model for these seeds")
    p.add_argument("--max-patches", type=int, default=0,
                   help="only use the first N patches of each set (quick tests)")
    p.add_argument("--data-root", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--input", choices=["post", "both"], default="post",
                   help="post: the 2024 image only; both: 2024 and 2019 (change detection)")
    args = p.parse_args()

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    out = C.results_dir()
    out.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    print(f"results go to: {out}")

    csv = out / args.results_file

    # learning rate per initialisation: given (--lrs) or taken from an earlier search (--lr-from-tag)
    lrs_for = {i: args.lrs for i in args.inits}
    if args.lr_from_tag:
        if not csv.exists():
            raise SystemExit(f"--lr-from-tag: {csv} not found")
        found = pd.read_csv(csv)
        for i in args.inits:
            name = i.partition("=")[0]
            rows = found[(found.tag == args.lr_from_tag) & (found.init == name)]
            if rows.empty:
                raise SystemExit(f"--lr-from-tag: no '{args.lr_from_tag}' runs for {name} in {csv}")
            per_lr = rows.groupby("lr").amazon_val_iou_t.mean()
            lrs_for[i] = [float(per_lr.idxmax())]
            print(f"learning rate for {name} from '{args.lr_from_tag}' runs: {lrs_for[i][0]:g} "
                  f"(mean Amazon validation IoU per learning rate: "
                  f"{', '.join(f'{lr:g}: {v:.4f}' for lr, v in per_lr.items())})")

    # which runs still need doing (seed by seed, so a partial job still gives complete seeds)
    todo = [(i, lr, f, s) for s in args.seeds for i in args.inits for lr in lrs_for[i]
            for f in args.fracs]
    if csv.exists():
        done = pd.read_csv(csv)
        done_input = done["input"].fillna("post") if "input" in done else pd.Series("post", index=done.index)
        done_keys = set(zip(done.tag, done.init, done_input, done.lr.round(9), done.frac.round(6),
                            done.seed, done.steps))
        todo = [(i, lr, f, s) for i, lr, f, s in todo
                if (args.tag, i.partition("=")[0], args.input, round(lr, 9), round(f, 6), s, args.steps)
                not in done_keys]
    print(f"{len(todo)} runs to do")
    if not todo:
        return

    # load the data once for all runs
    index = C.labeled_index(args.data_root)

    def subset_of(region, split):
        df = index[(index.region == region) & (index.split == split)]
        return df.iloc[: args.max_patches] if args.max_patches else df

    print("loading patches into memory...")
    train_df = subset_of("amazon", "train")
    keys = ["post", "pre", "label"] if args.input == "both" else ["post", "label"]
    train = C.load_patches(train_df.path, keys, name="amazon_train")
    evals = {}
    for set_name, region, split in EVAL_SETS:
        df = subset_of(region, split)
        evals[set_name] = C.load_patches(df.path, keys, name=set_name)
        evals[set_name]["ids"] = df.patch_id.values
    if args.input == "both":   # no clear 2019 view: such pixels cannot be scored for change
        for data in [train] + list(evals.values()):
            for i in range(len(data["label"])):
                data["label"][i][(data["pre"][i] == 0).all(axis=0)] = C.IGNORE

    for init, lr, frac, seed in todo:
        run_one(init, lr, frac, seed, train, evals, args, device, out)

    res = pd.read_csv(csv)
    res = res[res.tag == args.tag]
    cols = (["init", "lr", "frac", "seed"] + [f"{s}_iou_t" for s, _, _ in EVAL_SETS]
            + ["congo_test_ap"])
    print("\nAll runs with tag", args.tag)
    print(res[cols].to_string(index=False))


if __name__ == "__main__":
    main()
