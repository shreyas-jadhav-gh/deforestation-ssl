"""
Self-supervised pretraining (MoCo v2) of the ResNet-18 encoder on the unlabeled Sentinel-2
patches, adapted to 13-band imagery. The saved encoder has exactly the same layout as the
U-Net's encoder, so train_seg.py can start from it:  --inits moco_both=PATH/encoder.pt

How it works (MoCo v2, He et al. 2020 / Chen et al. 2020):
  * every patch gives two random views (crop, rotation/flip, brightness/contrast, band jitter,
    'grayscale', blur); the query encoder must match its view to the other view's embedding
    from a slowly-updated copy (the key encoder), and tell it apart from a queue of
    embeddings of other patches (contrastive InfoNCE loss)
  * adaptations for a single GPU and a small dataset:
      - split BatchNorm + key shuffling instead of multi-GPU shuffle BN (prevents a shortcut)
      - the queue remembers which patch each key came from, and keys of the same patch are
        never used as negatives
      - key-encoder momentum 0.99 (shorter schedule than ImageNet MoCo)
  * all augmentation runs on the GPU; patches are read from one memory-mapped file, so two
    pretraining processes (one per GPU) share the same RAM

Examples (from the repository root):
    python src/pretrain_moco.py --name moco_both --regions amazon_u congo_u
    python src/pretrain_moco.py --name moco_amazon --regions amazon_u
    python src/pretrain_moco.py --build-cache            # only build the memory-mapped file
    python src/pretrain_moco.py --name smoke --steps 20 --batch-size 16 --queue 64 \
        --splits 4 --max-patches 64 --log-every 5        # quick test
"""

import argparse
import concurrent.futures as cf
import copy
import math
import queue
import sys
import threading
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
from train_seg import SSL4EO_URL, load_encoder, lr_factor  # noqa: E402

FEATURE_DIM = 512   # ResNet-18 output channels

# the 8 rotations/flips of a square, as 2x2 matrices
DIHEDRAL = torch.tensor([[[1, 0], [0, 1]], [[0, -1], [1, 0]], [[-1, 0], [0, -1]], [[0, 1], [-1, 0]],
                         [[-1, 0], [0, 1]], [[1, 0], [0, -1]], [[0, 1], [1, 0]], [[0, -1], [-1, 0]]],
                        dtype=torch.float32)


# -----------------------------------------------------------------------------
# Data: all unlabeled patches in one memory-mapped uint16 file
# -----------------------------------------------------------------------------
def select_index(args):
    full = C.unlabeled_index(args.data_root)
    if args.max_patches:   # quick tests: a few patches from every region
        per_region = max(1, args.max_patches // full.region.nunique())
        full = full.groupby("region").head(per_region).reset_index(drop=True)
    return full


def unlabeled_array(index, workers=8):
    """Open (building it first if needed) one .npy file holding every patch in `index`."""
    path = C.cache_dir() / f"unlabeled_{len(index)}.npy"
    done = path.with_name(path.name + ".done")
    if not done.exists():
        print(f"building {path} from {len(index)} patches (one-off, a few minutes)...")
        arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint16,
                                        shape=(len(index), len(C.BANDS), 256, 256))

        def fill(i):
            with np.load(index.path.iloc[i]) as d:
                arr[i] = d["image"]

        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            for n, _ in enumerate(pool.map(fill, range(len(index))), 1):
                if n % 1000 == 0:
                    print(f"  {n}/{len(index)}")
        arr.flush()
        del arr
        done.touch()
    arr = np.load(path, mmap_mode="r")
    print(f"patches: {arr.shape[0]} in {path} ({arr.nbytes / 1e9:.1f} GB, memory-mapped)")
    return arr


class Prefetcher:
    """Background thread that gathers random batches from the array into (pinned) memory."""

    def __init__(self, arr, indices, batch_size, steps, seed, pin):
        self.q = queue.Queue(maxsize=3)
        args = (arr, np.asarray(indices), batch_size, steps, seed, pin)
        threading.Thread(target=self._run, args=args, daemon=True).start()

    def _run(self, arr, indices, batch_size, steps, seed, pin):
        try:
            stream = C.batch_stream(indices, batch_size, np.random.default_rng(seed))
            for _ in range(steps):
                idx = np.sort(next(stream))
                # uint16 has no torch equivalent: pass the bits as int16 and fix them on the GPU
                x = torch.from_numpy(np.ascontiguousarray(arr[idx]).view(np.int16))
                self.q.put((x.pin_memory() if pin else x, torch.from_numpy(idx)))
        except Exception as e:  # hand the error to the main thread
            self.q.put(e)

    def get(self):
        item = self.q.get(timeout=600)
        if isinstance(item, Exception):
            raise item
        return item


def to_reflectance(x_int16, device):
    x = x_int16.to(device, non_blocking=True).float()
    x = torch.where(x < 0, x + 65536.0, x)      # undo the int16 view of uint16 values
    return x / C.SCALE


# -----------------------------------------------------------------------------
# Augmentation (on the GPU, different random settings for every patch)
# -----------------------------------------------------------------------------
def gaussian_blur(x, sigma, ksize=23):
    """Blur each image with its own sigma (separable Gaussian, reflect padding)."""
    b, c, h, w = x.shape
    half = ksize // 2
    t = torch.arange(-half, half + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-t[None] ** 2 / (2 * sigma[:, None] ** 2))
    k = (k / k.sum(1, keepdim=True)).repeat_interleave(c, 0)          # (b*c, ksize)
    y = x.reshape(1, b * c, h, w)
    y = F.conv2d(F.pad(y, (half, half, 0, 0), mode="reflect"), k[:, None, None, :], groups=b * c)
    y = F.conv2d(F.pad(y, (0, 0, half, half), mode="reflect"), k[:, None, :, None], groups=b * c)
    return y.reshape(b, c, h, w)


def random_view(x, crop):
    """One augmented view per patch: x is (B, 13, H, W) reflectance on the GPU."""
    b, c = x.shape[:2]
    dev = x.device

    def uniform(lo, hi, *shape):
        return torch.empty(*shape, device=dev).uniform_(lo, hi)

    # random resized crop (20-100% of the area, aspect 3:4-4:3) + random rotation/flip
    area = uniform(0.2, 1.0, b)
    ratio = torch.exp(uniform(math.log(3 / 4), math.log(4 / 3), b))
    cw = torch.sqrt(area * ratio).clamp(max=1.0)
    ch = torch.sqrt(area / ratio).clamp(max=1.0)
    cx = torch.rand(b, device=dev) * (1 - cw) + cw / 2
    cy = torch.rand(b, device=dev) * (1 - ch) + ch / 2
    rot = DIHEDRAL.to(dev)[torch.randint(0, 8, (b,), device=dev)]
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, :, :2] = torch.diag_embed(torch.stack([cw, ch], 1)) @ rot
    theta[:, 0, 2] = 2 * cx - 1
    theta[:, 1, 2] = 2 * cy - 1
    grid = F.affine_grid(theta, (b, c, crop, crop), align_corners=False)
    v = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    # brightness and contrast (the 'colour jitter' of MoCo v2), each with probability 0.8
    on = (torch.rand(b, device=dev) < 0.8).float()
    v = v * (1 + on * uniform(-0.4, 0.4, b))[:, None, None, None]
    on = (torch.rand(b, device=dev) < 0.8).float()
    mean = v.mean((1, 2, 3), keepdim=True)
    v = (v - mean) * (1 + on * uniform(-0.4, 0.4, b))[:, None, None, None] + mean
    # band jitter: every band scaled by 0.9-1.1, probability 0.5
    on = (torch.rand(b, 1, device=dev) < 0.5).float()
    v = v * (1 + on * uniform(-0.1, 0.1, b, c))[:, :, None, None]
    # 'grayscale': every band replaced by the mean of all bands, probability 0.2
    gray = torch.rand(b, device=dev) < 0.2
    v = torch.where(gray[:, None, None, None], v.mean(1, keepdim=True).expand_as(v), v)
    # Gaussian blur (sigma 0.1-2 pixels), probability 0.5
    blur = torch.rand(b, device=dev) < 0.5
    if blur.any():
        v[blur] = gaussian_blur(v[blur], uniform(0.1, 2.0, int(blur.sum())))
    return v.clamp_(min=0)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class SplitBatchNorm(nn.BatchNorm2d):
    """BatchNorm over `num_splits` separate groups of the batch (single-GPU shuffle BN).
    Its saved weights look exactly like an ordinary BatchNorm2d."""

    def __init__(self, num_features, num_splits, **kwargs):
        super().__init__(num_features, **kwargs)
        self.num_splits = num_splits

    def forward(self, x):
        n, c, h, w = x.shape
        s = self.num_splits
        if not self.training or n % s:
            return super().forward(x)
        mean, var = self.running_mean.repeat(s), self.running_var.repeat(s)
        out = F.batch_norm(x.reshape(n // s, c * s, h, w), mean, var, self.weight.repeat(s),
                           self.bias.repeat(s), True, self.momentum, self.eps)
        self.running_mean.copy_(mean.view(s, c).mean(0))
        self.running_var.copy_(var.view(s, c).mean(0))
        return out.view(n, c, h, w)


def to_split_bn(module, num_splits):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d) and not isinstance(child, SplitBatchNorm):
            new = SplitBatchNorm(child.num_features, num_splits, eps=child.eps, momentum=child.momentum)
            new.load_state_dict(child.state_dict())
            setattr(module, name, new)
        else:
            to_split_bn(child, num_splits)
    return module


def make_encoder(init_from):
    """The U-Net's own encoder class, so the weights load straight into train_seg.py."""
    enc = smp.Unet("resnet18", encoder_weights=None, in_channels=len(C.BANDS), classes=2).encoder
    if init_from == "ssl4eo":
        state = torch.hub.load_state_dict_from_url(SSL4EO_URL, map_location="cpu", progress=False,
                                                   check_hash=True, weights_only=True)
        load_encoder(enc, state)
    return enc


class MoCo(nn.Module):
    def __init__(self, encoder, dim, queue_size, momentum, temperature, splits):
        super().__init__()
        self.m, self.t, self.k = momentum, temperature, queue_size
        self.encoder_q = to_split_bn(encoder, splits)
        self.head_q = nn.Sequential(nn.Linear(FEATURE_DIM, FEATURE_DIM), nn.ReLU(inplace=True),
                                    nn.Linear(FEATURE_DIM, dim))
        self.encoder_k = copy.deepcopy(self.encoder_q)
        self.head_k = copy.deepcopy(self.head_q)
        for p in list(self.encoder_k.parameters()) + list(self.head_k.parameters()):
            p.requires_grad = False
        self.register_buffer("queue", F.normalize(torch.randn(dim, queue_size), dim=0))
        self.register_buffer("queue_ids", torch.full((queue_size,), -1, dtype=torch.long))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    def online_parameters(self):
        return list(self.encoder_q.parameters()) + list(self.head_q.parameters())

    @staticmethod
    def embed(encoder, head, x):
        features = encoder(x)[-1].mean((2, 3))          # global average pool of the last stage
        return F.normalize(head(features).float(), dim=1)

    @torch.no_grad()
    def momentum_update(self):
        pairs = zip(list(self.encoder_q.parameters()) + list(self.head_q.parameters()),
                    list(self.encoder_k.parameters()) + list(self.head_k.parameters()))
        for pq, pk in pairs:
            pk.mul_(self.m).add_(pq.detach(), alpha=1 - self.m)

    @torch.no_grad()
    def enqueue(self, keys, ids):
        b, p = keys.shape[0], int(self.ptr)
        self.queue[:, p:p + b] = keys.T
        self.queue_ids[p:p + b] = ids
        self.ptr[0] = (p + b) % self.k

    def forward(self, view_q, view_k, ids):
        q = self.embed(self.encoder_q, self.head_q, view_q)
        with torch.no_grad():
            self.momentum_update()
            perm = torch.randperm(view_k.shape[0], device=view_k.device)   # key shuffle
            k = self.embed(self.encoder_k, self.head_k, view_k[perm])[torch.argsort(perm)]
        with torch.autocast(q.device.type, enabled=False):
            l_pos = (q * k).sum(1, keepdim=True)
            l_neg = q @ self.queue.clone()
            # never use a key of the same patch as a negative
            l_neg = l_neg.masked_fill(self.queue_ids[None, :] == ids[:, None], float("-inf"))
            logits = torch.cat([l_pos, l_neg], 1) / self.t
        self.enqueue(k, ids)
        return logits, q


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train(args):
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    out = C.results_dir() / "pretrain" / args.name
    out.mkdir(parents=True, exist_ok=True)
    final, last, log_csv = out / "encoder.pt", out / "last.pt", out / "log.csv"
    if final.exists():
        print(f"{final} already exists, nothing to do")
        return
    if args.queue % args.batch_size or args.batch_size % args.splits:
        raise SystemExit("--queue must be a multiple of --batch-size, and --batch-size of --splits")

    index = select_index(args)
    arr = unlabeled_array(index)
    use = np.flatnonzero(index.region.isin(args.regions).values)
    if len(use) == 0:
        raise SystemExit(f"no patches for regions {args.regions}; found {sorted(index.region.unique())}")
    print(f"pretraining '{args.name}' on {len(use)} patches from {args.regions}, "
          f"{args.steps} steps of {args.batch_size} ({args.steps * args.batch_size / len(use):.0f} epochs)")

    model = MoCo(make_encoder(args.init_from), args.dim, args.queue, args.momentum,
                 args.temperature, args.splits).to(device)
    lr = args.lr * args.batch_size / 256
    opt = torch.optim.SGD(model.online_parameters(), lr=lr, momentum=0.9, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, args.steps, args.warmup))
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start = 0
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start = ck["step"]
        print(f"resuming from step {start}")

    feed = Prefetcher(arr, use, args.batch_size, args.steps - start, args.seed + start, pin=amp)
    log, t0, seen = [], time.time(), 0
    if amp:
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    for step in range(start + 1, args.steps + 1):
        xb, ids = feed.get()
        x = to_reflectance(xb, device)
        ids = ids.to(device, non_blocking=True)
        with torch.no_grad():
            view_q, view_k = random_view(x, args.crop), random_view(x, args.crop)
        del x
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits, q = model(view_q, view_k, ids)
        loss = F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=device))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        seen += len(ids)

        if step % args.log_every == 0 or step == args.steps:
            row = {"step": step,
                   "loss": round(loss.item(), 4),
                   # how often the positive beats every negative (rises as the task is learned)
                   "acc": round((logits.argmax(1) == 0).float().mean().item(), 4),
                   # spread of embeddings: about 1 when well spread, near 0 if they collapse
                   "emb_std": round((q.detach().std(0).mean() * math.sqrt(q.shape[1])).item(), 3),
                   "lr": sched.get_last_lr()[0],
                   "img_s": round(seen / (time.time() - t0)),
                   "minutes": round((time.time() - t0) / 60, 1)}
            log.append(row)
            print(f"  step {step}/{args.steps}  loss {row['loss']:.3f}  acc {row['acc']:.3f}  "
                  f"emb_std {row['emb_std']:.2f}  lr {row['lr']:.3g}  {row['img_s']} img/s", flush=True)
            pd.DataFrame(log).to_csv(log_csv, mode="a", header=not log_csv.exists(), index=False)
            log = []
        if step % args.save_every == 0 and step < args.steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "step": step}, last)

    state = {k: v.cpu() for k, v in model.encoder_q.state_dict().items()}
    torch.save({"encoder": state, "config": vars(args)}, final)
    if last.exists():
        last.unlink()
    # self-check: the encoder must load into the segmentation U-Net
    unet = smp.Unet("resnet18", encoder_weights=None, in_channels=len(C.BANDS), classes=2)
    load_encoder(unet.encoder, torch.load(final, map_location="cpu", weights_only=True)["encoder"])
    peak = torch.cuda.max_memory_allocated(device) / 1e9 if amp else 0
    print(f"saved {final}  ({(time.time() - t0) / 60:.1f} min, peak GPU memory {peak:.1f} GB)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="moco_both", help="output folder: results/pretrain/NAME")
    p.add_argument("--regions", nargs="+", default=["amazon_u", "congo_u"])
    p.add_argument("--init-from", choices=["random", "ssl4eo"], default="random",
                   help="start from random weights or continue from the SSL4EO weights")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.03, help="for batch 256; scaled linearly")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--crop", type=int, default=224)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--queue", type=int, default=4096)
    p.add_argument("--momentum", type=float, default=0.99)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--splits", type=int, default=8, help="BatchNorm splits (single-GPU shuffle BN)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--max-patches", type=int, default=0, help="quick tests: use only N patches")
    p.add_argument("--data-root", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--build-cache", action="store_true", help="only build the memory-mapped file")
    args = p.parse_args()
    if args.build_cache:
        unlabeled_array(select_index(args))
        return
    train(args)


if __name__ == "__main__":
    main()
