"""缓存档位敏感性：成员按正确 cache 档位 vs 一律 512-native（错误档位）的概率差。

dist4 口径下 GPU 裁剪按 meta["cache_size"] 取采样边，因此「错误档位」等价于
`crop at 512 -> 插值到 img_size`，正是 512-native 的输入。

  python tools/finals_cache_group_delta.py --pkg finals --images <图目录> [--n 20]

输出：每成员与集成的 max|Δp| / mean|Δp| / 0.5 翻转数。
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default="finals")
    ap.add_argument("--images", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pkg = os.path.join(ROOT, args.pkg) if not os.path.isabs(args.pkg) else args.pkg
    sys.path.insert(0, os.path.join(pkg, "code"))
    import predict_and_cost as P
    from model import Ensemble, FundusNet
    from utils.weights import load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = sorted(glob.glob(os.path.join(args.images, "*.[jJ][pP][gG]")))
    paths += sorted(glob.glob(os.path.join(args.images, "*.png")))
    paths = paths[:args.n]
    if not paths:
        raise SystemExit(f"no image under {args.images}")

    model_dir = os.path.join(pkg, "model")
    base = []
    for fn, w, cache, img_size, tta in P.MEMBERS:
        sd, meta = load_checkpoint(os.path.join(model_dir, fn))
        m = FundusNet(arch=meta["arch"]).to(device).eval()
        m.load_state_dict(sd, strict=True)
        meta = dict(meta, cache_size=cache, img_size=img_size, tta=1)
        base.append((fn, m, meta, w))

    def decode(path):        # 与包内同一口径: 单次 draft384
        with Image.open(path) as im:
            im.draft("RGB", (P.DRAFT_TARGET, P.DRAFT_TARGET))
            return np.array(im.convert("RGB"))

    batch = np.stack([decode(p) for p in paths])
    ages = [P.AGE_MEDIAN] * len(paths)

    res = {}
    probs_bad, probs_ok = [], []
    for fn, m, meta, w in base:
        cache = int(meta["cache_size"])
        p_ok = Ensemble([(m, meta, 1.0)], device=device).predict(batch, ages)
        meta_bad = dict(meta, cache_size=512)
        p_bad = Ensemble([(m, meta_bad, 1.0)], device=device).predict(batch, ages)
        d = np.abs(p_ok - p_bad)
        res[fn] = {
            "cache": cache,
            "max_abs_dp": float(d.max()),
            "mean_abs_dp": float(d.mean()),
            "flips_0.5": int(((p_ok >= 0.5) != (p_bad >= 0.5)).sum()),
        }
        probs_bad.append(w * p_bad)
        probs_ok.append(w * p_ok)
        print(fn, "cache", cache, res[fn], flush=True)

    ens_bad = np.sum(probs_bad, axis=0)
    ens_ok = np.sum(probs_ok, axis=0)
    d = np.abs(ens_ok - ens_bad)
    res["ensemble"] = {
        "n_images": len(paths),
        "max_abs_dp": float(d.max()),
        "mean_abs_dp": float(d.mean()),
        "flips_0.5": int(((ens_ok >= 0.5) != (ens_bad >= 0.5)).sum()),
    }
    print("ensemble", res["ensemble"])
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
