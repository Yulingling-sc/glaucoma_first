"""dist lineage 预处理漂移核对：参考口径（按 cache 分组 draft 解码 + BICUBIC crop）
vs 出包口径（单次 draft384 解码 + GPU bbox/裁剪 + 双线性下采样）。

同一套权重/配比/TTA，只换预处理路径，输出逐图 |Δp| / Spearman / 0.5 翻转数。

  python tools/finals_dist_parity.py --images <dir> --meta <csv> [--limit 20]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(HERE, "finals", "code")
sys.path.insert(0, CODE)

from model import Ensemble, FundusNet  # noqa: E402
from utils import AGE_MEDIAN, clean_age  # noqa: E402
from utils.preprocess import crop_resize  # noqa: E402
from utils.weights import load_checkpoint  # noqa: E402

MODEL_DIR = os.path.join(HERE, "finals", "model")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def read_members():
    with open(os.path.join(MODEL_DIR, "members.json"), encoding="utf-8") as f:
        return json.load(f)["members"]


def build(device, members):
    loaded = []
    for m in members:
        sd, meta = load_checkpoint(os.path.join(MODEL_DIR, m["file"]))
        meta = dict(meta, cache_size=int(m["cache_size"]), img_size=int(m["img_size"]),
                    tta=int(m.get("tta", 1)))
        net = FundusNet(arch=meta["arch"]).to(device)
        net.load_state_dict(sd, strict=True)
        net.to(memory_format=torch.channels_last).eval()
        loaded.append((net, meta, float(m["weight"])))
    return Ensemble(loaded, device=device)


def ref_probs(model, path, age, device):
    """参考口径: 每个 cache 档位各自 draft 解码 + np 裁剪 BICUBIC。"""
    out = 0.0
    cache_seen = {}
    with torch.inference_mode():
        for net, meta, w in model.members:
            cache = int(meta["cache_size"])
            if cache not in cache_seen:
                draft = max(768, int(cache * 1.5))
                with Image.open(path) as im:
                    im.draft("RGB", (draft, draft))
                    arr = np.asarray(im.convert("RGB"))
                cache_seen[cache] = crop_resize(Image.fromarray(arr), arr, cache)
            x = cache_seen[cache]
            if int(meta["img_size"]) != x.shape[-1]:
                x = F.interpolate(x[None], size=(int(meta["img_size"]),) * 2,
                                  mode="bilinear", align_corners=False)
            else:
                x = x[None]
            x = ((x - MEAN) / STD).to(device).contiguous(memory_format=torch.channels_last)
            a = torch.tensor([[(age - 58.0) / 15.0]], device=device)
            p = torch.sigmoid(net(x, a)).float()
            out = out + w * p
    return float(out.item())


def pkg_probs(model, path, age, device):
    with Image.open(path) as im:
        im.draft("RGB", (384, 384))
        arr = np.array(im.convert("RGB"))
    return float(model.predict(arr[None], [age], use_fp16=True)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    members = read_members()
    model = build(device, members)
    print(f"[init] device={device} members={[(m['file'], m['weight'], m['cache_size'], m['img_size'], m.get('tta', 1)) for m in members]}")

    rows = list(csv.DictReader(open(args.meta, encoding="utf-8-sig")))
    if args.limit:
        rows = rows[:args.limit]

    pr, pk, names = [], [], []
    for r in rows:
        name = r["Image_name"]
        stem = name.rsplit(".", 1)[0]
        path = None
        for ext in (".jpg", ".JPG", ".jpeg", ".png", ".PNG"):
            cand = os.path.join(args.images, stem + ext)
            if os.path.exists(cand):
                path = cand
                break
        if path is None:
            print(f"[skip] {name}")
            continue
        age = clean_age(r.get("Age", AGE_MEDIAN))
        pr.append(ref_probs(model, path, age, device))
        pk.append(pkg_probs(model, path, age, device))
        names.append(name)
        print(f"  {name}  ref={pr[-1]:.5f}  pkg={pk[-1]:.5f}  d={pk[-1]-pr[-1]:+.5f}", flush=True)

    pr, pk = np.asarray(pr), np.asarray(pk)
    d = pk - pr
    flips = int((((pr >= 0.5) != (pk >= 0.5))).sum())
    rho = float(spearmanr(pr, pk).statistic)
    print("\n==== summary ====")
    print(f"n={len(d)}  max|dp|={np.abs(d).max():.5f}  mean|dp|={np.abs(d).mean():.5f}")
    print(f"spearman={rho:.5f}  flips@0.5={flips}  ref_mean={pr.mean():.5f}  pkg_mean={pk.mean():.5f}")


if __name__ == "__main__":
    main()
