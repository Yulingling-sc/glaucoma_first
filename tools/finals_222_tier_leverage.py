"""#222 分辨率/几何轴在真实测试分布上的杠杆量级（无标签，只测灵敏度）。

本地 test100（决赛 test 前 100 张，含非方图）上用 deployed 同源 cb384 逐 draft 档推理，
量化档位切换引起的概率位移；配合 LB 锚点（draft250 相对 draft384 −0.24）判断这条
"valid 筛不到"的轴还剩多少杠杆。

  python tools/finals_222_tier_leverage.py --images D:/glaucoma_work/final_national_test/images \
      --ages D:/glaucoma_work/final_national_test/test100.csv \
      --weights submit_pkg/cb384.pt --tiers 250,384,768 --out docs/finals/TIER_LEVERAGE_222.json
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr

Image.MAX_IMAGE_PIXELS = None

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "submit_pkg"))
import predict as P  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--ages", default=None, help="csv with Image_name,Age (else median age)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tiers", default="250,384,768")
    ap.add_argument("--ref", type=int, default=384)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="docs/finals/TIER_LEVERAGE_222.json")
    args = ap.parse_args()

    tiers = [int(t) for t in args.tiers.split(",") if t.strip()]
    files = sorted(f for f in os.listdir(args.images)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if args.limit:
        files = files[: args.limit]
    ages = {}
    if args.ages and os.path.exists(args.ages):
        with open(args.ages, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                ages[r["Image_name"]] = P.clean_age(r.get("Age", P.AGE_MEDIAN))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = P.load_model(args.weights, device)
    img = int(meta["img_size"])
    mean, std = MEAN.to(device), STD.to(device)
    print(f"device={device} meta={meta} n_img={len(files)}", flush=True)

    shapes, probs = [], {}
    for ti, tier in enumerate(tiers):
        ps = []
        for fn in files:
            path = os.path.join(args.images, fn)
            if ti == 0:
                with Image.open(path) as im:
                    shapes.append((fn, list(im.size)))
            crop = P.load_fundus_crop(path, size=args.size, draft=tier)
            x = torch.from_numpy(np.asarray(crop, dtype=np.float32) / 255.0)
            x = x.permute(2, 0, 1).unsqueeze(0)
            x = (x - mean) / std
            if x.shape[-1] != img:
                x = torch.nn.functional.interpolate(x, size=(img, img), mode="bilinear",
                                                    align_corners=False)
            av = (ages[fn] - P.AGE_MEDIAN) / 15.0 if fn in ages else 0.0
            a = torch.tensor([[av]], dtype=torch.float32).to(device)
            with torch.no_grad():
                logits = model(x, a)
                ps.append(float(torch.sigmoid(logits.float()).cpu().numpy().ravel()[0]))
        probs[tier] = np.array(ps)
        print(f"tier {tier}: mean p={probs[tier].mean():.4f} max={probs[tier].max():.4f}",
              flush=True)

    n_sq = sum(1 for _, s in shapes if s[0] == s[1])
    rows = []
    for tier in tiers:
        if tier == args.ref:
            continue
        d = np.abs(probs[tier] - probs[args.ref])
        rows.append({"tier": tier,
                     "mean_abs_dp": float(d.mean()), "max_abs_dp": float(d.max()),
                     "n_gt_0.01": int((d > 0.01).sum()),
                     "flips_0p5": int(((probs[tier] >= 0.5) != (probs[args.ref] >= 0.5)).sum()),
                     "spearman": float(spearmanr(probs[tier], probs[args.ref]).statistic),
                     "mean_p": float(probs[tier].mean())})
    out = {"n_img": len(files), "n_square": n_sq, "n_nonsquare": len(files) - n_sq,
           "device": device, "weights": os.path.basename(args.weights),
           "img_size": img, "tiers": tiers, "ref": args.ref,
           "mean_p": {str(t): float(probs[t].mean()) for t in tiers},
           "delta_vs_ref": rows, "sizes": dict(shapes[:10])}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps({"n_img": len(files), "nonsquare": len(files) - n_sq,
                      "delta_vs_ref": rows}, ensure_ascii=False, indent=1), flush=True)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
