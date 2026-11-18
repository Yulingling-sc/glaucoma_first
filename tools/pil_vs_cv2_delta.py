"""PIL BICUBIC vs cv2 INTER_CUBIC 概率差量化（#207 零-2）。

同一个 3 成员集成、同一批真实眼底图（ACRIMA 705 张），只替换预处理实现：
  PIL : 包内 utils/preprocess.py（Image.draft + Image.BICUBIC）
  cvA : cv2 全分辨率解码 + cv2.resize(INTER_CUBIC)
  cvB : cv2 缩解码（IMREAD_REDUCED_COLOR_2，长边>1024）+ INTER_CUBIC
输出 max|Δp| / 均值 / 超 1e-3 比例 / 排序相关 / AUPRC（ACRIMA 全量，辅助口径）。

  python tools/pil_vs_cv2_delta.py --pkg finals --images finals_t4_out/acrima/Database/Images \
      --out finals_t4_out/pil_vs_cv2.json [--n20 20]
"""
import argparse
import csv
import json
import os
import random
import sys

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))

AGE = 58.0


def pil_pre(pkg, path, size):
    sys.path.insert(0, os.path.join(os.path.abspath(pkg), "code"))
    from utils.preprocess import preprocess  # noqa: E402
    return preprocess(path, cache_size=size)


def pil_crop_arr(pkg, path, size):
    """与包内 crop_resize 完全相同的 bbox/裁剪，只在最后一步换成 cv2.resize。"""
    sys.path.insert(0, os.path.join(os.path.abspath(pkg), "code"))
    from utils.preprocess import decode  # noqa: E402
    draft = max(768, int(size * 1.5))
    im, arr = decode(path, draft)
    w, h = im.size
    gray = arr.mean(axis=2)
    mask = gray > 15
    frac = float(mask.mean())
    if frac < 0.02 or frac > 0.999:
        bbox = (0, 0, w, h)
    else:
        ys, xs = np.nonzero(mask)
        y0, y1 = np.percentile(ys, [0.2, 99.8])
        x0, x1 = np.percentile(xs, [0.2, 99.8])
        bbox = (float(x0), float(y0), float(x1), float(y1))
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * 1.02
    nx0, ny0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    nx1, ny1 = int(round(cx + side / 2)), int(round(cy + side / 2))
    pad_l, pad_t = max(0, -nx0), max(0, -ny0)
    pad_r, pad_b = max(0, nx1 - w), max(0, ny1 - h)
    crop = arr[max(0, ny0):min(h, ny1), max(0, nx0):min(w, nx1)]
    if pad_l or pad_t or pad_r or pad_b:
        crop = np.pad(crop, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="constant")
    return crop


def cv2_arr(path, reduced, size):
    flags = cv2.IMREAD_COLOR
    if reduced:
        im0 = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_2)
        if im0 is None:
            im0 = cv2.imread(path, cv2.IMREAD_COLOR)
        bgr = im0
    else:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = arr.shape[:2]
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mask = gray > 15
    frac = float(mask.mean())
    if frac < 0.02 or frac > 0.999:
        bbox = (0, 0, w, h)
    else:
        ys, xs = np.nonzero(mask)
        y0, y1 = np.percentile(ys, [0.2, 99.8])
        x0, x1 = np.percentile(xs, [0.2, 99.8])
        bbox = (float(x0), float(y0), float(x1), float(y1))
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * 1.02
    nx0, ny0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    nx1, ny1 = int(round(cx + side / 2)), int(round(cy + side / 2))
    pad_l, pad_t = max(0, -nx0), max(0, -ny0)
    pad_r, pad_b = max(0, nx1 - w), max(0, ny1 - h)
    crop = arr[max(0, ny0):min(h, ny1), max(0, nx0):min(w, nx1)]
    if pad_l or pad_t or pad_r or pad_b:
        crop = np.pad(crop, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="constant")
    out = cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)
    return out


def to_tensor(pkg, arr):
    import torch
    sys.path.insert(0, os.path.join(os.path.abspath(pkg), "code"))
    import torchvision.transforms.functional as TF  # noqa: E402
    return TF.to_tensor(Image.fromarray(arr))


def run(model, pkg, paths, size, mode):
    import torch
    tens = []
    for p in paths:
        if mode == "pil":
            tens.append(pil_pre(pkg, p, size))
        elif mode == "cvR":
            crop = pil_crop_arr(pkg, p, size)
            tens.append(to_tensor(pkg, cv2.resize(
                crop, (size, size), interpolation=cv2.INTER_CUBIC)))
        else:
            tens.append(to_tensor(pkg, cv2_arr(p, reduced=(mode == "cvB"), size=size)))
    out = []
    with torch.no_grad():
        for i in range(0, len(tens), 32):
            out.append(model.predict(tens[i:i + 32], [AGE] * len(tens[i:i + 32])))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default="finals")
    ap.add_argument("--images", default="finals_t4_out/acrima/Database/Images")
    ap.add_argument("--out", default="finals_t4_out/pil_vs_cv2.json")
    ap.add_argument("--n20", type=int, default=20)
    ap.add_argument("--subset", type=int, default=0, help=">0 时只跑前 N 张（快速调试）")
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.abspath(args.pkg), "code"))
    import torch  # noqa: E402
    from predict_and_cost import load_model  # noqa: E402

    files = sorted(f for f in os.listdir(args.images)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if args.subset:
        files = files[:args.subset]
    paths = [os.path.join(args.images, f) for f in files]
    labels = np.array([1 if "_g_" in f else 0 for f in files])
    print(f"{len(paths)} images, pos={labels.sum()}", flush=True)

    model = load_model(model_dir=os.path.join(os.path.abspath(args.pkg), "model"))
    size = model.cache_size
    print("device:", model.device, "cache:", size, flush=True)

    p_pil = run(model, args.pkg, paths, size, "pil")
    p_cva = run(model, args.pkg, paths, size, "cvA")
    p_cvb = run(model, args.pkg, paths, size, "cvB")
    p_cvr = run(model, args.pkg, paths, size, "cvR")

    rng = random.Random(42)
    idx20 = rng.sample(range(len(paths)), min(args.n20, len(paths)))

    def cmp(a, b, sel=None):
        ia = a if sel is None else a[sel]
        ib = b if sel is None else b[sel]
        d = np.abs(ia - ib)
        lab = labels if sel is None else labels[sel]
        return {
            "n": int(len(d)),
            "max_abs_dp": float(d.max()),
            "mean_abs_dp": float(d.mean()),
            "frac_over_1e-3": float((d > 1e-3).mean()),
            "label_flips_0.5": int(((ia >= 0.5) != (ib >= 0.5)).sum()),
            "spearman": float(spearmanr(ia, ib).statistic),
            "auprc_a": float(average_precision_score(lab, ia)) if 0 < lab.sum() < len(lab) else None,
            "auprc_b": float(average_precision_score(lab, ib)) if 0 < lab.sum() < len(lab) else None,
        }

    res = {
        "image_source": "ACRIMA 705 (非官方 test，辅助口径)",
        "n_all": len(paths),
        "n20_seed42": args.n20,
        "cvR_resize_only": cmp(p_pil, p_cvr),
        "cvR_resize_only_20": cmp(p_pil, p_cvr, idx20),
        "cvA_full_decode": cmp(p_pil, p_cva),
        "cvB_reduced_decode": cmp(p_pil, p_cvb),
        "cvA_full_decode_20": cmp(p_pil, p_cva, idx20),
        "cvB_reduced_decode_20": cmp(p_pil, p_cvb, idx20),
    }
    for k in ("cvR_resize_only", "cvA_full_decode", "cvB_reduced_decode"):
        res[k]["auprc_delta"] = (
            None if res[k]["auprc_a"] is None
            else res[k]["auprc_b"] - res[k]["auprc_a"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    for k in ("cvR_resize_only_20", "cvA_full_decode_20", "cvB_reduced_decode_20",
              "cvR_resize_only", "cvA_full_decode", "cvB_reduced_decode"):
        r = res[k]
        print(f"{k}: n={r['n']} max|dp|={r['max_abs_dp']:.3e} mean={r['mean_abs_dp']:.3e} "
              f">1e-3={r['frac_over_1e-3']*100:.1f}% flips={r['label_flips_0.5']} "
              f"rho={r['spearman']:.6f} dAUPRC={r.get('auprc_delta')}", flush=True)


if __name__ == "__main__":
    main()