"""#221 预处理 parity 门禁：pkg3/dist4 线（共享 512 PIL BICUBIC）vs 包内服务代码。

参照实现 `ref_crop_resize` 逐行拷贝 ed6bda3:finals/code/utils/preprocess.py（pkg3 出包
同源代码），独立于包内 utils.preprocess，防止两边一起改坏而静默通过。

覆盖两组 fixture：
  - square   : 1024x1024（默认复用 finals_t4_out/smoke/imgs，缺省时合成）
  - nonsquare: 768x512、1024x680、1024x512 非居中偏移（确定性合成，写入磁盘复用）
门槛（每组）：mean|Δp| <= 0.005 且 0 个 0.5 阈值翻转。
同时跑：
  - 回归演示：merged-main 的 (y,x) 转置实现（buggy）vs 参照实现 —— 非方图必须被判红；
  - --archive 锚点（若归档输出存在）：候选包代码 vs pkg3 归档 CSV —— 证明与归档一致。

  python tools/finals_prep_parity.py [--limit N] [--no-archive] [--out <report.json>]

退出码：门禁失败返回 1。
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = os.path.join(ROOT, "finals_t4_out", "smoke")
FIXTURES = os.path.join(SMOKE, "fixtures221")
PKG = os.path.join(ROOT, "finals")
CODE = os.path.join(PKG, "code")
sys.path.insert(0, CODE)

import predict_and_cost as entry  # noqa: E402
from model import Ensemble, FundusNet  # noqa: E402
from utils import AGE_MEDIAN, clean_age  # noqa: E402
from utils.weights import load_checkpoint  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SHARED = 512
GATE_MEAN = 0.005
MODEL_DIR = os.path.join(PKG, "model")


# ---------------------------------------------------------------- 参照实现（frozen pkg3）
def ref_crop_resize(arr, size=SHARED):
    """ed6bda3 utils/preprocess.crop_resize 的逐行拷贝（PIL BICUBIC, 整数框）。"""
    im = Image.fromarray(np.ascontiguousarray(arr, dtype=np.uint8))
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
    out = Image.fromarray(crop).resize((size, size), Image.BICUBIC)
    return np.asarray(out, dtype=np.float32).transpose(2, 0, 1) / 255.0


# ---------------------------------------------------------------- buggy 基线（merged main）
MASK_THRESHOLD = 15.0 / 255.0


def buggy_prep(arr_rgb, size=SHARED):
    """HEAD(5003bea) 的 GPU bbox/裁剪：nonzero 的 (y,x) 直接按 (x0,y0,x1,y1) 解包。"""
    x = torch.from_numpy(np.ascontiguousarray(arr_rgb)).float().to(DEV).div_(255.0)
    x = x.permute(2, 0, 1).unsqueeze(0)
    _, _, h, w = x.shape
    mask = x.mean(dim=1)[0] > MASK_THRESHOLD
    frac = float(mask.float().mean())
    if frac < 0.02 or frac > 0.999:
        box = (0.0, 0.0, float(w - 1), float(h - 1))
    else:
        idx = mask.nonzero(as_tuple=False).float()          # (N,2) -> (y,x)
        qs = torch.quantile(idx, torch.tensor([0.002, 0.998], device=DEV), dim=0)
        box = (float(qs[0, 0]), float(qs[0, 1]), float(qs[1, 0]), float(qs[1, 1]))
    x0, y0, x1, y1 = box                                     # 实际为 (y0,x0,y1,x1) 转置
    side = max(x1 - x0, y1 - y0) * 1.02
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    t = torch.linspace(0.0, 1.0, size, device=DEV)
    gx = (2.0 * ((cx - side * 0.5) + t * side) / max(w - 1, 1) - 1.0).view(1, -1).expand(size, size)
    gy = (2.0 * ((cy - side * 0.5) + t * side) / max(h - 1, 1) - 1.0).view(-1, 1).expand(size, size)
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return out[0].cpu().numpy()


# ---------------------------------------------------------------- fixtures
def make_fundus(h, w, cx_frac, cy_frac, seed):
    """合成眼底图：暗背景 + 亮盘（可偏移/部分出框）+ 纹理噪声。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = cx_frac * w, cy_frac * h
    r = 0.42 * min(h, w)
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    disc = (d < r).astype(np.float32) * np.clip(1.3 - d / r, 0.0, 1.0)
    tex = rng.normal(0.0, 12.0, (h, w)).astype(np.float32)
    base = np.stack([170.0 * disc, 110.0 * disc, 70.0 * disc], axis=-1)
    base += (disc[..., None] * tex[..., None]) + rng.normal(0.0, 2.0, (h, w, 3))
    return np.clip(base, 0, 255).astype(np.uint8)


def build_fixtures():
    """返回 {group: [(name, rgb_array, age)]}；同时落盘供复用/审计。"""
    os.makedirs(FIXTURES, exist_ok=True)
    groups = {}

    sq = sorted(glob.glob(os.path.join(SMOKE, "imgs", "*.jpg")))
    if sq:
        rows = []
        for i, p in enumerate(sq):
            with Image.open(p) as im:
                rows.append((os.path.basename(p), np.array(im.convert("RGB")), 50 + i))
        groups["square_1024"] = rows
    else:
        rows = [("sq_%02d.jpg" % i, make_fundus(1024, 1024, 0.5 + 0.06 * (i - 2), 0.5 - 0.04 * i, 100 + i),
                 50 + i) for i in range(4)]
        groups["square_1024"] = rows

    specs = [
        ("nonsq_768x512", lambda i: make_fundus(768, 512, 0.5 + 0.05 * i, 0.5 - 0.06 * i, 200 + i)),
        ("nonsq_1024x680", lambda i: make_fundus(1024, 680, 0.5 - 0.05 * i, 0.5 + 0.06 * i, 300 + i)),
        ("nonsq_1024x512_off", lambda i: make_fundus(1024, 512, 0.30 + 0.04 * i, 0.62 - 0.10 * i, 400 + i)),
    ]
    for gname, fn in specs:
        groups[gname] = [("%s_%d.jpg" % (gname, i), fn(i), 46 + 7 * i) for i in range(3)]

    meta_lines = [("group,Image_name,Age")]
    for gname, rows in groups.items():
        gdir = os.path.join(FIXTURES, gname)
        os.makedirs(gdir, exist_ok=True)
        for name, arr, age in rows:
            Image.fromarray(arr).save(os.path.join(gdir, name), quality=92)
            meta_lines.append(f"{gname},{name},{age}")
    with open(os.path.join(FIXTURES, "meta.csv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(meta_lines) + "\n")
    return groups


# ---------------------------------------------------------------- 模型与推理
def read_members(path=None):
    p = path or os.path.join(MODEL_DIR, "members.json")
    items = json.load(open(p, encoding="utf-8"))["members"]
    return [(m["file"], float(m["weight"]), int(m.get("img_size", 512)),
             int(m.get("tta", 1))) for m in items]


def load_pool(files):
    pool = {}
    for fn in files:
        if fn in pool:
            continue
        sd, meta = load_checkpoint(os.path.join(MODEL_DIR, fn))
        m = FundusNet(arch=meta["arch"]).to(DEV)
        m.load_state_dict(sd, strict=True)
        m.to(memory_format=torch.channels_last).eval()
        pool[fn] = m
    return pool


def ref_probs(pool, spec, prepped, ages, combine="prob"):
    """参照推理：共享 512 -> 成员 bilinear 内插 -> 归一化 -> sigmoid -> 概率加权。
    combine="logit" 时按 pkg3 归档的 logits 加权再 sigmoid（仅用于核对归档组合口径）。"""
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
    age = torch.tensor([(a - 58.0) / 15.0 for a in ages], dtype=torch.float32,
                       device=DEV).view(-1, 1)
    x = torch.from_numpy(np.stack(prepped)).float().to(DEV)
    out = None
    with torch.inference_mode():
        for fn, w, img_size, tta in spec:
            xi = x if img_size == SHARED else F.interpolate(
                x, size=(img_size, img_size), mode="bilinear", align_corners=False)
            xi = ((xi - mean) / std).contiguous(memory_format=torch.channels_last)
            flips = {1: [0], 2: [0, 1]}[tta]
            acc = 0.0
            for fl in flips:
                xf = torch.flip(xi, dims=[3]) if fl & 1 else xi
                with torch.amp.autocast("cuda", dtype=torch.float16,
                                        enabled=(DEV.type == "cuda")):
                    logits = pool[fn](xf, age)
                v = logits.float() if combine == "logit" else torch.sigmoid(logits.float())
                acc = acc + v
            contrib = w * acc / len(flips)
            out = contrib if out is None else out + contrib
    return torch.sigmoid(out).cpu().numpy() if combine == "logit" else out.cpu().numpy()


def stats(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = np.abs(a - b)
    flips = int(np.sum((a >= 0.5) != (b >= 0.5)))
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    rho = float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else 1.0
    return {"mean": float(d.mean()), "max": float(d.max()), "flips": flips, "rho": rho}


def decode(path, draft):
    with Image.open(path) as im:
        im.draft("RGB", (draft, draft))
        return np.array(im.convert("RGB"))


def run_group(pool, spec, rows):
    prepped_ref = [ref_crop_resize(a) for _n, a, _age in rows]
    prepped_bug = [buggy_prep(a) for _n, a, _age in rows]
    ages = [age for _n, _a, age in rows]
    r = ref_probs(pool, spec, prepped_ref, ages)
    model = Ensemble([(pool[fn], {"img_size": img, "cache_size": SHARED, "tta": tta}, w)
                      for fn, w, img, tta in spec], device=DEV)
    x = np.stack(prepped_ref)
    cand = entry.predict_batch(model, x, DEV, True, meta_rows=[{"Age": str(a)} for a in ages])
    bug = ref_probs(pool, spec, prepped_bug, ages)
    return stats(cand, r), stats(bug, r)


def run_archive(pool, spec, img_dir, meta_path, ref_csv, draft=768, limit=None):
    rows = list(csv.DictReader(open(meta_path, encoding="utf-8-sig")))
    ref = {r["Image_name"]: float(r["Probability"])
           for r in csv.DictReader(open(ref_csv, encoding="utf-8-sig"))}
    use = [r for r in rows if os.path.isfile(os.path.join(img_dir, r["Image_name"]))][:limit]
    model = Ensemble([(pool[fn], {"img_size": img, "cache_size": SHARED, "tta": tta}, w)
                      for fn, w, img, tta in spec], device=DEV)
    probs, want, prepped, ages = [], [], [], []
    for r in use:                                  # 逐图: draft 解码后尺寸可能不同
        arr = decode(os.path.join(img_dir, r["Image_name"]), draft)
        probs.append(entry.predict_batch(model, arr[None], DEV, True,
                                         meta_rows=[{"Age": r.get("Age", AGE_MEDIAN)}])[0])
        prepped.append(ref_crop_resize(arr))
        ages.append(clean_age(r.get("Age", AGE_MEDIAN)))
        want.append(ref[r["Image_name"]])
    out = {"candidate_vs_archive": stats(want, probs),
           "ref_logit_vs_archive": stats(want, ref_probs(pool, spec, prepped, ages, "logit"))}
    return out, len(use)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="每组最多几张（调试用）")
    ap.add_argument("--no-archive", action="store_true")
    ap.add_argument("--out", default=os.path.join(FIXTURES, "parity221.json"))
    args = ap.parse_args()

    groups = build_fixtures()
    spec = read_members()
    pool = load_pool([m[0] for m in spec])
    print(f"[parity] device={DEV} members={[(m[0], m[1], m[2], m[3]) for m in spec]}")

    report, ok = {"gate": {"mean_max": GATE_MEAN, "flips_max": 0}, "groups": {}}, True
    for gname, rows in groups.items():
        rows = rows[:args.limit] if args.limit else rows
        cand, bug = run_group(pool, spec, rows)
        passed = cand["mean"] <= GATE_MEAN and cand["flips"] == 0
        ok = ok and passed
        report["groups"][gname] = {"n": len(rows), "candidate_vs_ref": cand,
                                   "buggy_vs_ref": bug, "gate": "PASS" if passed else "FAIL"}
        print(f"[{gname:18s}] n={len(rows)} cand mean={cand['mean']:.5f} max={cand['max']:.5f} "
              f"flips={cand['flips']} rho={cand['rho']:.4f} -> {'PASS' if passed else 'FAIL'}"
              f" | buggy mean={bug['mean']:.5f} max={bug['max']:.5f} flips={bug['flips']}")

    if not args.no_archive:
        report["archive"] = {}
        checks = [
            ("acrima20_vs_out_pkg3", os.path.join(ROOT, "finals_t4_out", "acrima", "Database", "Images"),
             os.path.join(SMOKE, "acrima20.csv"),
             os.path.join(SMOKE, "out_pkg3", "submission.csv"),
             [("cb384.pt", 0.667, 384, 1), ("cs_hi512_all_ra.pt", 0.333, 512, 1)]),
            ("imgs1024_vs_out_pkg", os.path.join(SMOKE, "imgs"),
             os.path.join(SMOKE, "meta.csv"),
             os.path.join(SMOKE, "out_pkg", "submission.csv"),
             [("cb384.pt", 0.4, 384, 1), ("cs_hi512_all_fp16.pt", 0.3, 512, 1),
              ("cs_hi512_all_ra.pt", 0.3, 512, 1)]),
        ]
        extra = sorted({fn for _t, _i, _m, _r, s in checks for fn, *_ in s})
        pool.update(load_pool(extra))
        for tag, img_dir, meta_path, ref_csv, aspec in checks:
            if not (os.path.isdir(img_dir) and os.path.isfile(meta_path)
                    and os.path.isfile(ref_csv)):
                print(f"[archive {tag}] 归档缺失, 跳过")
                continue
            st, n = run_archive(pool, aspec, img_dir, meta_path, ref_csv, limit=args.limit)
            cand = st["candidate_vs_archive"]
            passed = cand["mean"] <= GATE_MEAN and cand["flips"] == 0
            ok = ok and passed
            report["archive"][tag] = {"n": n, **st, "gate": "PASS" if passed else "FAIL"}
            fl = st["ref_logit_vs_archive"]
            print(f"[archive {tag:22s}] n={n} cand mean={cand['mean']:.5f} "
                  f"max={cand['max']:.5f} flips={cand['flips']} rho={cand['rho']:.4f} "
                  f"-> {'PASS' if passed else 'FAIL'} | ref-logit mean={fl['mean']:.6f}")

    report["overall"] = "PASS" if ok else "FAIL"
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[parity] overall={report['overall']} report={args.out}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()