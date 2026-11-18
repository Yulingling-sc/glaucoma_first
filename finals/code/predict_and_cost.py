"""决赛统一入口：三接口与官方模板一致，全量预测与单张计时共用同一套数值路径。

    load_model(device) -> Ensemble
    preprocess(img_bgr) -> tensor[3,S,S]        # 官方模板按 BGR ndarray 传入
    predict_batch(model, batch, device, use_fp16, meta_rows=None) -> np.ndarray

CLI（解压后任意目录，脚本会定位 ../model）：
    python code/predict_and_cost.py --images <图片目录> --meta <meta.csv> --out_dir out
常用参数：
    --mode both|predict|cost   both = 先测 cost 再跑全量（默认）
    --num_workers 4 --batch_size 32 --use_fp16 1
    --tta 1|2|4
输出：
    out/submission.csv   Image_name,Probability,Pred_Label
    out/cost.json        5 张预热 + 随机 100 张、batch=1、端到端均值（avg_cost / avg_cost_ms）
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import Ensemble, FundusNet  # noqa: E402
from utils import AGE_MEDIAN, clean_age  # noqa: E402
from utils.preprocess import DEFAULT_CACHE_SIZE, crop_resize, decode  # noqa: E402
from utils.weights import load_checkpoint  # noqa: E402

ROOT = os.path.dirname(HERE)  # 包根目录（含 model/）
MEMBERS = [
    ("cb384.pt", 0.40),
    ("cs_hi512_all_fp16.pt", 0.30),
    ("cs_hi512_all_ra.pt", 0.30),
]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def _read_members(model_dir):
    """model/members.json 覆盖默认成员表：{"members":[{"file":..., "weight":...}]}"""
    p = os.path.join(model_dir, "members.json")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    items = data["members"]
    return [(m["file"], float(m.get("weight", 1.0 / len(items)))) for m in items]


# ------------------------------------------------------------------ 三接口
def load_model(device=None, model_dir=None, members=None):
    """加载集成成员，返回 Ensemble。"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_dir is None:
        model_dir = os.path.join(ROOT, "model")
    if members is None:
        members = _read_members(model_dir) or MEMBERS
    loaded = []
    for fn, w in members:
        sd, meta = load_checkpoint(os.path.join(model_dir, fn))
        m = FundusNet(arch=meta["arch"]).to(device)
        m.load_state_dict(sd, strict=True)
        m.eval()
        print(f"model: {fn} arch={meta['arch']} img_size={meta['img_size']} "
              f"cache={meta['cache_size']} use={meta['use']} w={w}", flush=True)
        loaded.append((m, meta, w))
    return Ensemble(loaded, device=device)


def preprocess(img_bgr, cache_size=DEFAULT_CACHE_SIZE):
    """官方模板接口：BGR ndarray -> tensor[3,S,S]。

    同时兼容路径与 PIL.Image（内部 CLI 走 JPEG draft 解码）。
    """
    if isinstance(img_bgr, str):
        im, arr = decode(img_bgr, max(768, int(cache_size * 1.5)))
    elif isinstance(img_bgr, Image.Image):
        im = img_bgr.convert("RGB")
        arr = np.asarray(im)
    else:
        arr = np.asarray(img_bgr)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = np.ascontiguousarray(arr[..., ::-1])  # BGR -> RGB
        im = Image.fromarray(arr)
    return crop_resize(im, arr, cache_size)


def _ages(meta_rows, n):
    if meta_rows is None:
        return [AGE_MEDIAN] * n
    out = []
    for r in meta_rows:
        out.append(clean_age(r.get("Age", AGE_MEDIAN)) if isinstance(r, dict) else float(r))
    return out


def predict_batch(model, batch, device=None, use_fp16=True, meta_rows=None, tta=1):
    """一批 tensor 一次前向 + sigmoid，返回 numpy 概率。"""
    return model.predict(batch, _ages(meta_rows, len(batch)), tta=tta, use_fp16=use_fp16)


# ------------------------------------------------------------------ 数据侧
def read_meta(path):
    names, ages = [], []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            names.append(r["Image_name"])
            ages.append(clean_age(r.get("Age", AGE_MEDIAN)))
    return names, ages


def resolve(images_dir, name):
    stem = name.rsplit(".", 1)[0]
    for ext in IMG_EXTS:
        p = os.path.join(images_dir, stem + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"image not found: {name} under {images_dir}")


# ------------------------------------------------------------------ 计时
def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def measure_cost(model, paths, ages, cache_size, tta=1, use_fp16=True,
                 warmup=5, n=100, seed=42):
    """5 张预热（不计时）+ 随机 n 张单轮；分解 decode / crop+resize / forward / post。

    forward / post 就是 predict_batch 内部调用的同一个 forward_logits + postprocess。
    """
    idx = list(range(len(paths)))
    random.Random(seed).shuffle(idx)
    warm, timed = idx[:warmup], idx[warmup:warmup + n]
    device = model.device
    draft = max(768, int(cache_size * 1.5))

    def once(i):
        p = paths[i]
        t0 = time.perf_counter()
        im, arr = decode(p, draft)
        t1 = time.perf_counter()
        tens = crop_resize(im, arr, cache_size)
        t2 = time.perf_counter()
        _sync(device)
        t3 = time.perf_counter()
        logits = model.forward_logits([tens], [ages[i]], tta=tta, use_fp16=use_fp16)
        _sync(device)
        t4 = time.perf_counter()
        model.postprocess(logits)
        _sync(device)
        t5 = time.perf_counter()
        return {"file": os.path.basename(p),
                "decode": (t1 - t0) * 1e3, "crop+resize": (t2 - t1) * 1e3,
                "forward": (t4 - t3) * 1e3, "post": (t5 - t4) * 1e3,
                "e2e": (t5 - t0) * 1e3}

    for i in warm:
        once(i)
    recs = [once(i) for i in timed]

    def stat(key):
        a = np.asarray([r[key] for r in recs])
        return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
                "p95": float(np.percentile(a, 95)), "max": float(a.max())}

    return {"n": len(recs), "warmup": len(warm), "batch_size": 1, "tta": tta,
            "cache_size": cache_size, "use_fp16": bool(use_fp16),
            "avg_cost": stat("e2e")["mean"],
            "avg_cost_ms": stat("e2e")["mean"],
            "phases": {k: stat(k) for k in ("decode", "crop+resize", "forward", "post", "e2e")},
            "slowest": sorted(recs, key=lambda r: -r["e2e"])[:5]}


# ------------------------------------------------------------------ 全量
def run_predict(model, paths, ages, out_csv, batch_size=32, num_workers=4, tta=1,
                use_fp16=True):
    cache = model.cache_size
    probs = np.zeros(len(paths), dtype=np.float64)

    def prep(p):
        return preprocess(p, cache_size=cache)

    pool = ThreadPoolExecutor(max_workers=max(1, num_workers))
    from collections import deque

    src = iter(range(len(paths)))
    futs = deque()
    lookahead = max(batch_size, num_workers * 2)
    for _ in range(min(lookahead, len(paths))):
        futs.append(pool.submit(prep, paths[next(src)]))
    for start in range(0, len(paths), batch_size):
        idx = list(range(start, min(start + batch_size, len(paths))))
        tensors = [futs.popleft().result() for _ in idx]
        for _ in idx:
            j = next(src, None)
            if j is not None:
                futs.append(pool.submit(prep, paths[j]))
        probs[idx] = predict_batch(model, tensors, model.device, use_fp16,
                                   meta_rows=[{"Age": ages[i]} for i in idx], tta=tta)
    pool.shutdown()
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--meta", required=True, help="csv: Image_name[,Age]")
    ap.add_argument("--out_dir", "--out", dest="out_dir", default="out")
    ap.add_argument("--mode", default="both", choices=["both", "predict", "cost"])
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--use_fp16", type=int, default=1, choices=[0, 1])
    ap.add_argument("--tta", type=int, default=1, choices=[1, 2, 4])
    args = ap.parse_args()
    use_fp16 = bool(args.use_fp16)

    os.makedirs(args.out_dir, exist_ok=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    model = load_model(device=args.device, model_dir=args.model_dir)
    names, ages = read_meta(args.meta)
    paths = [resolve(args.images, n) for n in names]
    print(f"{len(names)} images from {args.images}", flush=True)

    if args.mode in ("both", "cost"):
        cost = measure_cost(model, paths, ages, model.cache_size, tta=args.tta,
                            use_fp16=use_fp16)
        with open(os.path.join(args.out_dir, "cost.json"), "w", encoding="utf-8") as f:
            json.dump(cost, f, ensure_ascii=False, indent=2)
        ph = cost["phases"]
        print(f"cost: avg={cost['avg_cost_ms']:.2f}ms "
              f"p95={ph['e2e']['p95']:.2f}ms "
              f"(decode {ph['decode']['mean']:.2f} / crop+resize {ph['crop+resize']['mean']:.2f} "
              f"/ forward {ph['forward']['mean']:.2f} / post {ph['post']['mean']:.2f})", flush=True)

    if args.mode in ("both", "predict"):
        t0 = time.time()
        probs = run_predict(model, paths, ages, args.out_dir,
                            batch_size=args.batch_size, num_workers=args.num_workers,
                            tta=args.tta, use_fp16=use_fp16)
        out_csv = os.path.join(args.out_dir, "submission.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["Image_name", "Probability", "Pred_Label"])
            for n, p in zip(names, probs):
                wr.writerow([n, f"{p:.6f}", 1 if p >= 0.5 else 0])
        print(f"wrote {out_csv}: {len(names)} rows in {time.time()-t0:.0f}s "
              f"(pos_rate={float((probs >= 0.5).mean())*100:.2f}%)", flush=True)


if __name__ == "__main__":
    main()