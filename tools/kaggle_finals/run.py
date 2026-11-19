"""#207 T4 实测 v3（Kaggle script kernel，单文件）。

每个组合独立子进程：清页面缓存 -> 加载权重 -> 5 张预热 -> 100 张单轮计时。
v3 把 worker 合进本文件（kaggle kernels push 只上传 code_file），
子进程通过 `python run.py --worker ...` 自调用。
"""
import argparse
import glob
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
import zipfile

WORK = "/kaggle/working"
REPO = os.path.join(WORK, "glaucoma_first")
W = os.path.join(WORK, "weights")
SELF = os.path.abspath(__file__)
RUNNER = None
REPO_URL = "https://github.com/Yulingling-sc/glaucoma_first.git"
REL = "https://github.com/Yulingling-sc/glaucoma_first/releases/download"
ASSETS = [
    ("submission-degraded-110", "submission_pkg_2model.zip", 325_822_686),
    ("compliant_pkg-1028", "cs_hi512_all_ra.pt", 99_831_694),
    ("weights-1016", "cs_hi512_all_fp16.pt", 199_660_651),
]
COMBOS = [
    ("cnxb_all_pseudo (TTA1)", "cnxb", 1),
    ("cs_hi512_all_ra (TTA1)", "cs_hi", 1),
    ("cb384+cs_hi512_all_ra (TTA1)", "cb384,cs_hi", 1),
    ("cb384+cnxb (TTA1)", "cb384,cnxb", 1),
    ("3models_pseudo 0.5/0.25/0.25 (TTA1)", "cb384,cnxb,cs_hi", 1),
    ("3models_clean 0.4/0.3/0.3 (TTA1)", "cb384,cs_hi_all,cs_hi", 1),
    ("cb384 (TTA1, warm cache)", "cb384", 1),
]
WARMUP, N, SEED = 5, 100, 42
FILES = {
    "cb384": "cb384.pt",
    "cnxb": "cnxb_384_all_pseudo.pt",
    "cs_hi": "cs_hi512_all_ra.pt",
    "cs_hi_all": "cs_hi512_all_fp16.pt",
}


def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kw)


def ensure_repo():
    if not os.path.isdir(REPO):
        sh(["git", "clone", "-q", "--depth", "1", REPO_URL, REPO], check=True)


def dl(tag, name, want):
    dst = os.path.join(W, name)
    if os.path.isfile(dst) and os.path.getsize(dst) == want:
        return dst
    url = f"{REL}/{tag}/{name}"
    print("download", url, flush=True)
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, dst)
            break
        except Exception as e:  # noqa: BLE001
            print("retry", attempt, e, flush=True)
            time.sleep(5)
    print(f"  {name}: {os.path.getsize(dst)/1e6:.1f} MB", flush=True)
    return dst


def collect_images():
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".PNG")
    cands = []
    for root in glob.glob("/kaggle/input/*"):
        files = []
        for dirpath, _dirs, names in os.walk(root):
            for fn in names:
                if fn.endswith(exts):
                    files.append(os.path.join(dirpath, fn))
            if len(files) >= 3000:
                break
        if len(files) >= WARMUP + N:
            cands.append((len(files), root, files))
    if not cands:
        raise SystemExit("no images under /kaggle/input")
    cands.sort(reverse=True)
    _, root, files = cands[0]
    files.sort()
    random.Random(SEED).shuffle(files)
    return root, files[: WARMUP + N]


# ---------------------------------------------------------------- worker mode
def worker_main():
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchvision
    from PIL import Image

    sys.path.insert(0, os.path.join(REPO, "submit_pkg"))
    from predict import load_model  # noqa: E402

    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--combo", required=True, help="cb384,cnxb,cs_hi")
    ap.add_argument("--tta", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    device = "cuda"

    r = subprocess.run(["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
                       capture_output=True, text=True)
    rock = r.returncode == 0
    print(f"drop_caches: {'ok' if rock else 'FAILED ' + r.stderr.strip()}", flush=True)

    root, imgs = collect_images()
    print(f"images: {root} (n={len(imgs)})", flush=True)
    warm, timed = imgs[:WARMUP], imgs[WARMUP:WARMUP + N]

    keys = args.combo.split(",")
    models, metas = [], []
    for k in keys:
        m, meta = load_model(os.path.join(W, FILES[k]), device)
        models.append(m)
        metas.append(meta)
        print(f"loaded {k}: {meta}", flush=True)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    age_b = torch.tensor([[0.0]], dtype=torch.float32, device=device)
    flips = {1: [0], 2: [0, 1], 4: [0, 1, 2, 3]}[args.tta]

    def decode(path, draft):
        t0 = time.perf_counter()
        im = Image.open(path)
        im.draft("RGB", (draft, draft))
        im = im.convert("RGB")
        arr = np.asarray(im)
        return im, arr, (time.perf_counter() - t0) * 1000.0

    def crop_resize(im, arr, cache_size):
        t0 = time.perf_counter()
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
        out = Image.fromarray(crop).resize((cache_size, cache_size), Image.BICUBIC)
        return torchvision.transforms.functional.to_tensor(out), (time.perf_counter() - t0) * 1000.0

    def infer(path):
        cache = metas[0]["cache_size"]
        draft = max(768, int(cache * 1.5))
        im, arr, td = decode(path, draft)
        base, tc = crop_resize(im, arr, cache)
        tf = tp = 0.0
        for model, meta in zip(models, metas):
            x = base.unsqueeze(0).to(device)
            x = F.interpolate(x, size=(meta["img_size"],) * 2, mode="bilinear",
                              align_corners=False)
            x = (x - mean) / std
            with torch.no_grad():
                for fl in flips:
                    xf = x
                    if fl & 1:
                        xf = torch.flip(xf, dims=[3])
                    if fl & 2:
                        xf = torch.flip(xf, dims=[2])
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        logits = model(xf, age_b)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    torch.sigmoid(logits.float()).item()
                    torch.cuda.synchronize()
                    t2 = time.perf_counter()
                    tf += (t1 - t0) * 1000
                    tp += (t2 - t1) * 1000
        return {"file": os.path.basename(path), "decode": td, "crop+resize": tc,
                "forward": tf, "post": tp, "e2e": td + tc + tf + tp}

    for p in warm:
        infer(p)
    torch.cuda.synchronize()

    recs = [infer(p) for p in timed]

    def stats(key):
        a = np.asarray([r[key] for r in recs])
        return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
                "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
                "max": float(a.max())}

    slow = sorted(recs, key=lambda r: -r["e2e"])[:5]
    out = {"combo": args.combo, "tta": args.tta, "drop_caches": rock,
           "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "cuda": torch.version.cuda,
           "stats": {k: stats(k) for k in ("decode", "crop+resize", "forward", "post", "e2e")},
           "slowest": slow}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: {kk: round(vv, 2) for kk, vv in v.items()}
                      for k, v in out["stats"].items()}, ensure_ascii=False), flush=True)
    print("slowest:", json.dumps(slow, ensure_ascii=False), flush=True)


# ------------------------------------------------------------------- driver
def main():
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    ensure_repo()
    os.makedirs(W, exist_ok=True)
    # 子进程用持久路径调用自身（/kaggle/src 下的脚本可能在执行后被清理）
    src = SELF if os.path.isfile(SELF) else os.path.abspath(sys.argv[0])
    global RUNNER
    RUNNER = os.path.join(WORK, "self_runner.py")
    with open(src, "rb") as fi, open(RUNNER, "wb") as fo:
        fo.write(fi.read())
    print("runner copy:", RUNNER, os.path.getsize(RUNNER), flush=True)
    for tag, name, want in ASSETS:
        dl(tag, name, want)
    with zipfile.ZipFile(os.path.join(W, "submission_pkg_2model.zip")) as z:
        for fn in ("cb384.pt", "cnxb_384_all_pseudo.pt"):
            z.extract(fn, W)
    for fn in ("cb384.pt", "cnxb_384_all_pseudo.pt", "cs_hi512_all_ra.pt",
               "cs_hi512_all_fp16.pt"):
        print(f"{fn}: {os.path.getsize(os.path.join(W, fn))/1e6:.1f} MB", flush=True)

    rows = []
    for idx, (label, combo, tta) in enumerate(COMBOS):
        out = os.path.join(WORK, f"combo_{idx}.json")
        print(f"\n=== {label} ===", flush=True)
        rc = sh([sys.executable, RUNNER, "--worker", "--combo", combo,
                 "--tta", str(tta), "--out", out], cwd=WORK, check=False).returncode
        if rc != 0 or not os.path.isfile(out):
            print(f"!! combo failed rc={rc}", flush=True)
            continue
        with open(out, encoding="utf-8") as f:
            rows.append((label, json.load(f)))

    md = ["# T4 实测耗时（决赛口径）", ""]
    if rows:
        r0 = rows[0][1]
        md += [f"- 设备：{r0['gpu']}",
               f"- torch {r0['torch']} / CUDA {r0['cuda']}",
               "- 口径：随机 5 张预热 + 随机 100 张单轮、batch=1、端到端算术平均；seed=42",
               "- 每个组合独立进程运行，计时前清页面缓存（冷盘口径）",
               f"- 计时前清缓存：{'成功' if r0['drop_caches'] else '失败（热缓存，数字偏乐观）'}",
               "- 图像来源：/kaggle/input 公开眼底数据集（非官方 test 集，仅作 T4 量级证据）", ""]
    md += ["| 组合 | 端到端 mean | p50 | p95 | p99 | max | 判定 |",
           "|---|---|---|---|---|---|---|"]
    for label, d in rows:
        e = d["stats"]["e2e"]
        verdict = "PASS" if e["mean"] <= 100 else "FAIL"
        margin = "OK(<=85)" if e["mean"] <= 85 else ("RISK(85-100)" if e["mean"] <= 100 else "OVER")
        md.append(f"| {label} | {e['mean']:.2f} | {e['p50']:.2f} | {e['p95']:.2f} | "
                  f"{e['p99']:.2f} | {e['max']:.2f} | {verdict} ({margin}) |")
    md += ["", "## 分段拆解（ms/张，mean）", "",
           "| 组合 | decode | crop+resize | forward | post | e2e |",
           "|---|---|---|---|---|---|"]
    for label, d in rows:
        s = d["stats"]
        md.append(f"| {label} | {s['decode']['mean']:.2f} | {s['crop+resize']['mean']:.2f} | "
                  f"{s['forward']['mean']:.2f} | {s['post']['mean']:.2f} | {s['e2e']['mean']:.2f} |")
    md += ["", "## 最慢 5 张（e2e ms）", ""]
    for label, d in rows:
        slow = ", ".join(f"{r['file']}={r['e2e']:.1f}" for r in d["slowest"])
        md.append(f"- {label}: {slow}")

    with open(os.path.join(WORK, "T4_LATENCY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(WORK, "t4_latency.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("\n".join(md), flush=True)

    # 输出目录只留结果，避免 425MB 权重拖慢 kernels output
    for fn in ("submission_pkg_2model.zip", "cb384.pt", "cnxb_384_all_pseudo.pt",
               "cs_hi512_all_ra.pt", "cs_hi512_all_fp16.pt"):
        p = os.path.join(W, fn)
        if os.path.exists(p):
            os.remove(p)
    for p in glob.glob(os.path.join(W, "*.zip")):
        os.remove(p)
    print("cleaned weights from /kaggle/working", flush=True)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker_main()
    else:
        main()