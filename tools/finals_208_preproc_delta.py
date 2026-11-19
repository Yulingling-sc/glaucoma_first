# cv2 vs PIL preprocessing comparison for the finals package (issue #208 §0 / #207 §0.2).
# Part A: stage-wise pixel deltas on the 512x512 model input (cv2 reconstruction vs
#         submit_pkg PIL pipeline), over N valid images.
# Part B: 20-test-image probability deltas for the regional C0 trio (TTA1, batch=1).
import argparse, csv, json, os, sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "submit_pkg"))
import predict as P  # noqa: E402


def pil_crop(path, size, draft):
    return np.asarray(P.load_fundus_crop(path, size=size, draft=draft), dtype=np.float64)


def pick_reduce(w, h, target):
    # mirror PIL draft semantics: largest 2^n shrink whose result is not smaller than target
    for r in (8, 4, 2, 1):
        if w // r >= target and h // r >= target:
            return r
    return 1


def cv2_crop(path, size, draft):
    import cv2
    with open(path, "rb") as fh:
        data = np.frombuffer(fh.read(), dtype=np.uint8)
    head = cv2.imdecode(data, cv2.IMREAD_COLOR)  # full decode once to learn size
    h0, w0 = head.shape[:2]
    r = pick_reduce(w0, h0, draft)
    flag = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2,
            4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}[r]
    im = cv2.imdecode(data, flag)
    arr = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    w, h = arr.shape[1], arr.shape[0]
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
    out = cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)
    return out.astype(np.float64), r


def part_a(n_img, size, draft):
    import torch
    torch.set_num_threads(8)
    csv_path = os.path.join(_REPO, "glaucoma_first", "valid.csv")
    img_dir = os.path.join(_REPO, "glaucoma_first", "valid")
    with open(csv_path, encoding="utf-8") as f:
        rows = [r["Image_name"] for r in csv.DictReader(f)][:n_img]
    dmax = dmean = 0.0
    reduces = {}
    per_img = []
    for name in rows:
        stem = name.rsplit(".", 1)[0]
        p = None
        for ext in (".JPG", ".jpg"):
            cand = os.path.join(img_dir, stem + ext)
            if os.path.exists(cand):
                p = cand
                break
        a = pil_crop(p, size, draft)
        b, r = cv2_crop(p, size, draft)
        reduces[r] = reduces.get(r, 0) + 1
        d = np.abs(a - b)
        per_img.append(float(d.max()))
        dmax = max(dmax, float(d.max()))
        dmean += float(d.mean())
    dmean /= len(rows)
    return {"n": len(rows), "input_size": size, "draft_target": draft,
            "cv2_reduce_used": reduces, "max_abs_delta": dmax,
            "mean_abs_delta": dmean,
            "per_img_max_top5": sorted(per_img, reverse=True)[:5]}


def part_b(n_img, weights, blend, out_json):
    import torch
    import torch.nn.functional as F
    import torchvision
    torch.set_num_threads(16)
    csv_path = os.path.join(_REPO, "glaucoma_first", "test.csv")
    img_dir = os.path.join(_REPO, "glaucoma_first", "test")
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    step = max(1, len(rows) // n_img)
    rows = rows[::step][:n_img]

    models = []
    for wpath in weights:
        m, meta = P.load_model(wpath, "cpu")
        models.append((m, meta))
        print(f"loaded {wpath}: {meta}", flush=True)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    results = []
    for r in rows:
        stem = r["Image_name"].rsplit(".", 1)[0]
        p = None
        for ext in (".JPG", ".jpg"):
            cand = os.path.join(img_dir, stem + ext)
            if os.path.exists(cand):
                p = cand
                break
        age = P.clean_age(r.get("Age", P.AGE_MEDIAN))
        age_t = torch.tensor([[(age - P.AGE_MEDIAN) / 15.0]], dtype=torch.float32)
        probs = {"pil": [], "cv2": []}
        for variant in ("pil", "cv2"):
            if variant == "pil":
                cache_size = models[0][1]["cache_size"]
                draft = max(768, int(cache_size * 1.5))
                arr = pil_crop(p, cache_size, draft)
            else:
                arr, _ = cv2_crop(p, models[0][1]["cache_size"],
                                  max(768, int(models[0][1]["cache_size"] * 1.5)))
            base = torchvision.transforms.functional.to_tensor(
                Image_fromarray(arr.astype(np.uint8))).unsqueeze(0)
            for m, meta in models:
                x = F.interpolate(base, size=(meta["img_size"], meta["img_size"]),
                                  mode="bilinear", align_corners=False)
                x = (x - mean) / std
                with torch.no_grad():
                    logits = m(x, age_t)
                probs[variant].append(float(torch.sigmoid(logits).item()))
        pp = float(np.dot(probs["pil"], blend))
        pc = float(np.dot(probs["cv2"], blend))
        results.append({
            "image": r["Image_name"],
            "p_pil": round(pp, 6),
            "p_cv2": round(pc, 6),
            "abs_delta": round(abs(pp - pc), 6),
            "per_model_pil": [round(v, 6) for v in probs["pil"]],
            "per_model_cv2": [round(v, 6) for v in probs["cv2"]],
        })
        print(f"{r['Image_name']} p_pil={pp:.5f} p_cv2={pc:.5f} d={abs(pp - pc):.6f}", flush=True)

    deltas = [x["abs_delta"] for x in results]
    flips = sum(1 for x in results if (x["p_pil"] >= 0.5) != (x["p_cv2"] >= 0.5))
    summary = {"n": len(results), "blend": blend, "max_abs_delta": max(deltas),
               "mean_abs_delta": float(np.mean(deltas)), "label_flips": flips}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": results}, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary), flush=True)


def Image_fromarray(arr):
    from PIL import Image
    return Image.fromarray(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "b", "ab"], default="ab")
    ap.add_argument("--n-valid", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--weights", nargs="+", default=[
        os.path.join(_REPO, "submit_pkg", "cb384.pt"),
        r"D:\glaucoma_work\finals_208_assets\cs_hi512_all_fp16.pt",
        os.path.join(_REPO, "submit_pkg", "cs_hi512_all_ra.pt"),
    ])
    ap.add_argument("--blend", nargs="+", type=float, default=[0.5, 0.25, 0.25])
    ap.add_argument("--out-json", default=r"D:\glaucoma_work\finals_208_assets\preproc_delta_20test.json")
    args = ap.parse_args()

    if args.part in ("a", "ab"):
        res = part_a(args.n_valid, size=512, draft=768)
        print("PART A:", json.dumps(res), flush=True)
        with open(args.out_json.replace(".json", "_partA.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    if args.part in ("b", "ab"):
        part_b(args.n_test, args.weights, args.blend, args.out_json)


if __name__ == "__main__":
    main()
