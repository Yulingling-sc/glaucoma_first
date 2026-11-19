# CPU valid/test inference under the PIL submission pipeline (submit_pkg/predict.py).
# Computes TTA1 and TTA4 probabilities in one pass (tensor flips).
# Usage:
#   python tools/finals_208_valid_pil.py --weights W.pt --split valid --out out.npy
import argparse, csv, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "submit_pkg"))
import predict as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--csv", default=os.path.join(_REPO, "glaucoma_first", "valid.csv"))
    ap.add_argument("--images", default=os.path.join(_REPO, "glaucoma_first", "valid"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--prep-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only first N rows (smoke)")
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = auto)")
    ap.add_argument("--force-cache-size", type=int, default=0, choices=[0, 512, 768],
                    help="override the checkpoint's cache_size (0 = infer); needed when "
                         "the checkpoint carries a wrong/absent cache_size (issue #212 trap)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>.partial.npz written by an interrupted run")
    args = ap.parse_args()

    torch.set_num_threads(args.threads or max(4, (os.cpu_count() or 8) - 2))
    device = "cpu"

    names, ages, paths = [], [], []
    with open(args.csv, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            stem = r["Image_name"].rsplit(".", 1)[0]
            p = None
            for ext in (".JPG", ".jpg", ".jpeg", ".png", ".PNG"):
                cand = os.path.join(args.images, stem + ext)
                if os.path.exists(cand):
                    p = cand
                    break
            if p is None:
                raise SystemExit(f"image not found for {r['Image_name']}")
            names.append(r["Image_name"])
            ages.append(P.clean_age(r.get("Age", P.AGE_MEDIAN)))
            paths.append(p)
    if args.limit:
        names, ages, paths = names[: args.limit], ages[: args.limit], paths[: args.limit]
    n = len(paths)
    print(f"rows={n}", flush=True)

    model, meta = P.load_model(args.weights, device)
    if args.force_cache_size:
        meta["cache_size"] = args.force_cache_size
    print(f"meta: {meta}", flush=True)
    cache_size = meta["cache_size"]
    draft = max(768, int(cache_size * 1.5))
    img_size = meta["img_size"]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    age_all = torch.tensor([[(a - P.AGE_MEDIAN) / 15.0] for a in ages], dtype=torch.float32)

    acc1 = np.zeros(n, dtype=np.float64)
    acc4 = np.zeros(n, dtype=np.float64)
    begin = 0
    if args.resume:
        ppath = args.out + ".partial.npz"
        if os.path.exists(ppath):
            z = np.load(ppath)
            begin = int(z["done"][0])
            acc1[:] = z["probs_tta1"][:n]
            acc4[:] = z["probs_tta4"][:n]
            print(f"resume: {begin}/{n} from {ppath}", flush=True)
    t0 = time.time()
    for start in range(begin, n, args.batch_size):
        idx = list(range(start, min(start + args.batch_size, n)))
        base = torch.stack([
            torchvision.transforms.functional.to_tensor(
                P.load_fundus_crop(paths[i], size=cache_size, draft=draft))
            for i in idx])
        x = F.interpolate(base, size=(img_size, img_size), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        age_b = age_all[idx]
        with torch.no_grad():
            for fl, acc in ((0, acc1), (1, acc4), (2, acc4), (3, acc4)):
                xf = x
                if fl & 1:
                    xf = torch.flip(xf, dims=[3])
                if fl & 2:
                    xf = torch.flip(xf, dims=[2])
                logits = model(xf, age_b)
                pr = torch.sigmoid(logits.float()).numpy()
                acc[idx] += pr
        if start % (args.batch_size * 20) == 0:
            el = time.time() - t0
            done = start + len(idx)
            print(f"{done}/{n} {el:.0f}s ({el / done * 1000:.0f} ms/img)", flush=True)
        if (start // args.batch_size) % 100 == 0 and start > 0:
            np.savez(args.out + ".partial", probs_tta1=acc1, probs_tta4=acc4,
                     names=np.array(names), done=np.array([start + len(idx)]))
    acc1 /= 1.0
    acc4 /= 4.0
    np.savez(args.out, probs_tta1=acc1, probs_tta4=acc4, names=np.array(names))
    print(f"saved {args.out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
