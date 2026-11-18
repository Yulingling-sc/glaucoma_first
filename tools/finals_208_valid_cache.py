# CPU valid inference from the rebuilt PIL cache512 (pipeline-isolation baseline).
# Mirrors tools/finals_208_valid_pil.py but skips decode+crop (reads cache JPGs directly).
import argparse, csv, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "submit_pkg"))
import predict as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--csv", default=os.path.join(_REPO, "glaucoma_first", "valid.csv"))
    ap.add_argument("--cache", default=r"D:\glaucoma_work\cache512_valid\valid")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = auto)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads or max(4, (os.cpu_count() or 8) - 2))
    names, ages, paths = [], [], []
    with open(args.csv, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            stem = r["Image_name"].rsplit(".", 1)[0]
            p = os.path.join(args.cache, stem + ".jpg")
            if not os.path.exists(p):
                raise SystemExit(f"cache image missing: {p}")
            names.append(r["Image_name"])
            ages.append(P.clean_age(r.get("Age", P.AGE_MEDIAN)))
            paths.append(p)
    n = len(paths)
    print(f"rows={n}", flush=True)

    model, meta = P.load_model(args.weights, "cpu")
    print(f"meta: {meta}", flush=True)
    img_size = meta["img_size"]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    age_all = torch.tensor([[(a - P.AGE_MEDIAN) / 15.0] for a in ages], dtype=torch.float32)

    acc1 = np.zeros(n, dtype=np.float64)
    acc4 = np.zeros(n, dtype=np.float64)
    t0 = time.time()
    for start in range(0, n, args.batch_size):
        idx = list(range(start, min(start + args.batch_size, n)))
        base = torch.stack([torchvision_read(paths[i]) for i in idx])
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
                acc[idx] += torch.sigmoid(logits.float()).numpy()
        if start % (args.batch_size * 20) == 0:
            el = time.time() - t0
            done = start + len(idx)
            print(f"{done}/{n} {el:.0f}s ({el / done * 1000:.0f} ms/img)", flush=True)
        if (start // args.batch_size) % 100 == 0 and start > 0:
            np.savez(args.out + ".partial", probs_tta1=acc1, probs_tta4=acc4,
                     names=np.array(names), done=np.array([start + len(idx)]))
    acc4 /= 4.0
    np.savez(args.out, probs_tta1=acc1, probs_tta4=acc4, names=np.array(names))
    print(f"saved {args.out} in {time.time() - t0:.0f}s", flush=True)


def torchvision_read(path):
    import torchvision
    from PIL import Image
    return torchvision.transforms.functional.to_tensor(Image.open(path).convert("RGB"))


if __name__ == "__main__":
    main()
