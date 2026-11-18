"""禁 cv2 smoke：屏蔽 cv2 后按官方命令形态跑包内 code/predict_and_cost.py。

  python tools/no_cv2_smoke.py --pkg finals --images <图目录> --meta <csv> --out <目录> [--add-missing]

断言：
  - 进程内任何 import cv2 / opencv 都会抛 ImportError；
  - 三接口签名 = load_model(device) / preprocess(img_bgr) /
    predict_batch(model, batch, device, use_fp16, meta_rows...)；
  - 官方 CLI（--data_root/--meta/--out_dir）跑通，submission.csv 行数=meta 行数、无 NaN；
  - cost.json 含 avg_cost；
  - （--add-missing）meta 追加一条不存在的图，输出仍覆盖该行且概率=0.5。
"""
import argparse
import csv
import importlib.util
import inspect
import json
import os
import runpy
import sys

BLOCKED = ("cv2", "opencv", "cv2.cv2")
EXPECTED = {
    "load_model": ["device"],
    "preprocess": ["img_bgr"],
    "predict_batch": ["model", "batch", "device", "use_fp16", "meta_rows"],
}
MISSING_NAME = "__missing_for_smoke__.jpg"


class _BlockCv2:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in BLOCKED or fullname.startswith("cv2."):
            raise ImportError(f"cv2 is blocked by no_cv2_smoke.py ({fullname})")
        return None


def _write_meta_with_missing(src, dst):
    with open(src, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    rows.append([MISSING_NAME, "58"])
    with open(dst, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    return len(rows) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--add-missing", action="store_true")
    args = ap.parse_args()

    sys.meta_path.insert(0, _BlockCv2())
    for m in list(sys.modules):
        if m == "cv2" or m.startswith("cv2."):
            del sys.modules[m]

    os.makedirs(args.out, exist_ok=True)
    meta = os.path.abspath(args.meta)
    expect = sum(1 for _ in csv.DictReader(open(meta, encoding="utf-8-sig")))
    if args.add_missing:
        meta = os.path.join(os.path.abspath(args.out), "meta_with_missing.csv")
        expect = _write_meta_with_missing(args.meta, meta)

    entry = os.path.join(os.path.abspath(args.pkg), "code", "predict_and_cost.py")
    spec = importlib.util.spec_from_file_location("finals_entry", entry)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fn, prefix in EXPECTED.items():
        params = list(inspect.signature(getattr(mod, fn)).parameters)
        assert params[:len(prefix)] == prefix, f"{fn} signature {params} != {prefix}..."
    assert not any(m == "cv2" or m.startswith("cv2.") for m in sys.modules), "cv2 imported"

    sys.argv = [entry, "--data_root", os.path.abspath(args.images),
                "--meta", meta, "--out_dir", os.path.abspath(args.out)]
    runpy.run_path(entry, run_name="__main__")

    csv_path = os.path.join(args.out, "submission.csv")
    cost_path = os.path.join(args.out, "cost.json")
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    assert os.path.isfile(cost_path), "missing cost.json"
    assert len(rows) == expect, f"rows {len(rows)} != meta {expect}"
    assert all(r["Probability"].lower() != "nan" for r in rows), "NaN in submission"
    cost = json.load(open(cost_path, encoding="utf-8"))
    assert "avg_cost" in cost, "cost.json missing avg_cost"
    if args.add_missing:
        hit = [r for r in rows if r["Image_name"] == MISSING_NAME]
        assert hit, "missing-image row not covered"
        assert abs(float(hit[0]["Probability"]) - 0.5) < 1e-6, "missing image != 0.5"
    print(f"SMOKE OK: rows={len(rows)} avg_cost={cost['avg_cost']:.2f}ms "
          f"n_fail={cost.get('n_fail')} (cv2 blocked, never imported)")


if __name__ == "__main__":
    main()