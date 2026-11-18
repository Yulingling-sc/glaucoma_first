"""禁 cv2 smoke：屏蔽 cv2 后跑包内 code/predict_and_cost.py 的 cost+predict 全流程。

  python tools/no_cv2_smoke.py --pkg "D:/_finals_zip/extract/队长名_提交材料" \
      --images finals/_smoke/imgs --meta finals/_smoke/meta.csv --out <临时目录>

断言：进程内任何 import cv2 / opencv 都会抛 ImportError；脚本须跑通并产出
submission.csv（行数=meta 行数、无 NaN）与 cost.json。
"""
import argparse
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


class _BlockCv2:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in BLOCKED or fullname.startswith("cv2."):
            raise ImportError(f"cv2 is blocked by no_cv2_smoke.py ({fullname})")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.meta_path.insert(0, _BlockCv2())
    for m in list(sys.modules):
        if m == "cv2" or m.startswith("cv2."):
            del sys.modules[m]

    entry = os.path.join(os.path.abspath(args.pkg), "code", "predict_and_cost.py")
    spec = importlib.util.spec_from_file_location("finals_entry", entry)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fn, prefix in EXPECTED.items():
        params = list(inspect.signature(getattr(mod, fn)).parameters)
        assert params[:len(prefix)] == prefix, f"{fn} signature {params} != {prefix}..."
    assert not any(m == "cv2" or m.startswith("cv2.") for m in sys.modules), "cv2 imported"

    sys.argv = [entry, "--images", os.path.abspath(args.images),
                "--meta", os.path.abspath(args.meta), "--out_dir", os.path.abspath(args.out),
                "--mode", "both"]
    runpy.run_path(entry, run_name="__main__")

    csv_path = os.path.join(args.out, "submission.csv")
    cost_path = os.path.join(args.out, "cost.json")
    rows = open(csv_path, encoding="utf-8").read().strip().splitlines()
    expect = len(open(args.meta, encoding="utf-8-sig").read().strip().splitlines())
    assert os.path.isfile(cost_path), "missing cost.json"
    assert len(rows) == expect, f"rows {len(rows)} != meta {expect}"
    assert "nan" not in " ".join(rows).lower(), "NaN in submission"
    cost = json.load(open(cost_path, encoding="utf-8"))
    print(f"SMOKE OK: rows={len(rows)-1} avg_cost={cost['avg_cost_ms']:.2f}ms "
          f"(cv2 blocked, never imported)")


if __name__ == "__main__":
    main()