"""按决赛结构打包：#207 第 2 项。

  python tools/finals_build_pkg.py --src finals --name "<队长名>_提交材料" [--out D:/]

校验并生成：
  - SHA256SUMS.txt（包内相对路径 -> sha256）
  - <name>.zip（≤2G；解压后 code+model+licenses+xlsx ≤500MB）
  - 结构检查：code/{train/,predict_and_cost.py,model.py,utils/,requirements.txt}
               model/ licenses/ 模型核验材料与数据申明清单.xlsx
               不允许 pretrain_model/、submission.csv
"""
import argparse
import hashlib
import os
import sys
import zipfile

REQUIRED = [
    "code/predict_and_cost.py",
    "code/model.py",
    "code/requirements.txt",
    "code/train/",
    "code/utils/",
    "model/",
    "licenses/",
]
FORBIDDEN = ["pretrain_model", "submission.csv"]
XLSX = "模型核验材料与数据申明清单.xlsx"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="finals")
    ap.add_argument("--name", required=True, help='zip 名（不含 .zip），如 "陈家豪_提交材料"')
    ap.add_argument("--out", default=None, help="zip 输出目录（默认 src 的上一级）")
    ap.add_argument("--skip-zip", action="store_true")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    errors, warns = [], []
    for r in REQUIRED:
        if not os.path.exists(os.path.join(src, r)):
            errors.append(f"missing {r}")
    if not os.path.exists(os.path.join(src, XLSX)):
        errors.append(f"missing {XLSX}")

    total = 0
    files = []
    for dirpath, dirs, names in os.walk(src):
        dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith("_")]
        for fn in names:
            if fn.endswith(".pyc") or fn == "SHA256SUMS.txt":
                continue
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, src).replace("\\", "/")
            if rel.startswith(("pretrain_model", "submission.csv")):
                errors.append(f"forbidden file in package: {rel}")
            total += os.path.getsize(p)
            files.append((p, rel))

    if total > 500 * 1000 * 1000:
        errors.append(f"uncompressed size {total/1e6:.1f} MB > 500 MB")

    sha_lines = [f"{sha256(p)}  {rel}" for p, rel in sorted(files, key=lambda x: x[1])]
    with open(os.path.join(src, "SHA256SUMS.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(sha_lines) + "\n")

    print(f"files={len(files)} uncompressed={total/1e6:.1f} MB")
    for w in warns:
        print("WARN:", w)
    for e in errors:
        print("ERR :", e)
    if errors:
        sys.exit(1)

    if args.skip_zip:
        return
    out_dir = os.path.abspath(args.out or os.path.dirname(src))
    os.makedirs(out_dir, exist_ok=True)
    zpath = os.path.join(out_dir, args.name + ".zip")
    root_prefix = args.name
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for p, rel in sorted(files, key=lambda x: x[1]):
            z.write(p, f"{root_prefix}/{rel}")
        z.write(os.path.join(src, "SHA256SUMS.txt"),
                f"{root_prefix}/SHA256SUMS.txt")
    print(f"wrote {zpath} ({os.path.getsize(zpath)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()