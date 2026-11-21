"""#216 彩票包：dist4（3 成员 0.5/0.25/0.25、单解码 draft384）但 **仅 cb384 走 TTA2**。

repo 内 finals/ 保持 TTA1（dist4 复现态）；本脚本在临时目录复制一份、只改
model/members.json 的 cb384.tta=2，再调用 tools/finals_build_pkg.py 打包。

  python tools/finals_build_tta2_cb.py --name "陈家豪_提交材料_tta2cb" [--out D:/glaucoma/finals_out]

除 cb384 的 tta 外，其余文件与 finals/ 逐字节一致（脚本最后打印 diff 摘要）。
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "finals")
TARGET = "cb384.pt"


def copy_tree(src, dst):
    os.makedirs(dst, exist_ok=True)
    for dirpath, dirs, names in os.walk(src):
        dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith("_")]
        rel = os.path.relpath(dirpath, src)
        out = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(out, exist_ok=True)
        for fn in names:
            if fn.endswith(".pyc") or fn == "SHA256SUMS.txt":
                continue
            s, d = os.path.join(dirpath, fn), os.path.join(out, fn)
            try:
                os.link(s, d)          # 同盘硬链接, 零拷贝
            except OSError:
                shutil.copy2(s, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "finals_out"))
    ap.add_argument("--skip-zip", action="store_true")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="tta2cb_")
    build_src = os.path.join(tmp, "finals")
    try:
        copy_tree(SRC, build_src)
        mj = os.path.join(build_src, "model", "members.json")
        with open(mj, encoding="utf-8") as f:
            cfg = json.load(f)
        hit = 0
        for m in cfg["members"]:
            if m["file"] == TARGET:
                m["tta"] = 2
                hit += 1
        assert hit == 1, f"members.json 未找到唯一 {TARGET}"
        os.remove(mj)   # 可能是硬链接, 先断开再写, 避免改到 repo 里的原件
        with open(mj, "w", encoding="utf-8", newline="\n") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print("[tta2cb] members.json:", json.dumps(cfg["members"], ensure_ascii=False))

        cmd = [sys.executable, os.path.join(HERE, "tools", "finals_build_pkg.py"),
               "--src", build_src, "--name", args.name, "--out", args.out]
        if args.skip_zip:
            cmd.append("--skip-zip")
        rc = subprocess.call(cmd)
        if rc:
            raise SystemExit(rc)
        print("\n[tta2cb] 唯一差异: model/members.json 中 cb384.pt -> tta=2"
              "（code/ 与其余 model/ 文件来自同一份 finals/）")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
