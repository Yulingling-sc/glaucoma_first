#!/usr/bin/env bash
# 官方镜像干净沙箱验收（#207 零-4）。需要：Docker + nvidia-container-toolkit + T4。
#
#   bash tools/finals_sandbox_check.sh \
#     --pkg /path/to/解压后的/董明坤_提交材料 \
#     --images /path/to/test_images --meta /path/to/test.csv --out /tmp/finals_sandbox_out
#
# 断言：不联网、不 pip install、无 ImportError；out/submission.csv 行数=meta 行数、
#       Probability 无 NaN、out/cost.json 存在、avg_cost ≤ 100ms、关键日志齐全。
set -euo pipefail

IMAGE="crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete:3.0"
PKG=""; IMAGES=""; META=""; OUT=""

while [ $# -gt 0 ]; do
  case "$1" in
    --pkg) PKG="$2"; shift 2;;
    --images) IMAGES="$2"; shift 2;;
    --meta) META="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --image) IMAGE="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

[ -n "$PKG" ] && [ -n "$IMAGES" ] && [ -n "$META" ] && [ -n "$OUT" ] || {
  echo "usage: $0 --pkg DIR --images DIR --meta CSV --out DIR [--image IMAGE]" >&2; exit 2; }

PKG=$(cd "$PKG" && pwd); IMAGES=$(cd "$IMAGES" && pwd); META=$(cd "$(dirname "$META")" && pwd)/$(basename "$META")
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
LOG="$OUT/run.log"

echo "[1/3] run inside official image (offline, no pip) — official CLI shape"
docker run --rm --gpus all --network none \
  -v "$PKG":/pkg:ro -v "$IMAGES":/images:ro -v "$META":/meta.csv:ro -v "$OUT":/out \
  "$IMAGE" bash -lc \
  "cd /pkg && python code/predict_and_cost.py --data_root /images --meta /meta.csv --out_dir /out --num_workers 4 --batch_size 32" \
  2>&1 | tee "$LOG"

echo "[2/3] assertions"
python - "$META" "$OUT" <<'PY'
import csv, json, os, sys
meta, out = sys.argv[1], sys.argv[2]
log = open(os.path.join(out, "run.log"), encoding="utf-8", errors="replace").read()
assert "ImportError" not in log, "ImportError in run.log"
assert "[cost] avg_cost=" in log, "missing cost log line"
assert "[done]" in log, "missing [done] log line"
csv_path = os.path.join(out, "submission.csv")
cost_path = os.path.join(out, "cost.json")
assert os.path.isfile(csv_path), "missing submission.csv"
assert os.path.isfile(cost_path), "missing cost.json"
rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
n_meta = sum(1 for _ in csv.DictReader(open(meta, encoding="utf-8-sig")))
assert len(rows) == n_meta, f"rows {len(rows)} != meta {n_meta}"
assert list(rows[0].keys()) == ["Image_name", "Probability", "Pred_Label"], rows[0].keys()
assert all(r["Probability"].lower() != "nan" for r in rows), "NaN probability"
assert all(r["Pred_Label"] in ("0", "1") for r in rows), "bad Pred_Label"
cost = json.load(open(cost_path, encoding="utf-8"))
avg = cost["avg_cost"]
assert avg <= 100, f"avg_cost {avg:.2f}ms > 100ms"
print(f"SANDBOX OK: rows={len(rows)} avg_cost={avg:.2f}ms "
      f"infer={cost['avg_cost_infer']:.2f}ms n={cost['n_samples']} fail={cost['n_fail']}")
PY

echo "[3/3] done: $OUT/submission.csv $OUT/cost.json $LOG"
