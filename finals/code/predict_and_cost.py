#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_and_cost.py —— 决赛官方优化模板（大数据·决赛）
【决赛包】选手代码区为 FundusNet 三成员集成；预处理全程 PIL，不依赖 cv2。

接口约定（与评测流水线 final_pipeline.py 完全兼容）:
    python code/predict_and_cost.py \
        --data_root ./test_second \
        --meta      ./test_second.csv \
        --out_dir   ./output \
        [--num_workers 4] [--batch_size 32]
    num_workers（默认 4）与 batch_size（默认 32）保持默认即可, 不建议修改。

产出（流水线依赖，勿改文件名/字段名）:
    output/submission.csv  列: Image_name, Probability（阳性概率, 0~1 连续值）,
                              Pred_Label（0/1 硬标签, Probability >= 0.5）
    output/cost.json       字段: avg_cost（毫秒, 流水线据此判 >100ms 报错）

元数据:
    meta 至少含 Image_name 列; Age 等其他列原样保留, 通过 predict_batch 的可选参数
    meta_rows 按行提供（默认 None, 不需要可忽略; 字段值均为字符串）。

口径说明:
    avg_cost = 单张"读图 + 预处理 + 推理 + 后处理"端到端耗时（batch=1）,
    随机抽 105 张, 前 5 张预热(不计时), 后 100 张计时; 剔除读图失败的样本。
    后处理（概率校准、TTA 融合、裁剪等）请写在 predict_batch 内 model(...) 之后,
    会自动计入该口径; 模板不另设独立的 postprocess 接口。
    固定配置（fp16、Pred_Label 阈值、cost 张数/预热张数/抽样种子）写死在脚本顶部,
    不从命令行传入。
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np
import pandas as pd
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # 包根目录（含 model/）
sys.path.insert(0, HERE)

from model import Ensemble, FundusNet  # noqa: E402
from utils import AGE_MEDIAN, clean_age  # noqa: E402
from utils.preprocess import crop_resize  # noqa: E402
from utils.weights import load_checkpoint  # noqa: E402


# ======================================================================
# 固定配置（写死, 不从命令行传入; 需要调整时直接改这里）
# ======================================================================
USE_FP16 = True          # CUDA 上启用 fp16 混合精度; CPU 自动退化为 fp32
PRED_THRESHOLD = 0.5     # Probability >= 该阈值判为阳性(1), 写入 Pred_Label
COST_IMAGES = 100        # 阶段一正式计时张数
COST_WARMUP = 5          # 阶段一预热张数（不计时）
COST_SEED = 2026         # 阶段一抽样随机种子

# 选手配置（固定, 不从命令行传入）
CACHE_SIZE = 512         # 训练缓存分辨率, 与权重一致
DRAFT = max(768, int(CACHE_SIZE * 1.5))  # JPEG draft 解码目标边, 与训练缓存一致
TTA = 1                  # 单尺度单次前向
MEMBERS = [
    ("cb384.pt", 0.40),
    ("cs_hi512_all_fp16.pt", 0.30),
    ("cs_hi512_all_ra.pt", 0.30),
]


# ======================================================================
# 选手代码区（正式包由选手实现; 本测试包用内置 demo 模型填充）
# 需要替换的接口: load_model / preprocess / predict_batch。
# predict_batch 的可选参数 meta_rows 会逐行传入 meta 原始字段（如 Age）, 默认 None。
# 后处理（概率校准、TTA 融合等）写在 predict_batch 内 model(...) 之后,
# 与读图/预处理/推理一起计入 avg_cost 端到端口径。
# ======================================================================

def _read_members(model_dir):
    """model/members.json 覆盖默认成员表：{"members":[{"file":..., "weight":...}]}"""
    p = os.path.join(model_dir, "members.json")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    items = data["members"]
    return [(m["file"], float(m.get("weight", 1.0 / len(items)))) for m in items]


def load_model(device):
    """加载模型并返回 torch.nn.Module（集成封装）。
    整个脚本只调用一次, cost 与全量预测共用同一模型对象, 不要在这里计时。"""
    device = torch.device(device)
    model_dir = os.path.join(ROOT, "model")
    members = _read_members(model_dir) or MEMBERS
    loaded = []
    for fn, w in members:
        sd, meta = load_checkpoint(os.path.join(model_dir, fn))
        m = FundusNet(arch=meta["arch"]).to(device)
        m.load_state_dict(sd, strict=True)
        m.eval()
        print(f"[model] {fn} arch={meta['arch']} img_size={meta['img_size']} "
              f"cache={meta['cache_size']} use={meta['use']} w={w}", flush=True)
        loaded.append((m, meta, w))
    return Ensemble(loaded, device=device)


def _preprocess_rgb(arr_rgb):
    """RGB uint8 HWC -> (3, S, S) float32 [0,1] numpy（与训练缓存同一裁剪/缩放）。"""
    arr = np.ascontiguousarray(arr_rgb)
    return crop_resize(Image.fromarray(arr), arr, CACHE_SIZE).numpy()


def preprocess(img_bgr):
    """单张图预处理: BGR ndarray -> (3, S, S) float32 numpy。
    在读图线程中执行（CPU, 不要碰 CUDA）; 逻辑与训练缓存一致：
    灰度>15 bbox（0.2/99.8 分位）-> 居中方形 -> PIL BICUBIC。"""
    arr = np.asarray(img_bgr)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expect HxWx3 BGR image, got {arr.shape}")
    return _preprocess_rgb(arr[..., ::-1])


def predict_batch(model, batch, device, use_fp16, meta_rows=None):
    """对一个 batch 推理（含后处理）。batch: (B, 3, H, W) float32 CPU 张量。
    meta_rows: 与 batch 逐行对齐的 list[dict], 字段值均为字符串。
    返回长度 B 的 float32 numpy 数组, 取值 [0, 1]。"""
    if isinstance(batch, torch.Tensor):
        tensors = [batch[i] for i in range(batch.shape[0])]
    else:
        tensors = [t if torch.is_tensor(t) else torch.from_numpy(np.asarray(t))
                   for t in batch]
    if meta_rows is None:
        ages = [AGE_MEDIAN] * len(tensors)
    else:
        ages = [clean_age(r.get("Age", AGE_MEDIAN)) for r in meta_rows]
    return model.predict(tensors, ages, tta=TTA, use_fp16=use_fp16)


# ======================================================================
# 以下为官方框架代码, 选手一般无需修改
# ======================================================================

STOP = None  # 任务队列结束哨兵


_WARNED = set()


def _warn_once(key, msg):
    """同一类问题只提示一次, 避免上万张图刷屏。"""
    if key not in _WARNED:
        _WARNED.add(key)
        print(msg, flush=True)


def build_file_index(data_root):
    """数据集文件名索引：容忍 meta 名与实际文件名的扩展名大小写/前后缀差异。
    返回 (精确文件名集合, {主干小写: 实际文件名}, {尾部数字编号: [实际文件名, ...]})。"""
    exact = set()
    by_stem = {}
    by_num = {}
    for f in os.listdir(data_root):
        exact.add(f)
        p = Path(f)
        by_stem.setdefault(p.stem.lower(), f)
        m = re.search(r"(\d+)$", p.stem)
        if m:
            by_num.setdefault(int(m.group(1)), []).append(f)
    return exact, by_stem, by_num


def resolve_image(name, data_root, index):
    """把 meta 里的 Image_name 解析成实际图片路径。
    依次尝试: 精确名 -> 文件名主干(大小写无关) -> 尾部数字编号(仅唯一命中时)。
    数字回退可能把预测错配到别的图上, 因此一旦启用就打印警告; 编号有歧义时拒绝回退。"""
    exact, by_stem, by_num = index
    if name in exact:
        return data_root / name
    stem = Path(name).stem.lower()
    f = by_stem.get(stem)
    if f:
        return data_root / f
    m = re.search(r"(\d+)", stem)
    if m:
        cands = by_num.get(int(m.group(1)), [])
        if len(cands) == 1:
            _warn_once(
                ("num_fallback", name, cands[0]),
                f"[warn] meta 名 {name!r} 未直接匹配, 按尾部编号回退为 {cands[0]!r}",
            )
            return data_root / cands[0]
        if len(cands) > 1:
            _warn_once(
                ("num_ambiguous", name, tuple(cands[:5])),
                f"[warn] meta 名 {name!r} 未直接匹配, 尾部编号 {m.group(1)} "
                f"命中多张图 {cands[:5]}..., 拒绝回退",
            )
            return data_root / name
    return data_root / name


def read_one(path):
    """读 + 预处理单张图; 失败返回 None。
    PIL JPEG draft 解码（与训练缓存同口径）, 不依赖 cv2。"""
    try:
        with Image.open(str(path)) as im:
            im.draft("RGB", (DRAFT, DRAFT))
            arr = np.asarray(im.convert("RGB"))
    except Exception:
        return None
    if arr.ndim != 3 or arr.shape[2] != 3:
        return None
    return _preprocess_rgb(arr)


def run_cost(model, meta, data_root, device, use_fp16,
             n_samples=100, warmup=5, seed=2026):
    """cost 基准: 从 meta 随机抽 n_samples + warmup 张,
    前 warmup 张预热(不计时), 后 n_samples 张逐个端到端计时
    （读图+预处理+推理+后处理, batch=1）, 返回 cost.json 内容。"""
    names = meta["Image_name"].dropna().astype(str).tolist()
    meta_rows = meta.loc[meta["Image_name"].notna()].to_dict("records")  # 与 names 逐行对齐
    row_by_name = {}
    for name, row in zip(names, meta_rows):
        row_by_name.setdefault(name, row)
    rng = random.Random(seed)
    rng.shuffle(names)
    index = build_file_index(data_root)
    total = warmup + n_samples
    picked = []
    for name in names:                       # 顺序补抽, 跳过数据集里不存在的图
        if len(picked) >= total:
            break
        if resolve_image(name, data_root, index).is_file():
            picked.append(name)
    warm_names = picked[:warmup]             # 前 warmup 张: 只预热, 不计时
    timed_names = picked[warmup:warmup + n_samples]  # 后 n_samples 张: 正式计时

    def once(name):
        arr = read_one(resolve_image(name, data_root, index))
        if arr is None:
            return None
        x = torch.from_numpy(arr[None])      # (1, 3, H, W)
        return predict_batch(model, x, device, use_fp16,
                             meta_rows=[row_by_name[name]])[0]

    for name in warm_names:                  # warmup 不计时（首帧含 CUDA 初始化）
        try:
            once(name)
        except Exception:
            pass

    times, infer_times, n_fail = [], [], 0
    for name in timed_names:                 # 端到端: 读图+预处理 → 推理+后处理
        t0 = time.perf_counter()
        try:
            x = read_one(resolve_image(name, data_root, index))
            if x is None:
                n_fail += 1
                continue
            xi = torch.from_numpy(x[None])
            t1 = time.perf_counter()
            predict_batch(model, xi, device, use_fp16,
                          meta_rows=[row_by_name[name]])
        except Exception:
            n_fail += 1
            continue
        t2 = time.perf_counter()
        times.append((t2 - t0) * 1000.0)
        infer_times.append((t2 - t1) * 1000.0)

    if not times:
        raise RuntimeError("cost 基准全部失败: 无一成样本可用")
    return {
        "avg_cost": round(float(np.mean(times)), 3),          # 流水线只认这个字段
        "avg_cost_infer": round(float(np.mean(infer_times)), 3),  # t1→t2: H2D+前向+后处理+D2H
        "n_samples": len(times),
        "n_fail": n_fail,
        "warmup": warmup,
        "seed": seed,
        "metric": "ms, 单张端到端(读图+预处理+推理+后处理), batch=1",
    }


def run_prediction(model, meta, data_root, out_dir, device,
                   num_workers, batch_size, use_fp16, pred_threshold=0.5,
                   progress_every=2000):
    """全量预测: 读图/预处理线程流水线 + 主线程动态攒 batch（推理含后处理）,
    输出 submission.csv（Image_name / Probability / Pred_Label, 按 meta 行序）。"""
    names = meta["Image_name"].dropna().astype(str).tolist()
    meta_rows = meta.loc[meta["Image_name"].notna()].to_dict("records")  # 与 names 逐行对齐
    n = len(names)
    results = np.full(n, np.nan, dtype=np.float32)           # 按索引回填, 天然保序
    index = build_file_index(data_root)
    task_q = Queue(maxsize=num_workers * 4)                  # 待读任务（有界）
    data_q = Queue(maxsize=batch_size * 2)                   # 已预处理数据（有界, 防止内存爆）

    # 读图线程只做 读图+预处理; 后处理在 predict_batch 内, 与推理同在主线程执行
    def reader_worker():
        while True:
            item = task_q.get()
            try:
                if item is STOP:
                    return
                idx, name = item
                try:
                    data_q.put((idx, read_one(Path(name))))
                except Exception:
                    data_q.put((idx, None))
            finally:
                task_q.task_done()

    def feeder():
        for idx, name in enumerate(names):
            task_q.put((idx, str(resolve_image(name, data_root, index))))
        for _ in range(num_workers):
            task_q.put(STOP)

    readers = [Thread(target=reader_worker, daemon=True) for _ in range(num_workers)]
    for t in readers:
        t.start()
    Thread(target=feeder, daemon=True).start()

    batch_idx, batch_arr = [], []
    n_missing, done = 0, 0
    t_start = time.perf_counter()

    def flush():
        """把当前攒好的 batch 送入 predict_batch（推理+后处理）, 按原索引回填结果。"""
        nonlocal batch_idx, batch_arr
        if not batch_arr:
            return
        x = torch.from_numpy(np.stack(batch_arr))
        probs = predict_batch(model, x, device, use_fp16,
                              meta_rows=[meta_rows[i] for i in batch_idx])
        for i, p in zip(batch_idx, probs):
            results[i] = p
        batch_idx, batch_arr = [], []

    while done < n:
        idx, arr = data_q.get()
        done += 1
        if arr is None:
            n_missing += 1
        else:
            batch_idx.append(idx)
            batch_arr.append(arr)
            if len(batch_arr) >= batch_size:
                flush()
        if done % progress_every == 0 or done == n:
            speed = done / max(time.perf_counter() - t_start, 1e-6)
            print(f"[predict] {done}/{n} ({speed:.1f} 张/s)", flush=True)

    flush()  # 收尾: 刷出不足一个 batch 的尾部数据(否则 15512%32=24 条尾部样本丢失)

    n_nan = int(np.isnan(results).sum())
    if n_nan:
        results = np.where(np.isnan(results), 0.5, results)  # 兜底: 个别图不可读时给中性概率, 保证覆盖全部 meta 行
        print(f"[predict] {n_nan} 张不可读, 已按 0.5 中性概率兜底", flush=True)

    labels = (results >= pred_threshold).astype(np.int64)   # 0/1 硬标签, 与 Probability 同口径
    sub = pd.DataFrame({"Image_name": names, "Probability": results,
                        "Pred_Label": labels})
    sub_path = out_dir / "submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"[predict] 完成: 覆盖 {len(sub)}/{n}, 读图失败 {n_missing}, 输出 {sub_path}", flush=True)
    return len(sub)


def main():
    ap = argparse.ArgumentParser(description="决赛预测+耗时评测（官方优化模板）")
    ap.add_argument("--data_root", "--data-root", "--images", dest="data_root",
                    required=True, help="图片目录（官方参数名 --data_root）")
    ap.add_argument("--meta", required=True, help="元数据 csv（含 Image_name 列, 可含 Age 等附加列）")
    ap.add_argument("--out_dir", "--out-dir", "--out", dest="out_dir",
                    required=True, help="输出目录")
    # num_workers / batch_size 均有默认值, 一般无需传入, 不建议修改
    ap.add_argument("--num_workers", type=int, default=4,
                    help="读图线程数（默认 4, 不建议修改）")
    ap.add_argument("--batch_size", type=int, default=32,
                    help="推理(含后处理) batch 大小（默认 32, 不建议修改）")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = USE_FP16 and device.type == "cuda"
    torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # utf-8-sig 兼容 Excel 导出的 BOM; dtype=str 防止纯数字 ID 被推断成 int 丢前导零
    meta = pd.read_csv(args.meta, encoding="utf-8-sig", dtype=str)
    if "Image_name" not in meta.columns:
        raise SystemExit(f"meta 缺少 Image_name 列, 实际列: {list(meta.columns)}")
    print(f"[init] device={device} fp16={use_fp16} num_workers={args.num_workers} "
          f"batch_size={args.batch_size} 样本数={len(meta)}", flush=True)

    t0 = time.perf_counter()
    model = load_model(device)
    model.eval()
    print(f"[init] 模型加载完成, 耗时 {time.perf_counter() - t0:.1f}s", flush=True)

    # 阶段一: cost 基准（与全量预测共用模型）
    cost = run_cost(model, meta, data_root, device, use_fp16,
                    n_samples=COST_IMAGES, warmup=COST_WARMUP, seed=COST_SEED)
    (out_dir / "cost.json").write_text(
        json.dumps(cost, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[cost] avg_cost={cost['avg_cost']}ms "
          f"(其中推理+后处理 {cost['avg_cost_infer']}ms, n={cost['n_samples']}, 失败 {cost['n_fail']})", flush=True)

    # 阶段二: 全量预测
    run_prediction(model, meta, data_root, out_dir, device,
                   num_workers=args.num_workers, batch_size=args.batch_size,
                   use_fp16=use_fp16, pred_threshold=PRED_THRESHOLD)

    print("[done] output/submission.csv 与 output/cost.json 已生成", flush=True)


if __name__ == "__main__":
    main()
