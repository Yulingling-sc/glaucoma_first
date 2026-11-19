# T4 耗时实测（决赛口径，2026-11-18）

> 设备 **Kaggle Tesla T4 15GB**；镜像内 torch 2.11.0+cu130 / Python 3.13，
> **不是**官方决赛镜像（torch 2.10.0 / CUDA 12.8 / Python 3.12）。
> 本表用于选型与合规判断；官方镜像数字以决赛平台实测为准。

## 口径

- 随机 5 张预热（不计时）+ 随机 100 张、只跑 1 轮、`batch=1`、端到端**算术平均**（决赛口径）。
- 每张图 e2e = decode + crop+resize + forward + post(sigmoid)；forward 前后 `torch.cuda.synchronize()`。
- 每个组合独立进程运行；组合间尝试清页面缓存（Kaggle 容器 `/proc/sys/vm/drop_caches` 只读，**实际未清**，
  第一个跑的组合要吃冷盘）。
- 图像：ACRIMA 公开眼底数据集 105 张（`seed=42`），**非官方 test 集**；官方 15512 张若分辨率更大，decode/crop 会更高。
- 权重均为 fp16 瘦身权重重载为 fp32，全部 cache512。

## 结果（v4，ms/张）

| 组合 | 端到端 mean | p50 | p95 | p99 | max | 判定（mean≤100） |
|---|---|---|---|---|---|---|
| cnxb_all_pseudo（TTA1） | 79.69* | 75.30 | 117.65 | 135.67 | 139.58 | PASS*（冷缓存，修正 ≈44） |
| cs_hi512_all_ra（TTA1） | 41.21 | 34.62 | 67.61 | 83.45 | 108.37 | PASS（≤85 目标内） |
| cb384+cs_hi512_all_ra（TTA1） | 58.30 | 51.94 | 85.54 | 100.38 | 113.27 | PASS（≤85 目标内） |
| cb384+cnxb（TTA1） | 57.78 | 51.46 | 85.88 | 100.74 | 111.38 | PASS（≤85 目标内） |
| **3 模型 clean 0.4/0.3/0.3（TTA1）** | **74.40** | 68.47 | 103.58 | 116.61 | 137.29 | **PASS（目标 ≤85 达成）** |
| 3 模型 pseudo 0.5/0.25/0.25（TTA1） | 75.36 | 69.43 | 102.80 | 118.28 | 139.73 | PASS（目标 ≤85 达成） |
| cb384（TTA1，热缓存复测） | 41.80 | 35.95 | 66.82 | 91.00 | 103.47 | PASS（≤85 目标内） |
| 3 模型（TTA2，v3 实测） | 133.44 | 128.08 | 159.75 | 176.06 | 193.93 | **FAIL（超线 33%）** |

\* cnxb 是 v4 第一个跑的组合，decode 均值 39.12ms（其余 3.5–3.7ms）；按热缓存口径修正 ≈44.1ms（3.6+19.5+20.9+0.1）。

## 分段拆解（ms/张，mean）

| 组合 | decode | crop+resize | forward | post | e2e |
|---|---|---|---|---|---|
| cnxb_all_pseudo（冷缓存） | 39.12 | 19.53 | 20.91 | 0.12 | 79.69 |
| cs_hi512_all_ra | 3.60 | 19.87 | 17.62 | 0.11 | 41.21 |
| cb384+cs_hi512_all_ra | 3.53 | 19.54 | 35.01 | 0.21 | 58.30 |
| cb384+cnxb | 3.59 | 19.50 | 34.48 | 0.21 | 57.78 |
| 3 模型 clean | 3.63 | 19.37 | 51.08 | 0.31 | 74.40 |
| 3 模型 pseudo | 3.70 | 19.50 | 51.85 | 0.31 | 75.36 |
| cb384（热缓存） | 3.58 | 19.48 | 18.62 | 0.11 | 41.80 |

单成员 forward：cb384 ≈18.6–22.0，cs_hi512_all_ra ≈17.6，cnxb ≈19.6–20.9（同一 ConvNeXt-Base@384 的两次测量有 ±2ms 抖动）。
链路可加性成立，未测组合可按 `e2e ≈ decode + crop + Σforward + post` 估算。

## 结论

1. **推荐的干净 3 成员包（cb384 + cs_hi512_all + cs_hi512_all_ra）TTA1 = 74.40ms，满足 ≤85ms 目标**，
   100ms 红线余量 26%；p95 103.58ms 略超 100ms，但官方口径取算术平均，p95 不参与判定。
2. **TTA2 不可行**（133.44ms，forward 翻倍）。减少 1 个成员后 TTA2 仍 ≈115ms，压不回 100ms。
3. **干净 2 成员（cb384 + cs_hi512_all_ra）= 58.30ms**，是官方镜像数字不达标时最稳的降级档（包体 276MB）。
4. 主要非模型开销是 **crop+resize ≈19.4ms（p95 可到 44ms）**，其次 decode（热缓存 ≈3.6ms）；
   官方 test 图若比 ACRIMA 大，这两项会涨，是当前最大的不确定性。
5. 环境差异：官方 torch 2.10.0/CUDA 12.8 vs 本表 torch 2.11.0/CUDA 13.0，conv 前后向可能差 ±10%；
   74.40ms 即使慢 15% 也 ≈85.6ms，仍低于 100ms。
6. **适用边界（#207 P1-5）**：74.40ms 基于 ACRIMA 349px 小图，decode 仅 3.6ms；官方决赛图大概率
   2000–3000px，`draft(768)` 解码与灰度分位 bbox 的开销都会上涨，**decode+crop 预估 +10–20ms**。
   p95 已 103.58ms，官方按 mean 判但余量有限；拿到真实尺寸眼底图后需用 10–20 张复测一次。

## 复现

- 内核：`jiahaochen7711/g207-t4-latency`（私有脚本，`tools/kaggle_finals/run.py`，单文件 v4）。
- 一键重跑：`python -m kaggle kernels push -p tools/kaggle_finals`。
- 包内自查（任意 GPU）：`python code/predict_and_cost.py --data_root ... --meta ... --out_dir out`，
  `out/cost.json` 与 T4 分段同一套函数（`avg_cost` / `avg_cost_infer`）。

## 附：cv2 审计（#207 零，2026-11-18）

- 本表测的 `submit_pkg/predict.py` 与决赛包 `code/utils/preprocess.py` 是同一套 **PIL** 流程
  （`Image.draft` + `Image.BICUBIC`），**全链路无 cv2**，因此上表分段对本包有效。
- 决赛包 `grep -rn "cv2\|opencv" code/` 为空；`requirements.txt` 无 opencv。
- PIL↔cv2 概率差与训练同源说明见 `docs/finals/PIL_VS_CV2.md`。