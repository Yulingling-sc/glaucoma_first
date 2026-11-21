# T4 耗时实测（决赛口径，2026-11-18）

> 设备 **Kaggle Tesla T4 15GB**；镜像内 torch 2.11.0+cu130 / Python 3.13，
> **不是**官方决赛镜像（torch 2.10.0 / CUDA 12.8 / Python 3.12）。
> 本表用于选型与合规判断；官方镜像数字以决赛平台实测为准。

## ⚠️ 2026-11-18 更正：下表全部作废（单 cache 口径）

- 下表由 `submit_pkg/predict.py` + 旧权重文件名运行，`cs_hi512_all*` 被 `infer_cache_size`
  判成 **cache512**，两组共用一次 512 解码 → 实际是 **512-native**，不是训练口径 cache768→512。
- 因此 `3 模型 74.40ms` **不能**作为严格分组口径的耗时依据（少了一整套 1152 解码），已作废。
- 严格分组口径（512 组 draft768、768 组 draft1152）的复测结果见文末
  「严格分组口径（本机 5060）」；T4 + 官方镜像复测待补（本机无 Docker、分支未推送前
  Kaggle 内核拉不到新代码）。

## 严格分组口径（本机 5060，2026-11-18）

同一张 20 图 ACRIMA 集（349px 小图，n=15 计时；5 张预热，seed 2026，batch=1，fp16）：

| 组合 | avg_cost (ms) | avg_cost_infer (ms) | 包体 |
|---|---|---|---|
| **2 成员 `cb384 + cs_hi512_all_ra`（分支版，分组双解码）** | **45.06** | 27.34 | 276.3 MB |
| 3 成员 `cb384 + cs_hi512_all + cs_hi512_all_ra`（分支版） | 56.76 | 39.85 | 475.9 MB |
| 2 成员 @ 2048px 合成大图（压力档，非真实眼底） | 186.09 | 29.97 | — |

> 注意：本节测的是**本分支**的按 cache 分组双解码；线上 87.13 的 dist3 是「单解码 draft384 + GPU 裁剪」，
> 两者不同口径，本表不能当作 dist3 的复现数字。

- `cost.json`（定稿 2 成员，成品包解压后跑官方 CLI）：
  `{"avg_cost": 45.062, "avg_cost_infer": 27.337, "n_samples": 15, "n_fail": 0, "warmup": 5, "seed": 2026}`
- 3 成员档：`{"avg_cost": 56.761, "avg_cost_infer": 39.846, "n_samples": 15, "n_fail": 0, "warmup": 5, "seed": 2026}`
- 本机是 RTX 5060（非 T4），绝对值不可直接当平台分；但两档差 ≈11.7ms，与
  「多一套 1152 decode + 一次 small@512 forward」的分段估算一致。
- 2048px 档说明：大图 decode 受 `draft(1152)` 限制仍会显著抬升 e2e，官方 test 分辨率是
  剩余最大不确定性；`ISSUE_81` 全量 5855 图单模型实测 mean 32.65ms，说明官方图大概率不大。

## #211 复核：3 成员 forward 51ms 与 draft 备选档（2026-11-20）

### 1) 51ms 的可复现性

现存原始记录（`finals_t4_out/combo_*.json`，Tesla T4 / torch 2.11.0+cu130 / CUDA 13.0，ACRIMA 349px）：

| combo | 成员 | forward mean | e2e mean |
|---|---|---|---|
| combo_0 | cb384 | 22.00 | 89.64（冷缓存 decode 46ms） |
| combo_1 | cnxb_all_pseudo | 19.59 | 44.73 |
| combo_2 | cs_hi512_all_ra | 19.57 | 46.12 |
| combo_3 | cb384 + cnxb | 38.43 | 63.46 |
| combo_4 | cb384 + cnxb + cs_hi512_all_ra | **55.35** | 79.78 |
| combo_5 | 同上，TTA2 | 108.44 | 133.44 |

- 结论：本仓库保留的原始数据只能复现 **3 成员 55.35ms**（第三个成员是 cnxb_base@512，比 cs_hi512_all 的
  small@512 更重）；文档里的 51.08ms **没有对应的原始记录**，不应作为精确值引用。
- 采用区间 **forward ≈ 51–56ms**；官方镜像 torch 2.10 vs 实测 2.11 按 ±10% → **46–62ms**。
- 该区间不改变结论方向：加上平台 CPU 侧（单次解码 + H2D，按 dist4=103ms 反推 ≈45–50ms）与实测噪声
  ≥19ms，3 成员总量 ≈ **110–130ms**，与 ID 3764（103ms）/ 3773（122ms）一致，draft 档位压不回 100ms。

### 2) draft 320 备选档（本机 CPU，非 T4）

`tools/finals_draft_scan.py`：把首张图放大到 S 后存 JPEG q95，`open+draft+convert` 取 5 次中位数。

| S | draft250 解码边/ms | draft320 解码边/ms | draft384 解码边/ms |
|---|---|---|---|
| 2000 | 250 / 3.63 | 500 / 4.59 | 500 / 4.35 |
| 2048 | 256 / 3.61 | 512 / 4.87 | 512 / 4.69 |
| 2500 | 313 / 5.03 | 625 / 6.77 | 625 / 7.03 |
| 3000 | 375 / 7.09 | 375 / 6.82 | 750 / 9.34 |
| 4096 | 512 / 12.38 | 512 / 12.21 | 512 / 12.07 |

- 与 #211 里 dongmingkun 口径一致：**draft320 在 S∈[2000,2559] 和 S≥4096 与 draft384 解码边完全相同**，
  只在 S≈3000 省一次 2× 解码（本机 ≈3ms；T4 CPU 预计同量级）。
- 概率侧（本机 5060，2 成员集成，合成放大图 8 张）：S=2500 时 draft250 vs 384 max|Δp|=0.0121；
  S=3000 时 draft250/320 vs 384 max|Δp|=0.0219，**均 0/8 翻转**。
- 判断：**draft320 不是可用杠杆**——最好情况省 3–6ms，救不了 103ms；draft250 已有 122ms 反证。

### 3) 合规提醒（dist8/dist9）

#211 里 dist8/dist9 提到「cost 阶段读图/预处理放入后台线程与上一张推理重叠」。官方模板里
`run_cost` 是固定框架代码、逐张串行计时；若为重叠而改动 `run_cost`，`cost.json` 报出的已不是
单张墙钟，属于计时口径变更，审计风险高。建议：框架区保持与官方模板逐字一致，重叠只放在
`run_prediction`（全量预测）侧；`run_cost` 若必须用同一路径，应放弃重叠。

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

## 结论（基于下方旧表，注意单 cache 口径）

1. ~~**推荐的干净 3 成员包（cb384 + cs_hi512_all + cs_hi512_all_ra）TTA1 = 74.40ms**~~ —— **作废**，
   旧表未按 cache 分组；严格口径的同机对照见上方 56.76ms（3 成员）/ 45.06ms（2 成员）。
2. **TTA2 不可行**（133.44ms，forward 翻倍）。减少 1 个成员后 TTA2 仍 ≈115ms，压不回 100ms。
3. **定稿 2 成员（cb384 + cs_hi512_all_ra）= 45.06ms（本机 5060，严格分组）**，包体 276.3 MB；
   T4 复测待补。
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
## #215/#216 复核：dist4 重建与 TTA2-cb 受控 A/B（2026-11-21，本机 5060）

### 1) dist4 lineage 重建件

- `finals/` 已改为 **单解码 draft384 + GPU bbox/裁剪/插值 + 3 成员 0.5/0.25/0.25（TTA1）**，
  与线上 dist4（md5 `709eb089…`，87.19 / 0.9912）同 lineage、同体积（475.9 MB）。
- **不承诺 md5 一致**：仓库无 dist4 原始产物，重建件为 `finals_out/陈家豪_提交材料_dist4rebuild.zip`
  （md5 `d02031ae…`，sha256 `23ffa27f…`，含 #218 xlsx 修复）。出分包身份以线上 md5 为准。
- `no_cv2_smoke.py`（cv2 阻断 + 21 行含 1 张缺失图 → 0.5）：PASS；本机 5060 上
  ACRIMA 小图 avg_cost 44.9ms、2048² 合成图 73.5ms（非 T4，仅作链路自检）。
  整包单次 smoke 的绝对耗时在 60~85ms 间波动（笔记本 GPU 降频），**不作门禁依据**。

### 2) TTA2-cb（#216 彩票包）受控 A/B

同一进程、同一 20 张 2048² 图、同一模型对象重建（`finals/model`），只切 cb384 的 tta：

| 配置 | avg_cost (ms) | avg_cost_infer (ms) | 相对 TTA1 |
|---|---|---|---|
| dist4 TTA1（cb=1, small=1,1） | 61.2 – 64.5 | 55.4 – 59.1 | 基线 |
| TTA2-cb（cb=2, small=1,1） | 84.0 – 84.9 | 78.6 – 79.6 | **+20.4 ~ +22.8**（两次受控 A/B） |

- 逐图概率差 max|Δp| = 0.0145、mean 0.0052（TTA 确实生效，非 no-op）。
- T4 折算：cb384 forward 22.0ms（combo_0）→ 安静态 **≈ 125ms**（dist4 实测 103ms + 一次 cb384 前向 22ms）。
  **大概率吃 100ms 门禁**；作为零分数代价彩票提交，失败不改变 87.19。
- 早年结论「TTA2 不可行」（combo_5 133.44ms）是 **3 成员全 TTA2**；本包只有 cb384 翻倍，
  但仍无法回到 100ms 以内。
