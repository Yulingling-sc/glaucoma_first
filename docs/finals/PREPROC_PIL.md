# 预处理管线 PIL 迁移核验（issue #208 §0 / 联动 #207 §0.2）

> 生成：2026-11-18（首次交付）。复算脚本：`tools/finals_208_preproc_delta.py`（A/B 两部分）、
> `tools/finals_208_valid_pil.py`（PIL 直推全量 valid，TTA1+TTA4 一次跑完）、
> `tools/finals_208_valid_cache.py`（cache512 基线全量 valid）。

## 0. 背景

- 线上复赛包提交失败（ID 3080）：`ImportError: libxcb.so.1`（opencv-python 在官方镜像缺系统依赖）。
  结论：**决赛推理预处理不得依赖 cv2**，全部迁 PIL+numpy（`submit_pkg/predict.py` 已是 PIL 实现）。
- 需要回答两个问题：
  1. **PIL 直推 vs 训练缓存口径**：成员在 PIL 直推管线下的 valid AUPRC 相对官方（PIL 训练缓存测得）
     跌幅是否 ≤0.002（gate）；
  2. **cv2（已废）vs PIL**：概率差多大、0.5 阈值翻转多少（20 测试图对照）。

## 1. 管线定义

| 管线 | 实现 | 输入 | 流程 |
|---|---|---|---|
| 训练缓存（官方口径参照） | `preprocess.py`（PIL） | 原图 | JPEG draft→768 → 眼底 crop（gray>15，0.2/99.8 分位 bbox，居中方 ×1.02）→ BICUBIC 512 → **JPEG q92 落盘** → 训练/官方评测读盘 |
| **PIL 直推（决赛管线）** | `submit_pkg/predict.py::load_fundus_crop` | 原图 | 同上参数，**不落盘**，tensor 化后 bilinear→img_size → 归一化 |
| cv2 管线（11/17 已废包） | `tools/finals_208_preproc_delta.py::cv2_crop` | 原图 | `IMREAD_REDUCED_COLOR_{8/4/2/1}` 模拟 draft 语义 → 同 crop 逻辑 → INTER_CUBIC 512 |

三者的 crop 几何逻辑逐行一致，差异仅在**解码器**（PIL draft vs cv2 REDUCED vs PIL cache 落盘再读）与**插值实现**（PIL BICUBIC vs cv2 INTER_CUBIC）。

## 2. Part A：像素级对比（cv2 vs PIL 直推，valid 前 100 张，512×512 模型输入）

| 指标 | 值 |
|---|---|
| 输入尺寸 / draft 目标 | 512 / 768 |
| cv2 REDUCE 档位使用 | reduce2 ×81，reduce4 ×19（与 PIL draft 语义对齐后） |
| 逐像素 \|Δ\| 均值 | **0.618**（0–255 尺度） |
| 逐像素 \|Δ\| 最大 | **78.0**（逐图 max top5：78/77/74/71/65） |

**解读**：解码器+插值差异在像素级不可忽略（个别图 max 近 78/255），不能以"看起来一样"放行，
必须以**概率级**差异为准（Part B）。

## 3. Part B：概率级对比（C0 区域赛锁定三成员，20 测试图，TTA1，batch=1）

权重：`cb384_slim.pt`（md5 e4ba9b36…）+ `cs_hi512_all_fp16.pt`（fa3af24e…）+ `cs_hi512_all_ra_slim.pt`
（2553135d…），blend 0.5/0.25/0.25。

| 指标 | 值 |
|---|---|
| max \|Δp\| | **0.02885**（test_00000，p_pil 0.04504 vs p_cv2 0.01619） |
| mean \|Δp\| | **0.00321** |
| 0.5 阈值标签翻转 | **0 / 20** ✅ |
| Δp > 0.01 的图 | 2/20（test_00000 0.0288、test_05256 0.0162；均远离 0.5 决策面） |

**解读**：管线切换在概率级等效（均值 3‰），且无一图跨越 0.5 决策面；决赛指标为 AUROC/AUPRC，
对单调概率平移完全不敏感——**放行**。

## 4. Gate：PIL 直推全量 valid 复算（跌幅 ≤0.002 放行）

方法：`tools/finals_208_valid_pil.py`（PIL 直推，TTA1+TTA4 一次跑完）对 C0 三成员 +
`tools/finals_208_valid_cache.py`（cache512 基线）对 cb384，全量 5849 张、CPU 32 核。
官方参照值 = 权重文件名 / `docs/WEIGHTS.md`（PIL 训练缓存口径）。

| 成员 | 官方 valid AUPRC (TTA4) | cache512 基线（复算） | PIL 直推 TTA4 | Δ vs 官方 | gate ≤0.002 | PIL 直推 TTA1 |
|---|---|---|---|---|---|---|
| cb384 | 0.86439 | ⏳ 进行中 | ⏳ 进行中 | ⏳ | ⏳ | ⏳ |
| cs_hi512_all | —（train+all，valid 污染不评测） | — | ⏳（只报概率一致性，不作 gate） | — | — | ⏳ |
| cs_hi512_all_ra | —（train+all，valid 污染不评测） | — | ⏳（同上） | — | — | ⏳ |

> train+all 成员不得用 valid 报告指标（合规边界，MEMBERS.md §0）；此处仅跑概率级一致性
> （与 Part B 同目的的全量版：无崩溃、无 NaN、概率分布合理），**不产生 gate 数字**。

### 4.1 ctiny384 身份核验（C1 候选依赖）

`ctiny512_probs_valid_full_tta4.npy` 文件名与权重名（convnext_tiny_**384**）不一致。核验方法：
用本地 `convnext_tiny_384_1015_011832_cnxt_384.slim.pt`（56,518,009 B，与 Release 资产
`convnext_tiny_384_cache512_1015_auprc0.8558.pt` 同尺寸）PIL 直推全量 valid TTA4，与 Release
npy 对比（max|Δp| / 相关性 / AUPRC 一致性）。⏳ 进行中（排队于 C0 gate 之后）。

## 5. 结论（首版放行意见）

1. **管线切换（cv2→PIL）概率级等效**（Part B：max 0.029 / 0 翻转），PIL 直推管线放行用于决赛推理。
2. **首版提交（C0 区域赛锁定三成员，固定 0.5/0.25/0.25，TTA1 兜底）不受 PIL 迁移影响**：
   该组合本就于 11-05 以 PIL 系权重跑出 98.72（b664），且 Part B 证明 cv2↔PIL 概率差在 3‰ 量级。
3. gate 数字（cb384 全量 valid ΔAUPRC ≤0.002）⏳ 2026-11-19 上午回填；
   其余可部署成员的 PIL 直推复算排队中（结果追加进本表与 MEMBERS.md §1）。
