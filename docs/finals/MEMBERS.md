# 决赛成员盘点（issue #208 · 任务一）

> 生成：2026-11-18。数据来源：`docs/audit_98_87/manifest_98_87.json`（分支 `materials-98-87`）、
> GitHub Release `weights-1016`（48 资产）与 `compliant_pkg-1028`、本地概率缓存 `D:/glaucoma_work/ens8_probs`（2026-10-17 会话）、
> `docs/WEIGHTS.md`、`docs/FINAL_HANDOFF.md`、`SUBMISSIONS_LOG.md`。
> 复算脚本：`tools/finals_208_search.py`（valid 指标复算）、`tools/finals_208_valid_pil.py`（PIL 管线全量复算，进行中）。

## 0. 合规边界（规则第 3 条，先行声明）

- **train-only 成员**（valid 未并入训练）：可用 valid 做 checkpoint 选择、融合权重搜索。
- **train+all 成员**（57245 = train 51396 ∪ valid 5849，valid 已并入训练）：valid 指标**污染、一律不作为选择依据**；进入候选时权重必须是事先固定值（等权或区域赛锁定值），并注明"未用 valid 调参"。
- test（5855）概率**不参与任何选择**（规则第 5 条）；下表 test 概率文件仅用于工程复现与一致性核对。

## 1. 12 个 checkpoint 逐条盘点（对应 manifest_98_87.json）

valid 指标有两列口径：

- **官方口径（TTA4 单尺度）**：权重文件名 / `docs/WEIGHTS.md` 记录值（保存时测得，与 Release 资产哈希绑定）。
- **ens8 缓存口径**：本地 `ens8_probs` 概率今日用 valid.csv 标签复算。注意该缓存是**多尺度 TTA**（ConvNeXt 三成员 448/512/576 或 352/384/448 × 4 翻转；effnet/seed2 单尺度 TTA4），与官方口径不可直接比大小，仅供融合搜索内部一致使用。

| # | 角色 | 成员名 | arch | img | cache | 训练范围 | fp16 体积 (B) | valid AUROC (TTA4) | valid AUPRC (TTA4) | ens8 口径 AUPRC/AUROC | test 概率文件 | 可部署 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ens9/ens10 | cb512 | convnext_base | 512 | cache768 | **train-only** | 176,328,486 | 0.99089 | **0.87646** | 0.87906 / 0.99042（多尺度） | 本地 npy（Release 缺 CSV） | ✅ |
| 2 | ens9/ens10 | hi512_v2 | convnext_small | 512 | cache768 | **train-only** | 99,843,809 | 0.99085 | **0.87520** | 0.87795 / 0.99070（多尺度） | 本地 npy（Release 缺 CSV） | ✅ |
| 3 | ens9/ens10 | cb384 | convnext_base | 384 | cache512 | **train-only** | 176,328,125 | 0.99117 | **0.86439** | 0.87129 / 0.99147（多尺度） | 本地 npy（Release 缺 CSV） | ✅ |
| 4 | ens9/ens10 | b512 | efficientnet_b3 | 512 | cache512 | **train-only** | 23,544,137 | 0.98370 | **0.81415** | 0.81410 / 0.98369（单尺度，两口径一致） | 本地 npy（Release 缺 CSV） | ✅ |
| 5 | ens10 家族 | cs_hi512_all | convnext_small | 512 | cache_all768 | **train+all(57245)** | 199,660,651 | —（污染） | —（污染） | — | ✅ Release `cs_hi512_all_test_probs_tta4.csv` | ✅ |
| 6 | ens10 家族 | cs_hi512_all_ra | convnext_small | 512 | cache_all768 | **train+all(57245)+RA** | 199,662,799（1016 版）/ 99,831,694（1028 版） | —（污染） | —（污染） | — | ✅ Release `cs_hi512_all_ra_test_probs_tta4.csv` | ✅ |
| 7 | seedavg | seed7 | convnext_base | 512 | cache_all768 | **train+all(57245)+RA** 16ep | 705,015,055（fp32，fp16 未出 ≈352.5MB） | —（污染） | —（污染） | — | ✅ Release `cnxb_512_e16_ra_all_test_tta4.csv` | ✅（需 fp16 瘦身） |
| 8 | seedavg | seed42 | convnext_base | 512 | cache_all768 | **train+all(57245)+RA** 16ep | 705,015,055（fp32，fp16 未出 ≈352.5MB） | —（污染） | —（污染） | — | ✅ Release `cnxb_512_e16_ra_all_s42_test_tta4.csv` | ✅（需 fp16 瘦身） |
| 9 | RA_avg | ra_best | convnext_base | 512 | cache512 | **train-only**+RA 16ep | 705,225,551（fp32，fp16 未出 ≈352.6MB） | 0.99322（PR #132 审计） | **未归档**（缺 valid TTA4 概率文件） | — | ✅ Release `cnxb_512_e16_ra_test_best_tta4.csv` | ✅（需 fp16 瘦身） |
| 10 | RA_avg | ra_last | convnext_base | 512 | cache512 | **train-only**+RA 16ep | 705,225,551（fp32，fp16 未出 ≈352.6MB） | 0.99284（PR #132 审计） | **未归档**（缺 valid TTA4 概率文件） | — | ✅ Release `cnxb_512_e16_ra_test_last_tta4.csv` | ✅（需 fp16 瘦身） |
| 11 | Swin_pair | swinb_384_e16 | swin_base | 384 | cache512 | **train-only** 16ep | **checkpoint 未公开**（估算 fp16 ≈349.9MB） | 0.99230* | **0.85930*** | — | ✅ Release `swinb_384_test_probs_tta4.csv` | ❌ 无权重 |
| 12 | Swin_pair | swinb_384_e16_s7 | swin_base | 384 | cache512 | **train-only** 16ep seed7 | **checkpoint 未公开**（估算 fp16 ≈349.9MB） | 0.99190* | **0.85680*** | — | ✅ Release `swinb_384_e16_s7_test_probs_tta4.csv` | ❌ 无权重 |

\* #11/#12 的 valid 数字为 2026-11-18 用 Release `*_valid_tta4.csv`（5849 行，与 valid.csv 名序对齐）复算所得。

**盘点覆盖检查**：12/12 个 checkpoint 训练范围无空缺。✅

## 2. 可部署的 train-only 补充成员（不在 12 内，但候选搜索可用）

| 成员名 | arch | img | cache | 训练范围 | fp16 体积 (B) | valid AUROC (TTA4) | valid AUPRC (TTA4) | ens8 口径 | test 概率文件 | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|
| seed2 | convnext_small | 512 | cache768 | **train-only**（seed7 热启动 hi512_v2 续训 6ep，云 4090D） | 99,844,892 | 0.99064 | **0.87503** | 0.87503 / 0.99064（单尺度，两口径一致） | 本地 npy（Release 缺 CSV） | 多样性成员（#35） |
| cnxs384 | convnext_small | 384 | cache512 | **train-only** | 99,842,726 | 0.98994 | **0.86371** | 缺（valid 概率未归档） | 缺 | 候选可用但需先补 valid 概率 |
| ctiny384 | convnext_tiny | 384 | cache512 | **train-only** | 56,518,009 | 0.98945 | **0.85584** | 0.85442 / 0.98559（Release npy，文件名 `ctiny512_*`，身份核验进行中） | ✅ Release `ctiny512_test_probs_tta4.csv`（命名同疑点） | 小体积多样性成员 |
| b448 | efficientnet_b3 | 448 | cache512 | **train-only** | 23,544,137 | 0.98210 | **0.82672** | 0.82673 / 0.98210（一致） | 本地 npy（Release 缺 CSV） | 分辨率多样性 |
| b384 | efficientnet_b3 | 384 | cache512 | **train-only** | 23,547,284 | 0.98504 | **0.81592** | 0.81589 / 0.98503（一致） | 本地 npy（Release 缺 CSV） | — |

## 3. 仅存概率、权重未发布的成员（盘点如实记录，不可部署）

| 成员名 | arch/img | 训练范围（如可考） | valid AUPRC/AUROC（今日复算） | 概率文件 | 权重 |
|---|---|---|---|---|---|
| swinl_384_e16 | swin_large 384 | train-only 16ep（#77 阶段1） | 0.84477 / 0.98734 | ✅ Release valid+test CSV | ❌ |
| cnxb_384_22k | convnext_base 384（ImageNet-22k 初始化） | train-only | 0.86706 / 0.98926 | ✅ Release valid+test CSV | ❌ |
| cnxb_384_pw3 | convnext_base 384（pw3 变体） | train-only（如可考） | 0.86419 / 0.98922 | ✅ Release valid+test CSV | ❌ |
| hi640 | convnext_small 640 | train-only（多尺度 TTA） | 0.87278 / 0.98733 | 本地 npy（Release 缺） | ❌ |
| ra_seed7/42/123（概率文件） | — | **provenance 未定**（早于 10-22 RA 线，无对应已发布权重） | 0.88026 / 0.87262 / 0.88386 | ✅ Release npy | ❌ 不对应发布权重，**不入选** |

## 4. 伪标签家族（决赛合规禁用/需队长确认，盘点备查）

`cnxb_384_all_pseudo`、`cnxb_384_all_pseudo_s7/s123`、`swinb_384_all_v3`（#173 已标注 pseudo，禁止最终合规使用）均为 train+all+测试域伪标签，**不计入候选**。复赛包 `submit_pkg` 曾含 `cnxb_384_all_pseudo.pt`，决赛兜底按 b664 三成员（无伪标签）执行。

## 5. 缺口清单（诚实记录）

1. **Swin_pair 两个 checkpoint 未公开**：无法部署，Swin 线在 ≤500MB 决赛包内关闭（除非重训并发布）。
2. **ens10 逐成员构建脚本未入库**：权威件为 `submission_20251018_ens10_cal.csv`（md5 ce91bbf4…）；ens10 的 6 个 fp16 成员合计 875MB，本身超预算。
3. **cb512 / hi512_v2 / cb384 / b512 / b448 / b384 / seed2 的 test 概率 CSV 未上传 Release**（仅本地 `ens8_probs` npy）：定稿后需随成员权重一并上传（命名 `*_test_probs_tta4.csv`），供工程一致性核对。
4. **ra_best / ra_last 的 valid AUPRC 未归档**（只有 AUROC）：RA 单成员 fp16 ≈352.6MB，可挤进预算但 valid AUPRC 缺失，本工单暂不入候选；如需启用，先在 GPU 上补 valid TTA4 概率。
5. **cnxs384 的 valid 概率未归档**：候选池暂不含；如需启用同样先补。
6. **12 成员多尺度 TTA 口径差异**：ens8 缓存对 ConvNeXt 三成员为多尺度 TTA，与官方 TTA4 单尺度口径差 +0.002~+0.007（见 §1 两列）。**统一 TTA4 口径的成员概率与候选终值需 GPU 复算**（与 T4 实测同批，见 CANDIDATES.md 待办）。
7. **PIL 管线 valid AUPRC 复算进行中**：`tools/finals_208_valid_pil.py` + `tools/finals_208_valid_cache.py`，cb384 全量 5849 张 CPU 复算，结果回填 `docs/finals/PREPROC_PIL.md`。
