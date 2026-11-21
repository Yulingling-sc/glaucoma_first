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
| ctiny384 | convnext_tiny | 384 | cache512 | **train-only** | 56,518,009 | 0.98945 | **0.85584** | 0.85442 / 0.98559（Release npy `ctiny512_*`——**2026-11-19 身份核验 FAIL**，与该权重不匹配，此列数字不可归属，见 PREPROC_PIL.md §4.1） | ✅ Release `ctiny512_test_probs_tta4.csv`（**命名疑点已证实**，同 §4.1） | 小体积多样性成员（如启用须重算概率） |
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
7. **PIL 管线 valid AUPRC 复算进行中**：`tools/finals_208_valid_pil.py` + `tools/finals_208_valid_cache.py`，cb384 全量 5849 张 CPU 复算，结果回填 `docs/finals/PREPROC_PIL.md`。（2026-11-19 更新：cb384/ctiny384 gate 已回填 PASS，见 PREPROC_PIL.md §4；其余成员按需排队。）
8. **Release npy `ctiny512_probs_valid_full_tta4.npy` 身份核验 FAIL**（2026-11-19，PREPROC_PIL.md §4.1）：与 ctiny384 权重不匹配（Spearman 0.783、39 处 0.5 翻转、npy 质量低于权重官方值），provenance 未定；如需启用 ctiny384（含 C1 类候选、工程一致性核对），须以本地权重重算 valid/test 概率并重新上传 Release。

## 6. 决赛实际出分包（dist3 重试包 2026-11-18 提交；**终版 = dist4，见 §6.0**）

### 6.0 终版声明与 draft 档位判决（#215 B.2，2026-11-21）

- **最终出分包 = dist4**：`董明坤_提交材料.zip`（475.9MB），**ID 3764 / SubmitId 8891227，主 87.19 / 副 0.9912**（首测 103ms 判耗时失败 → 平台重评分出分），包 md5 `709eb0895d9ecb815476e556eb175c3d`。
- dist4 配置：3 成员 cb384 / cs_hi512_all_fp16 / cs_hi512_all_ra = **0.5/0.25/0.25**（ID 32301 配比，未用 valid 调参——含 train+all 成员本不允许调），单解码 draft384 + GPU bbox/裁剪/插值，TTA1，PIL 无 cv2；members.json 显式 cache_size/img_size/tta（§6.2 陷阱修正已在包内落实，见 `finals/20261118` 分支）。
- **draft 档位判决**：draft384 = 87.19（ID 3764）vs draft250 = 86.95（ID 4031，dist8：draft250 + fp16 权重存储 + channels_last，门禁干净通过）→ **−0.24**；dist7（draft250，122ms 未评分，判定平台抖动）旁证。**服务口径定格 draft384，禁止再降档。**
- 权重溯源注：dist4 的 cb384 = weights-1016 Release 资产本体（176,328,125 B，sha256 `a389f41c…`，与包内 SHA256SUMS.txt 一致）；dist3 所用为本仓库另一瘦身产物（176,314,205 B）。同一 ckpt（auprc0.8644）的两种 slim 产物，dist3/dist4 各自自洽。
- 本节其余小节（§6.1/§6.2）保留 dist3（2 成员 0.667/0.333 版，87.13/0.9910）作为历史记录。

平台记录（dist3）：**ID 3453 / SubmitId 8890916，主 87.13 / 副 0.9910**（已正确评分，耗时门槛通过）。
包 `董明坤_提交材料.zip` md5 `525e78143f972b04b8df0aea0f9e3a5e`。
前一次提交 ID 3080（2026-11-17 22:45）评分错误 0.00：`ImportError: libxcb.so.1`（cv2 GUI wheel），
由此全预处理迁 PIL（#207）。

### 6.1 出分成员行（2 成员）

| 成员 | arch / img / 真实 cache | 权重文件 | fp16 (B) | 比例 | 训练范围 |
|---|---|---|---|---|---|
| cb384 | convnext_base / 384 / cache512 | `convnext_base_384_cache512_1015_auprc0.8644.pt`（瘦身版） | 176,314,205 | **0.667** | train-only |
| cs_hi512_all_ra | convnext_small / 512 / **cache_all768** | `cs_hi512_all_ra.pt`（compliant_pkg-1028 版） | 99,831,694 | **0.333** | **train+all(57245)+RA** |

- 0.667/0.333 = 2/3 : 1/3 **机械归一**（2 成员等权的概率归一化），**未用 valid 调参** ✅——ra 为 train+all，规则第 3 条本就不允许调
- 推理口径：**单解码 draft384**（draft 只降不升：3000px→750²、4096px→512²）+ **bbox/裁剪/插值全上 GPU**（uint8 上传 → gray>15 掩码 → torch.quantile）；这是 413ms 根因（双 draft 解码 1500²×2 + 双 CPU 全分辨率裁剪×2）的修复，见 `docs/finals/T4_LATENCY.md`——**其适用边界为 ACRIMA 小图（349px），官方大图 decode+crop 会涨**
- 相对 C0 三成员（ID 32301，0.5/0.25/0.25，区域赛 98.72）：**去掉 cs_hi512_all**（省 ~200MB 与一路大解码/裁剪）换延迟达标；损失量见 §7 gate（非阻塞）

### 6.2 cache_size 陷阱列（部署必须 members.json 显式覆盖）

2026-11-19 用 `torch.load` 直读三份部署权重的实测元数据：

| 成员（真实 cache） | 内嵌 `cache_size` 字段 | `args.cache_root` | 回退链结果 | 正确档 |
|---|---|---|---|---|
| cb384（512） | 无 | 无 | 文件名无 `cache768` → 默认 512，**恰好正确** | 512 ✅ |
| cs_hi512_all（768） | 无（fp16 瘦身剥离） | 无 | 默认 512 | **768 ❌** |
| cs_hi512_all_ra（768） | **512（slim 工具写入的错值）** | 无 | 第 1 级读到错值 512 | **768 ❌** |

- `predict.py::infer_cache_size` 的三级回退（ckpt 字段 → `args.cache_root` 含 768 → 文件名含 `cache768` → 默认 512）在这两个 train+all 成员上**全部失效或给错值**；对外分发若再改名，第 3 级文件名推断也断档
- 错档后果：预处理按 512（draft768）走，而模型在 cache768（draft1152）上训练 → 输入分布漂移、概率不可信
- **结论：部署侧不得依赖内嵌元数据或文件名推断，必须在 `members.json` 为每个成员显式写 `cache_size`/`img_size`**（dist3 重试包已按此执行；同步回 `finals/20261118` 分支由 #212 分工 JiahaoChen 执行）
- 同类陷阱在候选侧已有一例：`ctiny512_probs_valid_full_tta4.npy` 文件名与 384 权重不一致（§2 ctiny384 行、CANDIDATES.md §1 C1 caveat）——**2026-11-19 核验 FAIL，C1 已作废退回 C3**（PREPROC_PIL.md §4.1）

### 6.3 核验材料与数据申明 xlsx ↔ dist4 一致性抽查（#215 B.3，2026-11-21）

对照 `finals/20261118` 分支 `SHA256SUMS.txt` + `model/members.json` + 台账终版（SUBMISSIONS_LOG）：

| 检查项 | 结果 |
|---|---|
| 三权重 sha256/大小 vs SHA256SUMS.txt | ✅ 逐字节一致：cb384 `a389f41c…` / 176,328,125 B；cs_hi512_all_fp16 `3e04d190…` / 199,660,651 B；cs_hi512_all_ra `5a04ebe3…` / 99,831,694 B；合计 475,820,470 B ↔ 台账 475.9MB / 475.82MB |
| members.json 配比与显式覆盖 | ✅ 0.5/0.25/0.25 + cache_size/img_size/tta（cb384 512/384/1；两个 small 768/512/1）——§6.2 陷阱修正落实 |
| **xlsx 配比声明** | ❌ **E 表 C3 与 A 表 C7 仍为旧包口径 0.4/0.3/0.3**（xlsx 生成于 10-28 旧 3 成员包时代，未随 dist4 更新） |
| xlsx 其余声明（T4 实测 74.40ms、SHA256、包体大小、无测试统计量） | ✅ 与 dist4 一致 |

**修复**：PR #218（→ `finals/20261118` 分支）——`tools/finals_xlsx.py` E2/A6 改 0.5/0.25/0.25 并回填 §7 gate 终值与 dist4 线上成绩，重新生成 xlsx（逐单元格 diff 恰 4 处预期变更，T4 实测 / SHA256 / 大小全部不变）。

## 7. 完整 valid ΔAUPRC gate：blend2（dist3）vs blend3（dist4 终版配比）—— 已完成（2026-11-19）

- 方法：cb384 / cs_hi512_all / cs_hi512_all_ra 三成员 **PIL 直推全量 valid**（5849 张，TTA1+TTA4 一次跑完，
  `tools/finals_208_valid_pil.py --force-cache-size`，512/768 各归其位——即 §6.2 陷阱修正后的口径），CPU 2 路并行（11-19 完成；中断后经 `--resume` 续跑）；
  对比 blend2 = 0.667·cb384 + 0.333·ra 与 blend3 = 0.5·cb384 + 0.25·cs_all + 0.25·ra
- 口径注意：两个 blend 都含 train+all 成员，valid 指标偏乐观；**只回答"去掉 cs_hi512_all 损失多少"，不用于任何调参**（规则第 3 条）

| blend | 配比 | 对应包 | valid AUPRC (TTA1) | valid AUROC (TTA1) | valid AUPRC (TTA4) | valid AUROC (TTA4) |
|---|---|---|---|---|---|---|
| blend2 | 0.667·cb384 + 0.333·ra | dist3（ID 3453，87.13 / 0.9910） | 0.88463 | 0.99328 | 0.88731 | 0.99329 |
| **blend3** | **0.5·cb384 + 0.25·cs_all + 0.25·ra** | **dist4（ID 3764，87.19 / 0.9912，终版）** | **0.89797** | **0.99404** | **0.89927** | **0.99406** |

- **ΔAUPRC（blend3 − blend2）= +0.01334（TTA1）/ +0.01196（TTA4）**：cs_all 贡献显著为正，与平台方向一致（dist4 87.19 vs dist3 87.13 = +0.06）→ **dist4 配比选择被 gate 验证**。
- 成员单模（PIL 直推，train+all 两名成员指标偏乐观，仅作概率合理性/一致性记录，不作 gate）：cb384 TTA1 0.99063/0.86145、TTA4 0.99148/0.86739（vs 官方 0.99117/0.86439，gate PASS 见 PREPROC_PIL.md §4）；cs_hi512_all TTA1 0.99515/0.92162、TTA4 0.99565/0.92315；cs_hi512_all_ra TTA1 0.99387/0.90244、TTA4 0.99397/0.90680。三成员概率 0 NaN、分布合理（mean 0.032–0.047，阳性率 3.27% 量级）。
- **#216 TTA2-cb 引用注意**：blend2 TTA4 0.88731 vs TTA1 0.88463 的 +0.27pp 是**两名成员都做 4 翻转**的口径；dist11 彩票包是**仅 cb384 加 hflip（TTA2）**且配比不同（0.5/0.25/0.25）。TTA2-cb 的全量 valid 精确复算已完成（2026-11-21，deployed 同源 cb384）：blend3-TTA2cb 0.89769 vs blend3-TTA1 0.89797，**ΔAUPRC −0.00029**（ΔAUROC −0.0001，噪声内；全成员 TTA4 参考 +0.00130）——**dist11 配置下 TTA2-cb 的 valid 期望 ≈ 0（点估计略负）**，精确表见 #216 评论。
