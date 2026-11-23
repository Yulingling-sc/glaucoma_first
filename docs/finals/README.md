# 决赛提交包（#207）

> 目标：一个可离线运行、无 cv2、≤100ms/张、≤500MB 的 `队长名_提交材料.zip`。
> 打包脚本 `tools/finals_build_pkg.py`；沙箱验收 `tools/finals_sandbox_check.sh`。

## 目录结构

```
<队长名>_提交材料/
├── code/
│   ├── predict_and_cost.py      # 官方三接口 + CLI（全量预测 & 计时同一数值路径）
│   ├── model.py                 # FundusNet / Ensemble（fp16 开关）
│   ├── utils/
│   │   ├── preprocess.py        # PIL decode(draft) + bbox + BICUBIC（无 cv2）
│   │   ├── weights.py           # checkpoint 解析（arch/img_size/cache/use）
│   │   └── age.py               # 年龄清洗
│   ├── train/                   # 训练脚本与最小复现依赖（推理不依赖）
│   └── requirements.txt         # torch==2.10.0 / torchvision==0.25.0 / numpy / pillow / pandas
├── model/
│   ├── cb384.pt                 # 176,328,125 B（cache512 / img384）
│   ├── cs_hi512_all_fp16.pt     # 199,660,651 B（cache768 / img512，fp16 存储）
│   ├── cs_hi512_all_ra.pt       #  99,831,694 B（cache768 / img512，compliant_pkg-1028 slim）
│   └── members.json             # 3 成员 0.5/0.25/0.25 + 显式 cache_size/img_size/tta
├── licenses/                    # torchvision / pytorch BSD-3-Clause 全文
├── 模型核验材料与数据申明清单.xlsx
└── SHA256SUMS.txt
```

不含 `pretrain_model/`、不含 `submission.csv`。权重不入库，走 Release：

| 文件 | Release 直链 | SHA-256 |
|---|---|---|
| cb384.pt | `releases/download/weights-1016/convnext_base_384_cache512_1015_auprc0.8644.pt` | `a389f41c…aff710e` |
| cs_hi512_all_fp16.pt | `releases/download/compliant_pkg-1028/cs_hi512_all_fp16.pt` | `3e04d190…105f492e` |
| cs_hi512_all_ra.pt | `releases/download/compliant_pkg-1028/cs_hi512_all_ra.pt` | `5a04ebe3…154c86e6` |

（仓库 `dongmingkun/glaucoma_first`；完整哈希见包内 `SHA256SUMS.txt` 与 `SHA256SUMS_weights.txt`。）

## 成员与预处理（#221 后服务口径：pkg3/dist4 线）

| 成员 | 权重 | arch / img_size | 共享 cache | tta | 训练范围 |
|---|---|---|---|---|---|
| `cb384.pt` | 0.50 | convnext_base / 384 | 512 | 1 | train-only |
| `cs_hi512_all_fp16.pt` | 0.25 | convnext_small / 512 | 512 | 1 | train+all |
| `cs_hi512_all_ra.pt` | 0.25 | convnext_small / 512 | 512 | 1 | train+all + RA |

- 配比来源：平台 ID 32301（2026-11-05，LB 98.72，0.5/0.25/0.25）。
- **单次解码 + CPU 侧 pkg3 裁剪**：每张图只 `Image.draft(384)` 解码一次；bbox（灰度>15，
  0.2/99.8 分位）、居中方形裁剪（side × 1.02）与 PIL **BICUBIC 缩放到共享 512** 在 CPU 上
  按 `(x, y)` 显式计算（`utils.preprocess.crop_resize`，与 pkg3 出包同源）；各成员再在
  Ensemble 内双线性内插到自己的 `img_size`。**非方图不存在 x/y 转置**（#221 根因）。
- 集成为**逐成员 sigmoid 概率 + TTA 平均 + 加权平均**（与 dist3/dist4 出分包同口径）；
  年龄按训练口径 `(age-58)/15` 归一化。
- `members.json` 保留 `cache_size` 字段（历史兼容/日志）；#221 后预处理统一共享 512。
- 线上对照：dist4（87.19/0.9912）为 GPU 裁剪线，方图行为可对齐；非方图按其 `_boxes`
  转置实现会裁错（#221 实测 mean|Δp| 0.019~0.084），本口径为修正后行为。

## 预处理 parity 门禁（#221）

`tools/finals_prep_parity.py` 以 **frozen pkg3 参照实现**（`ed6bda3` 的 `crop_resize`
逐行拷贝）对照包内服务代码，覆盖方图 + 非方图 fixture，门槛 **mean|Δp| ≤ 0.005、0 个
0.5 翻转**；同时把 merged-main 的转置实现作为**回归演示**跑红非方图。
详见 [PARITY_221.md](PARITY_221.md)。

## 三接口（与官方模板逐行对齐）

```python
load_model(device)                                              # -> Ensemble（已 eval）
preprocess(img_bgr)                                             # BGR -> (3,512,512) float32 numpy, [0,1]（含 pkg3 裁剪）
predict_batch(model, batch, device, use_fp16, meta_rows=None)   # -> np.ndarray 概率
```

- 框架侧 `read_one` 用 PIL `draft(384)` 单次解码（不依赖 cv2），把整图交给 `predict_batch`；
  `preprocess` 做 BGR→RGB + pkg3 裁剪（共享 512），供外部 runner 直接调用。
- `predict_batch` 接受 `(B,H,W,3)` uint8（draft 解码结果）或已按共享 512 预处理的
  `(B,3,512,512)` float；raw 输入内部逐图 `crop_resize` 到共享 512，再调用
  `Ensemble.forward_probs`（成员内插 + 前向 + TTA + 加权概率）与 `postprocess`。
- cost 计时的 forward/post 与全量预测走同一函数；`run_cost` 的抽样/预热/计时口径与官方模板一致。


## CLI（官方命令形态）

```bash
python code/predict_and_cost.py \
  --data_root ./test_images --meta ./test.csv --out_dir ./output \
  --num_workers 4 --batch_size 32
# --data_root 为官方参数名（同时接受 --data-root / --images 别名）
# --out_dir 不存在时自建
```

输出 `out/submission.csv`（`Image_name,Probability,Pred_Label`，覆盖 **全部 meta 行**，
读图失败的图按 0.5 中性概率兜底）与 `out/cost.json`
（`avg_cost` / `avg_cost_infer`，5 张预热 + 随机 100 张单轮 batch=1 均值）。

## 体积口径

- 解压后 **475.9 MB**（3 成员 dist4 口径）：`finals_out/陈家豪_提交材料_dist4rebuild.zip`
  475,933,580 B；md5 `d02031aef4f799548729e9639e209302`；
  sha256 `23ffa27fe9ee0e4657923a63f0b0f67c468d914fbb1bc6090696beddde3915ee`
  （含 #218 xlsx 修复，包内清单 xlsx 为 `eae9b865…`）。
- 官方写 "≤500MB"：十进制余量 **24.1 MB**；若按 500 MiB 计余量 46.1 MiB。
- ⚠️ **重建件不等于出分包**：dist4 线上包 md5 为 `709eb0895d9ecb815476e556eb175c3d`，
  本仓库无原始产物（无 Docker/无历史 zip），只能重建到「同 lineage、同体积、同配比」，
  md5 不可复现；以线上 md5 为准。
- 2 成员版（276.3 MB，md5 `5307878e`）仅作降耗备选，未取得线上验证。
- ⚠️ **#221 代码修复后需重新出包**：`model.py`/`predict_and_cost.py`/`members.json`
  已变更，上述 zip 哈希作废；重打包（`tools/finals_build_pkg.py`）后回填体积与 md5/sha256。

## #216 彩票包（TTA2-cb）

`python tools/finals_build_tta2_cb.py --name "陈家豪_提交材料_tta2cb" --out finals_out`

- 与 dist4 **唯一差异**：`model/members.json` 中 cb384 `tta=2`（原图 + 水平翻转），
  两个 small 成员保持 `tta=1`；draft384 解码、fp16 权重、channels_last 全部不变。
- 产物：475.9 MB（475,933,315 B）；md5 `ef93e21e9707a742dc2979011f410c74`；
  sha256 `0c9579fd0f47d57f8ab06670cf184205a31e25d819af917f1bcfb2d3a69b0d5c`
  （基于 #218 修复后的 `finals/`；早前无 xlsx 修复的构建为 `dcef9f25…`，已作废）。
- 本机 5060 受控 A/B（同 20 图、同进程、仅切换 cb384 的 tta）：TTA1 61–65ms / TTA2-cb 84–85ms →
  **+20~23ms**；整包单次 smoke 的绝对耗时在 60–85ms 间波动（笔记本 GPU 降频），不作门禁依据。
  按 T4 的 cb384 forward 22.0ms 折算安静态 **≈ 125ms**（dist4 实测 103ms + 一次 cb384 前向），
  大概率吃 100ms 门禁（零分数代价彩票：失败不改变 87.19）。
  **2026-11-21/23 精算更正**：#216 全量 valid 与 #222 复核均为 **ΔAUPRC −0.00029**（噪声内、点估计略负），
  原记 "成功上限 +0.1~0.18" 来自 blend2 双成员 4 翻转口径，不适用于 dist11 的 TTA2-cb ⇒ 期望按 **≈0** 计。
  **延迟风险实测校准**：dist12（ID 4243）即以 `cb tta=2 / smalls tta=1` 出包且**被正确评分**
  （−2.48 的根因是 #221 的 bbox 转置，非耗时）⇒ TTA2-cb 的延迟风险为**中等**，不是必死。

## 证据

- `docs/finals/T4_LATENCY.md`：T4 实测（旧表为 512-native 单 cache 口径，已标注作废）；
  严格分组口径本机 5060 实测与两档 `cost.json`；#216 TTA2-cb 受控 A/B。
- `docs/finals/BLEND_SEARCH_222.md`：#222 冲分三轴（权重/融合/TTA）成本中性候选核验与候选排序表
  （含 test100 非方图 93% 与 draft 档位灵敏度）；机器可读 `BLEND_SEARCH_222.json`、`TIER_LEVERAGE_222.json`。
- `docs/finals/PIL_VS_CV2.md`：PIL↔cv2 概率差与训练同源说明。
- `tools/no_cv2_smoke.py`：屏蔽 cv2 后按官方 CLI（`--data_root/--meta/--out_dir`）跑通成品包，
  21 行全覆盖（含 1 张刻意缺失的图 → Probability=0.5），三接口签名断言通过。
- `tools/finals_dist_parity.py`：单解码 vs 分组解码的逐图漂移（见上表）。
- cache 档位敏感性（`tools/finals_cache_group_delta.py`，6 张 1024² 图，正确档位 vs 一律
  512-native）：`cs_hi512_all_fp16` max|Δp| = 0.0098、`cs_hi512_all_ra` 0.0024、
  集成 0.0031、0/6 翻转 → 成员档位必须按 `members.json` 显式指定。