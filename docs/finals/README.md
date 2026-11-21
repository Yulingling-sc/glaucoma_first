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

## 成员与预处理（dist4 口径）

| 成员 | 权重 | arch / img_size | cache 档位 | tta | 训练范围 |
|---|---|---|---|---|---|
| `cb384.pt` | 0.50 | convnext_base / 384 | 512 | 1 | train-only |
| `cs_hi512_all_fp16.pt` | 0.25 | convnext_small / 512 | 768 | 1 | train+all |
| `cs_hi512_all_ra.pt` | 0.25 | convnext_small / 512 | 768 | 1 | train+all + RA |

- 配比来源：平台 ID 32301（2026-11-05，LB 98.72，0.5/0.25/0.25）。
- **单次解码**：每张图只 `Image.draft(384)` 解码一次；bbox（灰度>15，0.2/99.8 分位）、
  居中方形裁剪（side × 1.02）、按 cache 档位上采样、再插值到 img_size **全部在 GPU 上做**
  （`F.grid_sample`，`padding_mode="zeros"`）。这是 dist3/dist4 的服务口径，也是 103ms / 87.19 的来源。
- 集成为**逐成员 sigmoid 概率 + TTA 平均 + 加权平均**（与 `submit_pkg/predict.py` 同口径）；
  年龄按训练口径 `(age-58)/15` 归一化。
- `members.json` 显式覆盖 slim 权重不可靠的元数据（`cs_hi512_all_ra.pt` 内嵌 `cache_size=512`）。

## 与参考分组口径的漂移（单解码代价）

同一套权重/配比/TTA，只把预处理从「按 cache 分组 draft768/1152 解码 + numpy BICUBIC 裁剪」
换成「单解码 draft384 + GPU 裁剪/双线性」，20 图对比（`tools/finals_dist_parity.py`）：

| 图集 | 尺寸 | max\|Δp\| | mean\|Δp\| | Spearman | 0.5 翻转 |
|---|---|---|---|---|---|
| smoke/imgs | 1024² | 0.00021 | 0.00017 | 1.00000 | 0 |
| smoke/big20 | 2048² | 0.0535 | 0.0150 | 0.98496 | 0 |

- 小图（真实眼底常见尺寸）几乎无差异；2048² 合成压力档的漂移即单解码省掉一整套
  draft1152 解码的代价，与 dist3（87.13）/ dist4（87.19）线上结果一致，可接受。

## 三接口（与官方模板逐行对齐）

```python
load_model(device)                                              # -> Ensemble（已 eval）
preprocess(img_bgr)                                             # BGR ndarray -> (3,H,W) float32 numpy, [0,1]
predict_batch(model, batch, device, use_fp16, meta_rows=None)   # -> np.ndarray 概率
```

- 框架侧 `read_one` 用 PIL `draft(384)` 单次解码（不依赖 cv2），把整图交给 `predict_batch`；
  `preprocess` 只做 BGR→RGB 与 dtype 转换，供外部 runner 直接调用。
- `predict_batch` 接受 `(B,H,W,3)` uint8 或 `(B,3,H,W)` float（单张 3 维亦可），
  内部调用 `Ensemble.forward_probs`（GPU 预处理 + 前向 + TTA + 加权概率）与 `postprocess`。
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
  475,933,333 B；md5 `51afc84149004e2d2f1f85e9aa8740cc`；
  sha256 `0e815b9a9e719cbe283e43a1e257fc1d17eedc90cd9c4b5874714d4ec8c129e5`。
- 官方写 "≤500MB"：十进制余量 **24.1 MB**；若按 500 MiB 计余量 46.1 MiB。
- ⚠️ **重建件不等于出分包**：dist4 线上包 md5 为 `709eb0895d9ecb815476e556eb175c3d`，
  本仓库无原始产物（无 Docker/无历史 zip），只能重建到「同 lineage、同体积、同配比」，
  md5 不可复现；以线上 md5 为准。
- 2 成员版（276.3 MB，md5 `5307878e`）仅作降耗备选，未取得线上验证。

## #216 彩票包（TTA2-cb）

`python tools/finals_build_tta2_cb.py --name "陈家豪_提交材料_tta2cb" --out finals_out`

- 与 dist4 **唯一差异**：`model/members.json` 中 cb384 `tta=2`（原图 + 水平翻转），
  两个 small 成员保持 `tta=1`；draft384 解码、fp16 权重、channels_last 全部不变。
- 产物：475.9 MB（475,933,068 B）；md5 `dcef9f25fc97fe42e9e3c9d8a59a6e7f`；
  sha256 `b02dcc5f33a85dd57f6deeb214885171eb69a160bae253b615dd22cf0a1dbab6`。
- 本机 5060 受控 A/B（同 20 图、同进程、仅切换 cb384 的 tta）：TTA1 61–65ms / TTA2-cb 84–85ms →
  **+20~23ms**；整包单次 smoke 的绝对耗时在 60–85ms 间波动（笔记本 GPU 降频），不作门禁依据。
  按 T4 的 cb384 forward 22.0ms 折算安静态 **≈ 125ms**（dist4 实测 103ms + 一次 cb384 前向），
  大概率吃 100ms 门禁（零分数代价彩票：失败不改变 87.19，成功上限约 +0.1~0.18 分）。

## 证据

- `docs/finals/T4_LATENCY.md`：T4 实测（旧表为 512-native 单 cache 口径，已标注作废）；
  严格分组口径本机 5060 实测与两档 `cost.json`；#216 TTA2-cb 受控 A/B。
- `docs/finals/PIL_VS_CV2.md`：PIL↔cv2 概率差与训练同源说明。
- `tools/no_cv2_smoke.py`：屏蔽 cv2 后按官方 CLI（`--data_root/--meta/--out_dir`）跑通成品包，
  21 行全覆盖（含 1 张刻意缺失的图 → Probability=0.5），三接口签名断言通过。
- `tools/finals_dist_parity.py`：单解码 vs 分组解码的逐图漂移（见上表）。
- cache 档位敏感性（`tools/finals_cache_group_delta.py`，6 张 1024² 图，正确档位 vs 一律
  512-native）：`cs_hi512_all_fp16` max|Δp| = 0.0098、`cs_hi512_all_ra` 0.0024、
  集成 0.0031、0/6 翻转 → 成员档位必须按 `members.json` 显式指定。