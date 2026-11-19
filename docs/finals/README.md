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
│   ├── cb384.pt                 # 176,328,125 B
│   ├── cs_hi512_all_fp16.pt     # 199,660,651 B
│   ├── cs_hi512_all_ra.pt       # 99,831,694 B
│   └── members.json             # 0.40 / 0.30 / 0.30
├── licenses/                    # torchvision / pytorch BSD-3-Clause 全文
├── 模型核验材料与数据申明清单.xlsx
└── SHA256SUMS.txt
```

不含 `pretrain_model/`、不含 `submission.csv`。权重不入库，走 Release：

| 文件 | Release 直链 | SHA-256 |
|---|---|---|
| cb384.pt | `releases/download/weights-1016/convnext_base_384_cache512_1015_auprc0.8644.pt` | `a389f41c…aff710e` |
| cs_hi512_all_fp16.pt | `releases/download/weights-1016/cs_hi512_all_fp16.pt` | `3e04d190…105f492e` |
| cs_hi512_all_ra.pt | `releases/download/compliant_pkg-1028/cs_hi512_all_ra.pt` | `5a04ebe3…154c86e6` |

（仓库 `dongmingkun/glaucoma_first`；完整哈希见包内 `SHA256SUMS.txt`。）

## 三接口（与官方模板逐行对齐）

```python
load_model(device)                                              # -> Ensemble（已 eval）
preprocess(img_bgr)                                             # BGR ndarray -> (3,512,512) float32 numpy, [0,1]
predict_batch(model, batch, device, use_fp16, meta_rows=None)   # -> np.ndarray 概率
```

- `preprocess` 返回 **numpy**（官方模板经 `np.stack` 攒 batch），BGR→RGB 用 `arr[..., ::-1]`；
  内部 `crop_resize(...).numpy()`，与训练缓存同一 bbox + PIL BICUBIC。
- `predict_batch` 内部调用 `Ensemble.forward_logits + postprocess`，`use_fp16` 控制 autocast；
  年龄按训练口径 `(age-58)/15` 归一化（对应 `train.py:age_tensor`），缺失/非法年龄填 58。
- cost 计时的 forward/post 与全量预测走同一函数。

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

- 解压后 **475,928,272 B** = 475.93 MB（十进制）= 453.88 MiB（2026-11-18 模板对齐后重打包）。
- 官方写 "≤500MB"：按 500 MB 或 500 MiB 计均通过（余量分别 24.07 MB / 48.36 MiB）。
- zip md5 `1389c89626a4afb3e24fa9fddfa43cfc`；sha256 `2a77354f6eb3ad0b7d06dad57325d6617f0adce49e103e0d277cff5d098dccf2`。

## 证据

- `docs/finals/T4_LATENCY.md`：Kaggle T4 实测（三成员 74.40ms）。
- `docs/finals/PIL_VS_CV2.md`：PIL↔cv2 概率差与训练同源说明。
- `tools/no_cv2_smoke.py`：屏蔽 cv2 后按官方 CLI（`--data_root/--meta/--out_dir`）跑通成品包，
  7 行全覆盖（含 1 张刻意缺失的图 → Probability=0.5），三接口签名断言通过。