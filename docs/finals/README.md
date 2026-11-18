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
│   └── requirements.txt         # torch==2.10.0 / torchvision==0.25.0 / numpy / pillow
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

## 三接口（对齐官方模板）

```python
load_model(device)                                              # -> Ensemble
preprocess(img_bgr)                                             # BGR ndarray -> tensor[3,512,512]
predict_batch(model, batch, device, use_fp16, meta_rows=None)   # -> np.ndarray 概率
```

- `preprocess` 兼容路径 / `PIL.Image`（内部走 JPEG draft 解码），BGR→RGB 用 `arr[..., ::-1]`。
- `predict_batch` 内部调用 `Ensemble.forward_logits + postprocess`，`use_fp16` 控制 autocast。
- cost 计时的 forward/post 与全量预测走同一函数，`max|Δp| = 0.0`。

## CLI

```bash
python code/predict_and_cost.py --images <图片目录> --meta <meta.csv> --out_dir out
# 默认 --num_workers 4 --batch_size 32 --use_fp16 1 --tta 1
# --out 为 --out_dir 的别名；out_dir 不存在时自建
```

输出 `out/submission.csv`（`Image_name,Probability,Pred_Label`）与 `out/cost.json`
（含 `avg_cost` / `avg_cost_ms`，5 张预热 + 随机 100 张单轮 batch=1 均值）。

## 体积口径

- 解压后 **475,919,507 B** = 475.92 MB（十进制）= 453.87 MiB。
- 官方写 "≤500MB"：按 500 MB 或 500 MiB 计均通过（余量分别 24.08 MB / 48.37 MiB）。

## 证据

- `docs/finals/T4_LATENCY.md`：Kaggle T4 实测（三成员 74.40ms）。
- `docs/finals/PIL_VS_CV2.md`：PIL↔cv2 概率差与训练同源说明。
- `tools/no_cv2_smoke.py`：屏蔽 cv2 后跑通成品包 + 三接口签名断言。