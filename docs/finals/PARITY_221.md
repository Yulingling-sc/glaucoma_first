# #221 预处理 parity 报告（非方图转置修复）

日期：2026-11-23　分支：`finals/20261118`　范围：选手区 `model.py` / `predict_and_cost.py` /
`model/members.json`（`run_cost` 计时协议未动）

## 1. 根因与修复

- 根因：merged-main 的 `Ensemble._boxes` 把 `mask.nonzero()` 的 `(y, x)` 分位直接按
  `(x0, y0, x1, y1)` 解包，非方图裁剪中心 x/y 互换（768×512 → 裁到 (256,384)，正确 (384,256)），
  叠加底部零填充 → dist12（ID 4243）84.71，较 dist4 −2.48。
- 修复方式（issue 建议 B）：**服务代码回退到 pkg3/dist4 线**
  - `model.py::Ensemble` 删除 GPU `_boxes`/`_crop`，不再在服务侧计算 bbox；输入为已预处理
    的共享 512 张量，成员只做 `F.interpolate(bilinear)` 到各自 `img_size`。
  - `predict_and_cost.py` 新增 `_prep_rgb` / `_to_prepped`：raw draft 解码结果逐图走
    `utils.preprocess.crop_resize`（灰度>15 bbox（0.2/99.8 分位）→ 居中方形 → **PIL BICUBIC
    共享 512**）。该函数与 pkg3 出包同源（`ed6bda3:finals/code/utils/preprocess.py`），
    bbox 按 `(x, y)` 显式求解，**结构上不存在转置**。
  - `members.json` 的 `cache_size` 统一 512（字段保留兼容）；tta 字段保留，默认 1。
- 关键代码位置：`model.py:88`（`forward_probs`）、`predict_and_cost.py:135`（`_prep_rgb`）、
  `:150`（`preprocess`）、`:159`（`_to_prepped`）、`:197`（`predict_batch`）。

## 2. Fixture 清单

`tools/finals_prep_parity.py` 每次运行按固定种子确定性重建并落盘（`finals_t4_out/smoke/fixtures221/`）：

| 组 | 尺寸 (H×W) | 张数 | 说明 |
|---|---|---|---|
| `square_1024` | 1024×1024 | 6 | 复用 `finals_t4_out/smoke/imgs`（缺失时合成） |
| `nonsq_768x512` | 768×512 | 3 | 合成，亮盘偏心 0~±6% |
| `nonsq_1024x680` | 1024×680 | 3 | 合成，亮盘偏心反向 |
| `nonsq_1024x512_off` | 1024×512 | 3 | 合成，亮盘强偏移（cx≈0.30–0.38） |

合成图：暗背景 + 亮度梯度圆盘（部分可出框）+ 噪声纹理，保证 `gray>15` 面积占比落在
分位裁剪生效区间。扁平副本 + meta：`finals_t4_out/smoke/nonsq_all/`（供 CLI smoke）。

## 3. Parity 结果（门槛：mean|Δp| ≤ 0.005 且 0 个 0.5 翻转）

参照实现 = `ref_crop_resize`（pkg3 `crop_resize` 逐行拷贝，独立于包内代码）。
候选 = 包内 `predict_batch`（当前 `members.json`：3 成员 0.5/0.25/0.25，tta1，概率平均）。

| 组 | n | 候选 vs 参照 mean | max | 翻转 | 门禁 | 旧转置实现 vs 参照 mean | max |
|---|---|---|---|---|---|---|---|
| square_1024 | 6 | **0.00000** | 0.00000 | 0 | PASS | 0.01169 | 0.01179 |
| nonsq_768x512 | 3 | **0.00000** | 0.00000 | 0 | PASS | 0.02766 | 0.05117 |
| nonsq_1024x680 | 3 | **0.00000** | 0.00000 | 0 | PASS | 0.06337 | 0.07339 |
| nonsq_1024x512_off | 3 | **0.00000** | 0.00000 | 0 | PASS | 0.08434 | 0.11200 |

- 候选与 pkg3 参照**逐位一致**（同一 `crop_resize` + 同一成员内插/概率加权）。
- 旧转置实现（`buggy_prep`，HEAD 的 `_boxes/_crop` 原样）在方图上已有 0.012 偏差
  （偏心盘 bbox 非对称），非方图升到 0.028–0.084，**门禁能判红**（回归演示）。
- 与 issue 实测（非方图 mean 0.0192 / max 0.1029）同量级；差异来自 fixture 集合不同。

### 归档锚点（候选 vs pkg3 归档 CSV，draft768 解码，归档当时的配比）

| 归档 | n | 候选 mean | max | 翻转 | 门禁 | 同配比参照(logit) mean |
|---|---|---|---|---|---|---|
| `out_pkg3`（ACRIMA 20 图，2 成员 0.667/0.333，概率平均） | 20 | 0.00256 | 0.00817 | 0 | PASS | 0.00692 |
| `out_pkg`（smoke/imgs 1024²，3 成员 0.4/0.3/0.3，logit 平均） | 6 | 0.00474 | 0.00489 | 0 | PASS | **0.000028** |

- 方图归档：同配比 + logit 组合时与归档**逐位重合（2.8e-5）**，证明参照 prep == pkg3 归档；
  候选（概率平均，dist4 线）与其差 0.0047，纯粹是组合方式差异，仍在门禁内。
- ACRIMA 锚点：候选按归档配比（2 成员 + 概率平均）复现到 0.00256 / 0 翻转。
- 说明：`out_pkg3` 是 11-18 本地 2 成员 PIL 构建的 smoke 归档，与最终 dist3/dist4 线上包
  （GPU 裁剪线）不是同一构建；本表用于「预处理口径」对照，不用于复刻线上分数。

## 4. 端到端验证

- `tools/no_cv2_smoke.py`（屏蔽 cv2 + 官方 CLI，CPU）：
  - 方图 6 张 + 1 缺失：7/7 行输出，缺失行 Probability=0.5，`cost.json` 正常 → SMOKE OK。
  - 非方图 9 张 + 1 缺失：10/10 行输出，缺失行 0.5 → SMOKE OK。
- GPU（本机 5060，官方 CLI，draft384 单解码 + 共享 512):
  - `smoke/big20`（2048²，n=15）：**avg_cost 56.8ms**（dist4 归档同集 73.5ms）。
  - `smoke/nonsq_all`（9 张非方图，n=4）：avg_cost 61.5ms。
  - `smoke/imgs`（1024²，n=1）：avg_cost 55.0ms。
  - 三者均低于 100ms；TTA2-cb 彩票包不在本单范围，重打包后需另行复测。
- `run_cost` 抽样/预热/计时代码零改动（diff 仅覆盖选手区与 docstring）。

## 5. 复跑

```bash
# parity 门禁（GPU；输出 JSON + 表格，失败退出码 1）
python tools/finals_prep_parity.py
# 归档缺失时可跳过锚点
python tools/finals_prep_parity.py --no-archive

# 端到端 no-cv2 smoke（CPU）
CUDA_VISIBLE_DEVICES=-1 python tools/no_cv2_smoke.py --pkg finals \
  --images finals_t4_out/smoke/nonsq_all --meta finals_t4_out/smoke/nonsq_all/meta.csv \
  --out finals_t4_out/smoke/out_221cpu_nonsq --add-missing
```

报告 JSON：`finals_t4_out/smoke/fixtures221/parity221.json`。

## 6. 边界与后续

- **未重新打包**：本单只改服务代码 + 门禁；包内 `SHA256SUMS.txt` 已按新代码刷新，
  zip 需按 `tools/finals_build_pkg.py` 重新构建并回填体积/哈希。
- #216 TTA2-cb 彩票包基于旧 `_boxes` 线，重打包后才能使用（本单闭环前按 issue 禁令不重出包）。
- draft 仍为 384（dist4 定格档位，未动 `read_one`）；与 pkg3 归档的 draft768 解码差异在本
  fixture 上 ≤0.0005（同配比 logit：draft768 2.8e-5 / draft384 5e-4）。