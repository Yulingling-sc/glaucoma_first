# 训练复现（最小依赖）

`train.py` 即夺冠方案训练脚本（ImageNet 预训练骨干 + 年龄 MLP 融合头，pos_weight BCE、AMP、EMA、cosine+warmup）。
`paths.py` 只负责路径/标签读取；`preprocess.py` 从官方图片生成训练缓存。

## 数据布局

```
<DATA_ROOT>/
  train.csv  valid.csv  test.csv        # 官方发布，列：Image_name, Age, Label(0/1)
  images/                               # 官方 train/valid 图片
  cache512/train/*.jpg  cache512/test/*.jpg   # python preprocess.py 生成
```

## 三个成员（与 model/ 权重一一对应）

```bash
# cb384（train-only，cache512；对应 #65 的热启动基底）
python train.py --arch convnext_base --img-size 384 --cache-root <DATA_ROOT>/cache512 \
  --epochs 8 --batch-size 8 --accum 6 --lr 1e-4 --head-lr 5e-4 \
  --loader workers --workers 2 --seed 42 --out-dir runs/cb384

# cs_hi512_all（train+all，cache768；#54 配方）
python train.py --arch convnext_small --img-size 512 --cache-root <DATA_ROOT>/cache768 \
  --epochs 8 --batch-size 8 --accum 6 --lr 3e-5 --head-lr 1.5e-4 \
  --loader workers --workers 3 --train-on-all \
  --init-from <hi512_v2>.pt --seed 42 --out-dir runs/cs_hi512_all

# cs_hi512_all_ra（同上 + RandAugment，成员 #8）
python train.py --arch convnext_small --img-size 512 --cache-root <DATA_ROOT>/cache768 \
  --epochs 8 --batch-size 8 --accum 6 --lr 3e-5 --head-lr 1.5e-4 \
  --randaugment 2 9 --loader workers --workers 3 --train-on-all \
  --init-from <hi512_v2>.pt --seed 42 --out-dir runs/cs_hi512_all_ra
```

`train-on-all` 把 valid 并入训练、不做早停（固定 last.pt + EMA），与提交包 checkpoint 的 `use=ema` 一致。
权威配方以仓库 issue #8 / #54 / #65 的评论为准。

训练额外依赖 `scikit-learn`（指标计算）；推理只用 `requirements.txt` 里的 torch / torchvision / numpy / pillow，
不安装训练依赖也能跑 `code/predict_and_cost.py`。