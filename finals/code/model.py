"""FundusNet：与训练一致的骨干 + 年龄 MLP 融合头（推理侧只依赖 torchvision）。

#215/#216 口径：单解码 draft384 -> **GPU 侧** bbox/裁剪/插值，每成员可配 TTA。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


ARCH_INFO = {
    "efficientnet_b0": (torchvision.models.efficientnet_b0, 1280),
    "efficientnet_b3": (torchvision.models.efficientnet_b3, 1536),
    "efficientnet_b4": (torchvision.models.efficientnet_b4, 1792),
    "convnext_tiny": (torchvision.models.convnext_tiny, 768),
    "convnext_small": (torchvision.models.convnext_small, 768),
    "convnext_base": (torchvision.models.convnext_base, 1024),
    "resnet50": (torchvision.models.resnet50, 2048),
}


class FundusNet(nn.Module):
    def __init__(self, arch="convnext_base", age_dim=16, dropout=0.3):
        super().__init__()
        if arch not in ARCH_INFO:
            raise ValueError(f"unknown arch: {arch}")
        ctor, feat_dim = ARCH_INFO[arch]
        m = ctor(weights=None)
        if arch.startswith("efficientnet"):
            self.features = m.features
            self.avgpool = m.avgpool
            self.norm = nn.Identity()
        elif arch.startswith("convnext"):
            self.features = m.features
            self.avgpool = m.avgpool
            self.norm = m.classifier[0]
        elif arch.startswith("resnet"):
            self.features = nn.Sequential(*list(m.children())[:-2])
            self.avgpool = m.avgpool
            self.norm = nn.Identity()
        else:
            raise ValueError(arch)
        self.age_mlp = nn.Sequential(
            nn.Linear(1, age_dim), nn.SiLU(), nn.Linear(age_dim, age_dim), nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(feat_dim + age_dim, 512), nn.BatchNorm1d(512), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(512, 1),
        )

    def forward(self, x, age):
        f = self.features(x)
        f = self.avgpool(f)
        f = self.norm(f)
        f = f.flatten(1)
        a = self.age_mlp(age)
        return self.head(torch.cat([f, a], dim=1)).squeeze(1)


AGE_MEAN = 58.0
AGE_STD = 15.0
MASK_THRESHOLD = 15.0 / 255.0


class Ensemble:
    """3 成员加权概率平均；bbox/裁剪/插值全部在 GPU 上做，CPU 只负责一次 decode。

    - 输入是单次 draft384 解码得到的整图 batch（B,3,H,W，[0,1]），各成员按自己的
      cache_size 从**同一份**解码结果上采样裁剪，再插值到 img_size；
      等价于训练缓存口径「灰度>15 bbox（0.2/99.8 分位）-> 居中方形 -> 缓存档位 -> img_size」。
    - 每成员按 meta["tta"] 独立做翻转 TTA（#216：cb384 TTA2，两个 small 保持 TTA1），
      概率先 sigmoid、TTA 内平均，再按权重相加（与 submit_pkg/predict.py 同口径）。
    - forward_probs 做前向+sigmoid（计时用 forward），postprocess 只做 CPU 回传（计时用 post）。
    - 年龄入模前按训练口径归一化 (age - 58) / 15（对应 train.py 的 age_tensor）。
    """

    def __init__(self, members, device="cuda"):
        # members: list of (FundusNet, meta dict, weight float)
        # meta: cache_size / img_size / tta
        self.members = members
        self.device = torch.device(device)
        self.ttas = {1: [0], 2: [0, 1], 4: [0, 1, 2, 3]}
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    # ---------------------------------------------------------------- GPU 预处理
    @staticmethod
    def _boxes(x):
        """整图 batch -> 每图 (x0, y0, x1, y1)；与训练缓存同一分位口径，在 GPU 上算。"""
        b, _, h, w = x.shape
        mask = x.mean(dim=1) > MASK_THRESHOLD
        boxes = []
        for i in range(b):
            m = mask[i]
            frac = float(m.float().mean())
            if frac < 0.02 or frac > 0.999:
                boxes.append((0.0, 0.0, float(w - 1), float(h - 1)))
                continue
            idx = m.nonzero(as_tuple=False).float()          # (N,2) -> (y,x)
            qs = torch.quantile(idx, torch.tensor([0.002, 0.998], device=x.device), dim=0)
            boxes.append((float(qs[0, 0]), float(qs[0, 1]),
                          float(qs[1, 0]), float(qs[1, 1])))
        return boxes

    @staticmethod
    def _crop(x, boxes, size):
        """按 bbox 居中方形采样到 (B,3,size,size)；框外补 0（与训练 np.pad 一致）。"""
        b, _, h, w = x.shape
        t = torch.linspace(0.0, 1.0, size, device=x.device)
        grids = []
        for x0, y0, x1, y1 in boxes:
            side = max(x1 - x0, y1 - y0) * 1.02
            cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
            px = (cx - side * 0.5) + t * side
            py = (cy - side * 0.5) + t * side
            gx = (2.0 * px / max(w - 1, 1) - 1.0).view(1, -1).expand(size, size)
            gy = (2.0 * py / max(h - 1, 1) - 1.0).view(-1, 1).expand(size, size)
            grids.append(torch.stack([gx, gy], dim=-1))
        grid = torch.stack(grids, dim=0)                      # (B,size,size,2)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros",
                             align_corners=True)

    def forward_probs(self, batch, ages, use_fp16=True):
        """batch: (B,3,H,W) tensor/numpy，值域 [0,1]（read_one 的单次解码结果）。"""
        if not torch.is_tensor(batch):
            batch = torch.from_numpy(np.asarray(batch))
        if batch.ndim == 3:
            batch = batch.unsqueeze(0)
        if batch.shape[-1] == 3 and batch.shape[1] != 3:   # HWC -> CHW
            batch = batch.permute(0, 3, 1, 2)
        batch = batch.to(self.device, non_blocking=True).float()
        if batch.max() > 1.001:
            batch = batch / 255.0
        age = torch.as_tensor(ages, dtype=torch.float32, device=self.device).view(-1, 1)
        age = (age - AGE_MEAN) / AGE_STD

        boxes = self._boxes(batch)
        crops = {}
        out = None
        with torch.inference_mode():
            for model, meta, w in self.members:
                cache = int(meta["cache_size"])
                if cache not in crops:
                    crops[cache] = self._crop(batch, boxes, cache)
                xi = crops[cache]
                if int(meta["img_size"]) != xi.shape[-1]:
                    xi = F.interpolate(xi, size=(int(meta["img_size"]),) * 2,
                                       mode="bilinear", align_corners=False)
                xi = (xi - self._mean) / self._std
                xi = xi.contiguous(memory_format=torch.channels_last)
                flips = self.ttas[int(meta.get("tta", 1))]
                acc = 0.0
                for fl in flips:
                    xf = xi
                    if fl & 1:
                        xf = torch.flip(xf, dims=[3])
                    if fl & 2:
                        xf = torch.flip(xf, dims=[2])
                    with torch.amp.autocast("cuda", dtype=torch.float16,
                                            enabled=(use_fp16 and self.device.type == "cuda")):
                        logits = model(xf, age)
                    acc = acc + torch.sigmoid(logits.float())   # 与参考口径同: fp32 下 sigmoid
                contrib = w * acc / len(flips)
                out = contrib if out is None else out + contrib
        return out.float()

    def eval(self):
        for model, _meta, _w in self.members:
            model.eval()
        return self

    @staticmethod
    def postprocess(probs):
        return probs.clamp(0.0, 1.0).cpu().numpy()

    def predict(self, batch, ages, use_fp16=True):
        return self.postprocess(self.forward_probs(batch, ages, use_fp16))
