"""FundusNet：与训练一致的骨干 + 年龄 MLP 融合头（推理侧只依赖 torchvision）。

#221 口径：服务代码回退到 pkg3/dist4 线——CPU 侧 PIL BICUBIC 把视野裁成居中方形并
缩放到共享 512（utils.preprocess.crop_resize），成员再内插到各自 img_size；TTA 默认 1。
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


class Ensemble:
    """3 成员加权概率平均（pkg3/dist4 线：共享 512 预处理 + 成员内插）。

    - 输入是已预处理的整批张量（B,3,512,512，[0,1]）：裁剪/缩放（灰度>15 bbox
      （0.2/99.8 分位）-> 居中方形 -> PIL BICUBIC 512）由 CPU 侧 preprocess 完成，
      这里只按成员 img_size 做双线性内插（与 pkg3 出分包一致）。bbox 按 (x,y)
      显式求解，非方图不存在 x/y 转置。
    - 每成员按 meta["tta"] 独立做翻转 TTA（默认 1）；概率先 sigmoid、TTA 内平均，
      再按权重相加（与 dist3/dist4 出分包同口径）。
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

    def forward_probs(self, batch, ages, use_fp16=True):
        """batch: (B,3,S,S) 预处理后的 tensor/numpy（S=共享缓存 512），值域 [0,1]。"""
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

        out = None
        with torch.inference_mode():
            for model, meta, w in self.members:
                xi = batch
                img_size = int(meta["img_size"])
                if img_size != xi.shape[-1]:
                    xi = F.interpolate(xi, size=(img_size, img_size),
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
