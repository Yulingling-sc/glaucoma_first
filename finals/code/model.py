"""FundusNet：与训练一致的骨干 + 年龄 MLP 融合头（推理侧只依赖 torchvision）。"""
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


class Ensemble:
    """加权概率平均：全部成员共用一次预处理结果（同一 cache 档位）。

    forward_logits 只做前向（计时用 forward），postprocess 只做 sigmoid（计时用 post）；
    predict 是两者串联，全量预测与 cost 计时因此走完全相同的数值路径。
    """

    def __init__(self, members, device="cuda"):
        # members: list of (FundusNet, meta dict, weight float)
        self.members = members
        self.device = device
        self.ttas = {1: [0], 2: [0, 1], 4: [0, 1, 2, 3]}
        self.cache_size = members[0][1]["cache_size"]
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def forward_logits(self, tensors, ages, tta=1, use_fp16=True):
        x = torch.stack(tensors).to(self.device, non_blocking=True)
        age = torch.as_tensor(ages, dtype=torch.float32, device=self.device).view(-1, 1)
        flips = self.ttas[tta]
        out = None
        with torch.inference_mode():
            for model, meta, w in self.members:
                xi = x if meta["img_size"] == self.cache_size else F.interpolate(
                    x, size=(meta["img_size"],) * 2, mode="bilinear", align_corners=False)
                xi = (xi - self._mean) / self._std
                acc = 0.0
                for fl in flips:
                    xf = xi
                    if fl & 1:
                        xf = torch.flip(xf, dims=[3])
                    if fl & 2:
                        xf = torch.flip(xf, dims=[2])
                    with torch.amp.autocast("cuda", dtype=torch.float16,
                                            enabled=(use_fp16 and self.device == "cuda")):
                        acc = acc + model(xf, age)
                contrib = w * acc / len(flips)
                out = contrib if out is None else out + contrib
        return out.float()

    @staticmethod
    def postprocess(logits):
        return torch.sigmoid(logits).cpu().numpy()

    def predict(self, tensors, ages, tta=1, use_fp16=True):
        return self.postprocess(self.forward_logits(tensors, ages, tta, use_fp16))