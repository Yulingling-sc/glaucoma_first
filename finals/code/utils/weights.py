"""权重解析：arch / img_size / cache_size / use 元信息，兼容瘦身 fp16 单状态权重。"""
import os

import torch

DEFAULT_CACHE_SIZE = 512


def _infer_cache_size(ckpt, path):
    if isinstance(ckpt, dict):
        cs = ckpt.get("cache_size")
        if cs in (512, 768):
            return int(cs)
        args = ckpt.get("args")
        if isinstance(args, dict) and "768" in str(args.get("cache_root", "")):
            return 768
    if "cache768" in os.path.basename(str(path)):
        return 768
    return DEFAULT_CACHE_SIZE


def load_checkpoint(path):
    """返回 (state_dict, meta)；meta 含 arch / img_size / cache_size / use。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        return ckpt, {"arch": "convnext_base", "img_size": 384,
                      "cache_size": DEFAULT_CACHE_SIZE, "use": "raw"}
    arch = ckpt.get("arch", "convnext_base")
    img_size = int(ckpt.get("img_size", 384))
    cache_size = _infer_cache_size(ckpt, path)
    use = ckpt.get("use")
    if use not in ("ema", "model"):
        use = "ema" if ckpt.get("auroc_ema", -1) >= ckpt.get("auroc", -1) else "model"
    sd = ckpt.get(use) or ckpt.get("state_dict") or ckpt.get("model")
    if sd is None:
        sd = ckpt
    return sd, {"arch": arch, "img_size": img_size,
                "cache_size": cache_size, "use": use}