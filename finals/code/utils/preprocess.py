"""眼底图预处理：JPEG draft 解码 -> 圆形视野裁剪 -> 居中方形 -> cache_size。

推理与计时走同一套函数（cost 拆解时分别计时 decode / crop_resize）：
    preprocess(path) == crop_resize(*decode(path, ...))
"""
import numpy as np
import torchvision
from PIL import Image

DEFAULT_CACHE_SIZE = 512


def decode(path, draft):
    """JPEG draft 解码（RGB）。返回 (PIL.Image, np.ndarray[H,W,3])。"""
    im = Image.open(path)
    im.draft("RGB", (draft, draft))
    im = im.convert("RGB")
    return im, np.asarray(im)


def crop_resize(im, arr, size=DEFAULT_CACHE_SIZE):
    """视野 bbox（gray>15，0.2/99.8 分位去边）-> 居中方形 -> BICUBIC 到 size。"""
    w, h = im.size
    gray = arr.mean(axis=2)
    mask = gray > 15
    frac = float(mask.mean())
    if frac < 0.02 or frac > 0.999:
        bbox = (0, 0, w, h)
    else:
        ys, xs = np.nonzero(mask)
        y0, y1 = np.percentile(ys, [0.2, 99.8])
        x0, x1 = np.percentile(xs, [0.2, 99.8])
        bbox = (float(x0), float(y0), float(x1), float(y1))
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * 1.02
    nx0, ny0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    nx1, ny1 = int(round(cx + side / 2)), int(round(cy + side / 2))
    pad_l, pad_t = max(0, -nx0), max(0, -ny0)
    pad_r, pad_b = max(0, nx1 - w), max(0, ny1 - h)
    crop = arr[max(0, ny0):min(h, ny1), max(0, nx0):min(w, nx1)]
    if pad_l or pad_t or pad_r or pad_b:
        crop = np.pad(crop, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="constant")
    out = Image.fromarray(crop).resize((size, size), Image.BICUBIC)
    return torchvision.transforms.functional.to_tensor(out)


def preprocess(image, cache_size=DEFAULT_CACHE_SIZE):
    """单图 -> float32 tensor (3, cache_size, cache_size)，值域 [0,1]。

    image 可为路径（走 JPEG draft）或已解码的 PIL.Image / np.ndarray。
    """
    if isinstance(image, str):
        draft = max(768, int(cache_size * 1.5))
        im, arr = decode(image, draft)
    elif isinstance(image, Image.Image):
        im, arr = image.convert("RGB"), np.asarray(image.convert("RGB"))
    else:
        arr = np.asarray(image)
        im = Image.fromarray(arr)
    return crop_resize(im, arr, cache_size)