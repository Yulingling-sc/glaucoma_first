"""按决赛 3 成员包更新《模型核验材料与数据申明清单.xlsx》。

  python tools/finals_xlsx.py --t4-clean3 86.0 --t4-pseudo3 79.8 --t4-clean2 66.6 \
      --src docs/模型核验材料与数据申明清单_20261028.xlsx \
      --out finals/模型核验材料与数据申明清单.xlsx
"""
import argparse
import os

import openpyxl
from openpyxl.styles import Alignment

A1 = """cb384 = ConvNeXt-Base@384（convnext_base_384_cache512_1015_auprc0.8644.pt，train-only，use=model）
cs_hi512_all = ConvNeXt-Small@512 train+all（cs_hi512_all_fp16.pt，use=ema）
cs_hi512_all_ra = ConvNeXt-Small@512 train+all + RandAugment（cs_hi512_all_ra.pt，use=ema）
三者均为 ImageNet-1k 预训练骨干 + 本赛题官方 train/valid 微调，无自监督、无医学预训练、无测试伪标签。"""

A2 = """cb384: releases/download/weights-1016/convnext_base_384_cache512_1015_auprc0.8644.pt → 176,328,125 B
cs_hi512_all: releases/download/weights-1016/cs_hi512_all_fp16.pt → 199,660,651 B
cs_hi512_all_ra: releases/download/compliant_pkg-1028/cs_hi512_all_ra.pt → 99,831,694 B
仓库：https://github.com/Yulingling-sc/glaucoma_first/releases"""

A5 = """【预训练骨干】
convnext_base-6075fbad.pth｜354,486,097 B｜6075fbad31c688f9ed013922a59daf0fc2f4dc12e6e43bc01fa840c7bd1ca2cd
convnext_small-0c510722.pth｜201,014,929 B｜0c510722adfd92966a2bd72b92f785ca05966bbac03cafe2f7a90b1f54bfab9a
【微调权重】
cb384｜176,328,125 B｜a389f41cb367132b4c23e2f9195bf15ac87c9df1214d359656f36ee63aff710e
cs_hi512_all｜199,660,651 B｜3e04d1906965742d8b13657b16e539063b05dab622c5272454939f57105f492e
cs_hi512_all_ra｜99,831,694 B｜5a04ebe3152416022b0e07c9d42f7af57532b9cb366771c09889173a154c86e6
（与包内 SHA256SUMS.txt 逐字节一致）"""

A6 = """cb384：valid TTA4 AUROC 0.99117 / AUPRC 0.86439（weights-1016 记录）
cs_hi512_all：train+all 家族成员，check PASS
cs_hi512_all_ra：check PASS 0.9132（历史最高成员）
集合（0.4/0.3/0.3，TTA1）valid 指标与决赛 AUPRC 预期由 #208 定稿回填；本表不重复登记。"""

E1 = """最终 3 成员包（cb384 + cs_hi512_all + cs_hi512_all_ra）不含任何测试集统计量、不含跨图归一化参数、不含测试伪标签；预处理逐图独立（code/utils/preprocess.py）。"""

E2 = """3 模型加权概率平均（0.4/0.3/0.3，见 model/members.json），TTA1（单尺度单次前向）；计时 batch=1，全量预测 batch=32；BN 保持 eval 模式（不更新 running stats）。"""

F2 = """cost 路径与全量预测调用同一组函数（code/predict_and_cost.py 的 preprocess / predict_batch）：本地 6 张连续两次运行 max|Δp| = 0.0（≤1e-3），二值判定一致；官方逐张比对待决赛复测。"""

APPENDIX = {
    2: ("https://github.com/Yulingling-sc/glaucoma_first/releases/download/weights-1016/"
        "convnext_base_384_cache512_1015_auprc0.8644.pt", "176,328,125",
        "a389f41cb367132b4c23e2f9195bf15ac87c9df1214d359656f36ee63aff710e"),
    3: ("https://github.com/Yulingling-sc/glaucoma_first/releases/download/weights-1016/"
        "cs_hi512_all_fp16.pt", "199,660,651",
        "3e04d1906965742d8b13657b16e539063b05dab622c5272454939f57105f492e"),
    4: ("https://github.com/Yulingling-sc/glaucoma_first/releases/download/compliant_pkg-1028/"
        "cs_hi512_all_ra.pt", "99,831,694",
        "5a04ebe3152416022b0e07c9d42f7af57532b9cb366771c09889173a154c86e6"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="docs/模型核验材料与数据申明清单_20261028.xlsx")
    ap.add_argument("--out", default="finals/模型核验材料与数据申明清单.xlsx")
    ap.add_argument("--t4-clean3", type=float, required=True, help="干净 3 模型 TTA1 均值 ms")
    ap.add_argument("--t4-pseudo3", type=float, default=None, help="含伪标签 3 模型 TTA1 均值 ms")
    ap.add_argument("--t4-clean2", type=float, default=None, help="干净 2 模型 TTA1 均值 ms")
    args = ap.parse_args()

    wb = openpyxl.load_workbook(args.src)
    wrap = Alignment(wrap_text=True, vertical="top")

    ws = wb["说明"]
    ws.cell(4, 2).value = "2026-11-18"
    ws.cell(5, 2).value = "@JiahaoChen-star-JJ（执行）｜对应工单 #207"
    ws.cell(6, 2).value = ("cb384（ConvNeXt-Base@384）+ cs_hi512_all（ConvNeXt-Small@512）"
                           "+ cs_hi512_all_ra（ConvNeXt-Small@512 RA），统一入口 code/predict_and_cost.py")
    ws.cell(9, 2).value = ("A1–A6、B1–B3、C1–C2、D、E1–E3、F1–F3；T4 实测见 F1；"
                           "决赛包结构见 docs/finals/T4_LATENCY.md 与本包 SHA256SUMS.txt")
    ws.cell(10, 2).value = ("官方镜像（torch 2.10.0 / CUDA 12.8）T4 复测待决赛平台实测（当前用 Kaggle T4 / torch 2.11.0）；"
                           "接口按决赛页三函数（load_model / preprocess / predict_batch）实现。")

    ws = wb["A 模型核验"]
    ws.cell(2, 3).value = A1
    ws.cell(2, 3).alignment = wrap
    ws.cell(3, 3).value = A2
    ws.cell(3, 3).alignment = wrap
    ws.cell(6, 3).value = A5
    ws.cell(6, 3).alignment = wrap
    ws.cell(6, 4).value = "已核实｜三个微调权重 SHA-256 与包内 SHA256SUMS.txt 一致"
    if ws.cell(7, 1).value != "A6":
        ws.insert_rows(7)
        ws.cell(7, 1).value = "A6"
        ws.cell(7, 2).value = "推理精度（valid）"
    ws.cell(7, 3).value = A6
    ws.cell(7, 3).alignment = wrap
    ws.cell(7, 4).value = "部分待 #208 回填"

    ws = wb["B-C-D 数据申明"]
    ws.cell(3, 3).value = ("最终 3 成员包未使用任何测试集标签或伪标签；test.csv 仅用于推理出概率。"
                           "历史 pseudo 实验（cnxb_384_all_pseudo）不进决赛包。")
    ws.cell(3, 4).value = "已核实｜包内 members.json 仅列 cb384 / cs_hi512_all / cs_hi512_all_ra"

    ws = wb["E 评测与推理"]
    ws.cell(2, 3).value = E1
    ws.cell(2, 3).alignment = wrap
    ws.cell(3, 3).value = E2
    ws.cell(3, 3).alignment = wrap
    t4 = (f"包体 475.82 MB（十进制，解压后；cb384 176.33 + cs_hi512_all 199.66 + cs_hi512_all_ra 99.83）≤500MB；"
          f"Kaggle T4 15GB 实测 3 模型 TTA1 端到端均值 {args.t4_clean3:.2f} ms/张（随机 5 预热 + 100 张、batch=1）")
    if args.t4_clean2:
        t4 += f"；干净 2 模型（cb384+cs_hi512_all_ra）{args.t4_clean2:.2f} ms"
    ws.cell(4, 3).value = t4
    ws.cell(4, 3).alignment = wrap
    ws.cell(4, 4).value = "已核实｜见 docs/finals/T4_LATENCY.md；官方镜像待复测"

    ws = wb["F 合规包"]
    f1 = (f"Kaggle Tesla T4 15GB（torch 2.11.0+cu130）实测：干净 3 模型 TTA1 均值 {args.t4_clean3:.2f} ms"
          f"（p95 待回填），≤100ms 红线；目标 ≤85ms 与红线之间需在官方镜像复测确认。")
    if args.t4_pseudo3:
        f1 += f" 含伪标签 3 模型 TTA1 {args.t4_pseudo3:.2f} ms。"
    if args.t4_clean2:
        f1 += f" 干净 2 模型 {args.t4_clean2:.2f} ms。"
    ws.cell(2, 3).value = f1
    ws.cell(2, 3).alignment = wrap
    ws.cell(2, 4).value = "已实测（Kaggle T4）｜官方镜像未实测"
    ws.cell(3, 3).value = F2
    ws.cell(3, 3).alignment = wrap
    ws.cell(4, 3).value = ("最终推理实际加载：cb384 176.33MB + cs_hi512_all 199.66MB + cs_hi512_all_ra 99.83MB "
                           "= 475.82MB（十进制，解压后）≤500MB；无其他被调用权重。")
    ws.cell(4, 4).value = "已核实｜包内 SHA256SUMS.txt 与 model/members.json 一致"

    ws = wb["附录-直链与SHA256"]
    for row, (url, size, sha) in APPENDIX.items():
        ws.cell(row, 2).value = url
        ws.cell(row, 2).alignment = wrap
        ws.cell(row, 3).value = size
        ws.cell(row, 4).value = sha

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    wb.save(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()