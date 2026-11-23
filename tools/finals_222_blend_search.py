"""#222 冲分配置研究：权重曲面 / 融合口径 / TTA 上界（全部 valid 后处理，零推理）。

  python tools/finals_222_blend_search.py --probs-dir .scratch_208 \
      --valid-csv glaucoma_first/valid.csv --boot 2000 --out docs/finals/BLEND_SEARCH_222.json
"""
import argparse
import csv
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

BASE_W = (0.5, 0.25, 0.25)
FILES = {
    "cb384": "cb384_pil.npy.npz",
    "cs_hi512_all": "cs_hi512_all_pil.npy.npz",
    "cs_hi512_all_ra": "cs_hi512_all_ra_pil.npy.npz",
    "ctiny384": "ctiny384_pil.npy.npz",
}
LABEL_COL = "True_Label"


def metrics(y, p):
    return roc_auc_score(y, p), average_precision_score(y, p)


def load_members(probs_dir, tta="probs_tta1"):
    out, names = {}, None
    for tag, fn in FILES.items():
        path = os.path.join(probs_dir, fn)
        if not os.path.exists(path):
            continue
        z = np.load(path)
        if names is None:
            names = list(z["names"])
        elif list(z["names"]) != names:
            raise SystemExit(f"order mismatch in {fn}")
        out[tag] = z[tta].astype(np.float64)
    return names, out


def labels(csv_path, names):
    lab = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            lab[r["Image_name"]] = int(float(r[LABEL_COL]))
    return np.array([lab[str(n)] for n in names], dtype=np.int64)


def grid_weights(lo_cb, hi_cb, lo_s, hi_s, step):
    def vals(lo, hi):
        n = int(round((hi - lo) / step)) + 1
        return [round(lo + i * step, 4) for i in range(n)]
    rows = []
    for a in vals(lo_cb, hi_cb):
        for b in vals(lo_s, hi_s):
            for c in vals(lo_s, hi_s):
                if abs(a + b + c - 1.0) < 1e-9:
                    rows.append((a, b, c))
    return rows


def boot_paired(y, p_cand, p_base, n_boot, rng):
    n = len(y)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() < 2:
            deltas[i] = np.nan
            continue
        deltas[i] = average_precision_score(y[idx], p_cand[idx]) - average_precision_score(
            y[idx], p_base[idx])
    d = deltas[~np.isnan(deltas)]
    return {
        "mean": float(np.mean(d)),
        "ci_lo": float(np.percentile(d, 2.5)),
        "ci_hi": float(np.percentile(d, 97.5)),
        "p_gt_0": float(np.mean(d > 0)),
        "n_boot": int(len(d)),
    }


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def boot_ap(y, p, n_boot, rng):
    n = len(y)
    vals = [average_precision_score(y[i], p[i])
            for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return {"mean": float(np.mean(vals)),
            "ci_lo": float(np.percentile(vals, 2.5)),
            "ci_hi": float(np.percentile(vals, 97.5))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs-dir", default=".scratch_208")
    ap.add_argument("--valid-csv", default="glaucoma_first/valid.csv")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--gate-delta", type=float, default=0.002)
    ap.add_argument("--out", default="docs/finals/BLEND_SEARCH_222.json")
    args = ap.parse_args()

    names, mem = load_members(args.probs_dir)
    y = labels(args.valid_csv, names)
    print(f"rows={len(y)} pos={int(y.sum())} rate={y.mean():.5f} members={sorted(mem)}")
    rng = np.random.default_rng(args.seed)

    cb, cs, ra = mem["cb384"], mem["cs_hi512_all"], mem["cs_hi512_all_ra"]
    base = BASE_W[0] * cb + BASE_W[1] * cs + BASE_W[2] * ra
    base_auroc, base_auprc = metrics(y, base)
    print(f"BASE {BASE_W}  AUROC {base_auroc:.5f} AUPRC {base_auprc:.5f}")
    base_ci = boot_ap(y, base, args.boot, rng)
    print(f"     基线 AUPRC bootstrap 95% CI [{base_ci['ci_lo']:.5f}, {base_ci['ci_hi']:.5f}]"
          f"（宽度 {base_ci['ci_hi'] - base_ci['ci_lo']:.5f} = 绝对噪声尺度参照）")

    # A. constrained weight grid
    rows = []
    for w in grid_weights(0.40, 0.60, 0.15, 0.35, args.step):
        p = w[0] * cb + w[1] * cs + w[2] * ra
        a, pr = metrics(y, p)
        rows.append({"w_cb": w[0], "w_cs": w[1], "w_ra": w[2],
                     "auprc": pr, "auroc": a, "d_auprc": pr - base_auprc})
    rows.sort(key=lambda r: r["auprc"], reverse=True)
    print(f"\n[A] 权重网格（sum=1，cb 0.40–0.60，smalls 0.15–0.35，步长 {args.step}）n={len(rows)}")
    print(f"    {'cb':>5} {'cs':>5} {'ra':>5}  {'AUPRC':>8} {'dAUPRC':>8} {'AUROC':>8}")
    for r in rows[:8] + [{"w_cb": "…"}] + rows[-3:]:
        if "auprc" not in r:
            print("      …")
            continue
        print(f"    {r['w_cb']:5.2f} {r['w_cs']:5.2f} {r['w_ra']:5.2f}  "
              f"{r['auprc']:8.5f} {r['d_auprc']:+8.5f} {r['auroc']:8.5f}")
    n_pass = sum(1 for r in rows if r["d_auprc"] >= args.gate_delta)
    print(f"    ΔAUPRC ≥ {args.gate_delta} 的格点：{n_pass}/{len(rows)}；"
          f"网格极差 {rows[0]['auprc'] - rows[-1]['auprc']:.5f}")

    perm = []
    for _ in range(args.perm):
        yp = rng.permutation(y)
        vals = [average_precision_score(yp, w[0] * cb + w[1] * cs + w[2] * ra)
                for w in [(r["w_cb"], r["w_cs"], r["w_ra"]) for r in rows]]
        base_p = average_precision_score(yp, sum(w * p for w, p in zip(BASE_W, [cb, cs, ra])))
        perm.append(max(vals) - base_p)
    perm = np.array(perm)
    obs_max = rows[0]["d_auprc"]
    p_val = float(np.mean(perm >= obs_max))
    print(f"    置换零假设（{args.perm} 次打乱标签，同网格搜最优）：null max Δ "
          f"中位 {np.median(perm):+.5f}，95 分位 {np.percentile(perm, 95):+.5f}，"
          f"观测 {obs_max:+.5f} → p={p_val:.3f}")
    perm_out = {"n_perm": args.perm, "null_median": float(np.median(perm)),
                "null_p95": float(np.percentile(perm, 95)),
                "observed_max": obs_max, "p_value": p_val}
    print("    最优格点 bootstrap vs 基线（配对 2000）：")
    top = []
    for r in rows[:5]:
        bs = boot_paired(y, r["w_cb"] * cb + r["w_cs"] * cs + r["w_ra"] * ra, base, args.boot, rng)
        top.append({**r, "boot": bs})
        print(f"      {r['w_cb']:.2f}/{r['w_cs']:.2f}/{r['w_ra']:.2f} Δ{r['d_auprc']:+.5f} "
              f"CI[{bs['ci_lo']:+.5f},{bs['ci_hi']:+.5f}] P(Δ>0)={bs['p_gt_0']:.3f}")

    # B. clean-pair cross-check (train-only cb384 + ctiny384, uncontaminated)
    print("\n[B] 干净成员对照（cb384 + ctiny384，均 train-only，valid 无污染）")
    clean = {}
    if "ctiny384" in mem:
        ct = mem["ctiny384"]
        for wb in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00):
            p = wb * cb + (1 - wb) * ct
            a, pr = metrics(y, p)
            clean[wb] = {"auprc": pr, "auroc": a}
            print(f"    cb {wb:.2f} + ctiny {1 - wb:.2f}  AUROC {a:.5f} AUPRC {pr:.5f}")
        best_wb = max(clean, key=lambda k: clean[k]["auprc"])
        bs = boot_paired(y, best_wb * cb + (1 - best_wb) * ct, 0.5 * cb + 0.5 * ct, args.boot, rng)
        print(f"    最优 cb {best_wb:.2f}/ctiny {1 - best_wb:.2f} vs 0.50/0.50 Δ "
              f"{clean[best_wb]['auprc'] - clean[0.50]['auprc']:+.5f} "
              f"CI[{bs['ci_lo']:+.5f},{bs['ci_hi']:+.5f}] P(Δ>0)={bs['p_gt_0']:.3f}")
        clean["best_vs_half_boot"] = {**bs, "best_w_cb": best_wb}
    else:
        print("    ctiny384 概率缺失，跳过")

    # C. fusion口径：概率/logit/秩/几何平均，污染三元与干净双元同表对照
    print("\n[C] 融合口径（同权重，prob / logit / rank / geo）")
    fusion = []
    labs = [("3 成员污染（0.5/0.25/0.25）", [cb, cs, ra], list(BASE_W)),
            ("3 成员污染（网格最优）", [cb, cs, ra],
             [rows[0]["w_cb"], rows[0]["w_cs"], rows[0]["w_ra"]])]
    if "ctiny384" in mem:
        labs.append(("干净双元（cb0.7/ctiny0.3）", [cb, mem["ctiny384"]], [0.7, 0.3]))
    for tag, ps, ws in labs:
        ref = sum(w * p for w, p in zip(ws, ps))
        lod = sum(w * logit(p) for w, p in zip(ws, ps))
        alt = {"logit": 1.0 / (1.0 + np.exp(-lod)),
               "rank": sum(w * np.argsort(np.argsort(p)) for w, p in zip(ws, ps)),
               "geo": np.exp(sum(w * np.log(np.clip(p, 1e-6, None)) for w, p in zip(ws, ps)))}
        row = {"lab": tag, "prob_auprc": metrics(y, ref)[1], "prob_auroc": metrics(y, ref)[0]}
        for name, p in alt.items():
            a, pr = metrics(y, p)
            bs = boot_paired(y, p, ref, args.boot, rng)
            flips = int(np.sum((ref >= 0.5) != (p >= 0.5)))
            row[name] = {"auprc": pr, "auroc": a, "d_auprc": pr - row["prob_auprc"],
                         "d_auroc": a - row["prob_auroc"], "flips_0p5": flips,
                         "boot_ci": [bs["ci_lo"], bs["ci_hi"]], "p_gt_0": bs["p_gt_0"]}
            print(f"    {tag:26s} {name:5s} AUPRC {pr:.5f} (Δ{pr - row['prob_auprc']:+.5f}) "
                  f"AUROC {a:.5f} 0.5翻转 {flips:3d} CI[{bs['ci_lo']:+.5f},{bs['ci_hi']:+.5f}] "
                  f"P(Δ>0)={bs['p_gt_0']:.3f}")
        fusion.append(row)
    lod_base = sum(w * logit(p) for w, p in zip(BASE_W, [cb, cs, ra]))
    p_log_base = 1.0 / (1.0 + np.exp(-lod_base))
    perm_log = []
    for _ in range(args.perm):
        yp = rng.permutation(y)
        perm_log.append(average_precision_score(yp, p_log_base)
                        - average_precision_score(yp, base))
    perm_log = np.array(perm_log)
    obs_log = fusion[0]["logit"]["d_auprc"]
    print(f"    logit vs prob 置换零假设：null Δ 中位 {np.median(perm_log):+.5f}，"
          f"95 分位 {np.percentile(perm_log, 95):+.5f}，观测 {obs_log:+.5f} → "
          f"p={float(np.mean(perm_log >= obs_log)):.3f}")
    perm_fusion = {"null_median": float(np.median(perm_log)),
                   "null_p95": float(np.percentile(perm_log, 95)),
                   "observed": obs_log, "p_value": float(np.mean(perm_log >= obs_log))}


    # D. TTA upper bound on the same blend (all-member 4-flip = family ceiling)
    print("\n[D] TTA 上界（同权重，全成员 4 翻转 = 整个 TTA 族的上界）")
    zt = np.load(os.path.join(args.probs_dir, "cb384_tta2_pil.npy.npz"))
    cb4p = np.load(os.path.join(args.probs_dir, "cb384_pil.npy.npz"))["probs_tta4"].astype(np.float64)
    cs4 = np.load(os.path.join(args.probs_dir, "cs_hi512_all_pil.npy.npz"))["probs_tta4"].astype(np.float64)
    ra4 = np.load(os.path.join(args.probs_dir, "cs_hi512_all_ra_pil.npy.npz"))["probs_tta4"].astype(np.float64)
    tta = []
    for tag, pcb in [("TTA1（现网）", cb),
                     ("TTA2-cb（cb 原图+hflip）", zt["probs_tta2"].astype(np.float64))]:
        p = BASE_W[0] * pcb + BASE_W[1] * cs + BASE_W[2] * ra
        a, pr = metrics(y, p)
        tta.append({"cfg": tag, "auprc": pr, "auroc": a, "d_auprc": pr - base_auprc})
        print(f"    {tag:24s} AUROC {a:.5f} AUPRC {pr:.5f} Δ {pr - base_auprc:+.5f}")
    p_full4 = BASE_W[0] * cb4p + BASE_W[1] * cs4 + BASE_W[2] * ra4
    p_small4 = BASE_W[0] * cb + BASE_W[1] * cs4 + BASE_W[2] * ra4
    p_cb4small1 = BASE_W[0] * cb4p + BASE_W[1] * cs + BASE_W[2] * ra
    for tag, p in [("TTA2-smalls 上界（cb1+smalls4）", p_small4),
                   ("TTA4-all（上界）", p_full4),
                   ("TTA4-cb（cb4+smalls1）", p_cb4small1)]:
        a, pr = metrics(y, p)
        tta.append({"cfg": tag, "auprc": pr, "auroc": a, "d_auprc": pr - base_auprc})
        print(f"    {tag:34s} AUROC {a:.5f} AUPRC {pr:.5f} Δ {pr - base_auprc:+.5f}")
    print("    成员自身 TTA1→TTA4 增益（单模，train+all 成员仅作方向参考）：")
    singles = {}
    for tag, p1, p4 in [("cb384", cb, cb4p), ("cs_hi512_all", cs, cs4),
                        ("cs_hi512_all_ra", ra, ra4)]:
        s1, s4 = metrics(y, p1)[1], metrics(y, p4)[1]
        singles[tag] = {"tta1": s1, "tta4": s4, "d": s4 - s1}
        print(f"      {tag:16s} TTA1 {s1:.5f} → TTA4 {s4:.5f}  Δ {s4 - s1:+.5f}")

    result = {"n_valid": int(len(y)), "n_pos": int(y.sum()),
              "baseline": {"w": list(BASE_W), "auprc": base_auprc, "auroc": base_auroc,
                           "boot_ci": base_ci},
              "gate_delta": args.gate_delta, "grid": rows, "grid_top_boot": top,
              "n_grid_pass_gate": n_pass, "grid_perm_null": perm_out,
              "fusion_perm_null": perm_fusion, "clean_pair": clean, "fusion": fusion, "tta": tta,
              "member_tta_singles": singles,
              "seed": args.seed, "boot": args.boot}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
