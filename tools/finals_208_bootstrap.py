# Issue #208: paired stratified bootstrap on valid (5849) for candidate AUPRC deltas.
# 2000 resamples, seed 20261118. Weights are the deployable rounded values from
# docs/finals/CANDIDATES.md (search-optimised values differ by <0.0002 AUPRC).
#
# Usage:
#   python tools/finals_208_bootstrap.py \
#     --ens8-dir D:/glaucoma_work/ens8_probs \
#     --assets-dir D:/glaucoma_work/finals_208_assets \
#     --csv-root glaucoma_first
import argparse, csv, os

import numpy as np
from sklearn.metrics import average_precision_score

B = 2000
SEED = 20261118

CANDIDATES = {
    "C1_greedy5": (["cb512", "hi512_v2", "b512", "ctiny384", "seed2"],
                   [0.33, 0.36, 0.08, 0.10, 0.13]),
    "C2_k3": (["cb512", "hi512_v2", "b512"], [0.40, 0.40, 0.20]),
    "C3_k4": (["cb512", "hi512_v2", "seed2", "b448"], [0.33, 0.37, 0.17, 0.13]),
    "ref_ens9": (["cb512", "hi512_v2", "cb384", "b512"], [0.591, 0.224, 0.141, 0.044]),
    "ref_trial3_equal": (["cb512", "cb384", "hi512_v2"], [1 / 3, 1 / 3, 1 / 3]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ens8-dir", default=r"D:\glaucoma_work\ens8_probs")
    ap.add_argument("--assets-dir", default=r"D:\glaucoma_work\finals_208_assets")
    ap.add_argument("--csv-root", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "glaucoma_first"))
    args = ap.parse_args()

    with open(os.path.join(args.csv_root, "valid.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    y = np.array([int(r["True_Label"]) for r in rows])

    names = sorted({m for lst, _ in CANDIDATES.values() for m in lst})
    P = {}
    for m in names:
        fname = ("ctiny512_probs_valid_full_tta4.npy" if m == "ctiny384"
                 else f"{m}_val.npy")
        for d in (args.ens8_dir, args.assets_dir):
            p = os.path.join(d, fname)
            if os.path.exists(p):
                P[m] = np.load(p).astype(np.float64)
                break
        if m not in P:
            raise SystemExit(f"missing prob file for {m}")

    def blend(tag):
        lst, w = CANDIDATES[tag]
        return sum(P[m] * x for m, x in zip(lst, w))

    print("point estimates (valid AUPRC):")
    for tag in CANDIDATES:
        print(f"  {tag:18s} {average_precision_score(y, blend(tag)):.5f}")

    rng = np.random.default_rng(SEED)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    print(f"\npaired bootstrap (AUPRC), {B} resamples, seed {SEED}:")
    for base, other in [("C1_greedy5", "ref_ens9"), ("C1_greedy5", "ref_trial3_equal"),
                        ("C3_k4", "ref_ens9"), ("C3_k4", "ref_trial3_equal"),
                        ("C2_k3", "ref_ens9"), ("C2_k3", "ref_trial3_equal"),
                        ("C1_greedy5", "C2_k3"), ("C1_greedy5", "C3_k4"),
                        ("C3_k4", "C2_k3")]:
        pb_full, po_full = blend(base), blend(other)
        d_obs = average_precision_score(y, pb_full) - average_precision_score(y, po_full)
        deltas = np.empty(B)
        for b in range(B):
            sp = rng.choice(pos_idx, len(pos_idx), replace=True)
            sn = rng.choice(neg_idx, len(neg_idx), replace=True)
            idx = np.concatenate([sp, sn])
            deltas[b] = (average_precision_score(y[idx], pb_full[idx])
                         - average_precision_score(y[idx], po_full[idx]))
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        print(f"  {base} - {other:18s} d_obs={d_obs:+.5f} "
              f"CI95=[{lo:+.5f},{hi:+.5f}] P(delta>0)={np.mean(deltas > 0):.3f}")


if __name__ == "__main__":
    main()
