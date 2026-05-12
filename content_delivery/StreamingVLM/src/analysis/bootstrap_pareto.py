"""
Bootstrap 95% CI for the plotQA Pareto curve.

For each result JSON, compute per-sample correctness vector, then resample
with replacement B=10000 times to get a 95% CI on accuracy.

Also computes PAIRED bootstrap CI on (r=X stack vs r=0.30 control) and
(r=X stack vs vanilla) since the samples are matched across configs.
"""

import json
import random
import sys

random.seed(42)

FILES = {
    'vanilla (r=1.00)': 'mlvu_baseline.json',
    'r=0.30 (control)': 'mlvu_stamptemporal_r0.3__c5_merge_multi7152331_tast32g0.1__n539_phase7control.json',
    'r=0.50': 'mlvu_stamptemporal_r0.5__c5_merge_multi7152331_tast32g0.1.json',
    'r=0.70': 'mlvu_stamptemporal_r0.7__c5_merge_multi7152331_tast32g0.1.json',
    'r=0.85': 'mlvu_p8_path1_r085_n539.json',
}

def load_correct_vec(path):
    """Return ordered list of 1/0 correctness per sample, keyed by (video, question)."""
    d = json.load(open(path))
    results = d.get('results', d) if isinstance(d, dict) else d
    out = {}
    for r in results:
        if not isinstance(r, dict) or 'correct' not in r:
            continue
        k = (r.get('video', ''), r.get('question', ''))
        out[k] = 1 if r['correct'] else 0
    return out


def bootstrap_acc(vec, B=10000, ci=0.95):
    """Resample list of 0/1 with replacement, return (point, low, high)."""
    n = len(vec)
    point = sum(vec) / n
    samples = []
    for _ in range(B):
        s = sum(vec[random.randint(0, n - 1)] for _ in range(n))
        samples.append(s / n)
    samples.sort()
    lo_idx = int(B * (1 - ci) / 2)
    hi_idx = int(B * (1 - (1 - ci) / 2))
    return point, samples[lo_idx], samples[hi_idx]


def bootstrap_paired_delta(vec_a, vec_b, B=10000, ci=0.95):
    """Paired bootstrap on per-sample (a_i, b_i) — return point delta + CI."""
    pairs = list(zip(vec_a, vec_b))
    n = len(pairs)
    point = (sum(a for a, _ in pairs) - sum(b for _, b in pairs)) / n
    samples = []
    for _ in range(B):
        idxs = [random.randint(0, n - 1) for _ in range(n)]
        a_acc = sum(pairs[i][0] for i in idxs) / n
        b_acc = sum(pairs[i][1] for i in idxs) / n
        samples.append(a_acc - b_acc)
    samples.sort()
    lo_idx = int(B * (1 - ci) / 2)
    hi_idx = int(B * (1 - (1 - ci) / 2))
    return point, samples[lo_idx], samples[hi_idx]


print("=" * 80)
print("plotQA Pareto bootstrap CI (n=539, B=10000, paired by (video,question))")
print("=" * 80)

vecs = {}
keys_union = None
for label, fname in FILES.items():
    d = load_correct_vec(fname)
    vecs[label] = d
    if keys_union is None:
        keys_union = set(d.keys())
    else:
        keys_union &= set(d.keys())

print(f"\nCommon samples across all 5 result files: {len(keys_union)}")
ordered_keys = sorted(keys_union)

for label in FILES:
    raw = vecs[label]
    aligned = [raw[k] for k in ordered_keys]
    point, lo, hi = bootstrap_acc(aligned)
    print(f"  {label:<22} {point*100:6.2f}%  CI95 [{lo*100:.2f}, {hi*100:.2f}]  ({100*(hi-lo)/2:+.2f} half-width)")

print()
print("=" * 80)
print("Paired deltas vs r=0.30 control (positive = better than control)")
print("=" * 80)
ctrl = [vecs['r=0.30 (control)'][k] for k in ordered_keys]
for label in ['r=0.50', 'r=0.70', 'r=0.85', 'vanilla (r=1.00)']:
    a = [vecs[label][k] for k in ordered_keys]
    d, lo, hi = bootstrap_paired_delta(a, ctrl)
    sig = "**" if (lo > 0 or hi < 0) else "  "
    print(f"  {label:<22} Δ={d*100:+6.2f} pp  CI95 [{lo*100:+.2f}, {hi*100:+.2f}]  {sig}")

print()
print("=" * 80)
print("Paired deltas vs vanilla r=1.00 (negative = pruning regression)")
print("=" * 80)
van = [vecs['vanilla (r=1.00)'][k] for k in ordered_keys]
for label in ['r=0.30 (control)', 'r=0.50', 'r=0.70', 'r=0.85']:
    a = [vecs[label][k] for k in ordered_keys]
    d, lo, hi = bootstrap_paired_delta(a, van)
    sig = "**" if (lo > 0 or hi < 0) else "  "
    print(f"  {label:<22} Δ={d*100:+6.2f} pp  CI95 [{lo*100:+.2f}, {hi*100:+.2f}]  {sig}")
