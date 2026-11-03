#!/usr/bin/env python3
"""Compare two avbd_backend_test dumps. usage: compare_avbd_dumps.py a.txt b.txt [tol]"""
import sys, math

def load(p):
    d, cur = {}, None
    for line in open(p):
        t = line.split()
        if len(t) == 2 and not t[0][0].isdigit() and t[0][0] not in '-+.':
            cur = t[0]; d[cur] = []
        elif t and t[0] == 'selfcoll_pairs':
            d[t[0]] = [float(x) for x in t[1:]]
        elif cur and t:
            d[cur].append(float(t[0]))
    return d

a, b = load(sys.argv[1]), load(sys.argv[2])
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
bad = 0
for k in a:
    if k not in b: print(f'{k:20s} MISSING in second'); bad += 1; continue
    x, y = a[k], b[k]
    if len(x) != len(y): print(f'{k:20s} LENGTH {len(x)} vs {len(y)}'); bad += 1; continue
    if not x: print(f'{k:20s} empty'); continue
    scale = max(max(abs(v) for v in x), 1e-12)
    err = max(abs(p - q) for p, q in zip(x, y)) / scale
    nan = any(math.isnan(v) for v in y)
    status = 'ok' if err <= tol and not nan else 'DIFF'
    if status != 'ok': bad += 1
    print(f'{k:20s} n={len(x):6d} max|a|={scale:10.4g} rel.err={err:9.3g} {status}')
sys.exit(1 if bad else 0)
