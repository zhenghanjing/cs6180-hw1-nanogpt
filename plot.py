import re, sys, os, matplotlib.pyplot as plt

def load(path):
    raw = open(path, 'rb').read()
    txt = raw.decode('utf-16') if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else raw.decode('utf-8', 'ignore')
    rows = re.findall(r'step (\d+): train loss ([\d.]+), val loss ([\d.]+)', txt)
    return [(int(s), float(t), float(v)) for s, t, v in rows]

out = sys.argv[1]
logs = sys.argv[2:]
plt.figure(figsize=(8, 5))
for p in logs:
    data = load(p)
    name = os.path.basename(p)[:-4]
    s, tr, va = zip(*data)
    line, = plt.plot(s, va, label=f'{name} val')
    plt.plot(s, tr, '--', color=line.get_color(), alpha=0.6, label=f'{name} train')
    best_i = min(range(len(va)), key=lambda i: va[i])
    print(f'{name:20s} best val loss = {va[best_i]:.4f} (at step {s[best_i]})')
plt.xlabel('iteration'); plt.ylabel('loss'); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(out, dpi=150)
print('saved', out)