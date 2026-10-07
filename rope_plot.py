"""Q2.2: plot the RoPE attention score A_ij against |i - j| for d = 128,
q_i = k_j = (1/sqrt(128)) * [1, ..., 1].

A_ij = (1/64) * sum_{t=0}^{63} cos(|i - j| * theta_t),  theta_t = 10000^(-2t/d)
"""
import numpy as np
import matplotlib.pyplot as plt

d = 128
theta = 10000.0 ** (-np.arange(0, d, 2) / d)        # 64 frequencies
delta = np.arange(0, 65537)                         # |i - j| = 0 ... 65536
A = np.cos(np.outer(delta, theta)).mean(axis=1)     # (1/64) * sum cos(.)

print(f"A at |i-j| = 0, 1, 10, 100, 1000, 10000: "
      f"{A[0]:.4f}, {A[1]:.4f}, {A[10]:.4f}, {A[100]:.4f}, {A[1000]:.4f}, {A[10000]:.4f}")
print(f"first |i-j| with A < 0.5: {np.argmax(A < 0.5)}; first with A < 0: {np.argmax(A < 0)}")
print(f"min A = {A.min():.4f} at |i-j| = {A.argmin()}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(delta, A, lw=0.4)
axes[0].set_xlabel("|i - j|")
axes[0].set_ylabel(r"$A_{i,j}$")
axes[0].set_title("Linear x-axis")
axes[0].axhline(0, color="gray", lw=0.6, ls="--")

axes[1].plot(delta[1:], A[1:], lw=0.6)
axes[1].set_xscale("log")
axes[1].set_xlabel("|i - j| (log scale)")
axes[1].set_ylabel(r"$A_{i,j}$")
axes[1].set_title("Log x-axis")
axes[1].axhline(0, color="gray", lw=0.6, ls="--")

for ax in axes:
    ax.grid(alpha=0.3)
fig.suptitle(r"RoPE attention score vs. relative distance ($d=128$, base $10000$)")
fig.tight_layout()
fig.savefig("rope_decay.png", dpi=200)
print("saved rope_decay.png")
