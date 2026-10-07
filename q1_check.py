"""Q1: numerically verify the convolution output size and the input-gradient formula
using PyTorch autograd.

dL/dX = pad(G, 2) * rot180(W)    (full convolution with the flipped kernel)
      = col2im(w^T g)            (matrix / im2col form)
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
X = torch.randn(1, 1, 4, 4, requires_grad=True)
W = torch.randn(1, 1, 3, 3)
G = torch.randn(1, 1, 2, 2)          # dL/dY

# Q1.1: output size
Y = F.conv2d(X, W)                   # stride=1, dilation=1, no padding
print("output shape:", tuple(Y.shape[-2:]))

# Q1.2: im2col forward equals conv2d
X_col = F.unfold(X, kernel_size=3)   # (1, 9, 4)
y_gemm = W.view(1, 9) @ X_col[0]     # (1, 4)
print("im2col GEMM matches conv2d:", torch.allclose(y_gemm.view(1, 1, 2, 2), Y, atol=1e-6))

# Q1.3: gradient w.r.t. the input
Y.backward(G)
dX_autograd = X.grad

dX_full = F.conv2d(F.pad(G, (2, 2, 2, 2)), torch.flip(W, dims=[2, 3]))
print("full conv with flipped kernel matches autograd:", torch.allclose(dX_autograd, dX_full, atol=1e-6))

dX_col = W.view(1, 9).t() @ G.view(1, 4)                          # (9, 4)
dX_fold = F.fold(dX_col.unsqueeze(0), output_size=(4, 4), kernel_size=3)
print("col2im(w^T g) matches autograd:", torch.allclose(dX_autograd, dX_fold, atol=1e-6))
