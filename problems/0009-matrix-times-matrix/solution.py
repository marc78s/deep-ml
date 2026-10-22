import torch

def matrixmul(a, b) -> torch.Tensor:
    """
    Multiply two matrices using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2D tensor of shape (m, n) or a scalar tensor -1 if dimensions mismatch.
    """
    len_a = a.size()
    len_b = b.size()
    if len_a[1] == len_b[0]:
        return a @ b
    return -1