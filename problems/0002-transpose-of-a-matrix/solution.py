import torch

def transpose_matrix(a) -> torch.Tensor:
    """
    Transpose a 2D matrix using PyTorch.
    
    Args:
        a: A 2D matrix (can be list, numpy array, or torch.Tensor)
    
    Returns:
        A transposed torch.Tensor
    """
    a_t = torch.as_tensor(a)
    # Your code here
    shape_a = a_t.size()
    a_transpose = torch.zeros(size=(shape_a[1],shape_a[0]))
    for i in range(shape_a[0]):
        for j in range(shape_a[1]):
            a_transpose[j][i] = a_t[i][j]
    return a_transpose