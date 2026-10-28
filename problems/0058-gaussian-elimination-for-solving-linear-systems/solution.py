import torch

# logique du problème 
# étape 1 -> combinaison des deux tensor
# étape 2 ->  parcourir chacune des lignes de A
#   étape 2.1 -> si la valeur est 0 pour notre ligne: regarder les autres lignes et l'invertir avec une ligne adéquate si aucune ligne n'est adéquate et que la valeur de b demande une demande adéquate, alors aucune solution
# étape 3 -> appliquer à chacune des valeurs la valeur suffisante pour la supprimer sur le niveau de ma valeur de la ligne et l'appliquer à toute la lignes en dessous (A et b)
# étape 4 -> de la dernière ligne à la première ligne remonter en assaissant les valeurs de a par les valeurs trouvé c'est à dire

# étape 2 -> clean_tensor : tensor -> tensor
#         -> get_pivot : tensor -> float64
# étape 3 -> remove_to_line : tensor x tensor x float -> void
# étape 4 -> get_x : tensor x tensor -> tensor



def check_if_existent_coeff(A : torch.Tensor, i : int ) -> bool:
    check = False
    j = 0
    while not check and j < len(A) - 1:
        check = (A[j][i] != 0)
        j += 1
    return check

def clean_tensor(A: torch.Tensor) -> torch.Tensor:
    # vérfication de la matrice qu'on m'anipule
    for i in range(len(A) - 1):
        actual_line = A[i] # la ligne actuel
        j = i + 1
        while actual_line[i] == 0.0 and j < len(A) - 1:
            second_line = A[j].clone()
            A[j].copy_(actual_line)
            actual_line.copy_(second_line)
            copy_b_val = A[-1][i].item()
            A[-1][i].copy_(A[-1][j])
            A[-1][j].copy_(copy_b_val)
            j += 1
        if j >= len(A) and not check_if_existent_coeff(A,i):
            return None
    return A

def remove_to_line(gauss: torch.Tensor, pivot: torch.Tensor, i: int, j: int) -> None:
    line_to_remove = gauss[j]
    other_line = gauss[i]
    val_pivot  = pivot.item()
    for  idx in range(len(gauss) - 1):
        line_to_remove[idx] -= other_line[idx] * val_pivot
    b = gauss[-1]
    b[j] -= b[i] * val_pivot


def get_x(gauss: torch.Tensor) -> torch.Tensor:
    x = torch.zeros_like(gauss[-1])
    b = gauss[-1]
    for k in range(len(gauss) - 2,-1, -1):
        for i in range(k + 1,len(gauss) - 1):
            b[k] = b[k] - gauss[k][i] * x[i]
        x[k] = b[k] / gauss[k][k]
    return x        
def gaussian_elimination(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    :param A: Coefficient matrix (torch.Tensor)
    :param b: Right-hand side vector (torch.Tensor)
    :return: Solution vector x (torch.Tensor)
    """
    b_mod = torch.tensor([b.tolist()], dtype=torch.float64)
    gauss = torch.cat((A,b_mod),dim=0)
    gauss = clean_tensor(gauss)
    for i in range (len(A)):
        val_pivot =gauss[i][i]
        if val_pivot == 0:
            continue
        for j in range(i + 1, len(A)):
            pivot = gauss[j][i] / val_pivot
            remove_to_line(gauss,pivot,i,j)
    return get_x(gauss)