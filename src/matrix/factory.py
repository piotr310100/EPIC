from .invertible import InvertibleMatrix
from .orthogonal import OrthogonalMatrix


def create_matrix(matrix_type: str, dim: int, device="cpu"):
    if matrix_type == "orthogonal":
        return OrthogonalMatrix(dim, device)
    elif matrix_type == "invertible":
        return InvertibleMatrix(dim, device)
    else:
        raise ValueError(f"Unsupported matrix type: {matrix_type}")
