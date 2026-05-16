
import numpy as np

def compute_complex_gradient(H, Xi_k, G, Y, Z, H1, Hm, C, Mr):
    """
    Calcul du gradient
    Inputs: 
        Xi_k:  H^k * G^k+1 * (G^k+1).H * (H^k).H
    Equation (14)
    """
    # Construire E
    E = Y - np.eye(Mr) - C*Xi_k + Z
    
    # Calcul de l'intérieur de vec_d
    T = H @ G
    Left = H1.conj().T @ (E @ T)
    Right = G.conj().T @ Hm.conj().T

    # Extraire la diagonale et calculer grad
    grad = -2.0 * C * np.einsum('ij,ji->i', Left, Right) # einsum permet de calculer uniquement les éléments diagonaux (pas besoin des autres)

    return grad

def water_filling_allocator(S, Ms, C):
    """
    Algorithme du water filling pour l'allocation de puissance de la matrice G.
    Résoud : max sum log(1 + C * lambda_j^2 * p_j)
            s.t. sum(p_j) = Ms, p_j >= 0
    Water level: p_j = max(0, mu - 1/(C * lambda_j^2))
    """
    gains = C * S**2 
    inv_gains = 1.0 / gains

    low, high = 0.0, (Ms + np.sum(inv_gains)) / len(S)
    for _ in range(100):
        mu = (low + high) / 2.0
        p = np.maximum(0.0, mu - inv_gains)
        if np.sum(p) > Ms:
            high = mu
        else:
            low = mu
    
    return np.maximum(0.0, (low + high) / 2.0 - inv_gains)

def solve_quadratic_eigenvalues(eig_vals, rho):
    """
    Résoud l'équation quadratique de l'Appendix A pour le calcul de Y
    """
    v = np.asarray(eig_vals)
    return (v + np.sqrt(v**2 + 4.0 / rho)) / 2.0
