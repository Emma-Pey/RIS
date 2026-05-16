
import numpy as np
from layer3 import compute_complex_gradient, solve_quadratic_eigenvalues, water_filling_allocator

def compute_effective_channel(H1, H2, Hm, theta):
    """
    Calcule la matrice totale du système, H

    Inputs:
        H1: RIS-user (Mr x Mi)
        H2: BS-user (Mr x Mt)
        Hm: BS-RIS (Mi x Mt)
        theta: Vecteur des phases de la RIS (Mi x 1)
        
    Returns:
        H: Matrice équivalente du système (Mr x Mt)
    """
    H = (H1 * theta) @ Hm + H2
    return H

def update_G_step(H, Ms, C): 
    """
    Update de la matrice de précodage G, basée sur la SVD (Singular Value Decomposition) de H et l'algorithme du water-filling
    Equation (7)
    """
    # SVD
    U, S, Vh = np.linalg.svd(H, full_matrices=False) # S ici = Λ du papier
    V = Vh.conj().T
    
    # Water-filling
    if Ms == 1: # pas besoin de faire de calcul dans ce cas-là
        A_diag = np.array([1.0])
    else:
        A_diag = water_filling_allocator(S[:Ms], Ms, C)
    
    # Calcul du nouveau G
    G = V[:, :Ms] @ np.diag(np.sqrt(A_diag))

    # Retourne tout ce dont on a besoin pour les updates de Y et theta
    return G, U, S, A_diag

def update_Y_step(U, S, A_diag, Z, C, rho, Mr, Ms):
    """
    Update de la matrice auxiliaire Y
    Equation (10), Appendix A
    """
    # Calcul de Q (Appendix A)
    gain_diag = (S[:Ms]**2) * A_diag 
    Xi = (U[:, :Ms] * gain_diag) @ U[:, :Ms].conj().T
    Q = np.eye(Mr) + C * Xi - Z
    
    # Calcul de U1 : Eigenvalue decomposition
    eig_vals, U1 = np.linalg.eigh(Q) 
    
    # Calcul de Y~ 
    Y_vals = solve_quadratic_eigenvalues(eig_vals, rho)
    
    # Reconstruction finale de Y
    Y = U1 @ np.diag(Y_vals) @ U1.conj().T
    
    return Y, Xi

def update_theta_step_apg(theta_k, theta_k_prev, tk, H_eff, Xi_k, G, Y, Z, H1, Hm, C, Mr, k, tau_apg):
    """
    Update de theta, méthode APG. 
    Equations (12) et (15)
    """    
    # Calcul du gradient
    grad = compute_complex_gradient(H_eff, Xi_k, G, Y, Z, H1, Hm, C, Mr)
    
    # Calcul du pas tau_k
    norm_grad = np.linalg.norm(grad)
    tau_k = norm_grad*tau_apg

    # Calcul de omega
    omega = theta_k + tk * (theta_k - theta_k_prev)

    # Descente de gradient
    theta_tilde = omega - 1 / tau_k * grad 

    # Projection pour avoir un module de 1
    theta_next = np.where(theta_tilde != 0, theta_tilde / np.abs(theta_tilde), 1.0 + 0j)
    
    return theta_next

def update_Z_step(Z_old, Y_new, H, G_new, C, Mr):
    """
    Update de la scaled dual matrix
    Equation (5d)
    """
    # Calcul de H*G*G^H*H^H avec le nouveau theta
    T_next = H @ G_new
    Xi_next = T_next @ T_next.conj().T
    
    # Update de Z
    Z_new = Z_old + (Y_new - np.eye(Mr) - C * Xi_next)
    
    return Z_new

def get_SE(H_k, G, C, Mr): 
    """
    Calcul de la SE 
    Equation (2)
    """           
    T_k = H_k @ G
    inner_mat = np.eye(Mr) + C * (T_k @ T_k.conj().T)
    _, logdet = np.linalg.slogdet(inner_mat)
    return logdet / np.log(2)
