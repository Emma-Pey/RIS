
import numpy as np

def compute_complex_gradient(H_eff, Xi_k, G, Y, Z, H1, Hm, C, Mr):
    """
    H_eff: H^k
    Xi_k: C * H^k * G^k+1 * (G^k+1).H * (H^k).H
    """
    # 1. Build E exactly as pictured in your screenshot
    # E = Y - I - Xi_k + Z
    #Mr = Y.shape[0]
    E = Y - np.eye(Mr) - C*Xi_k + Z
    
    # 2. Group terms for O(N^3) efficiency
    # grad = -2C * vecd( H1.H @ E @ H_eff @ G @ G.H @ Hm.H )
    
    # Left = H1.H @ E @ H_eff @ G (Mi x Ms)
    # Using H_eff @ G is faster than using H_eff and G separately if 
    # we pre-calculate T = H_eff @ G in the main loop.
    T = H_eff @ G
    Left = H1.conj().T @ (E @ T)
    
    # Right = G.H @ Hm.H (Ms x Mi)
    Right = G.conj().T @ Hm.conj().T
    
    # 3. Extract diagonal
    grad = -2.0 * C * np.einsum('ij,ji->i', Left, Right)

    # grad random pour voir si le grad est juste ou si c'est du bullshit
    #grad_random = (np.random.randn(Mr) + 1j*np.random.randn(Mr))
    
    return grad, E, T, Left, Right

def water_filling_allocator(S, Ms):
    """
    Allocates power pj to Ms streams such that sum(pj) = Ms.
    S: Singular values of the effective channel.
    """
    gains = S**2
    inv_gains = 1.0 / gains
    
    # Binary search for the water level 'mu'
    low, high = 0, (Ms + np.sum(inv_gains)) / len(S)
    for _ in range(50):
        mu = (low + high) / 2
        p = np.maximum(0, mu - inv_gains)
        if np.sum(p) > Ms:
            high = mu
        else:
            low = mu
    return np.maximum(0, (low + high)/2 - inv_gains)

def solve_quadratic_eigenvalues(eig_vals, rho):
    """
    Solves the optimal diagonal of Y (Eq. 20).
    v: eigenvalues of Q.
    """
    # (v + sqrt(v^2 + 4/rho)) / 2
    v = np.asarray(eig_vals)
    return (v + np.sqrt(v**2 + 4.0 / rho)) / 2.0

if __name__ == "__main__":
    # test water filling
    S_test = np.array([10, 1, 0.1])
    Ms_test = 2
    p_alloc = water_filling_allocator(S_test, Ms_test)
    print("Power Allocation (Water-Filling):", p_alloc) # the third should be 0

    # test quadratic solver
    eig_vals = [-1, 0, 1]
    rho = 1
    Y_vals = solve_quadratic_eigenvalues(eig_vals, rho)
    print("Optimal Y Diagonal:", Y_vals) # should be positive