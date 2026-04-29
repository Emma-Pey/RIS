
import numpy as np
from layer3 import compute_complex_gradient, solve_quadratic_eigenvalues, water_filling_allocator


def compute_effective_channel(H1, H2, Hm, theta):
    """
    Computes the effective channel matrix H_eff = H1 * diag(theta) * Hm + H2.
    
    Inputs:
        H1: Matrix from IRS to Receiver (Mr x Mi)
        H2: Direct link from Transmitter to Receiver (Mr x Mt)
        Hm: Matrix from Transmitter to IRS (Mi x Mt)
        theta: Vector of IRS phase shifts (Mi x 1)
        
    Returns:
        H_eff: Total effective channel (Mr x Mt)
    """
    # Scaling the columns of H1 by theta is equivalent to H1 @ diag(theta)
    # This is O(Mr * Mi) instead of O(Mr * Mi^2)
    H1_scaled = H1 * theta
    
    # Compute the cascaded link: (H1 * diag(theta)) @ Hm
    H_cascaded = H1_scaled @ Hm
    
    # Add the direct link
    return H_cascaded + H2

def update_G_step(H_eff, Ms, P): # P ??
    """
    Updates the precoding matrix G based on the SVD of the effective channel.
    Paper Ref: Eq. (7)
    """
    # 1. Compute Truncated SVD (returns V^H as Vh)
    # We use full_matrices=False to get the economy SVD (O(Mr^2 * Mt))
    U, S, Vh = np.linalg.svd(H_eff, full_matrices=False)
    
    # 2. Get V (Conjugate Transpose of Vh)
    V = Vh.conj().T
    
    # 3. Power Allocation (Layer 3 call)
    # The paper says sum(pj) = Ms. 
    # S contains singular values in descending order.
    A_diag = water_filling_allocator(S[:Ms], Ms)
    
    # 4. Construct G = V[:, :Ms] * diag(sqrt(A))
    # Slicing to Ms ensures we only use the primary streams
    G = V[:, :Ms] @ np.diag(np.sqrt(A_diag))
    
    # Return everything needed for Y and Z updates
    return G, U, S, A_diag

def update_Y_step(U, S, A_diag, Z, C, rho, Mr, Ms):
    """
    Updates the auxiliary matrix Y using the closed-form quadratic solution.
    Paper Ref: Eq. (20)
    """
    # 1. Efficiently compute Xi = H*G*G'*H' = U * Lambda * A * Lambda * U'
    # S[:Ms] are singular values (Lambda), A_diag is power allocation
    # Scaling columns of U by (s_i^2 * a_i)
    gain_diag = (S[:Ms]**2) * A_diag 
    Xi = (U[:, :Ms] * gain_diag) @ U[:, :Ms].conj().T
    
    # 2. Form Q = I + C*Xi - Z
    Q = np.eye(Mr) + C * Xi - Z
    
    # 3. Solve via EVD (using eigh for Hermitian matrices)
    eig_vals, U_q = np.linalg.eigh(Q) # U_q = U1 
    
    # 4. Apply the quadratic formula to eigenvalues (Layer 3 call)
    # Y_vals_i = (v_i + sqrt(v_i^2 + 4/rho)) / 2
    Y_vals = solve_quadratic_eigenvalues(eig_vals, rho)
    
    # 5. Reconstruct Y
    Y = U_q @ np.diag(Y_vals) @ U_q.conj().T
    
    return Y, Xi

def update_theta_step_apg(theta_k, theta_k_prev, tk, H_eff, Xi_k, G, Y, Z, H1, Hm, C, tau_apg, Mr):
    """
    Updates IRS phase shifts using Accelerated Projected Gradient.
    Paper Ref: Eq. (12) and (15)
    """
    # 1. Momentum Step (Eq. 12)
    omega = theta_k + tk * (theta_k - theta_k_prev)
    
    # 2. Compute Complex Gradient (Layer 3 call)
    # Gradient is evaluated at the momentum point omega
    grad, E, T, Left, Right = compute_complex_gradient(H_eff, Xi_k, G, Y, Z, H1, Hm, C, Mr)
    # 1. Calculer la norme du gradient
    norm_grad = np.linalg.norm(grad)

    step_size = 0.1 / (norm_grad + 1e-12) # ça ajoute de la complexité mais négligeable ? 
                                         # step_size adaptatif, permet de converger relativement rapidment, même quand la puissance est grande

    
    #step_size = 1000000000 #ce step size est pas mal pour la convergence rapide à P = 20 (plus c'est grand plus c'est rapide? )
    # 3. Gradient Descent Step
    # Normalize the gradient so its largest update is, say, 0.1 radians
    theta_tilde = omega - step_size * grad # à la place de step_size : (1.0 / tau_apg) # theta_tilde, c'est theta avant normalisation
    
    # 4. Projection to Unit Modulus (Eq. 15)
    # theta_n = xi_n / |xi_n| (with 0-check)
    theta_next = np.where(np.abs(theta_tilde) > 1e-12, theta_tilde / np.abs(theta_tilde), 1.0 + 0j)
    return theta_next, grad, E, T, Left, Right

def update_Z_step(Z_old, Y_new, H_next, G_new, C, Mr):
    """
    Updates the dual variable Z based on the updated channel state.
    Paper Ref: Algorithm 1 Step 7
    """
    # 1. Compute Xi_next using the new theta (H_next) and current G
    T_next = H_next @ G_new
    Xi_next = T_next @ T_next.conj().T
    
    # 2. Update Z = Z + rho * (Y - (I + C*Xi_next))
    # Note: Most papers use rho as a step size here. 
    # Check if your paper uses a coefficient or just Z + (Y - Phi)
    Z_new = Z_old + (Y_new - np.eye(Mr) - C * Xi_next)
    
    return Z_new

if __name__ == "__main__":
    # 1. SETUP PARAMETERS
    Mr, Mt, Mi, Ms = 4, 4, 32, 2
    P_dbm = 30
    sigma2_dbm = -80
    P = 10**(P_dbm / 10) / 1000  # Convert to Watts
    sigma2 = 10**(sigma2_dbm / 10) / 1000
    C = P / (sigma2 * Ms)

    # 2. GENERATE SYNTHETIC DATA
    def get_mock_data():
        # Complex Gaussian Channels
        H1 = (np.random.randn(Mr, Mi) + 1j*np.random.randn(Mr, Mi)) / np.sqrt(2)
        H2 = (np.random.randn(Mr, Mt) + 1j*np.random.randn(Mr, Mt)) / np.sqrt(2)
        Hm = (np.random.randn(Mi, Mt) + 1j*np.random.randn(Mi, Mt)) / np.sqrt(2)
        theta = np.exp(1j * np.random.uniform(0, 2*np.pi, Mi))
        return H1, H2, Hm, theta

    H1, H2, Hm, theta = get_mock_data()

    # 3. TEST THE LAYERS
    # Test Effective Channel
    H_eff = compute_effective_channel(H1, H2, Hm, theta)
    print(f"H_eff shape: {H_eff.shape}") # Should be (4, 4)

    # Test G update logic (truncated SVD)
    U, S, Vh = np.linalg.svd(H_eff, full_matrices=False)
    A_diag = water_filling_allocator(S[:Ms], Ms)
    G = Vh.conj().T[:, :Ms] @ np.diag(np.sqrt(A_diag))
    print(f"G shape: {G.shape}") # Should be (4, 2)

    # Test Objective Function (Spectral Efficiency)
    def compute_se(H, G, C):
        # R = log2(det(I + C * HGGH*H*))
        Mr = H.shape[0]
        inner = np.eye(Mr) + C * (H @ G @ G.conj().T @ H.conj().T)
        return np.real(np.log2(np.linalg.det(inner)))

    print(f"Initial Spectral Efficiency: {compute_se(H_eff, G, C):.2f} bps/Hz")