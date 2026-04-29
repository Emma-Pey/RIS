
import numpy as np
from layer2 import compute_effective_channel, update_G_step, update_Y_step, update_theta_step_apg, update_Z_step

def admm_apg_main(H1, H2, Hm, P, sigma_n2, Ms, Mr, Mt, Mi, K_max=100, tau_stopping=1e-3, rho=1.0, tau_apg=0.1000): # tau_apg inutile si on le calcule. Il semblerait que tau ne dépende que de P.
    """
    Main ADMM-APG Algorithm for IRS-assisted MIMO Spectral Efficiency Maximization.
    
    Inputs:
        H1, H2, Hm: Channel matrices (constant throughout the algorithm)
        P, sigma_n2: Total power and noise variance
        Ms: Number of data streams
        Mr, Mt, Mi: Dimensions of the channel matrices
        K_max: Maximum iterations
        tau_stopping: Convergence threshold for the outer loop
        rho: ADMM penalty parameter
        tau_apg: Step-size for the APG gradient update (the constant tau_k)
    """

    # --- Lists to track history ---
    se_history = []
    norm_history = []
    
    # --- 1. INITIALIZATION ---
    C = P / (sigma_n2 * Ms)
    
    # Variables
    theta = np.exp(1j * np.random.uniform(0, 2*np.pi, Mi)) # theta_k : Initialize with random phases
    theta_prev = theta.copy()                             # theta_k-1
    
    Y = np.eye(Mr, dtype=complex) # Y^k : Initialize as identity
    Z = np.zeros((Mr, Mr), dtype=complex) # Z^k : Initialize as zero
    
    # Pre-calculate Momentum Sequence (d and t) : step size for APG
    d = np.zeros(K_max + 1) 
    d[0] = 0 
    for i in range(1, K_max + 1):
        d[i] = (1 + np.sqrt(1 + 4 * d[i-1]**2)) / 2
    t_seq = np.zeros(K_max + 1)
    t_seq[1:] = (d[1:] - 1) / d[1:]

    norm_H1 = np.linalg.norm(H1, 'fro')**2
    norm_Hm = np.linalg.norm(Hm, 'fro')**2
    #tau_apg = 2.0 * (C**2) * norm_H1 * norm_Hm * Ms #lipschitz. Pbm : devient trop grand quand la puissance augmente = > c'est ce qui cause la lenteur de convergence

    """# Before the loop
    H_eff_init = compute_effective_channel(H1, H2, Hm, theta)
    G_init, _, _, _ = update_G_step(H_eff_init, Ms, P)
    T_init = H_eff_init @ G_init
    inner_mat = np.eye(Mr) + C * (T_init @ T_init.conj().T)
    _, logdet = np.linalg.slogdet(inner_mat)
    se_init = logdet / np.log(2)
    print(f"Initial SE: {se_init:.10f} bps/Hz")"""

    se_max, theta_init = get_upper_bound(H1, H2, Hm, Ms, P, 1.0)
    print(f"SE Max théorique (Phase align) : {se_max:.2f}")

    # --- 2. MAIN ADMM LOOP ---
    for k in range(1, K_max + 1):
        Y_old = Y.copy()
        
        # H^k is based on theta from the end of the previous loop
        H_eff_k = compute_effective_channel(H1, H2, Hm, theta)
        
        # Step 1 : Update G^{k+1} using H^k
        G, U, S, A_diag = update_G_step(H_eff_k, Ms, P)
        
        # Step 2 : Update Y^{k+1} using H^k (via reused U, S, A)
        Y, Xi_k = update_Y_step(U, S, A_diag, Z, C, rho, Mr, Ms)
        
        # Step 3 : Update theta^{k+1} using G^{k+1} and Y^{k+1}
        # Note: The gradient in APG will also use H_eff_k
        current_tk = t_seq[k]

        theta_next, grad, E, T, Left, Right = update_theta_step_apg(theta, theta_prev, current_tk, H_eff_k, Xi_k, 
                                           G, Y, Z, H1, Hm, C, tau_apg, Mr) 
        
        # RECOMPUTE Channel for Z update: H^{k+1} uses theta^{k+1}
        H_eff_next = compute_effective_channel(H1, H2, Hm, theta_next)
        
        # Step 4 : Update Z^{k+1} using H^{k+1} and G^{k+1}
        # We need a new Xi here: Xi_next = H^{k+1} G^{k+1} (G^{k+1})^H (H^{k+1})^H
        Z = update_Z_step(Z, Y, H_eff_next, G, C, Mr)
        
        # --- ADD TRACKING LINES HERE ---
        # 1. Spectral Efficiency: log2 det(I + C * H_next * G * G' * H_next')
        # We use H_eff_next and G because they are the most recent updates
        T_k = H_eff_next @ G
        inner_mat = np.eye(Mr) + C * (T_k @ T_k.conj().T)
        _, logdet = np.linalg.slogdet(inner_mat)
        se_history.append(logdet / np.log(2))

        # 2. Frobenius Norm (Constraint Residual): || Y - (I + C*H*G*G'H') ||_F
        # Note: update_Z_step uses the same residual logic
        Xi_k = (T_k @ T_k.conj().T)
        residual = Y - (np.eye(Mr) + C*Xi_k)
        norm_history.append(np.linalg.norm(residual, 'fro'))
        # -------------------------------

        if k<6:
            #print(f"k={k}")
            """print(f"  norm(Z)     = {np.linalg.norm(Z, 'fro'):.4e}")
            print(f"  norm(Y)     = {np.linalg.norm(Y, 'fro'):.4e}")
            print(f"  norm(G)     = {np.linalg.norm(G, 'fro'):.4e}")
            print(f"  norm(theta) = {np.linalg.norm(theta_next):.4e}")
            print(f"  norm(grad)  = {np.linalg.norm(grad):.4e}")
            print(f"  norm(E)     = {np.linalg.norm(E, 'fro'):.4e}")
            """
            """print(f"  norm(E@T)        = {np.linalg.norm(E @ T):.4e}")
            print(f"  norm(H1.H@(E@T)) = {np.linalg.norm(H1.conj().T @ (E @ T)):.4e}")
            print(f"  norm(Right)      = {np.linalg.norm(Right):.4e}")
            print(f"  norm(Left@Right) = {np.linalg.norm(Left @ Right):.4e}")
            print(f"  norm(diag only)  = {np.linalg.norm(np.einsum('ij,ji->i', Left, Right)):.4e}")"""

        if k == 1:
            T_k = H2 @ G
            inner_mat = np.eye(Mr) + C * (T_k @ T_k.conj().T)
            _, logdet = np.linalg.slogdet(inner_mat)
            print("se direct",logdet / np.log(2))

            T_k = H_eff_k @ G
            inner_mat = np.eye(Mr) + C * (T_k @ T_k.conj().T)
            _, logdet = np.linalg.slogdet(inner_mat)
            print("se random irs",logdet / np.log(2))
            
        # Update History
        theta_prev = theta.copy()
        theta = theta_next
        
        # --- 3. CONVERGENCE CHECK ---
        # Based on relative change of Y or Spectral Efficiency
        #error = np.linalg.norm(Y - Y_old, 'fro') / np.linalg.norm(Y_old, 'fro')
        #if error < tau_stopping:
            #print(f"Converged at iteration {k}")
            #break     

    assert np.allclose(np.abs(theta), 1.0), "Theta should have unit modulus"
    return G, theta, se_history, norm_history

# --- CODE A INSERER AVANT LA BOUCLE ADMM ---
def get_upper_bound(H1, H2, Hm, Ms, P, sigma2):
    # On cherche a aligner les phases de l'IRS sur le flux dominant de H2
    # C'est une estimation rapide (Heuristique de phase)
    Mr, Mt = H2.shape
    Mi = H1.shape[1]
    
    # On calcule le canal combine pour chaque element de l'IRS
    # Pour l'element n: H1[:,n] @ Hm[n,:]
    theta_ideal = np.zeros(Mi, dtype=complex)
    for n in range(Mi):
        # On aligne la phase de l'IRS pour qu'elle "somme" constructivement avec H2
        # On regarde la trace du produit pour maximiser la puissance globale
        link_n = H1[:, [n]] @ Hm[[n], :]
        # On cherche l'angle qui maximise la correlation avec H2
        angle = np.angle(np.trace(H2.conj().T @ link_n))
        theta_ideal[n] = np.exp(-1j * angle)
        
    
    H_total = H2 + H1 @ np.diag(theta_ideal) @ Hm
    # Calcul de la SE avec Water-Filling sur ce canal "ideal"
    U, S, Vh = np.linalg.svd(H_total)
    se_ub = np.sum(np.log2(1 + (P/Ms) * S**2 / sigma2))
    return se_ub, theta_ideal