
import numpy as np
from layer2 import compute_effective_channel, update_G_step, update_Y_step, update_theta_step_apg, update_Z_step
from time import time

def admm_apg_main(H1, H2, Hm, P, sigma_n2, Ms, Mr, Mt, Mi, K_max=100, tau_stopping=1e-3, rho=1.0): 
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
    """

    # --- Lists to track history ---
    se_history = []
    
    # --- 1. INITIALIZATION ---
    C = P / (sigma_n2 * Ms)
    
    # Variables
    #theta = np.exp(1j * np.random.uniform(0, 2*np.pi, Mi)) # theta_k : Initialize with random phases
    theta = np.exp(1j * np.random.uniform(0, 0, Mi)) # on met les phases à 0, ie exp(0) = 1
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

    norm_H1 = np.linalg.norm(H1)
    norm_Hm = np.linalg.norm(Hm)

    tau_apg = 2.0 * (C**2) * norm_H1 * norm_Hm * Ms #lipschitz. Pbm : devient trop grand quand la puissance augmente = > c'est ce qui cause la lenteur de convergence

    # H^k is based on theta from the end of the previous loop
    H_k = compute_effective_channel(H1, H2, Hm, theta)

    start = time()
    # --- 2. MAIN ADMM LOOP ---
    for k in range(1,  K_max +1):
        Y_old = Y.copy() # pour le critère de convergence basé sur Y (pas utile pour l'instant)

        # Step 1 : Update G
        G, U, S, A_diag = update_G_step(H_k, Ms, C)
        #print(k,":",G)
        if k == 1:
                #print(G)
            #if Ms==1:
                """# With unit variance channels, Ms=1, Mr=1
                H_eff = compute_effective_channel(H1, H2, Hm, theta)  # shape (1, Mt)
                sigma1 = np.linalg.norm(H_eff)  # only one singular value when Mr=1
                C = P / (sigma_n2 * 1)
                SE = np.log2(1 + C * sigma1**2)
                print(f"sigma1 = {sigma1:.4f}")
                print(f"C = {C:.4f}")
                print(f"C * sigma1^2 = {C * sigma1**2:.4f}")
                print(f"SE = {SE:.4f} bps/Hz")"""
            #else:
                #print("trace de G",np.trace(G @ G.conj().T))
                #G_initial = np.sqrt(P / Mt) * np.ones((Mt, 1))
                #print(G_initial)
                """print(f"Norme de H1 : {np.linalg.norm(H1)}")
                print(f"Norme de Hm : {np.linalg.norm(Hm)}")
                print(f"Norme de H2 : {np.linalg.norm(H2)}")
                print(f"Norme de theta : {np.linalg.norm(theta)}")
                print(f"Norme de H : {np.linalg.norm(H_k)}")
                T_k = H_k @ G
                inner_mat = np.eye(Mr) + C * (T_k @ T_k.conj().T)
                #print(C * (T_k @ T_k.conj().T))
                _, logdet = np.linalg.slogdet(inner_mat)
                print("SE initial",logdet / np.log(2))"""


        
        # Step 2 : Update Y^{k+1} using H^k (via reused U, S, A)
        # Xi_k = H^k * G^k+1 * G^k+1^H * H^k^H, pour pouvoir le réutiliser
        Y, Xi_k = update_Y_step(U, S, A_diag, Z, C, rho, Mr, Ms)
        
        # Step 3 : Update theta^{k+1} using G^{k+1} and Y^{k+1}
        current_tk = t_seq[k]
        theta_next = update_theta_step_apg(theta, theta_prev, current_tk, H_k, Xi_k, 
                                           G, Y, Z, H1, Hm, C, Mr, tau_apg) 
        
        # RECOMPUTE Channel for Z update: H^{k+1} uses theta^{k+1}
        # Hk+1
        H_k = compute_effective_channel(H1, H2, Hm, theta_next)
        
        # Step 4 : Update Z
        Z = update_Z_step(Z, Y, H_k, G, C, Mr)
        

        # --- 3. TRACK PROGRESS ---
        # Spectral Efficiency: log2 det(I + C * H_next * G * G' * H_next')
        # We use H_eff_next and G because they are the most recent updates
        T_k = H_k @ G
        inner_mat = np.eye(Mr) + C * (T_k @ T_k.conj().T)
        _, logdet = np.linalg.slogdet(inner_mat)
        se_history.append(logdet / np.log(2))
            
        # Update History
        theta_prev = theta.copy()
        theta = theta_next
        
        # --- 4. CONVERGENCE CHECK ---
        # Based on relative change of Y or Spectral Efficiency
        """error = np.linalg.norm(Y - Y_old, 'fro') / np.linalg.norm(Y_old, 'fro')
        if error < tau_stopping:
            print(f"Converged at iteration {k}")
            break """    

    end = time()
    return G, theta, se_history, end-start