
import numpy as np
from layer2 import compute_effective_channel, update_G_step, update_Y_step, update_theta_step_apg, update_Z_step, get_SE
from time import time

def admm_apg_main(H1, H2, Hm, P, sigma_n2, Ms, Mr, Mt, Mi, K_max=5000, tau_apg=0.01, rho=1.0, tau_stopping=1e-3, stop_when_converged=False): 
    """
    Fonction principale de l'algorithme ADMM-APG tiré du papier "Efficient Spectral Efficiency Maximization Design
    for IRS-aided MIMO Systems", Fuying Li et al., oct 2025. 
    
    Inputs:
        H1, H2, Hm: Matrices du canal (RIS-user, BS-user, BS-RIS)
        P, sigma_n2: Puissance d'émission et du bruit
        Ms: Nombre de data streams
        Mr, Mt, Mi: Nombre d'antennes en réception, en émission, nombre d'éléments de la RIS
        K_max: Nombre d'itérations maximum
        tau_apg: Facteur pour la step size d'APG (j'ai défini tau_k comme norme_gradient*tau_apg)
        rho: Penalty parameter de l'ADMM
        tau_stopping: Seuil de convergence pour l'algorithme
        stop_when_converged: False pour faire les K_max itérations, True pour s'arrêter quand la convergence est détectée
    
    Returns:
        G: matrice de précodage optimisée
        theta: matrice des phases optimisée
        se_history: évolution de l'efficacité spectrale en fonction des itérations
        duree: durée d'exécution de la boucle principale de l'algorithme
        stop: itération pour laquelle l'algorithme a convergé
    """
    ## Initialisation --------------------------------------------------------------------------

    # Liste pour garder l'historique de l'efficacité spectrale (SE)
    se_history = []

    # Rapport signal/bruit à l'émission 
    C = P / (sigma_n2 * Ms)
    
    # Variables : "Initialize Y, Φ and Z to feasible solutions", Φ=diag(theta)
    theta = np.exp(1j * np.random.uniform(0, 0, Mi)) # phases à 0 (ou aléatoires si changement des valeurs dans uniform)
    theta_prev = theta.copy()     # theta_k-1
    Y = np.eye(Mr, dtype=complex) 
    Z = np.zeros((Mr, Mr), dtype=complex) 
    
    # On précalcule les valeurs de (13) pour gagner du temps dans la boucle principale
    d = np.zeros(K_max + 1) 
    d[0] = 0 
    for i in range(1, K_max + 1):
        d[i] = (1 + np.sqrt(1 + 4 * d[i-1]**2)) / 2
    t_seq = np.zeros(K_max + 1)
    t_seq[1:] = (d[1:] - 1) / d[1:]

    # H^k matrice totale
    H_k = compute_effective_channel(H1, H2, Hm, theta)

    start = time() # calcul de la durée d'exécution de l'algo
    ## Boucle principale de l'algo 1 --------------------------------------------------------------------------
    for k in range(1,  K_max+1):

        # Etape 1 : Update G (7) + récupération des valeurs utiles pour la suite
        G, U, S, A_diag = update_G_step(H_k, Ms, C)

        # Calculer la SE avant la première boucle pour pouvoir comparer à la SE finale
        if k == 1:  
            se = get_SE(H_k, G, C, Mr)              
            se_history.append(se)
        
        # Etape 2 : Update Y (10). On réutilise les valeurs déjà calculées pour l'update de G 
        #+ on récupère Xi_k = U*Λ*A*Λ*U^H = H*G^{k+1}*(G^{k+1})^H*H^H (voir (9) et (8)) (utile pour la suite)
        Y, Xi_k = update_Y_step(U, S, A_diag, Z, C, rho, Mr, Ms)
        
        # Etape 3 : Update theta (12). On réutilise la valeur de Xi_k déjà calculée
        current_tk = t_seq[k]
        theta_next = update_theta_step_apg(theta, theta_prev, current_tk, H_k, Xi_k, 
                                           G, Y, Z, H1, Hm, C, Mr, k, tau_apg) 
                
        # Etape 4 : Update Z
        # On recalcule H_k (devient Hk+1) pour correspondre au nouveau canal
        H_k = compute_effective_channel(H1, H2, Hm, theta_next)
        Z = update_Z_step(Z, Y, H_k, G, C, Mr)
        
        # Mise à jour des indices
        theta_prev = theta.copy()
        theta = theta_next

        # Historique SE
        se = get_SE(H_k, G, C, Mr)              
        se_history.append(se)
        
        # Test de convergence
        if stop_when_converged:
            error = (abs(se_history[-1] - se_history[-2]))*100/(se_history[-2]) # en %
            if error < tau_stopping : # L'écart entre les deux dernières valeurs est inférieur à tau_stopping % de l'avant dernière valeur
                #print("Converge à l'itération",k)
                stop = k
                break     
        if k == K_max:
             stop = K_max

    end = time()
    return G, theta, se_history, end-start, stop
