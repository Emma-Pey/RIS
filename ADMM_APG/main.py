import numpy as np
from admm_apg import admm_apg_main

###############
#### Canal ####
###############

def generate_rician_channel(N_rx, N_tx, dist, C0, beta, gamma, m=np.pi, d_as=1):
    """Génère le canal suivant le modèle de Rice
    Equation (17)"""
    path_loss = C0 * (dist**-beta)

    # LoS component
    phi_t = np.random.uniform(0, 2*np.pi)
    phi_r = np.random.uniform(0, 2*np.pi)
    ar = np.exp(1j * m * d_as * np.sin(phi_r) * np.arange(N_rx)).reshape(-1, 1) / np.sqrt(N_rx) 
    at = np.exp(1j * m * d_as  * np.sin(phi_t) * np.arange(N_tx)).reshape(-1, 1) / np.sqrt(N_tx) 
    H_LoS = ar @ at.conj().T

    # NLoS component
    H_NLoS = (np.random.randn(N_rx, N_tx) + 1j*np.random.randn(N_rx, N_tx)) / np.sqrt(2)

    # Combiner
    H = np.sqrt(path_loss) * (np.sqrt(gamma/(1+gamma)) * H_LoS + np.sqrt(1/(1+gamma)) * H_NLoS)
    return H

if __name__=="__main__":
    ######################
    ##### Paramètres ##### 
    ######################

    ### Paramètres des antennes ---
    Mr, Mt, Mi, Ms = 4, 16, 100, 4 # taille des matrices
    P_dBW = 20 # Si sigma_n2 = 1, c'est aussi le SNR en dB 
    P_linear = 10**((P_dBW) / 10)

    # Espacement des antennes
    l = 3e8 / (3.5e9) # longueur d'onde pour f = 3.5 GHz (onde 5G)
    d_as = l/2 
    m = 2*np.pi/l # das*m = pi 

    ### Paramètres du canal ---
    # géométrie
    d_bs_user = 100   
    d_bs_ris = 60
    d_ris_user = 50

    # path loss. Voir équation (17)
    beta_bs_user = 3.6 # Path-loss exponent
    beta_bs_ris = 2.0
    beta_ris_user = 2.0
    C0_db = -30
    C0 = 10**(C0_db / 10) # path loss référence à 1m

    # Modèle de Rice
    gamma_rician_db = 10 
    gamma_rician = 10**(gamma_rician_db / 10)  # Convert dB to linear, pas besoin si 10

    # Bruit
    sigma_n2_dBW = -80 
    sigma_n2 = 10**(sigma_n2_dBW/10)


    #######################
    # Génération du canal # 
    #######################
    np.random.seed(None)
    H1 = generate_rician_channel(Mr, Mi, d_ris_user, C0, beta_ris_user, gamma_rician)
    Hm = generate_rician_channel(Mi, Mt, d_bs_ris, C0, beta_bs_ris, gamma_rician)
    H2 = generate_rician_channel(Mr, Mt, d_bs_user, C0, beta_bs_user, gamma_rician)


    #################
    # Algo ADMM-APG # 
    #################
    print("Rapport signal sur bruit à l'émission :",10*np.log10(P_linear/sigma_n2), "dB")
    G, theta, se_history, duree, stop = admm_apg_main(
        H1, H2, Hm, P_linear, sigma_n2, Ms, Mr, Mt, Mi
    )

    print("SE initiale :", se_history[0])
    print("SE finale :", se_history[-1])
    print("Augmentation de ", (se_history[-1]/se_history[0]-1)*100, "%")

