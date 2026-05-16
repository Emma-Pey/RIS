import numpy as np
from admm_apg import admm_apg_main
import matplotlib.pyplot as plt
from main import generate_rician_channel

#######################
######## TEST #########
#######################

def generate_test_channels(Mr, Mt, Mi):
    """Générer des canaux aléatoires simples, sans path loss"""
    # Standard complex Gaussian channels
    # H = (1/sqrt(2)) * (Real + j*Imag)
    H1 = (np.random.randn(Mr, Mi) + 1j * np.random.randn(Mr, Mi)) / np.sqrt(2)
    H2 = (np.random.randn(Mr, Mt) + 1j * np.random.randn(Mr, Mt)) / np.sqrt(2)
    Hm = (np.random.randn(Mi, Mt) + 1j * np.random.randn(Mi, Mt)) / np.sqrt(2)
    
    return H1, H2, Hm

def add_estimation_error(H, delta, Mt, Mr):
    """Ajouter une erreur à la matrice du canal pour tester la résistance aux erreurs d'estimation
    Equation (19)"""
    norm_H = np.linalg.norm(H, 'fro')
    gamma_sq = (delta * (norm_H**2)) / np.sqrt(Mt * Mr) 
    
    noise = np.sqrt(gamma_sq/2) * (np.random.randn(*H.shape) + 1j*np.random.randn(*H.shape))
    H_hat = H + noise

    return H_hat

def plot_se_par_iteration(SNR_dB_list, H1, H2, Hm, Ms, Mr, Mt, Mi):
    """Affiche l'efficacité spectrale en fonction du numéro d'itération, suivant différents SNR
    Figure 8. """
    for SNR_dB in SNR_dB_list:
        SNR_linear = 10**(SNR_dB / 10)        
        # Exécution de l'algorithme ADMM-APG
        _, _, se_history, _, _ = admm_apg_main(
        H1, H2, Hm, SNR_linear, 1, Ms, Mr, Mt, Mi, 
        )
        plt.plot(range(1, len(se_history) + 1), se_history,label=f'SNR = {SNR_dB} dB')

    plt.title("Évolution de l'Efficacité Spectrale par Itération")
    plt.xlabel("Itérations")
    plt.ylabel("Efficacité Spectrale (bps/Hz)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.show()

def plot_se_moy_par_iteration(num_realisations, SNR_dB_list, Ms, Mr, Mt, Mi, d_bs_user=30, d_bs_ris=16, d_ris_user=16, beta_bs_user = 4, beta_bs_ris = 2.0, beta_ris_user = 2.0, gamma_rician=10):
    """Affiche l'efficacité spectrale moyenne sur num_realisations générations de canal
    Figure 8."""
    plt.figure(figsize=(10, 6))

    for SNR_dB in SNR_dB_list:

        SNR_linear = 10**(SNR_dB / 10)
        # On initialise un tableau pour stocker l'historique de SE de chaque réalisation
        # Dimensions : (100 réalisations, 50 itérations)
        all_se_histories = np.zeros((num_realisations, 101))
        
        print(f"Simulation pour SNR = {SNR_dB} dB...")
        
        for r in range(num_realisations):   

            # on génère un nouveau canal 
            H1 = generate_rician_channel(Mr, Mi, d_ris_user, C0, beta_ris_user, gamma_rician) # IRS - Rx
            Hm = generate_rician_channel(Mi, Mt, d_bs_ris, C0, beta_bs_ris, gamma_rician) # BS - IRS 
            H2 = generate_rician_channel(Mr, Mt, d_bs_user, C0, beta_bs_user, gamma_rician) # BS - Rx (direct)

            # Exécution de l'algorithme
            _, _, se_history, _, _ = admm_apg_main(
                H1, H2, Hm, SNR_linear, sigma_n2, Ms, Mr, Mt, Mi, 
            )
            
            # On stocke l'historique (en s'assurant qu'il fait bien la taille K_max+1 avec l'itération 0)
            all_se_histories[r, :] = se_history[:]+[se_history[-1]]*(101-len(se_history))#K_max+1
        
        # Calcul de la moyenne sur l'axe des réalisations (axis=0)
        mean_se_history = np.mean(all_se_histories, axis=0)
        
        # Affichage de la courbe moyennée
        plt.plot(range(0, 101), mean_se_history, label=f'SNR = {SNR_dB} dB (avg)')
    plt.xlabel('Itérations ADMM')
    plt.ylabel('Efficacité Spectrale moyenne (bps/Hz)')
    plt.title(f'Convergence de l\'ADMM-APG ({num_realisations} réalisations)')
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_time_par_Mi(num_realisations, Mi_list, SNR_dB, Ms, Mr, Mt, Mi, d_bs_user=30, d_bs_ris=16, d_ris_user=16, beta_bs_user = 4, beta_bs_ris = 2.0, beta_ris_user = 2.0, gamma_rician = 10):
    """Affiche le temps de calcul en fonction du nombre d'éléments de la RIS. 1) Pour 100 itérations et 2) jusqu'à la convergence
    Figure 7."""
    plt.figure(figsize=(10, 6))
    SNR_linear = 10**(SNR_dB/10)

    # Initialisation des listes de résultats
    avg_times = [0] * len(Mi_list)
    avg_times_conv = [0] * len(Mi_list)

    for i, Mi in enumerate(Mi_list):
        print(f"Simulation pour Mi = {Mi} ...")
        
        for r in range(num_realisations):
            # 1. Génération unique des canaux pour cette réalisation
            H1 = generate_rician_channel(Mr, Mi, d_ris_user, C0, beta_ris_user, gamma_rician)
            Hm = generate_rician_channel(Mi, Mt, d_bs_ris, C0, beta_bs_ris, gamma_rician)
            H2 = generate_rician_channel(Mr, Mt, d_bs_user, C0, beta_bs_user, gamma_rician)
            
            # 2. Exécution version standard (nb_iter fixe)
            _, _, _, duree, _ = admm_apg_main(
                H1, H2, Hm, SNR_linear, 1, Ms, Mr, Mt, Mi, 
                stop_when_converged=False
            )
            avg_times[i] += duree

            # 3. Exécution version avec critère d'arrêt (convergence)
            _, _, _, duree_conv, _ = admm_apg_main(
                H1, H2, Hm, P_linear, sigma_n2, Ms, Mr, Mt, Mi, 
                stop_when_converged=True
            )
            avg_times_conv[i] += duree_conv

        # Moyennage en fin de boucle Mi
        avg_times[i] /= num_realisations
        avg_times_conv[i] /= num_realisations

    # Affichage des deux courbes
    plt.plot(Mi_list, avg_times, '-o', label=f'{nb_iter} itérations (Fixe)')
    plt.plot(Mi_list, avg_times_conv, '-s', label='Stop when converged')

    plt.xlabel('Nombre d\'éléments de l\'IRS (Mi)')
    plt.ylabel('Durée moyenne (s)')
    plt.title(f'Computation time (moyenne sur {num_realisations} réalisations)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_SE_par_Mt(num_realisations, Mt_list, SNR_dB, Ms, Mr, Mt, Mi, d_bs_user=30, d_bs_ris=16, d_ris_user=16, beta_bs_user = 4, beta_bs_ris = 2.0, beta_ris_user = 2.0, gamma_rician = 10):
    """Affiche l'efficacité spectrale en fonction du nombre d'antennes à l'émission
    Figure 4."""
    plt.figure(figsize=(10, 6))

    avg_SE = [0]*len(Mt_list) # Liste pour stocker la SE moyenne finale pour chaque Mt
    SNR_linear = 10**(SNR_dB/10)


    for i,Mt in enumerate(Mt_list):

        print(f"Simulation pour Mt = {Mt} ...")

        for r in range(num_realisations):  
            # on génère un nouveau canal pour la prochaine itération
            H1 = generate_rician_channel(Mr, Mi, d_ris_user, C0, beta_ris_user, gamma_rician)
            Hm = generate_rician_channel(Mi, Mt, d_bs_ris, C0, beta_bs_ris, gamma_rician)
            H2 = generate_rician_channel(Mr, Mt, d_bs_user, C0, beta_bs_user, gamma_rician)
            
            # Exécution de l'algorithme
            _, _, se_history, _, _ = admm_apg_main(
                H1, H2, Hm, SNR_linear, 1, Ms, Mr, Mt, Mi, 
            )
            
            # On stocke l'historique (en s'assurant qu'il fait bien la taille K_max)
            avg_SE[i] += se_history[-1] # on prend le dernier (meilleur)

        avg_SE[i]=avg_SE[i]/num_realisations
        
    # Affichage de la courbe 
    plt.plot(Mt_list,avg_SE, 'o-', label=f'SNR = {SNR_dB} dB')

    plt.xlabel('Mt')
    plt.ylabel('SE moyen')
    plt.title(f'Nombre d\'antennes')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_SE_erreur(num_realisations, delta_list, SNR_dB, Ms, Mr, Mt, Mi, d_bs_user=30, d_bs_ris=16, d_ris_user=16, beta_bs_user = 4, beta_bs_ris = 2.0, beta_ris_user = 2.0, gamma_rician=10):
    """Affiche l'efficacité spectrale en fonction de la taille de l'erreur sur le canal définie selon l'équation (19)
    Figure 6."""
    plt.figure(figsize=(10, 6))

    SNR_linear = 10**(SNR_dB/10)

    avg_SE = [0]*len(delta_list)        # Liste pour stocker la SE moyenne finale pour chaque Mt

    for i,delta in enumerate(delta_list):   
        print(f"Simulation pour delta = {delta} ...")


        for r in range(num_realisations):   
            H1_true = generate_rician_channel(Mr, Mi, d_ris_user, C0, beta_ris_user, gamma_rician)
            Hm_true = generate_rician_channel(Mi, Mt, d_bs_ris, C0, beta_bs_ris, gamma_rician)
            H2_true = generate_rician_channel(Mr, Mt, d_bs_user, C0, beta_bs_user, gamma_rician)

            
            # Création des estimations imparfaites (ce que l'algorithme "voit")
            H1_hat = add_estimation_error(H1_true, delta, Mi, Mr)
            Hm_hat = add_estimation_error(Hm_true, delta, Mt, Mi)
            H2_hat = add_estimation_error(H2_true, delta, Mt, Mr)

            # Exécution de l'algorithme
            G, theta, _, _, _ = admm_apg_main(
                H1_hat, H2_hat, Hm_hat, SNR_linear, 1, Ms, Mr, Mt, Mi, 
            )

            # le vrai canal obtenu :
            H_total_real = H2_true + (H1_true * theta) @ Hm_true

            # efficacité spectrale finale :
            T_k = H_total_real @ G
            inner_mat = np.eye(Mr) + SNR_linear/(Ms) * (T_k @ T_k.conj().T)
            _, logdet = np.linalg.slogdet(inner_mat)
            avg_SE[i] += logdet / np.log(2)

        avg_SE[i]=avg_SE[i]/num_realisations

        
        # Affichage de la courbe 
    plt.plot(delta_list,avg_SE, 'o-', label=f'SNR = {SNR_dB} dB')

    plt.xlabel('delta')
    plt.ylabel('Efficacité spectrale finale moyenne (bps/Hz)')
    plt.title(f'Efficacité spectrale avec erreurs d\'estimation du canal (sur {num_realisations} réalisations)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_augmentation_SE_par_beta(num_realisations, beta_list, SNR_dB, Ms, Mr, Mt, Mi, d_bs_user=30, d_bs_ris=16, d_ris_user=16, beta_bs_ris = 2.0, beta_ris_user = 2.0, gamma_rician=10):
    """Affiche le pourcentage d'augmentation d'efficacité spectrale gagné par la configuration de la RIS en fonction d'à quel point la canal direct BS-user est bloqué.
    Si pas de blocage (beta = 2), on a relativement peu d'augmentation d'efficacité spectrale.
    Plus le canal direct est bloqué (beta augmente), plus la RIS permet d'augmenter l'efficacité spectrale. """
    plt.figure(figsize=(10, 6))

    SNR_linear = 10**(SNR_dB/10)

    avg_augmentation_SE = [0]*len(beta_list) 

    for i,beta in enumerate(beta_list):   
        print(f"Simulation pour beta = {beta} ...")

        for r in range(num_realisations):    
            H1 = generate_rician_channel(Mr, Mi, d_ris_user, C0, beta_ris_user, gamma_rician)
            Hm = generate_rician_channel(Mi, Mt, d_bs_ris, C0, beta_bs_ris, gamma_rician)
            H2 = generate_rician_channel(Mr, Mt, d_bs_user, C0, beta, gamma_rician)

            # Exécution de l'algorithme
            _, _, se_history, _, _ = admm_apg_main(
                H1, H2, Hm, SNR_linear, 1, Ms, Mr, Mt, Mi, 
            )

            avg_augmentation_SE[i] += (se_history[-1]/se_history[0]-1)*100

        avg_augmentation_SE[i]=avg_augmentation_SE[i]/num_realisations

        
        # Affichage de la courbe 
    plt.plot(beta_list,avg_augmentation_SE, 'o-', label=f'SNR = {SNR_dB} dB')

    plt.xlabel('beta')
    plt.ylabel('Augmentation moyenne de l\'efficacité spectrale (%)')
    plt.title(f'Augmentation moyenne de l\'efficacité spectrale en fonction de l\'état de la ligne directe (sur {num_realisations} réalisations)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__=="__main__":
    ####### 1. SETUP PARAMETERS
    # System parameters
    Mr, Mt, Mi, Ms = 4, 16, 100, 4 
    P_dBW = 100 # Si sigma = 1, c'est aussi le SNR en dB 
    P_linear = 10**((P_dBW) / 10) 
    l = 2*np.pi / (3.5e9) # longueur d'onde pour f = 3.5 GHz (onde 5G)

    # Antenna spacing (directement mis dans la génération des matrices de canal) 
    d_as = l/2 
    m = 2*np.pi/l # das*m = pi

    # Channel parameters
    beta = 2.0          # Path-loss exponent
    gamma_rician_db = 10 
    gamma_rician = 10**(gamma_rician_db / 10)  
    C0_db = -30
    C0 = 10**(C0_db / 10) # Reference path loss at 1m
    sigma_n2 = 1 

    ####### 3. RUN ADMM-APG
    ## 3.1 Une seule réalisation 
    print("==== Évolution de l'Efficacité Spectrale par Itération ====")
    H1 = generate_rician_channel(Mr, Mi, 16, C0, beta, gamma_rician)
    Hm = generate_rician_channel(Mi, Mt, 16, C0, beta, gamma_rician)
    H2 = generate_rician_channel(Mr, Mt, 30, C0, 4, gamma_rician)
    SNR_list = [160, 170, 180]
    plot_se_par_iteration(SNR_list,H1,H2,Hm,Ms,Mr,Mt,Mi)
    print()

    ## 3.2 Moyenne SE sur x réalisations
    SNR_dB_list = [100,110,120] # SNR si sigma vaut 1
    num_realisations = 10  # Nombre d'itérations 
    print(f"==== Convergence de l\'ADMM-APG ({num_realisations} réalisations) ====")
    #plot_se_moy_par_iteration(num_realisations, SNR_dB_list, Ms, Mr, Mt, Mi)
    print()

    #3.3 Computation time en fonction de Mi
    Mi_list = range(25,526,100)
    num_realisations = 10  # Nombre d'itérations 
    nb_iter = 100
    SNR_dB = 100
    print(f'==== Computation time (moyenne sur {num_realisations} réalisations) ====')
    #compute_time_par_Mi(num_realisations, Mi_list, SNR_dB, Ms, Mr, Mt, Mi)
    print()

    #3.4 SE en fonction de Mt
    Mt_list = range(10,130,10)
    num_realisations = 10  # Nombre d'itérations 
    SNR_dB = 100
    print(f"==== SE en fonction du Nombre d\'antennes ({num_realisations} réalisations) ====")
    #plot_SE_par_Mt(num_realisations,Mt_list, SNR_dB, Ms,Mr,Mt,Mi)
    print()

    #3.5 SE en fonction du niveau d'erreur de connaissance du canal 
    SNR_dB=120
    delta_list = np.arange(0,1.1,0.25)
    num_realisations = 10  # Nombre d'itérations 
    print(f"==== Efficacité spectrale avec erreurs d\'estimation du canal (sur {num_realisations} réalisations) ====")
    #plot_SE_erreur(num_realisations, delta_list, SNR_dB, Ms, Mr, Mt,Mi)
    print()

    #3.6 Pourcentage d'augmentation de la SE grâce à la RIS en fonction de l'état du lien direct
    beta_list = np.arange(2.0,8.1,0.5)
    SNR_dB=120
    num_realisations = 10  # Nombre d'itérations
    print(f"==== Augmentation de l'Efficacité spectrale en fonction de l'atténuation de la ligne directe (sur {num_realisations} réalisations) ====")
    #plot_augmentation_SE_par_beta(num_realisations, beta_list, SNR_dB, Ms, Mr, Mt, Mi)
    print()
