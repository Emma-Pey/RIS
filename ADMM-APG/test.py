import numpy as np
from admm_apg import admm_apg_main
import matplotlib.pyplot as plt


#######################
######## TEST #########
#######################

def generate_test_channels(Mr, Mt, Mi):
    # Standard complex Gaussian channels
    # H = (1/sqrt(2)) * (Real + j*Imag)
    H1 = (np.random.randn(Mr, Mi) + 1j * np.random.randn(Mr, Mi)) / np.sqrt(2)
    H2 = (np.random.randn(Mr, Mt) + 1j * np.random.randn(Mr, Mt)) / np.sqrt(2)
    Hm = (np.random.randn(Mi, Mt) + 1j * np.random.randn(Mi, Mt)) / np.sqrt(2)
    
    # Optional: Apply path loss scaling
    # H1 = H1 * 10**(-dist_loss / 20)
    return H1, H2, Hm

'''def array_response(N, phi, d_as=0.5):
    """Computes a(phi) from Eq. (18)"""
    indices = np.arange(N)
    # Eq: 1/sqrt(N) * exp(j * 2*pi * d_as * n * sin(phi))
    a = (1/np.sqrt(N)) * np.exp(1j * 2 * np.pi * d_as * indices * np.sin(phi))
    return a.reshape(-1, 1)'''

"""def generate_rician_channel_nok(N_rx, N_tx, gamma_db):
    gamma = 10**(gamma_db / 10)
    phi_t = np.random.uniform(0, 2*np.pi)
    phi_r = np.random.uniform(0, 2*np.pi)
    ar = np.exp(1j * np.pi * np.sin(phi_r) * np.arange(N_rx)).reshape(-1, 1) / np.sqrt(N_rx)
    at = np.exp(1j * np.pi * np.sin(phi_t) * np.arange(N_tx)).reshape(-1, 1) / np.sqrt(N_tx)
    H_LoS  = ar @ at.conj().T
    H_NLoS = (np.random.randn(N_rx, N_tx) + 1j*np.random.randn(N_rx, N_tx)) / np.sqrt(2)
    return np.sqrt(gamma/(1+gamma)) * H_LoS + np.sqrt(1/(1+gamma)) * H_NLoS"""

def generate_rician_channel(N_rx, N_tx, dist, C0, beta, gamma):
    path_loss = C0 * (dist**-beta)
    #path_loss = (dist**-beta)

    #path_loss = 1 #en mettant ça, on retrouve à peu près les courbes du papier
    #path_loss = 0.5
    # LoS component
    phi_t = np.random.uniform(0, 2*np.pi)
    phi_r = np.random.uniform(0, 2*np.pi)
    # Ensure these are (N, 1) and (1, N)
    ar = np.exp(1j * np.pi * np.sin(phi_r) * np.arange(N_rx)).reshape(-1, 1) #/ np.sqrt(N_rx) #on ne divise pas pour inclure les gains directement dans le canal. Sinon pour calculer la pusisance totale reçue, il faudrait multiplier par les gains
    at = np.exp(1j * np.pi * np.sin(phi_t) * np.arange(N_tx)).reshape(-1, 1) #/ np.sqrt(N_tx) #le gain est proportionnel au nombre d'éléments
    H_LoS = ar @ at.conj().T
    #print(np.linalg.norm(H_LoS))
    # NLoS component
    H_NLoS = (np.random.randn(N_rx, N_tx) + 1j*np.random.randn(N_rx, N_tx)) /np.sqrt(2) #(np.sqrt(N_tx)*np.sqrt(N_rx)*
    #print(np.linalg.norm(H_NLoS))

    # Combine
    H = np.sqrt(path_loss) * (np.sqrt(gamma/(1+gamma)) * H_LoS + np.sqrt(1/(1+gamma)) * H_NLoS)
    #print(np.linalg.norm(H), N_rx, N_tx, np.sqrt(path_loss))
    return H

# ==============================================================================
#  STEERING VECTORS GEOMETRIQUES [eq. 3-4]
# ==============================================================================
def steering_vec(rx_pos, src_pos):
    """Vecteur directeur pour une source vers un reseau.
    rx_pos  : (K,3) positions des elements du reseau
    src_pos : (3,)  position de la source
    Retourne : (K,) complexe, norme sqrt(K) (non normalise)
    """
    diff  = src_pos - rx_pos                          # (K,3)
    dist  = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-12
    u     = diff / dist                               # vecteurs unitaires
    phase = (2 * np.pi / lam) * (u * rx_pos).sum(axis=1)
    return np.exp(1j * phase)                         # (K,)

# ==============================================================================
#  CANAUX RICIAN AVEC STEERING GEOMETRIQUE [eq. 1]
# ==============================================================================
def rician_geo(rx_pos, tx_pos_list, K, pl_scalar):
    """Canal Rician (rows=rx, cols=tx) avec steering geometrique.
    pl_scalar : path loss scalaire normalise (homogene a une puissance)
    Retourne  : (rows, cols) complexe
    """
    rows = len(rx_pos)
    cols = len(tx_pos_list)
    H    = np.zeros((rows, cols), dtype=complex)
    for j in range(cols):
        a_los  = steering_vec(rx_pos, tx_pos_list[j])   #composente directe du canal       # (rows,)
        nlos   = (np.random.randn(rows) + 1j * np.random.randn(rows)) / np.sqrt(2)
        h_col  = (np.sqrt(K / (K + 1)) * a_los
                  + np.sqrt(1.0 / (K + 1)) * nlos)
        # Normalisation : E[||h||^2] = rows  =>  diviser par sqrt(rows)
        H[:, j] = np.sqrt(pl_scalar) * h_col / np.sqrt(rows) 
    #print(np.linalg.norm(H),rows,cols)
    return H

def add_estimation_error(H, delta, Mt, Mr):
    norm_H = np.linalg.norm(H, 'fro')
    gamma_sq = (delta * (norm_H**2)) / np.sqrt(Mt * Mr)
    
    noise = np.sqrt(gamma_sq/2) * (np.random.randn(*H.shape) + 1j*np.random.randn(*H.shape))
    H_hat = H + noise
    
    # INDISPENSABLE : Normaliser pour garder la même puissance totale
    # On évite d'injecter de l'énergie avec le bruit
    H_hat = H_hat * (norm_H / np.linalg.norm(H_hat, 'fro'))
    return H_hat

def plot_results(se_history):
    iterations = range(1, len(se_history) + 1)
    
    plt.figure(figsize=(10, 5))
    
    plt.plot(iterations, se_history, 'b-o', markersize=3, label='Spectral Efficiency')
    plt.xlabel('Iteration')
    plt.ylabel('SE (bps/Hz)')
    plt.title('Objective Function Progress')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

#np.random.seed(42)
####### 1. SETUP PARAMETERS
# System parameters
Mr, Mt, Mi, Ms = 4, 16, 100, 4 # MISO si Mr, Ms = 1 #changer Mi augmente (multiplie par plus de 10 le pourcentage d'augmentation)
P_dB = 100 # Avec sigma = 1, c'est aussi le SNR en dB #voir liste plus bas
P_linear = 10**((P_dB) / 10) # Converting dB to Watts (linear)
# Antenna spacing (directement mis dans la génération des matrices de canal) 
# das = lambda/2 -> das_lambda = 0.5
# m = 2pi/lambda
# das*m = pi 

N_SIDE=10

lam   = 0.030   # longueur d'onde 30 mm
d_ant = lam / 2

# BS : ULA sur axe x, z=15 m
bs_pos = np.array([[i * d_ant, 0.0, 0.0] for i in range(Mt)])  # (L,3)

# RIS : grille N_SIDE x N_SIDE, y=60 m, z=15 m
ris_pos = np.array(
    [[30+ix * d_ant, np.sqrt(60**2-30**2), 0 + iz * d_ant]
     for iz in range(N_SIDE) for ix in range(N_SIDE)]
)  # (N,3)

# Utilisateurs [papier Example 1]
user_pos = np.array([
    [ 60.0, 0.0,  0.0]#,
    #[  0.0, 50.0,  0.0],
    #[-20.0, 55.0,  0.0],
])  # (M,3)

# Channel parameters
d = 30        # Side length of equilateral triangle (m)
beta = 2.0          # Path-loss exponent
gamma_rician_db = 10 # plus il est petit, plus la SE est grande. A -10dB, on retrouve les courbes du papier ??
gamma_rician = 10**(gamma_rician_db / 10)  # Convert dB to linear, pas besoin si 10
C0_db = -30
C0 = 10**(C0_db / 10) # Reference path loss at 1m
sigma_n2 = 1#1.3806e-21*300 = -180 dBW      # Noise power

path_loss=0.5 #valeur arbitraire

####### 2. GENERATE TEST CHANNELS
# Generate the 3 channels using the vertex distance d=30
H1 = generate_rician_channel(Mr, Mi, d, C0, beta, gamma_rician) # IRS - Rx
Hm = generate_rician_channel(Mi, Mt, d, C0, beta, gamma_rician) # BS - IRS 
H2 = generate_rician_channel(Mr, Mt, d, C0, beta, gamma_rician) # BS - Rx (direct)

#ne fonctionera pas pour Mr!=1
# A : Bs -> user
A = rician_geo(user_pos, bs_pos, gamma_rician, path_loss) #direct
# B : user <- RIS  (N x M)
B = rician_geo(user_pos, ris_pos, gamma_rician, path_loss) #RIS-user
# C : RIS <- BS     (L x N)
C = rician_geo(ris_pos, bs_pos, gamma_rician, path_loss)#BS-RIS

print(f"Rapport P/Sigma2 : {P_linear / sigma_n2}")

#H1,H2,Hm = generate_test_channels(Mr,Mt,Mi)
####### 3. RUN ADMM-APG
## 3.1 Une seule réalisation 
"""P_db_list = [P_dB]  # Les puissances à tester 
for P_db in P_db_list:
    P_linear = 10**(P_db / 10)
    
    # Exécution de l'algorithme ADMM-APG
    G, theta, se_history, duree = admm_apg_main(
    H1, H2, Hm, P_linear, sigma_n2, Ms, Mr, Mt, Mi, 
        K_max=100, rho=1, stop_when_converged=True
    )
    
    plt.plot(range(1, len(se_history) + 1), se_history, label=f'SNR = {P_db} dB')

####### PLOT RESULTS
plt.title("Évolution de l'Efficacité Spectrale par Itération")
plt.xlabel("Itérations")
plt.ylabel("Efficacité Spectrale (bps/Hz)")
plt.legend()
plt.grid(True, which='both', linestyle='--')
plt.show()
"""

## 3.2 Moyenne SE sur x réalisations

"""P_db_list = [100] #SNR si sigma vaut 1
num_realizations = 100  # Nombre d'itérations 
K_max = 100              # Nombre d'itérations de l'algorithme ADMM

plt.figure(figsize=(10, 6))

for P_db in P_db_list:

    P_linear = 10**(P_db / 10)
    # On initialise un tableau pour stocker l'historique de SE de chaque réalisation
    # Dimensions : (100 réalisations, 50 itérations)
    all_se_histories = np.zeros((num_realizations, K_max+1))
    
    print(f"Simulation pour P = {P_db} dBW...")
    
    for r in range(num_realizations):   

        # on génère un nouveau canal 
        H1 = generate_rician_channel(Mr, 800, d, C0, beta, gamma_rician) # IRS - Rx
        Hm = generate_rician_channel(800, Mt, d, C0, beta, gamma_rician) # BS - IRS 
        H2 = generate_rician_channel(Mr, Mt, d, C0, 3.5, gamma_rician) # BS - Rx (direct)

        # Exécution de l'algorithme
        G, theta, se_history, duree = admm_apg_main(
            H1, H2, Hm, P_linear, sigma_n2, Ms, Mr, Mt, 800, 
            K_max=K_max, rho=1.0
        )
        
        # On stocke l'historique (en s'assurant qu'il fait bien la taille K_max+1 avec l'itération 0)
        all_se_histories[r, :] = se_history[:K_max+1]
    
    # Calcul de la moyenne sur l'axe des réalisations (axis=0)
    mean_se_history = np.mean(all_se_histories, axis=0)
    
    # Affichage de la courbe moyennée
    plt.plot(range(0, K_max + 1), mean_se_history, label=f'P = {P_db} dBW (avg)')

plt.xlabel('Itérations ADMM')
plt.ylabel('Efficacité Spectrale moyenne (bps/Hz)')
plt.title(f'Convergence de l\'ADMM-APG ({num_realizations} réalisations)')
plt.legend()
plt.grid(True)
plt.show()
"""

#3.4 Computation time en fonction de Mi

Mi_list = range(25,526,100)
num_realizations = 100  # Nombre d'itérations 
nb_iter=100

plt.figure(figsize=(10, 6))

## Avec nb_iter itérations
avg_times = [0]*len(Mi_list)        # Liste pour stocker le temps moyen final pour chaque Mi

for i,Mi in enumerate(Mi_list):

    print(f"Simulation pour Mi = {Mi} ...")

    for r in range(num_realizations):    
        # on génère un nouveau canal 
        H1 = generate_rician_channel(Mr, Mi, d, C0, beta, gamma_rician) # IRS - Rx
        Hm = generate_rician_channel(Mi, Mt, d, C0, beta, gamma_rician) # BS - IRS 
        H2 = generate_rician_channel(Mr, Mt, d, C0, 3.5, gamma_rician) # BS - Rx (direct)    
        # Exécution de l'algorithme
        G, theta, se_history, duree = admm_apg_main(
            H1, H2, Hm, P_linear, sigma_n2, Ms, Mr, Mt, Mi, 
            K_max=nb_iter, rho=1.0
        )
        #if r == 1 or r == 2:
            #print(H2)
        # On stocke l'historique (en s'assurant qu'il fait bien la taille K_max)
        avg_times[i] += duree



    avg_times[i]=avg_times[i]/num_realizations
    
    # Calcul de la moyenne sur l'axe des réalisations (axis=0)
    #mean_se_history = np.mean(all_times_histories, axis=0)
    
    # Affichage de la courbe 
plt.plot(Mi_list,avg_times, '-', label=f'{nb_iter} itérations')


## Quand on a atteint la convergence
avg_times2 = [0]*len(Mi_list)        # Liste pour stocker le temps moyen final pour chaque Mi

for i,Mi in enumerate(Mi_list):

    print(f"Simulation pour Mi = {Mi} ...")

    for r in range(num_realizations):    
        # on génère un nouveau canal 
        H1 = generate_rician_channel(Mr, Mi, d, C0, beta, gamma_rician) # IRS - Rx
        Hm = generate_rician_channel(Mi, Mt, d, C0, beta, gamma_rician) # BS - IRS 
        H2 = generate_rician_channel(Mr, Mt, d, C0, 3.5, gamma_rician) # BS - Rx (direct)    
        # Exécution de l'algorithme
        G, theta, se_history, duree = admm_apg_main(
            H1, H2, Hm, P_linear, sigma_n2, Ms, Mr, Mt, Mi, 
            K_max=nb_iter, rho=1.0, stop_when_converged=True
        )
        #if r == 1 or r == 2:
            #print(H2)
        # On stocke l'historique (en s'assurant qu'il fait bien la taille K_max)
        avg_times2[i] += duree



    avg_times2[i]=avg_times2[i]/num_realizations
    
    # Calcul de la moyenne sur l'axe des réalisations (axis=0)
    #mean_se_history = np.mean(all_times_histories, axis=0)
    
    # Affichage de la courbe 
plt.plot(Mi_list,avg_times2, '-', label=f'Stop when converged')

plt.xlabel('Mi')
plt.ylabel('Durée moyenne (s)')
plt.title(f'Computation time (moyenne sur {num_realizations} réalisations)')
plt.legend()
plt.grid(True)
plt.show()
# si on a un bon tau, alors on peut stopper avant 100 itérations 



#3.5 SE en fonction de Mt

"""Mt_list = range(10,130,10)
num_realizations = 10  # Nombre d'itérations 

plt.figure(figsize=(10, 6))

avg_SE = [0]*len(Mt_list)        # Liste pour stocker la SE moyenne finale pour chaque Mt

for i,Mt in enumerate(Mt_list):

    print(f"Simulation pour Mt = {Mt} ...")

    for r in range(num_realizations):  
        # on génère un nouveau canal pour la prochaine itération
        H1 = generate_rician_channel(Mr, Mi, d, C0, beta, gamma_rician) # IRS - Rx
        Hm = generate_rician_channel(Mi, Mt, d, C0, beta, gamma_rician) # BS - IRS 
        H2 = generate_rician_channel(Mr, Mt, d, C0, beta, gamma_rician) # BS - Rx (direct)
        
        # Exécution de l'algorithme
        G, theta, se_history, duree = admm_apg_main(
            H1, H2, Hm, P_linear, sigma_n2, Ms, Mr, Mt, Mi, 
            K_max=100, rho=1.0
        )
        
        # On stocke l'historique (en s'assurant qu'il fait bien la taille K_max)
        avg_SE[i] += se_history[-1] # on prend le dernier (meilleur)


    avg_SE[i]=avg_SE[i]/num_realizations
    
    # Calcul de la moyenne sur l'axe des réalisations (axis=0)
    #mean_se_history = np.mean(all_times_histories, axis=0)
    
    # Affichage de la courbe 
plt.plot(Mt_list,avg_SE, 'o-', label=f'P = {P_dB} dBW')

plt.xlabel('Mt')
plt.ylabel('SE moyen')
plt.title(f'Nombre d\'antennes')
plt.legend()
plt.grid(True)
plt.show()

"""
#3.6 SE en fonction de P
"""
P_dB_list = range(-10,22,2)
num_realizations = 10  # Nombre d'itérations 

plt.figure(figsize=(10, 6))

avg_SE = [0]*len(P_dB_list)        # Liste pour stocker la SE moyenne finale pour chaque Mt

for i,P_dB in enumerate(P_dB_list):
    P_linear = 10**(P_dB/10)
    print(f"Simulation pour P = {P_dB} dBW ...")

    for r in range(num_realizations):  
        # on génère un nouveau canal pour la prochaine itération
        H1 = generate_rician_channel(Mr, Mi, d, C0, beta, gamma_rician) # IRS - Rx
        Hm = generate_rician_channel(Mi, Mt, d, C0, beta, gamma_rician) # BS - IRS 
        H2 = generate_rician_channel(Mr, Mt, d, C0, beta, gamma_rician) # BS - Rx (direct)
        
        # Exécution de l'algorithme
        G, theta, se_history, duree = admm_apg_main(
            H1, H2, Hm, P_linear, sigma_n2, Ms, Mr, Mt, Mi, 
            K_max=100, rho=1.0
        )
        
        # On stocke l'historique (en s'assurant qu'il fait bien la taille K_max)
        avg_SE[i] += se_history[-1] # on prend le dernier (meilleur)


    avg_SE[i]=avg_SE[i]/num_realizations
    
    # Calcul de la moyenne sur l'axe des réalisations (axis=0)
    #mean_se_history = np.mean(all_times_histories, axis=0)
    
    # Affichage de la courbe 
plt.plot(P_dB_list,avg_SE, 'o-')

plt.xlabel('P (dBW)')
plt.ylabel('SE moyen')
plt.title(f'Puissance émission')
plt.legend()
plt.grid(True)
plt.show()"""

#3.7 SE en fonction du niveau d'erreur de connaissance du canal ?

"""delta_list = np.arange(0,1.1,0.1)
num_realizations = 10  # Nombre d'itérations 

plt.figure(figsize=(10, 6))

avg_SE = [0]*len(delta_list)        # Liste pour stocker la SE moyenne finale pour chaque Mt

for i,delta in enumerate(delta_list):   
    print(f"Simulation pour delta = {delta} ...")


    for r in range(num_realizations):    
        H1_true = generate_rician_channel(Mr, Mi, d, C0, beta, gamma_rician) # IRS - Rx
        Hm_true = generate_rician_channel(Mi, Mt, d, C0, beta, gamma_rician) # BS - IRS 
        H2_true = generate_rician_channel(Mr, Mt, d, C0, beta, gamma_rician) # BS - Rx (direct)

        
        # Création des estimations imparfaites (ce que l'algorithme "voit")
        H1_hat = add_estimation_error(H1_true, delta, Mi, Mr)
        Hm_hat = add_estimation_error(Hm_true, delta, Mt, Mi)
        H2_hat = add_estimation_error(H2_true, delta, Mt, Mr)

        # Exécution de l'algorithme
        G, theta, se_history, duree = admm_apg_main(
            H1_hat, H2_hat, Hm_hat, P_linear, sigma_n2, Ms, Mr, Mt, Mi, 
            K_max=100, rho=1.0
        )

        # le vrai canal obtenu :
        H_total_real = H2_true + H1_true @ np.diag(theta) @ Hm_true

        # efficacité spectrale finale :
        T_k = H_total_real @ G
        inner_mat = np.eye(Mr) + P_linear/(sigma_n2*Ms) * (T_k @ T_k.conj().T)
        _, logdet = np.linalg.slogdet(inner_mat)
        avg_SE[i] += logdet / np.log(2)
        
        # On stocke l'historique (en s'assurant qu'il fait bien la taille K_max)
        #avg_SE[i] += se_history[-1] # on prend le dernier (meilleur)


    avg_SE[i]=avg_SE[i]/num_realizations
    
    # Calcul de la moyenne sur l'axe des réalisations (axis=0)
    #mean_se_history = np.mean(all_times_histories, axis=0)
    
    # Affichage de la courbe 
plt.plot(delta_list,avg_SE, 'o-', label=f'P = {P_dB} dBW')

plt.xlabel('delta')
plt.ylabel('SE moyen')
plt.title(f'Nombre d\'antennes')
plt.legend()
plt.grid(True)
plt.show()"""