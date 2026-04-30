import numpy as np
from admm_apg import admm_apg_main
import matplotlib.pyplot as plt


#######################
######## TEST #########
#######################

"""def generate_test_channels(Mr, Mt, Mi):
    # Standard complex Gaussian channels
    # H = (1/sqrt(2)) * (Real + j*Imag)
    H1 = (np.random.randn(Mr, Mi) + 1j * np.random.randn(Mr, Mi)) / np.sqrt(2)
    H2 = (np.random.randn(Mr, Mt) + 1j * np.random.randn(Mr, Mt)) / np.sqrt(2)
    Hm = (np.random.randn(Mi, Mt) + 1j * np.random.randn(Mi, Mt)) / np.sqrt(2)
    
    # Optional: Apply path loss scaling
    # H1 = H1 * 10**(-dist_loss / 20)
    return H1, H2, Hm"""

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
    path_loss = np.sqrt(C0 * (dist**-beta)) 
    #path_loss = 1 #en mettant ça, on retrouve les courbes du papier
    # LoS component
    phi_t = np.random.uniform(0, 2*np.pi)
    phi_r = np.random.uniform(0, 2*np.pi)
    # Ensure these are (N, 1) and (1, N)
    ar = np.exp(1j * np.pi * np.sin(phi_r) * np.arange(N_rx)).reshape(-1, 1) / np.sqrt(N_rx)
    at = np.exp(1j * np.pi * np.sin(phi_t) * np.arange(N_tx)).reshape(-1, 1) / np.sqrt(N_tx)
    H_LoS = ar @ at.conj().T
    # NLoS component
    H_NLoS = (np.random.randn(N_rx, N_tx) + 1j*np.random.randn(N_rx, N_tx)) / np.sqrt(2)
    # Combine
    H = path_loss * (np.sqrt(gamma/(1+gamma)) * H_LoS + np.sqrt(1/(1+gamma)) * H_NLoS)
    return H

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


####### 1. SETUP PARAMETERS
# System parameters
Mr, Mt, Mi, Ms = 4, 16, 100, 4 # MISO si Mr = 1
#P_dB = 20 # Avec sigma = 1, c'est aussi le SNR en dB #voir liste plus bas
#P_linear = 10**((P_dB) / 10) # Converting dB to Watts (linear)
# Antenna spacing (directement mis dans la génération des matrices de canal) 
# das = lambda/2 -> das_lambda = 0.5
# m = 2pi/lambda
# das*m = pi 

# Channel parameters
d = 30              # Side length of equilateral triangle (m)
beta = 2.0          # Path-loss exponent
gamma_rician_db = 10 # plus il est petit, plus la SE est grande. A -10dB, on retrouve les courbes du papier
gamma_rician = 10**(gamma_rician_db / 10)  # Convert dB to linear, pas besoin
C0_db = -30
C0 = 10**(C0_db / 10) # Reference path loss at 1m
sigma_n2 = 1      # Noise power

####### 2. GENERATE TEST CHANNELS
# Generate the 3 channels using the vertex distance d=30
H1 = generate_rician_channel(Mr, Mi, d, C0, beta, gamma_rician) # IRS - Rx
Hm = generate_rician_channel(Mi, Mt, d, C0, beta, gamma_rician) # BS - IRS 
H2 = generate_rician_channel(Mr, Mt, d, C0, beta, gamma_rician) # BS - Rx (direct)

####### 3. RUN ADMM-APG
## 3.1 Une seule réalisation 
"""P_db_list = [15]  # Les puissances à tester 
for P_db in P_db_list:
    P_linear = 10**(P_db / 10)
    
    # Exécution de l'algorithme ADMM-APG
    G, theta, se_history = admm_apg_main(
        H1, H2, Hm, P_linear, sigma_n2, Ms, Mr, Mt, Mi, 
        K_max=100, rho=1
    )
    
    plt.plot(range(1, len(se_history) + 1), se_history, label=f'P = {P_db} dB')

####### PLOT RESULTS
plt.title("Évolution de l'Efficacité Spectrale par Itération")
plt.xlabel("Itérations")
plt.ylabel("Efficacité Spectrale (bps/Hz)")
plt.legend()
plt.grid(True, which='both', linestyle='--')
plt.show()"""


## 3.2 Moyenne sur x réalisations

P_db_list = [0, 5, 10, 15]
num_realizations = 1  # Nombre d'itérations 
K_max = 50              # Nombre d'itérations de l'algorithme ADMM

plt.figure(figsize=(10, 6))

for P_db in P_db_list:
    P_linear = 10**(P_db / 10)
    # On initialise un tableau pour stocker l'historique de SE de chaque réalisation
    # Dimensions : (100 réalisations, 50 itérations)
    all_se_histories = np.zeros((num_realizations, K_max))
    
    print(f"Simulation pour P = {P_db} dB...")
    
    for r in range(num_realizations):        
        # Exécution de l'algorithme
        G, theta, se_history = admm_apg_main(
            H1, H2, Hm, P_linear, sigma_n2, Ms, Mr, Mt, Mi, 
            K_max=K_max, rho=1.0
        )
        
        # On stocke l'historique (en s'assurant qu'il fait bien la taille K_max)
        all_se_histories[r, :] = se_history[:K_max]

        # on génère un nouveau canal pour la prochaine itération
        H1 = generate_rician_channel(Mr, Mi, d, C0, beta, gamma_rician) # IRS - Rx
        Hm = generate_rician_channel(Mi, Mt, d, C0, beta, gamma_rician) # BS - IRS 
        H2 = generate_rician_channel(Mr, Mt, d, C0, beta, gamma_rician) # BS - Rx (direct)
    
    # Calcul de la moyenne sur l'axe des réalisations (axis=0)
    mean_se_history = np.mean(all_se_histories, axis=0)
    
    # Affichage de la courbe moyennée
    plt.plot(range(1, K_max + 1), mean_se_history, label=f'P = {P_db} dB (avg)')

plt.xlabel('Itérations ADMM')
plt.ylabel('Efficacité Spectrale moyenne (bps/Hz)')
plt.title(f'Convergence de l\'ADMM-APG ({num_realizations} réalisations)')
plt.legend()
plt.grid(True)
plt.show()

