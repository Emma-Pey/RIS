import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#INGREDIENTS 
# =============================================================================
# MODÈLE DE PHASE SHIFT PRATIQUE — Eq. (5)
# β(θ) = (1 - β_min) * ((sin(θ - φ) + 1) / 2)^α + β_min
# β_min : amplitude minimale (atteinte près de θ=0)
# α     : contrôle la pente de la courbe
# φ     : décalage horizontal
# =============================================================================
def beta(theta, beta_min=0.2, alpha=1.6, phi=0.43*np.pi):
    """Amplitude de réflexion βn(θn) — Eq. (5)"""
    return (1 - beta_min) * ((np.sin(theta - phi) + 1) / 2)**alpha + beta_min


# =============================================================================
# FONCTION OBJECTIF DU SOUS-PROBLÈME P2 — Eq. (18)
# f(θn) = β²(θn)*Ψnn  +  β(θn)*|φn|*cos(arg(φn) - θn)
#          ─────────────    ──────────────────────────────
#         terme A           terme B
#  terme A : puissance propre de l'élément n (gain du canal RIS →User)
#  terme B : gain d'alignement de phase avec les autres éléments + canal direct
# =============================================================================
def f_P2(theta_n, Psi_nn, phi_n, beta_min=0.2, alpha=1.6, phi_param=0.43*np.pi):
    """Eq. (18) : objectif P2 pour l'élément n"""
    b = beta(theta_n, beta_min, alpha, phi_param)
    return b**2 * Psi_nn + b * abs(phi_n) * np.cos(np.angle(phi_n) - theta_n)


# =============================================================================
# RÉGION DE CONFIANCE — Proposition 2
# On sait que θn* est entre arg(φn) (alignement parfait) et ±π (amplitude max)
# - arg(φn) >= 0 → θn* ∈ [arg(φn), π]
# - arg(φn) <  0 → θn* ∈ [-π, arg(φn)]
# =============================================================================
def trust_region(phi_n_angle):
    """Proposition 2 : bornes [θ_A, θ_C] de la région de confiance"""
    if phi_n_angle >= 0:
        return phi_n_angle, np.pi      # λ = 0
    else:
        return phi_n_angle, -np.pi     # λ = 1  (soit [-π, arg(φn)] en notation standard)


# =============================================================================
# SOLUTION CLOSED-FORM PAR FIT QUADRATIQUE — Proposition 3
# On approche f(θ) par une parabole passant par 3 points équirépartis
# dans la trust region, puis on prend le sommet de cette parabole comme θn*
# =============================================================================
def quadratic_fit(theta_A, theta_C, f_func):
    """
    Proposition 3 : sommet de la parabole ajustée sur 3 points
    θ_A = bord gauche (arg(φn))
    θ_B = milieu de la trust region
    θ_C = bord droit (±π)
    """
    # Étape 1 : 3 points équirépartis dans la trust region
    theta_B = (theta_A + theta_C) / 2.0

    # Étape 2 : évaluer f en chaque point
    f1 = f_func(theta_A)
    f2 = f_func(theta_B)
    f3 = f_func(theta_C)

    # Étape 3 : sommet de la parabole — Eq. (21)
    denom = 4 * (f1 - 2*f2 + f3)
    if abs(denom) < 1e-12:
        # Cas dégénéré (f quasi-linéaire) : prendre le meilleur des 3 points
        return [theta_A, theta_B, theta_C][np.argmax([f1, f2, f3])]

    theta_hat = (theta_A*(f1 - 4*f2 + 3*f3) + theta_C*(3*f1 - 4*f2 + f3)) / denom

    # Étape 4 : s'assurer que θ* reste dans la trust region
    lo, hi = min(theta_A, theta_C), max(theta_A, theta_C)
    return np.clip(theta_hat, lo, hi)

#ALGO
# =============================================================================
# ALGORITHME AO — Algorithme 1
# Maximise ||v^H * Φ + hd^H||² (gain de canal effectif = P1)
# en optimisant un θn à la fois, les autres fixés (Alternating Optimization)
#
# Précalculs :
#   Ψ    = Φ @ Φ^H          (N×N) — matrice de gain canal IRS
#   ĥd   = Φ @ hd            (N,)  — projection canal direct sur IRS
#   φn   = Σ_{m≠n} Ψnm*vm + 2*ĥd,n  — interaction élément n avec le reste
# =============================================================================
def AO_algorithm(Phi, hd, N, beta_min=0.2, alpha=1.6, phi_param=0.43*np.pi,
                 max_iter=200, tol=1e-6):
    """
    Algorithme 1 : AO pour résoudre P1
    Entrées :
        Phi      : matrice canal combinée AP→IRS→User, (N×M)
        hd       : canal direct AP→User, (M,)
        N        : nombre d'éléments IRS
        beta_min : amplitude minimale (0.2 = pratique, 1.0 = idéal)
    Sorties :
        v        : coefficients de réflexion optimisés (N,)
        theta    : déphasages optimisés (N,)
        obj      : valeur finale de ||v^H Φ + hd^H||²
        history  : historique de convergence
    """

    # ── INITIALISATION ────────────────────────────────────────────────────────
    # θn ∈ {π, -π} : amplitude maximale β(±π) ≈ 1 comme point de départ
    theta = np.random.choice([np.pi, -np.pi], size=N).astype(float)

    # Précalculs (constants pendant tout l'algo)
    Psi     = Phi @ Phi.conj().T    # Ψ = Φ*Φ^H, (N×N)
    h_hat_d = Phi @ hd              # ĥd = Φ*hd,  (N,)

    # Coefficients de réflexion initiaux : vn = β(θn) * e^(jθn)
    v = np.array([beta(t, beta_min, alpha, phi_param) * np.exp(1j*t)
                  for t in theta])

    obj_prev = -np.inf
    history  = []

    # ── BOUCLE PRINCIPALE AO ──────────────────────────────────────────────────
    for iteration in range(max_iter):

        # Pour chaque élément n, optimiser θn avec les autres vm fixés
        for n in range(N):

            # ÉTAPE 1 : Calculer φn — interaction de l'élément n avec le reste
            # φn = Σ_{m≠n} Ψnm*vm + 2*ĥd,n
            # = (Ψ[n,:] @ v) - Ψnn*vn + 2*ĥd,n  (on retire la contribution de vn)
            phi_n = np.dot(Psi[n, :], v) - Psi[n, n]*v[n] + 2*h_hat_d[n]

            # ÉTAPE 2 : Région de confiance (Proposition 2)
            # θn* ∈ [arg(φn), ±π] selon le signe de arg(φn)
            theta_A, theta_C = trust_region(np.angle(phi_n))

            # ÉTAPE 3 : Trouver θn* par fit quadratique (Proposition 3)
            Psi_nn = Psi[n, n].real
            def f_func(t, _phi=phi_n, _P=Psi_nn):
                return f_P2(t, _P, _phi, beta_min, alpha, phi_param)

            theta[n] = quadratic_fit(theta_A, theta_C, f_func)

            # ÉTAPE 4 : Mettre à jour vn = β(θn*) * e^(jθn*)
            v[n] = beta(theta[n], beta_min, alpha, phi_param) * np.exp(1j*theta[n])

        # ── CRITÈRE DE CONVERGENCE ────────────────────────────────────────────
        # Objectif P1 : ||v^H Φ + hd^H||²
        eff = v.conj() @ Phi + hd.conj()
        obj = np.real(np.dot(eff, eff.conj()))
        history.append(obj)

        if obj_prev > -np.inf and abs(obj - obj_prev) / (abs(obj_prev) + 1e-12) < tol:
            break
        obj_prev = obj

    return v, theta, obj, history


# =============================================================================
# GÉNÉRATION DES CANAUX — Modèle Rayleigh + path loss
# Géométrie (Fig. 5 du papier) :
#   AP en (dx, 0, 0), IRS en (0, dy, 0), User en (dx, d, 0)
#   Exposants de path loss : 2.2 (AP→IRS), 2.8 (IRS→User), 3.8 (AP→User)
#   Path loss à 1m : PL0 = -40 dB
# =============================================================================
def generate_channels(M, N, d_user, dx=2.0, dy=400.0, seed=None):
    rng = np.random.default_rng(seed)
    PL0 = 10**(-40/10)  # -40 dB à 1m

    # Distances géométriques
    d_AP_IRS  = np.sqrt(dx**2 + dy**2)
    d_IRS_usr = np.sqrt(dx**2 + (d_user - dy)**2)
    d_AP_usr  = np.sqrt(dx**2 + d_user**2)

    def rayleigh(n_rx, n_tx, dist, exp):
        """Canal Rayleigh avec path loss PL0/dist^exp"""
        PL = PL0 * (1.0 / dist)**exp
        h  = (rng.standard_normal((n_rx, n_tx)) +
              1j*rng.standard_normal((n_rx, n_tx))) / np.sqrt(2)
        return np.sqrt(PL) * h

    G  = rayleigh(N, M, d_AP_IRS,  2.2)   # AP→IRS,  (N×M)
    hr = rayleigh(N, 1, d_IRS_usr, 2.8)[:, 0]  # IRS→User, (N,)
    hd = rayleigh(M, 1, d_AP_usr,  3.8)[:, 0]  # AP→User,  (M,)

    # Φ = diag(hr^H) @ G — canal combiné AP→IRS→User, (N×M)
    Phi = np.diag(hr.conj()) @ G
    return Phi, hd


# =============================================================================
# CALCUL DE LA PUISSANCE MINIMALE — Eq. P* = γσ² / ||v^H Φ + hd^H||²
# =============================================================================
def puissance_dBm(v, Phi, hd, gamma, sigma2):
    """P* en dBm — puissance minimale pour satisfaire la contrainte SNR γ"""
    eff  = v.conj() @ Phi + hd.conj()
    gain = np.real(np.dot(eff, eff.conj()))
    return 10 * np.log10(gamma * sigma2 / gain * 1e3)


# =============================================================================
# SIMULATION MONTE CARLO — Figure 7 du papier
# Paramètres : N=40, M=4, SNR cible=10dB, σ²=-94dBm, 50 réalisations
# 4 schémas comparés :
#   1) IRS idéal : β=1 ∀θ (borne inférieure, meilleur cas)
#   2) AO pratique : β(θ) variable avec β_min=0.2 (algo proposé)
#   5) θ idéaux → IRS pratique : θn optimisés pour β=1, appliqués à β_min=0.2
#   6) Sans IRS : chemin direct AP→User uniquement
# =============================================================================
np.random.seed(0)
N      = 40
M      = 4
gamma  = 10**(10/10)        # SNR cible = 10 dB (linéaire)
sigma2 = 10**(-94/10)*1e-3  # bruit σ² = -94 dBm en Watts
n_MC   = 50                 # réalisations Monte Carlo

d_values = np.array([380, 383, 386, 390, 393, 397, 400], dtype=float) #On fait varier la position de l'utilisateur entre "près de l'IRS" (380m) et "loin de l'IRS / près du AP" (400m)

res_lb    = []  # 1) IRS idéal β=1
res_ao    = []  # 2) AO pratique β_min=0.2
res_ia    = []  # 5) θ idéaux → IRS pratique
res_noirs = []  # 6) sans IRS

print("Simulation en cours...")
for i_d, d in enumerate(d_values):
    print(f"  d = {d:.0f} m  [{i_d+1}/{len(d_values)}]")
    p_lb, p_ao, p_ia, p_noirs = [], [], [], []

    for r in range(n_MC):
        Phi, hd = generate_channels(M, N, d, seed=i_d*1000+r)

        # 1) IRS idéal : β=1 toujours → alignement de phase parfait
        v_lb, t_lb, _, _ = AO_algorithm(Phi, hd, N, beta_min=1.0)
        p_lb.append(puissance_dBm(v_lb, Phi, hd, gamma, sigma2))

        # 2) AO pratique : β(θ) variable, β_min=0.2 → compromis amplitude/phase
        v_ao, _, _, _ = AO_algorithm(Phi, hd, N, beta_min=0.2)
        p_ao.append(puissance_dBm(v_ao, Phi, hd, gamma, sigma2))

        # 5) θ optimisés pour β=1, appliqués à IRS réel (β_min=0.2)
        # → β(θn) sera sous-optimal car les θn ignorent la contrainte d'amplitude
        v_ia = np.array([beta(t_lb[n], 0.2) * np.exp(1j*t_lb[n]) for n in range(N)])
        p_ia.append(puissance_dBm(v_ia, Phi, hd, gamma, sigma2))

        # 6) sans IRS : gain = ||hd||² uniquement
        gain_direct = np.real(np.dot(hd.conj(), hd))
        p_noirs.append(10*np.log10(gamma*sigma2/gain_direct*1e3))

    res_lb.append(np.mean(p_lb))
    res_ao.append(np.mean(p_ao))
    res_ia.append(np.mean(p_ia))
    res_noirs.append(np.mean(p_noirs))

# =============================================================================
# TRACÉ
# =============================================================================
plt.figure(figsize=(8, 5))
plt.plot(d_values, res_lb,    'b-o',  label='1) IRS idéal : β=1 ∀θ (borne inf)')
plt.plot(d_values, res_ao,    'g--s', label='2) AO pratique : β(θ) variable, β_min=0.2')
plt.plot(d_values, res_ia,    'r:D',  label='5) θ optimisés pour β=1, appliqués à β_min=0.2')
plt.plot(d_values, res_noirs, 'k--x', label='6) Sans IRS (chemin direct uniquement)')
plt.xlabel('Distance AP–utilisateur d (m)')
plt.ylabel('Puissance AP (dBm)')
plt.title('Fig. 7 — Puissance AP vs distance (N=40, M=4, SNR=10dB)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
print("Figure sauvegardée.")
