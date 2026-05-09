import numpy as np

def beta(theta, beta_min=0.2, alpha=1.6, phi=0.43*np.pi):
    return (1 - beta_min) * ((np.sin(theta - phi) + 1) / 2)**alpha + beta_min

def f_P2(theta_n, Psi_nn, phi_n, beta_min=0.2, alpha=1.6, phi_param=0.43*np.pi):
    """
    Éq. (18) : fonction objectif du sous-problème P2
    f(θ_n) = β²(θ_n) * Ψ_nn  +  β(θ_n) * |φ_n| * cos(arg(φ_n) - θ_n)
    """
    b = beta(theta_n, beta_min, alpha, phi_param)
    return b**2 * Psi_nn + b * abs(phi_n) * np.cos(np.angle(phi_n) - theta_n)

def trust_region(phi_n_angle):
    """
    Proposition 2 : retourne les bornes [θ_A, θ_C] de la région de confiance
    """
    if phi_n_angle >= 0:
        return phi_n_angle, np.pi     # λ = 0
    else:
        return phi_n_angle, -np.pi    # λ = 1

# Test
print(f"φ_n = +π/4 → région : {trust_region(np.pi/4)}")
print(f"φ_n = -π/4 → région : {trust_region(-np.pi/4)}")

def quadratic_fit(theta_A, theta_C, f_func):
    """
    Proposition 3 : fit quadratique sur 3 points équidistants
    retourne le sommet de la parabole = solution approchée
    """
    theta_B = (theta_A + theta_C) / 2.0

    f1 = f_func(theta_A)
    f2 = f_func(theta_B)
    f3 = f_func(theta_C)

    denom = 4 * (f1 - 2*f2 + f3)

    if abs(denom) < 1e-12:
        # cas dégénéré : on prend juste le meilleur des 3 points
        return [theta_A, theta_B, theta_C][np.argmax([f1, f2, f3])]

    theta_hat = (theta_A*(f1 - 4*f2 + 3*f3) + theta_C*(3*f1 - 4*f2 + f3)) / denom

    # on s'assure que le résultat reste dans la région
    lo, hi = min(theta_A, theta_C), max(theta_A, theta_C)
    return np.clip(theta_hat, lo, hi)

# Test : même φ_n qu'avant
Psi_nn = 1.0
phi_n  = 0.5 * np.exp(1j * np.pi/4)
theta_A, theta_C = trust_region(np.angle(phi_n))


def f_test(t):
    return f_P2(t, Psi_nn, phi_n)

theta_star = quadratic_fit(theta_A, theta_C, f_test)

print(f"θ_A        = {theta_A:.3f}")
print(f"θ_C        = {theta_C:.3f}")
print(f"θ* trouvé  = {theta_star:.3f}")
print(f"f(θ*)      = {f_test(theta_star):.4f}")
print(f"f(θ_C=+π)  = {f_test(np.pi):.4f}")

def AO_algorithm(Phi, hd, N, beta_min=0.2, alpha=1.6, phi_param=0.43*np.pi,
                 max_iter=200, tol=1e-6):
    """
    Algorithme 1 — AO pour maximiser ||v^H * Phi + hd^H||²
    """
    # Initialisation : θ_n ∈ {π, -π} (amplitude maximale)
    theta = np.random.choice([np.pi, -np.pi], size=N).astype(float)

    # Précalcul de Ψ = Phi @ Phi^H  et  ĥ_d = Phi @ hd
    Psi     = Phi @ Phi.conj().T    # (N x N)
    h_hat_d = Phi @ hd              # (N,)

    # Coefficients de réflexion initiaux
    v = np.array([beta(t, beta_min, alpha, phi_param) * np.exp(1j*t)
                  for t in theta])

    obj_prev = -np.inf
    history  = []

    for iteration in range(max_iter):
        for n in range(N):

            # Étape 1 : calculer φ_n
            phi_n = np.dot(Psi[n, :], v) - Psi[n, n]*v[n] + 2*h_hat_d[n]

            # Étape 2 : région de confiance (Proposition 2)
            theta_A, theta_C = trust_region(np.angle(phi_n))

            # Étape 3 : fit quadratique (Proposition 3)
            Psi_nn = Psi[n, n].real
            def f_func(t, _phi=phi_n, _P=Psi_nn):
                return f_P2(t, _P, _phi, beta_min, alpha, phi_param)

            theta[n] = quadratic_fit(theta_A, theta_C, f_func)

            # Étape 4 : mettre à jour v_n
            v[n] = beta(theta[n], beta_min, alpha, phi_param) * np.exp(1j*theta[n])

        # Calcul de l'objectif ||v^H Phi + hd^H||²
        eff = v.conj() @ Phi + hd.conj()
        obj = np.real(np.dot(eff, eff.conj()))
        history.append(obj)

        # Critère d'arrêt
        if obj_prev > -np.inf and abs(obj - obj_prev) / (abs(obj_prev) + 1e-12) < tol:
            break
        obj_prev = obj

    return v, theta, obj, history

# Test de l'algorithme AO
np.random.seed(42)

N = 4   # nombre d'éléments IRS (petit pour tester)
M = 2   # nombre d'antennes au PA

# Canaux aléatoires simples
Phi = (np.random.randn(N, M) + 1j*np.random.randn(N, M)) / np.sqrt(2)
hd  = (np.random.randn(M)   + 1j*np.random.randn(M))   / np.sqrt(2)

# Lancer l'algorithme
v, theta, obj, history = AO_algorithm(Phi, hd, N)

print(f"Nombre d'itérations : {len(history)}")
print(f"Objectif final      : {obj:.4f}")
print(f"Déphasages θ*       : {np.round(theta, 3)}")
print(f"Amplitudes β(θ*)    : {np.round(np.abs(v), 3)}")

print("\nConvergence :")
for i, val in enumerate(history):
    print(f"  itération {i+1} : {val:.4f}")

def generate_channels(M, N, d_user, dx=2.0, dy=400.0, seed=None):
    rng = np.random.default_rng(seed)
    PL0 = 10**(-40/10)  # perte de trajet à 1m = -40 dB

    # Distances
    d_AP_IRS  = np.sqrt(dx**2 + dy**2)
    d_IRS_usr = np.sqrt(dx**2 + (d_user - dy)**2)
    d_AP_usr  = np.sqrt(dx**2 + d_user**2)

    def rayleigh(n_rx, n_tx, dist, exp):
        PL = PL0 * (1.0 / dist)**exp
        h  = (rng.standard_normal((n_rx, n_tx)) +
              1j*rng.standard_normal((n_rx, n_tx))) / np.sqrt(2)
        return np.sqrt(PL) * h

    G  = rayleigh(N, M, d_AP_IRS,  2.2)
    hr = rayleigh(N, 1, d_IRS_usr, 2.8)[:, 0]
    hd = rayleigh(M, 1, d_AP_usr,  3.8)[:, 0]

    Phi = np.diag(hr.conj()) @ G
    return Phi, hd

def puissance_dBm(v, Phi, hd, gamma, sigma2):
    """P* = γσ² / ||v^H Phi + hd^H||²  (en dBm)"""
    eff  = v.conj() @ Phi + hd.conj()
    gain = np.real(np.dot(eff, eff.conj()))
    return 10 * np.log10(gamma * sigma2 / gain * 1e3)


# ── Simulation Figure 7 ──────────────────────────────────────
import matplotlib.pyplot as plt

np.random.seed(0)
N      = 40
M      = 4
gamma  = 10**(10/10)        # SNR cible = 10 dB
sigma2 = 10**(-94/10)*1e-3  # bruit = -94 dBm en Watts
n_MC   = 500                # nombre de réalisations Monte Carlo

d_values = np.array([380, 383, 386, 390, 393, 397, 400], dtype=float)

res_lb  = []   # 1) borne inf  : IRS idéal β_min=1
res_ao  = []   # 2) AO Prop.3  : IRS pratique β_min=0.2
res_ia  = []   # 5) hypothèse idéale appliquée à IRS pratique
res_noirs = [] # 6) sans IRS

print("Simulation en cours...")
for i_d, d in enumerate(d_values):
    print(f"  d = {d:.0f} m  [{i_d+1}/{len(d_values)}]")
    p_lb, p_ao, p_ia, p_noirs = [], [], [], []

    for r in range(n_MC):
        Phi, hd = generate_channels(M, N, d, seed=i_d*1000+r)

        # 1) IRS idéal
        v_lb, t_lb, _, _ = AO_algorithm(Phi, hd, N, beta_min=1.0)
        p_lb.append(puissance_dBm(v_lb, Phi, hd, gamma, sigma2))

        # 2) IRS pratique AO
        v_ao, _, _, _ = AO_algorithm(Phi, hd, N, beta_min=0.2)
        p_ao.append(puissance_dBm(v_ao, Phi, hd, gamma, sigma2))

        # 5) déphasages idéaux → IRS pratique
        v_ia = np.array([beta(t_lb[n], 0.2) * np.exp(1j*t_lb[n]) for n in range(N)])
        p_ia.append(puissance_dBm(v_ia, Phi, hd, gamma, sigma2))

        # 6) sans IRS
        gain_direct = np.real(np.dot(hd.conj(), hd))
        p_noirs.append(10*np.log10(gamma*sigma2/gain_direct*1e3))

    res_lb.append(np.mean(p_lb))
    res_ao.append(np.mean(p_ao))
    res_ia.append(np.mean(p_ia))
    res_noirs.append(np.mean(p_noirs))

# ── Tracé ────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(d_values, res_lb,    'b-o',  label='1) Borne inf : IRS idéal')
plt.plot(d_values, res_ao,    'g--s', label='2) AO Prop.3 : β_min=0.2')
plt.plot(d_values, res_ia,    'r:D',  label='5) Hypothèse idéale → pratique')
plt.plot(d_values, res_noirs, 'k--x', label='6) Sans IRS')
plt.xlabel('Distance PA–utilisateur d (m)')
plt.ylabel('Puissance PA (dBm)')
plt.title('Fig. 7 — Puissance PA vs distance (N=40, M=4)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('figure7.png', dpi=150)
plt.show()
print("Figure sauvegardée : figure7.png")

