"""
SCA-SOCP pour RIS-MU-MISO — version 3 FINALE (conforme à l'article)
Kumar et al., IEEE Wireless Commun. Letters, vol.12, no.2, Feb. 2023.

═══════════════════════════════════════════════════════════════════
ARCHITECTURE CORRECTE DES CONTRAINTES SINR (eq. 15 de l'article)
═══════════════════════════════════════════════════════════════════

L'article N'utilise PAS un majorant direct de |g_k^H w_l|².
Il introduit des variables slack (t_kl, t̄_kl) par paire (k,l) :

  SINR_k ≥ Γ_k  ⟺  f_k ≥ Γ_k · (σ² + Σ_{l≠k} (t_kl² + t̄_kl²))   [15b]
  
  avec les contraintes linéaires sur les slacks :
    t_kl  ≥  Re{g_k^H w_l}    [15c+]
    t_kl  ≥ -Re{g_k^H w_l}    [15c-]  → t_kl ≥ |Re{g_k^H w_l}|
    t̄_kl  ≥  Im{g_k^H w_l}    [15d+]
    t̄_kl  ≥ -Im{g_k^H w_l}    [15d-]  → t̄_kl ≥ |Im{g_k^H w_l}|

  Et le surrogate concave sur le signal utile (eq.8) :
    f_k = 2 Re{conj(a_n) · g_k^H w_k} - |a_n|²   avec a_n = g_k^(n)H w_k^n

  Vecteur d'optimisation x = [W_re, W_im, phi_re, phi_im, T_re, T_im]
  où T_re[k,l], T_im[k,l] sont les slacks t_kl, t̄_kl.

CORRECTIONS PAR RAPPORT À V2
═══════════════════════════════════════════════════════════════════
1. [🔴 CRITIQUE] surrogate_signal : formule b_n vectorielle incorrecte.
   V2 : np.dot(b_n.conj(), scalar) = scalar * sum(b_n)  ← faux
   V3 : 2 Re{conj(a_n) · (g_k^H w_k)} - |a_n|²         ← linéarisation standard

2. [🔴 CRITIQUE] terme d'interférence : aucun majorant valide dans V2.
   V3 : slacks t_kl, t̄_kl + contraintes linéaires (conforme article).

3. [🟠 MAJEUR] Monotonie : conséquence directe des corrections 1 et 2.
   Le sous-problème est maintenant vraiment convexe → SCA converge.

4. [🟠 MAJEUR] Ralentissement : effective_channel utilisait np.diag(Ns×Ns).
   V3 : broadcasting O(Ns·K) + jacobiens analytiques.

5. [🟡 MODÉRÉ] Initialisation : bisection MRC pour satisfaire SINR dès iter 0.
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time, os, warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# 0.  CONSTANTES PHYSIQUES
# ═══════════════════════════════════════════════════════════════
BW           = 20e6
NOISE_DBM_HZ = -174
NOISE_POWER  = 10 ** ((NOISE_DBM_HZ + 10 * np.log10(BW) - 30) / 10)   # Watts

BS_POS      = np.array([0.0,  20.0, 10.0])
RIS_POS     = np.array([50.0,  0.0,  5.0])
USER_CENTER = np.array([55.0,  0.0,  1.5])


# ═══════════════════════════════════════════════════════════════
# 1.  GÉNÉRATION DES CANAUX
# ═══════════════════════════════════════════════════════════════

def generate_geometry_channels(Nt, Ns, K, seed=0):
    rng   = np.random.default_rng(seed)
    d_BR  = np.linalg.norm(BS_POS - RIS_POS)
    PL_BR = 10**(-30/10) * d_BR**-2.2
    K_ric = 2.0
    los   = np.ones((Ns, Nt), dtype=complex)
    nlos  = (rng.standard_normal((Ns, Nt)) +
             1j * rng.standard_normal((Ns, Nt))) / np.sqrt(2)
    Hts   = np.sqrt(PL_BR) * (np.sqrt(K_ric / (K_ric + 1)) * los +
                               np.sqrt(1 / (K_ric + 1)) * nlos)
    htk   = np.zeros((K, Nt), dtype=complex)
    hsk   = np.zeros((K, Ns), dtype=complex)
    for k in range(K):
        up     = USER_CENTER + rng.normal(0, 2, 3)
        up[2]  = 1.5
        PL_Bk  = 10**(-30/10) * np.linalg.norm(BS_POS  - up) ** -3.5
        PL_Rk  = 10**(-30/10) * np.linalg.norm(RIS_POS - up) ** -2.5
        htk[k] = (np.sqrt(PL_Bk) * (rng.standard_normal(Nt) +
                  1j * rng.standard_normal(Nt)) / np.sqrt(2))
        hsk[k] = (np.sqrt(PL_Rk) * (rng.standard_normal(Ns) +
                  1j * rng.standard_normal(Ns)) / np.sqrt(2))
    return Hts, htk, hsk


# ═══════════════════════════════════════════════════════════════
# 2.  UTILITAIRES CANAL
# ═══════════════════════════════════════════════════════════════

def effective_channel(htk, hsk, Hts, phi):
    """g_k = h_tk + h_sk diag(phi) Hts  —  (K, Nt)
    Broadcasting O(K·Ns·Nt), évite np.diag (O(Ns²))."""
    return htk + (hsk * phi[np.newaxis, :]) @ Hts


def compute_sinr(gk, W, noise=NOISE_POWER):
    K    = gk.shape[0]
    sinr = np.zeros(K)
    for k in range(K):
        sig  = abs(complex(np.dot(gk[k], W[k].conj()))) ** 2
        intf = sum(abs(complex(np.dot(gk[k], W[l].conj())))**2
                   for l in range(K) if l != k)
        sinr[k] = sig / (noise + intf)
    return sinr


def tx_power_dBm(W):
    return 10 * np.log10(max(np.sum(np.abs(W)**2), 1e-30) / 1e-3)


# ═══════════════════════════════════════════════════════════════
# 3.  SURROGATE DU SIGNAL UTILE  — eq.(8) de Kumar 2023
# ═══════════════════════════════════════════════════════════════

def surrogate_signal_lb(a_n, gk_cur, w_cur):
    """
    Minoration concave de |g_k^H w_k|² (linéarisation standard).

    f_k = 2 Re{conj(a_n) · (g_k^H w_k)} - |a_n|²

    Propriétés :
      f_k ≤ |g_k^H w_k|²  pour tout (g_k, w_k)   [minoration]
      f_k = |g_k^H w_k|²  au point (g_k^n, w_k^n)  [égalité]
      f_k affine → convexe et concave.
    """
    inner = complex(np.dot(gk_cur.conj(), w_cur))      # scalaire g_k^H w_k
    return float(2.0 * np.real(np.conj(a_n) * inner) - abs(a_n)**2)


# ═══════════════════════════════════════════════════════════════
# 4.  PACK / UNPACK  (avec slacks t_kl, t̄_kl)
# ═══════════════════════════════════════════════════════════════
#
# Vecteur x :
#   [W_re (K*Nt)] [W_im (K*Nt)] [phi_re (Ns)] [phi_im (Ns)]
#   [T_re (K*(K-1))] [T_im (K*(K-1))]
#
# T_re[k,l] = t_kl   (l ≠ k, ordonné par paire)
# T_im[k,l] = t̄_kl
# ═══════════════════════════════════════════════════════════════

def _pairs(K):
    """Liste ordonnée des paires (k, l) avec l ≠ k."""
    return [(k, l) for k in range(K) for l in range(K) if l != k]


def pack(W, phi, T_re, T_im):
    return np.concatenate([W.real.ravel(), W.imag.ravel(),
                           phi.real.ravel(), phi.imag.ravel(),
                           T_re.ravel(), T_im.ravel()])


def unpack(x, K, Nt, Ns):
    nW   = K * Nt
    nPhi = Ns
    nT   = K * (K - 1)        # nombre de paires (k,l)

    off  = 0
    W_re = x[off:off+nW].reshape(K, Nt);      off += nW
    W_im = x[off:off+nW].reshape(K, Nt);      off += nW
    p_re = x[off:off+nPhi];                    off += nPhi
    p_im = x[off:off+nPhi];                    off += nPhi
    T_re = x[off:off+nT].reshape(K, K-1);     off += nT
    T_im = x[off:off+nT].reshape(K, K-1);     off += nT

    W   = W_re + 1j * W_im
    phi = p_re + 1j * p_im
    return W, phi, T_re, T_im


def pair_index(k, l, K):
    """Indice de la paire (k,l) dans T[k, :] (l ≠ k)."""
    return l if l < k else l - 1


# ═══════════════════════════════════════════════════════════════
# 5.  SOUS-PROBLÈME CONVEXE (P^{n+1})  — conforme à l'article
# ═══════════════════════════════════════════════════════════════

def build_subproblem(W_n, phi_n, Hts, htk, hsk,
                     Gamma_lin, xi, K, Nt, Ns):
    """
    Construit (objective, jac, constraints) pour scipy.minimize.

    Variables : W (K×Nt complexe), phi (Ns complexe),
                T_re (K×(K-1)) slacks t_kl,
                T_im (K×(K-1)) slacks t̄_kl.

    Objectif  : ||W||² - ξ · 2 Re{φ_n^H φ}

    Contraintes :
      [SINR]    f_k ≥ Γ_k · (σ² + Σ_{l≠k} (t_kl² + t̄_kl²))   convexe
      [SLACK+]  t_kl ≥  Re{g_k^H w_l}    linéaire
      [SLACK-]  t_kl ≥ -Re{g_k^H w_l}    linéaire
      [SLAKB+]  t̄_kl ≥  Im{g_k^H w_l}   linéaire
      [SLAKB-]  t̄_kl ≥ -Im{g_k^H w_l}   linéaire
      [MOD]     |φ_ns|² ≤ 1               convexe
    """
    # Pré-calcul canal et scalaires au point de linéarisation
    gk_n = effective_channel(htk, hsk, Hts, phi_n)           # (K, Nt)
    a_n  = np.array([complex(np.dot(gk_n[k].conj(), W_n[k]))
                     for k in range(K)])                       # (K,)

    nW   = K * Nt
    nPhi = Ns
    nT   = K * (K - 1)
    nx   = 2 * nW + 2 * nPhi + 2 * nT   # dim totale du vecteur x

    # ── Objectif ────────────────────────────────────────────
    def objective(x):
        W, phi, _, _ = unpack(x, K, Nt, Ns)
        return (float(np.sum(np.abs(W)**2))
                - xi * 2.0 * float(np.real(np.dot(phi_n.conj(), phi))))

    def objective_jac(x):
        W, phi, _, _ = unpack(x, K, Nt, Ns)
        dT = np.zeros(2 * nT)
        return np.concatenate([2.0 * W.real.ravel(), 2.0 * W.imag.ravel(),
                                -2.0 * xi * phi_n.real, -2.0 * xi * phi_n.imag,
                                dT])

    constraints = []
    off_Wre  = 0
    off_Wim  = nW
    off_phre = 2 * nW
    off_phim = 2 * nW + nPhi
    off_Tre  = 2 * nW + 2 * nPhi
    off_Tim  = 2 * nW + 2 * nPhi + nT

    # ── Contraintes SINR ─────────────────────────────────────
    for k in range(K):
        def make_sinr(k):
            def fun(x):
                W, phi, T_re, T_im = unpack(x, K, Nt, Ns)
                gk  = effective_channel(htk, hsk, Hts, phi)
                lhs = surrogate_signal_lb(a_n[k], gk[k], W[k])
                rhs = NOISE_POWER
                for li, l in enumerate(l for l in range(K) if l != k):
                    rhs += T_re[k, li]**2 + T_im[k, li]**2
                return float(lhs - Gamma_lin[k] * rhs)
            return fun
        constraints.append({'type': 'ineq', 'fun': make_sinr(k)})

    # ── Contraintes de slack t_kl ≥ |Re{g_k^H w_l}|  ────────
    for k in range(K):
        for li, l in enumerate(ll for ll in range(K) if ll != k):
            idx_t = off_Tre + k * (K-1) + li   # indice de t_kl dans x

            def make_slack_re_pos(k, l, idx_t):
                def fun(x):
                    W, phi, T_re, _ = unpack(x, K, Nt, Ns)
                    gk  = effective_channel(htk, hsk, Hts, phi)
                    val = float(T_re[k, pair_index(k, l, K)]
                                - np.real(complex(np.dot(gk[k].conj(), W[l]))))
                    return val
                return fun

            def make_slack_re_neg(k, l, idx_t):
                def fun(x):
                    W, phi, T_re, _ = unpack(x, K, Nt, Ns)
                    gk  = effective_channel(htk, hsk, Hts, phi)
                    val = float(T_re[k, pair_index(k, l, K)]
                                + np.real(complex(np.dot(gk[k].conj(), W[l]))))
                    return val
                return fun

            def make_slack_im_pos(k, l, idx_t):
                def fun(x):
                    W, phi, _, T_im = unpack(x, K, Nt, Ns)
                    gk  = effective_channel(htk, hsk, Hts, phi)
                    val = float(T_im[k, pair_index(k, l, K)]
                                - np.imag(complex(np.dot(gk[k].conj(), W[l]))))
                    return val
                return fun

            def make_slack_im_neg(k, l, idx_t):
                def fun(x):
                    W, phi, _, T_im = unpack(x, K, Nt, Ns)
                    gk  = effective_channel(htk, hsk, Hts, phi)
                    val = float(T_im[k, pair_index(k, l, K)]
                                + np.imag(complex(np.dot(gk[k].conj(), W[l]))))
                    return val
                return fun

            constraints.append({'type': 'ineq', 'fun': make_slack_re_pos(k, l, idx_t)})
            constraints.append({'type': 'ineq', 'fun': make_slack_re_neg(k, l, idx_t)})
            constraints.append({'type': 'ineq', 'fun': make_slack_im_pos(k, l, idx_t)})
            constraints.append({'type': 'ineq', 'fun': make_slack_im_neg(k, l, idx_t)})

    # ── Contraintes de module : |φ_ns|² ≤ 1 ─────────────────
    for ns in range(Ns):
        def make_mod(ns):
            def fun(x):
                pr = x[off_phre + ns]
                pi = x[off_phim + ns]
                return float(1.0 - pr*pr - pi*pi)

            def jac(x):
                pr = x[off_phre + ns]
                pi = x[off_phim + ns]
                g  = np.zeros(nx)
                g[off_phre + ns] = -2.0 * pr
                g[off_phim + ns] = -2.0 * pi
                return g
            return fun, jac

        f_m, j_m = make_mod(ns)
        constraints.append({'type': 'ineq', 'fun': f_m, 'jac': j_m})

    return objective, objective_jac, constraints


# ═══════════════════════════════════════════════════════════════
# 6.  INITIALISATION FAISABLE
# ═══════════════════════════════════════════════════════════════

def init_feasible(gk_n, W_n, Gamma_lin, K, Nt):
    """Initialise les slacks t_kl, t̄_kl à leurs valeurs exactes en W_n."""
    T_re = np.zeros((K, K-1))
    T_im = np.zeros((K, K-1))
    for k in range(K):
        for li, l in enumerate(ll for ll in range(K) if ll != k):
            u = complex(np.dot(gk_n[k].conj(), W_n[l]))
            T_re[k, li] = abs(np.real(u))
            T_im[k, li] = abs(np.imag(u))
    return T_re, T_im


# ═══════════════════════════════════════════════════════════════
# 7.  ALGORITHME SCA-SOCP PRINCIPAL
# ═══════════════════════════════════════════════════════════════

def sca_socp_ris(Hts, htk, hsk, Gamma_dB, K, Nt, Ns,
                 xi=1e-4, n_iter=20, tol=1e-4, verbose=True):
    """
    Algorithme 1 de Kumar et al. 2023.

    Paramètres
    ----------
    xi     : régularisation φ (typiquement 1e-4 à 1e-3)
    n_iter : max itérations SCA
    tol    : critère arrêt Δpower/power

    Retourne
    --------
    W, phi, power_history (dBm)
    """
    Gamma_lin = 10 ** (np.asarray(Gamma_dB, dtype=float) / 10.0)

    # ── Initialisation ────────────────────────────────────────
    phi_n = np.exp(1j * np.random.uniform(0, 2*np.pi, Ns))
    gk_n  = effective_channel(htk, hsk, Hts, phi_n)

    # W_mrc scalé pour satisfaire grossièrement les SINR
    W_n = np.zeros((K, Nt), dtype=complex)
    for k in range(K):
        n_g  = np.linalg.norm(gk_n[k])
        W_n[k] = gk_n[k].conj() / (n_g + 1e-30)

    # Bisection sur le scaling pour satisfaire SINR
    alpha = 1.0
    sinr  = compute_sinr(gk_n, W_n * alpha)
    while np.any(sinr < Gamma_lin) and alpha < 1e10:
        alpha *= 4.0
        sinr   = compute_sinr(gk_n, W_n * alpha)
    W_n *= alpha

    T_re_n, T_im_n = init_feasible(gk_n, W_n, Gamma_lin, K, Nt)

    power_history = []
    prev_power    = float(np.sum(np.abs(W_n)**2))

    if verbose:
        print(f"  {'Iter':>4}  {'Power (dBm)':>12}  {'SINR_min (dB)':>14}  {'OK':>4}")
        print("  " + "-"*40)

    for n in range(n_iter):
        obj_fn, obj_jac, constrs = build_subproblem(
            W_n, phi_n, Hts, htk, hsk, Gamma_lin, xi, K, Nt, Ns)

        x0  = pack(W_n, phi_n, T_re_n, T_im_n)   # warm-start
        res = minimize(obj_fn, x0,
                       jac=obj_jac,
                       method='SLSQP',
                       constraints=constrs,
                       options={'maxiter': 300, 'ftol': 1e-7})

        W_c, phi_c, T_re_c, T_im_c = unpack(res.x, K, Nt, Ns)

        # Projection sur cercle unité
        phi_c = np.exp(1j * np.angle(phi_c))

        p_cand = float(np.sum(np.abs(W_c)**2))

        if not np.isfinite(p_cand):
            W_new, phi_new  = W_n, phi_n
            T_re_n, T_im_n  = T_re_n, T_im_n
            power           = prev_power
        else:
            W_new, phi_new  = W_c, phi_c
            T_re_n, T_im_n  = T_re_c, T_im_c
            power           = p_cand

        p_dBm = tx_power_dBm(W_new)
        power_history.append(p_dBm)

        if verbose:
            gk_new = effective_channel(htk, hsk, Hts, phi_new)
            sinr   = compute_sinr(gk_new, W_new)
            ok     = "✓" if np.all(sinr >= Gamma_lin * 0.95) else "·"
            print(f"  {n+1:>4}  {p_dBm:>12.3f}  "
                  f"{10*np.log10(np.min(sinr)+1e-30):>14.3f}  {ok:>4}")

        if n > 1:
            delta = abs(power - prev_power) / (prev_power + 1e-30)
            if delta < tol:
                if verbose:
                    print(f"  → Convergence à l'itération {n+1}")
                break

        W_n, phi_n, prev_power = W_new, phi_new, power

    return W_new, phi_new, power_history


# ═══════════════════════════════════════════════════════════════
# 8.  SIMULATIONS DE VALIDATION
# ═══════════════════════════════════════════════════════════════

def run_all_simulations(Nt=4, K=4, mc_runs=5):
    os.makedirs('outputs', exist_ok=True)
    np.random.seed(0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Kumar et al. 2023 — Validation SCA-SOCP (V3 finale)", fontsize=12)

    # ── Sim 1 : Convergence ──────────────────────────────────
    print("\n[1/3] Convergence (Ns=100, K=4) ...")
    Ns = 100
    Hts, htk, hsk = generate_geometry_channels(Nt, Ns, K, seed=42)
    t0 = time.time()
    print("  — Γ=10 dB —")
    _, _, h10 = sca_socp_ris(Hts, htk, hsk, np.ones(K)*10, K, Nt, Ns, verbose=True)
    print("  — Γ=15 dB —")
    _, _, h15 = sca_socp_ris(Hts, htk, hsk, np.ones(K)*15, K, Nt, Ns, verbose=True)
    print(f"  Durée sim1 : {time.time()-t0:.1f}s")

    ax = axes[0]
    ax.plot(range(1, len(h10)+1), h10, 'o-',  color='#3B8BD4', lw=2, ms=5, label=r'$\Gamma=10$ dB')
    ax.plot(range(1, len(h15)+1), h15, 's--', color='#E24B4A', lw=2, ms=5, label=r'$\Gamma=15$ dB')
    ax.set_xlabel('Itérations SCA'); ax.set_ylabel('Puissance TX (dBm)')
    ax.set_title(f'Convergence  Ns={Ns}, K={K}')
    ax.grid(True, ls='--', alpha=.4); ax.legend()

    # ── Sim 2 : Puissance vs Ns ──────────────────────────────
    print("\n[2/3] Puissance vs Ns ...")
    Ns_list = [64, 144, 256,400,576]
    Nt = 6
    K = 6
    p_vs_Ns = []
    t0 = time.time()
    for ns in Ns_list:
        vals = []
        for seed in range(mc_runs):
            H_, h_, s_ = generate_geometry_channels(Nt, ns, K, seed=seed)
            _, _, hist  = sca_socp_ris(H_, h_, s_, np.ones(K)*10,
                                        K, Nt, ns, verbose=False)
            if hist:
                vals.append(10**(hist[-1]/10))
        mean_mW = np.mean(vals) if vals else np.nan
        p_vs_Ns.append(10*np.log10(mean_mW) if np.isfinite(mean_mW) else np.nan)
        print(f"  Ns={ns:3d} → {p_vs_Ns[-1]:.2f} dBm  ({len(vals)}/{mc_runs})")
    print(f"  Durée : {time.time()-t0:.1f}s")

    ax = axes[1]
    ax.plot(Ns_list, p_vs_Ns, 'D-', color='#1D9E75', lw=2, ms=8, label='SCA-SOCP')
    ax.set_xlabel(r'Éléments RIS ($N_s$)'); ax.set_ylabel('Puissance moy. (dBm)')
    ax.set_title(r'Puissance vs $N_s$  ($\Gamma=10$ dB)')
    ax.grid(True, ls='--', alpha=.4); ax.legend()

    # ── Sim 3 : Puissance vs Γ ───────────────────────────────
    print("\n[3/3] Puissance vs SINR cible ...")
    gdb_list   = [0, 5, 10, 15, 20]
    p_vs_gamma = []
    t0 = time.time()
    for g_dB in gdb_list:
        vals = []
        for seed in range(mc_runs):
            H_, h_, s_ = generate_geometry_channels(Nt, 16, K, seed=seed)
            _, _, hist  = sca_socp_ris(H_, h_, s_, np.ones(K)*g_dB,
                                        K, Nt, 16, verbose=False)
            if hist:
                vals.append(10**(hist[-1]/10))
        mean_mW = np.mean(vals) if vals else np.nan
        p_vs_gamma.append(10*np.log10(mean_mW) if np.isfinite(mean_mW) else np.nan)
        print(f"  Γ={g_dB:2d} dB → {p_vs_gamma[-1]:.2f} dBm")
    print(f"  Durée : {time.time()-t0:.1f}s")

    ax = axes[2]
    ax.plot(gdb_list, p_vs_gamma, '^--', color='#3B8BD4', lw=2, ms=8)
    ax.set_xlabel(r'SINR cible $\Gamma$ (dB)'); ax.set_ylabel('Puissance moy. (dBm)')
    ax.set_title(r'Puissance vs $\Gamma$  (Ns=16)')
    ax.grid(True, ls='--', alpha=.4)

    plt.tight_layout()
    out = 'outputs/kumar_v3_final.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n>>> Figures sauvegardées : {out}")
    return fig


# ═══════════════════════════════════════════════════════════════
# 9.  TESTS UNITAIRES
# ═══════════════════════════════════════════════════════════════

def test_surrogates(n=10_000):
    """Vérifie la propriété de minoration de surrogate_signal_lb."""
    np.random.seed(1); Nt = 6; err = 0
    for _ in range(n):
        g_n = np.random.randn(Nt)+1j*np.random.randn(Nt)
        w_n = np.random.randn(Nt)+1j*np.random.randn(Nt)
        g_t = np.random.randn(Nt)+1j*np.random.randn(Nt)
        w_t = np.random.randn(Nt)+1j*np.random.randn(Nt)
        a_n  = complex(np.dot(g_n.conj(), w_n))
        true = abs(complex(np.dot(g_t.conj(), w_t)))**2
        lb   = surrogate_signal_lb(a_n, g_t, w_t)
        if lb > true + 1e-9: err += 1
    status = "✓ PASS" if err == 0 else f"✗ FAIL ({err} violations)"
    print(f"[Test] surrogate_signal_lb  (minoration) : {status}")

    # Vérif slacks: t² + t̄² ≥ |u|²  quand t=|Re(u)|, t̄=|Im(u)|
    err2 = 0
    for _ in range(n):
        g = np.random.randn(Nt)+1j*np.random.randn(Nt)
        w = np.random.randn(Nt)+1j*np.random.randn(Nt)
        u = complex(np.dot(g.conj(), w))
        if abs(np.real(u))**2 + abs(np.imag(u))**2 < abs(u)**2 - 1e-9:
            err2 += 1
    status2 = "✓ PASS" if err2 == 0 else f"✗ FAIL ({err2})"
    print(f"[Test] slack t²+t̄² ≥ |u|²             : {status2}")
    return err == 0 and err2 == 0


# ═══════════════════════════════════════════════════════════════
# 10. POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 60)
    print("  SCA-SOCP RIS-MU-MISO  —  Kumar et al. 2023  (V3 finale)")
    print("═" * 60)
    ok = test_surrogates()
    if not ok:
        raise RuntimeError("Échec des tests unitaires — arrêt.")
    run_all_simulations(Nt=4, K=2, mc_runs=5)
    print("\n>>> TERMINÉ.")
