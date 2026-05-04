"""
theta -> H (canal effectif global)-> cost (minimiser) -> gradient -> d -> mu -> nouveau theta
RIS (Reconfigurable Intelligent Surface) — Optimisation phases + beamforming
D'apres : Pitz et al., "A gradient-based algorithm for joint beamforming
and reflection design in RIS-assisted mobile communications"

Systeme uplink MU-MISO :
  BS : L antennes | RIS : N elements passifs | M utilisateurs mono-antenne

Approche proposee (Section 4) :
  - Cout J(theta) = sum_m log2(1 - Pm*||h_m||^2 / tr(R))  [eq. 35]
  - Gradient analytique exact                              [eq. 41]
  - Pas de descente via recherche de ligne Armijo          [Algorithm 2]
  - Mise a jour BFGS amortie pour accelerer la convergence [Algorithm 3]
  - Beamforming MVDR optimal en post-traitement            [eq. 15]

Canaux : steering vectors geometriques (positions Example 1 du papier)
         + path_loss normalises (sigma2=1, SNR_dB=20)
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
#  PARAMETRES
# ==============================================================================
SEED     = 42
L        = 16      # antennes BS  [papier Section 5]
N_SIDE   = 20       # cote grille RIS : N = N_SIDE^2  (mettre 20 pour l'article complet)
N        = N_SIDE ** 2
M        = 3       # utilisateurs [papier Example 1]
SNR_dB   = 20      # SNR de reference
K_rice   = 1e9     # K -> inf : LoS pur [papier Section 5.1]
KMAX     = 1000    # iterations max [papier Section 5]
EPS      = 1e-3    # tolerance [papier Section 5]
USE_BFGS = True

# Path loss normalises (ordre de grandeur coherent avec Fig.3)
PL_B = 0.20   # users -> RIS atténuation du canal
PL_C = 0.50   # RIS   -> BS

# Armijo [papier Section 5]
MU0  = 1.0# pas initial
BETA = 0.5 # facteur de reduction du pas
NU   = 1e-2 #constante armijo
# ==============================================================================

np.random.seed(SEED)
sigma2 = 1.0
Pm     = np.full(M, 10 ** (SNR_dB / 10) / M)

# ==============================================================================
#  GEOMETRIE [papier Section 5 — positions en metres]
# ==============================================================================
lam   = 0.030   # longueur d'onde 30 mm
d_ant = lam / 2

# BS : ULA sur axe x, z=15 m
bs_pos = np.array([[i * d_ant, 0.0, 15.0] for i in range(L)])  # (L,3)

# RIS : grille N_SIDE x N_SIDE, y=60 m, z=15 m
ris_pos = np.array(
    [[ix * d_ant, 60.0, 15.0 + iz * d_ant]
     for iz in range(N_SIDE) for ix in range(N_SIDE)]
)  # (N,3)

# Utilisateurs [papier Example 1]
user_pos = np.array([
    [ 30.0, 50.0,  5.0],
    [  0.0, 50.0,  0.0],
    [-20.0, 55.0,  0.0],
])  # (M,3)

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
    return H

# ==============================================================================
#  CANAUX DU SYSTEME
# ==============================================================================
# Example 1 : lien direct bloque [Section 5.1]
A = np.zeros((L, M), dtype=complex)

# B : users -> RIS  (N x M)
B = rician_geo(ris_pos, user_pos, K_rice, PL_B)

# C : RIS -> BS     (L x N)
C = rician_geo(bs_pos, ris_pos, K_rice, PL_C)

# ── Canal effectif [eq. 6] ────────────────────────────────────────────────────
def get_H(theta):
    """H = A + C * diag(exp(j*theta)) * B   shape (L, M)"""
    return A + C @ (np.exp(1j * theta)[:, None] * B)

# ── Fonction cout [eq. 35] ────────────────────────────────────────────────────
def cost(theta):
    H      = get_H(theta)
    norms2 = np.sum(np.abs(H) ** 2, axis=0)          # (M,)
    trR    = float(Pm @ norms2) + L * sigma2
    ratios = np.clip(Pm * norms2 / trR, 0.0, 1 - 1e-12)
    return float(np.sum(np.log2(1.0 - ratios)))

def sum_rate(theta):
    return -cost(theta)

# ── Gradient analytique [eq. 41] ─────────────────────────────────────────────
def gradient(theta):
    """Gradient exact de J [eq. 41] — sans inversion matricielle.

    Derivation depuis J = sum_m log2(1 - Pm*||hm||^2/tr(R)) :

      d(||hm||^2)/dtheta_n = -2 Im[ Tv_n * B[n,m] * conj((C^H hm)_n) ]
      d(tr(R))/dtheta_n    = -2 Im[ Tv_n * conj((C^H Q)_{n,n}) ]

    avec Q = sum_m Pm * hm * bm^H  [eq. 29], shape (L, N)

    => grad_n = (2/ln2) sum_m Pm/(tr(R)-Pm||hm||^2)
                * Im[ Tv_n * (B[n,m]*conj(CHhm_n) - ||hm||^2/tr(R)*conj(CHQ_nn)) ]
    """
    H      = get_H(theta)                              # (L, M)
    norms2 = np.sum(np.abs(H) ** 2, axis=0)           # ||hm||², shape (M,)
    trR    = float(Pm @ norms2) + L * sigma2           # scalaire

    Tv = np.exp(1j * theta)                            # (N,)

    # Q [eq. 29] : Q[l,n] = sum_m Pm * hm[l] * conj(B[n,m])
    Q = (H * Pm[None, :]) @ B.conj().T                # (L, N)

    # Diagonale de C^H Q : (C^H Q)_{n,n}  shape (N,)
    CHQ_diag = np.diag(C.conj().T @ Q)                # (N,)  — sans Tv

    grad = np.zeros(N)
    for m in range(M):
        hm    = H[:, m]                                # (L,)
        denom = trR - Pm[m] * norms2[m]               # scalaire > 0

        # (C^H hm)_n = sum_l conj(C[l,n]) * hm[l]
        CHhm = C.conj().T @ hm                         # (N,)

        # terme interieur: Tv_n * [B[n,m]*conj(CHhm_n) - ||hm||^2/trR*conj(CHQ_nn)]
        inner = Tv * (B[:, m] * np.conj(CHhm)
                      - (norms2[m] / trR) * np.conj(CHQ_diag))

        grad += Pm[m] / denom * np.imag(inner)

    return (2.0 / np.log(2.0)) * grad

# ── Armijo backtracking [Algorithm 2] ────────────────────────────────────────
def armijo(theta, direction, grad, J0,
           mu0=MU0, beta=BETA, nu=NU, max_ls=50):
    mu = mu0
    gd = float(grad @ direction)
    for _ in range(max_ls):
        if cost(theta + mu * direction) - J0 <= nu * mu * gd:
            return mu
        mu *= beta
    return mu

# ── BFGS amorti [Algorithm 3 du papier] ──────────────────────────────────────
def bfgs_update(D, mu, d, g_old, g_new):
    """
    Algorithm 3 :
      xi  = 1  si  d^T g_new <= 0.8 * d^T g_old
            0.8*(d^T g_old)/(d^T g_new)  sinon
      g(k) = xi*(g_new - g_old) - mu*(1-xi)*g_old
      tau  = mu * g(k)^T * d
      D(k+1) = [I - tau*mu*d*g^T] D [I - tau*mu*g*d^T] + tau*mu^2 * d*d^T
    """
    dTg_old = float(d @ g_old)
    dTg_new = float(d @ g_new)

    if dTg_new <= 0.8 * dTg_old:
        xi = 1.0
    else:
        xi = 0.8 * dTg_old / dTg_new if abs(dTg_new) > 1e-12 else 1.0

    gk  = xi * (g_new - g_old) - mu * (1.0 - xi) * g_old
    tau = mu * float(gk @ d)

    if abs(tau) < 1e-14:
        return D

    I     = np.eye(N)
    tmp   = I - tau * mu * np.outer(d, gk)
    return tmp @ D @ tmp.T + tau * mu ** 2 * np.outer(d, d)

# ── Boucle d'optimisation ─────────────────────────────────────────────────────
def optimize():
    theta = np.zeros(N)
    D     = np.eye(N)

    SR_hist   = []
    grad_hist = []

    SR0  = sum_rate(theta)
    g0   = gradient(theta)
    eps0 = np.linalg.norm(g0)

    label = "BFGS amorti" if USE_BFGS else "Steepest Descent"
    print(f"\n{'='*60}")
    print(f"  RIS Optimisation — {label}")
    print(f"  L={L} ant. | N={N} ({N_SIDE}x{N_SIDE}) | M={M} | SNR={SNR_dB} dB")
    print(f"{'='*60}")
    print(f"  Debit initial : {SR0:.3f} bps/Hz\n")

    converged = False
    for k in range(KMAX):
        g     = gradient(theta)
        gnorm = np.linalg.norm(g)
        SR    = sum_rate(theta)
        SR_hist.append(SR)
        grad_hist.append(gnorm)

        if k % 50 == 0 or gnorm / eps0 <= EPS:
            print(f"  k={k:4d} | ||g||/||g0||={gnorm/eps0:.2e} | SR={SR:.4f} bps/Hz")

        if gnorm / eps0 <= EPS:
            converged = True
            break

        # Direction de descente [eq. 21]
        if USE_BFGS:
            d = -(D @ g)
            if float(g @ d) >= 0.0:
                d = -g
                D = np.eye(N)
        else:
            d = -g

        # Pas Armijo
        J0        = cost(theta)
        mu        = armijo(theta, d, g, J0)
        theta_new = theta + mu * d

        # Mise a jour BFGS
        if USE_BFGS:
            g_new = gradient(theta_new)
            D     = bfgs_update(D, mu, d, g, g_new)

        theta = theta_new

    SR_final = sum_rate(theta)
    status   = "Converge" if converged else "Iterations max atteintes"
    print(f"\n  {status}")
    print(f"  Debit final : {SR_final:.3f} bps/Hz")
    print(f"  Gain RIS    : +{SR_final - SR0:.3f} bps/Hz")
    print(f"  Iterations  : {k + 1}")
    print(f"{'='*60}\n")
    return theta, SR_hist, grad_hist, SR0, SR_final

# ── SINR MVDR [eq. 17] ───────────────────────────────────────────────────────
def sinr_mvdr(theta):
    H  = get_H(theta)
    R  = sigma2 * np.eye(L) + (H * Pm[None, :]) @ H.conj().T
    Ri = np.linalg.inv(R)
    sinrs = []
    for m in range(M):
        hm    = H[:, m]
        q     = float(np.real(hm.conj() @ Ri @ hm))
        gamma = Pm[m] / max(1.0 / q - Pm[m], 1e-12)
        sinrs.append(10 * np.log10(gamma))
    return sinrs

# ── Execution ─────────────────────────────────────────────────────────────────
theta_opt, SR_hist, grad_hist, SR0, SR_final = optimize()
sinr_0   = sinr_mvdr(np.zeros(N))
sinr_opt = sinr_mvdr(theta_opt)

# ── Figures ───────────────────────────────────────────────────────────────────
label = "BFGS amorti" if USE_BFGS else "Steepest Descent"
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(
    f"RIS — {label}  |  L={L}, N={N} ({N_SIDE}²), M={M}, SNR={SNR_dB} dB\n",
    fontsize=11, fontweight='bold'
)

ax = axes[0, 0]
ax.plot(SR_hist, 'steelblue', lw=2)
ax.axhline(SR0,      color='gray',   ls='--', label=f'Initial : {SR0:.2f} bps/Hz')
ax.axhline(SR_final, color='tomato', ls='--', label=f'Final   : {SR_final:.2f} bps/Hz')
ax.set_xlabel('Iterations'); ax.set_ylabel('Debit somme (bps/Hz)')
ax.set_title('Convergence — debit somme')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.semilogy(grad_hist, 'darkorange', lw=2)
ax.set_xlabel('Iterations'); ax.set_ylabel('||grad J||')
ax.set_title('Norme du gradient')
ax.grid(alpha=0.3, which='both')

ax = axes[1, 0]
grid = (theta_opt % (2 * np.pi)).reshape(N_SIDE, N_SIDE)
im   = ax.imshow(grid, cmap='hsv', vmin=0, vmax=2 * np.pi)
plt.colorbar(im, ax=ax, label='Phase (rad)')
ax.set_title(f'Phases RIS ({N_SIDE}×{N_SIDE})')
ax.set_xlabel('Colonne'); ax.set_ylabel('Ligne')

ax = axes[1, 1]
x = np.arange(M); w = 0.35
ax.bar(x - w/2, sinr_0,   w, label='Sans optim.', color='steelblue', alpha=0.85)
ax.bar(x + w/2, sinr_opt, w, label='RIS optimise', color='tomato',   alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f'User {m+1}' for m in range(M)])
ax.set_ylabel('SINR MVDR (dB)'); ax.set_title('SINR avant / apres')
ax.legend(); ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(exist_ok=True)
out = output_dir / "ris_results.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Figure sauvegardee -> {out}")
