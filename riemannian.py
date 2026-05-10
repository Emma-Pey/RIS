import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# PARAMÈTRES
# =========================================================
M = 16
N = 2
K = 2

sigma2 = 0.05
alpha = 2.2

omega = np.ones(K)

# =========================================================
# INITIALISATION
# =========================================================
W = (
    np.random.randn(M, K)
    + 1j * np.random.randn(M, K)
) / np.sqrt(2)

W = W / np.linalg.norm(W)

theta = np.exp(1j * 2 * np.pi * np.random.rand(N))

# =========================================================
# MODÈLE DE CANAL
# =========================================================
def path_loss(d):
    return d ** (-alpha)


def rayleigh(shape):
    return (
        np.random.randn(*shape)
        + 1j * np.random.randn(*shape)
    ) / np.sqrt(2)


def channel(shape, d):
    return np.sqrt(path_loss(d)) * rayleigh(shape)

# =========================================================
# DISTANCES
# =========================================================
d_ap_user = 30
d_ris_user = 30
d_ap_ris = 30

# =========================================================
# CANAUX
# =========================================================
hd = []
Hr = []

# AP -> RIS
G = channel((N, M), d_ap_ris)

# matrice RIS
phi = 2 * np.pi * np.random.rand(N)

Theta = np.diag(np.exp(1j * phi))

for k in range(K):

    hd_k = channel((M,), d_ap_user)

    hr_k = channel((N,), d_ris_user)

    Hr_k = np.diag(np.conj(hr_k)) @ Theta @ G

    hd.append(hd_k)

    Hr.append(Hr_k)

# =========================================================
# CANAL ÉQUIVALENT
# =========================================================
def compute_hk(hd, Hr, theta):

    hk = []

    for k in range(K):

        h = hd[k] + Hr[k].conj().T @ theta

        hk.append(h)

    return hk

# =========================================================
# PARAMÈTRE PUISSANCE
# =========================================================
PT = 5

# =========================================================
# UPDATE W
# =========================================================
def update_W(hk, W):

    chi = np.zeros(K, dtype=complex)

    kappa = np.zeros(K, dtype=complex)

    # =========================
    # calcul chi
    # =========================
    for k in range(K):

        num = hk[k].conj() @ W[:, k]

        denom = 0

        for i in range(K):

            denom += np.abs(
                hk[k].conj() @ W[:, i]
            )**2

        denom += sigma2

        chi[k] = num / denom

    # =========================
    # calcul kappa
    # =========================
    for k in range(K):

        val = hk[k].conj() @ W[:, k]

        kappa[k] = 1 / (
            1 - np.conj(chi[k]) * val
        )

    # =========================
    # matrice A0
    # =========================
    A0 = np.zeros((M, M), dtype=complex)

    for i in range(K):

        hi = hk[i].reshape(-1,1)

        A0 += (
            omega[i]
            * np.abs(chi[i])**2
            * kappa[i]
            * (hi @ hi.conj().T)
        )

    # =========================
    # fonction puissance
    # =========================
    def compute_power(lambda_):

        eps = 1e-8

        A = (
            A0
            + (lambda_ + eps) * np.eye(M)
        )

        A_inv = np.linalg.inv(A)

        W_tmp = np.zeros(
            (M, K),
            dtype=complex
        )

        for k in range(K):

            W_tmp[:, k] = (
                omega[k]
                * chi[k]
                * kappa[k]
                * (A_inv @ hk[k])
            )

        return np.sum(
            np.linalg.norm(
                W_tmp,
                axis=0
            )**2
        )

    # =========================
    # recherche lambda
    # =========================
    if compute_power(0) <= PT:

        lambda_ = 0

    else:

        lambda_min = 0

        lambda_max = 1

        while (
            compute_power(lambda_max)
            > PT
        ):

            lambda_max *= 2

        for _ in range(30):

            lambda_ = (
                lambda_min
                + lambda_max
            ) / 2

            power = compute_power(
                lambda_
            )

            if power > PT:

                lambda_min = lambda_

            else:

                lambda_max = lambda_

    # =========================
    # update W final
    # =========================
    eps = 1e-8

    A = (
        A0
        + (lambda_ + eps) * np.eye(M)
    )

    A_inv = np.linalg.inv(A)

    W_new = np.zeros(
        (M, K),
        dtype=complex
    )

    for k in range(K):

        W_new[:, k] = (
            omega[k]
            * chi[k]
            * kappa[k]
            * (A_inv @ hk[k])
        )

    return W_new
# =========================================================
# CALCUL a ET b
# =========================================================
def compute_a_b(Hr, hd, W):

    a = {}

    b = {}

    for i in range(K):

        for k in range(K):

            a[(i,k)] = Hr[k] @ W[:, i]

            b[(i,k)] = (
                hd[k].conj() @ W[:, i]
            )

    return a, b

# =========================================================
# CALCUL Ak
# =========================================================
def compute_Ak(theta, a, b):

    A_list = []

    for k in range(K):

        # =========================
        # num1
        # =========================
        num1 = 0

        for i in range(K):

            ai = a[(i,k)]

            bi = b[(i,k)]

            num1 += (
                ai * (ai.conj().T @ theta)
                + ai * np.conj(bi)
            )

        # =========================
        # den1
        # =========================
        den1 = 0

        for i in range(K):

            val = (
                theta.conj().T @ a[(i,k)]
                + b[(i,k)]
            )

            den1 += np.abs(val)**2

        den1 += sigma2

        # =========================
        # num2
        # =========================
        num2 = 0

        for i in range(K):

            if i != k:

                ai = a[(i,k)]

                bi = b[(i,k)]

                num2 += (
                    ai * (ai.conj().T @ theta)
                    + ai * np.conj(bi)
                )

        # =========================
        # den2
        # =========================
        den2 = 0

        for i in range(K):

            if i != k:

                val = (
                    theta.conj().T @ a[(i,k)]
                    + b[(i,k)]
                )

                den2 += np.abs(val)**2

        den2 += sigma2

        Ak = num1 / den1 - num2 / den2

        A_list.append(Ak)

    return A_list

# =========================================================
# GRADIENT EUCLIDIEN
# =========================================================
def compute_gradient(theta, a, b):

    A_list = compute_Ak(theta, a, b)

    grad = np.zeros_like(
        theta,
        dtype=complex
    )

    for k in range(K):

        grad += 2 * omega[k] * A_list[k]

    return grad

# =========================================================
# GRADIENT RIEMANNIEN
# =========================================================
def grad_fC(theta, grad_euclid):

    return (
        grad_euclid
        - np.real(
            grad_euclid * np.conj(theta)
        ) * theta
    )

# =========================================================
# TRANSPORT
# =========================================================
def T(d_bar, theta):

    return (
        d_bar
        - np.real(
            d_bar * np.conj(theta)
        ) * theta
    )

# =========================================================
# DIRECTION
# =========================================================
def compute_d(
    grad_fC_val,
    d_bar,
    tau1,
    theta
):

    if d_bar is None:

        return -grad_fC_val

    else:

        return (
            -grad_fC_val
            + tau1 * T(d_bar, theta)
        )

# =========================================================
# RETRACTION
# =========================================================
def update_theta(theta, d, tau2):

    theta_new = theta + tau2 * d

    return theta_new / np.abs(theta_new)

# =========================================================
# WSR
# =========================================================
def compute_wsr(hk, W):

    wsr = 0

    for k in range(K):

        hk_H = hk[k].conj()

        signal = np.abs(
            hk_H @ W[:, k]
        )**2

        interference = 0

        for i in range(K):

            if i != k:

                interference += np.abs(
                    hk_H @ W[:, i]
                )**2

        sinr = signal / (
            interference + sigma2
        )

        wsr += omega[k] * np.log(1 + sinr)

    return np.real(wsr)

# =========================================================
# RCG CLASSIQUE
# =========================================================
def rcg_iteration(
    theta,
    a,
    b,
    d_bar,
    tau1=0.5,
    tau2=1e-3
):

    grad_euclid = compute_gradient(
        theta,
        a,
        b
    )

    grad_riem = grad_fC(
        theta,
        grad_euclid
    )

    if d_bar is None:

        d = -grad_riem

    else:

        d = (
            -grad_riem
            + tau1 * T(d_bar, theta)
        )

    theta_new = update_theta(
        theta,
        d,
        tau2
    )

    return theta_new, d

# =========================================================
# TAU2 ARMIJO
# =========================================================
def compute_tau2(
    theta,
    d,
    grad,
    f,
    tau_init=1
):

    tau = tau_init

    rho = 0.5

    c = 1e-4

    if np.real(np.vdot(grad, d)) >= 0:

        d = -grad

    f_theta = f(theta)

    for _ in range(20):

        theta_new = update_theta(
            theta,
            d,
            tau
        )

        if (
            f(theta_new)
            <= f_theta
            + c * tau * np.real(np.vdot(grad, d))
        ):

            return tau

        tau *= rho

    return tau

# =========================================================
# RCG ARMIJO
# =========================================================
def rcg_iteration2(
    theta,
    a,
    b,
    d_bar,
    f,
    tau1=0.5
):

    grad_euclid = compute_gradient(
        theta,
        a,
        b
    )

    grad = grad_fC(
        theta,
        grad_euclid
    )

    if d_bar is None:

        d = -grad

    else:

        d = (
            -grad
            + tau1 * T(d_bar, theta)
        )

    tau2 = compute_tau2(
        theta,
        d,
        grad,
        f
    )

    theta_new = update_theta(
        theta,
        d,
        tau2
    )

    return theta_new, d

# =========================================================
# COMPARAISON
# =========================================================
n_iter = 40

theta_init = theta.copy()

W_init = W.copy()

# =========================================================
# VERSION CONSTANTE
# =========================================================
theta1 = theta_init.copy()

W1 = W_init.copy()

wsr_const = []

d_bar1 = None

for it in range(n_iter):

    hk = compute_hk(
        hd,
        Hr,
        theta1
    )

    W1 = update_W(hk, W1)

    a, b = compute_a_b(
        Hr,
        hd,
        W1
    )

    theta1, d_bar1 = rcg_iteration(
        theta1,
        a,
        b,
        d_bar1
    )

    hk = compute_hk(
        hd,
        Hr,
        theta1
    )

    wsr = compute_wsr(hk, W1)

    wsr_const.append(wsr)

    print(
        f"[CONST] Iter {it:02d} | "
        f"WSR = {wsr:.4f}"
    )

# =========================================================
# VERSION ARMIJO
# =========================================================
theta2 = theta_init.copy()

W2 = W_init.copy()

wsr_armijo = []

d_bar2 = None

for it in range(n_iter):

    hk = compute_hk(
        hd,
        Hr,
        theta2
    )

    W2 = update_W(hk, W2)

    a, b = compute_a_b(
        Hr,
        hd,
        W2
    )

    def f_local(theta_local):

        hk_local = compute_hk(
            hd,
            Hr,
            theta_local
        )

        return -compute_wsr(
            hk_local,
            W2
        )

    theta2, d_bar2 = rcg_iteration2(
        theta2,
        a,
        b,
        d_bar2,
        f_local
    )

    hk = compute_hk(
        hd,
        Hr,
        theta2
    )

    wsr = compute_wsr(hk, W2)

    wsr_armijo.append(wsr)

    print(
        f"[ARMIJO] Iter {it:02d} | "
        f"WSR = {wsr:.4f}"
    )

# =========================================================
# PLOT
# =========================================================
plt.figure()

plt.plot(
    wsr_const,
    marker='o'
)


plt.xlabel("Itération")

plt.ylabel("WSR")

plt.title(
    "Convergence algorithme gradient"
)

plt.legend()

plt.grid()

plt.show()


