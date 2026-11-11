"""Hamilton (1989) two-state Markov regime-switching model, from scratch.

Model
-----
Returns are drawn from one of two Gaussian regimes — a calm state and a
turbulent state — and the regime itself follows a first-order Markov chain::

    r_t | s_t = i  ~  N(mu_i, sigma_i^2),     i in {0, 1}
    P(s_t = j | s_{t-1} = i) = P[i, j]

Estimation is by maximum likelihood via the EM algorithm, implemented here
from scratch: a scaled forward-backward pass (E-step) produces smoothed
state probabilities and pairwise transition counts; closed-form weighted
moments update the parameters (M-step). The Hamilton filter gives the
real-time (filtered) probability of the turbulent state — the quantity a
risk desk would actually watch — and the Viterbi algorithm recovers the
single most likely regime path.

States are reported ordered by volatility: state 0 = low-vol, state 1 =
high-vol.

Limitations: two states is a coarse description of market dynamics (crises
differ from ordinary turbulence); Gaussian emissions understate within-regime
tails; transition probabilities are constant, so regime risk does not vary
with the cycle. The model is a lens, not a forecast.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = [
    "fit",
    "filtered_probs",
    "smoothed_probs",
    "viterbi_path",
    "expected_durations",
    "simulate",
]

_LOG_2PI = np.log(2.0 * np.pi)


def _log_emission(returns: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Log Gaussian emission densities, shape (T, 2)."""
    r = returns[:, None]
    return -0.5 * (
        _LOG_2PI + 2.0 * np.log(sigma)[None, :] + ((r - mu[None, :]) / sigma[None, :]) ** 2
    )


def _forward_backward(
    returns: np.ndarray, mu: np.ndarray, sigma: np.ndarray, P: np.ndarray, pi: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Scaled forward-backward. Returns (alpha, beta, gamma, loglik).

    alpha/beta are row-normalised at every step (Rabiner scaling); the
    log-likelihood accumulates the logs of the normalising constants.
    """
    T = len(returns)
    log_b = _log_emission(returns, mu, sigma)
    b = np.exp(log_b - log_b.max(axis=1, keepdims=True))
    shift = log_b.max(axis=1)

    alpha = np.empty((T, 2))
    scale = np.empty(T)
    a = pi * b[0]
    scale[0] = a.sum()
    alpha[0] = a / scale[0]
    for t in range(1, T):
        a = (alpha[t - 1] @ P) * b[t]
        scale[t] = a.sum()
        alpha[t] = a / scale[t]
    loglik = float(np.sum(np.log(scale)) + np.sum(shift))

    beta = np.ones((T, 2))
    for t in range(T - 2, -1, -1):
        beta[t] = (P @ (b[t + 1] * beta[t + 1])) / scale[t + 1]

    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    return alpha, beta, gamma, loglik


def _xi_sum(
    returns: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    P: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Sum over t of P(s_t = i, s_{t+1} = j | all data), shape (2, 2)."""
    log_b = _log_emission(returns, mu, sigma)
    b_next = np.exp(log_b[1:] - log_b[1:].max(axis=1, keepdims=True))
    # Pairwise joints, unnormalised; the per-t shift cancels in the ratio.
    xi = (
        alpha[:-1][:, :, None]
        * P[None, :, :]
        * (b_next * beta[1:])[:, None, :]
    )
    xi /= xi.sum(axis=(1, 2), keepdims=True)
    return xi.sum(axis=0)


def _initial_params(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Data-driven start: split by |return| around its median."""
    med = np.median(np.abs(returns))
    calm = returns[np.abs(returns) <= med]
    turb = returns[np.abs(returns) > med]
    mu = np.array([calm.mean(), turb.mean()])
    sigma = np.array([max(calm.std(), 1e-4), max(turb.std(), 1e-4)])
    if sigma[1] < sigma[0]:  # guarantee state 1 starts as the high-vol state
        mu = mu[::-1]
        sigma = sigma[::-1]
    P = np.array([[0.97, 0.03], [0.05, 0.95]])
    pi = np.array([0.5, 0.5])
    return mu, sigma, P, pi


def fit(
    returns: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> dict:
    """Fit the 2-state Gaussian HMM by EM.

    Returns a dict with ``mu``, ``sigma`` (volatility-ordered), transition
    matrix ``P``, initial distribution ``pi``, final ``loglik``, the full
    ``loglik_history`` (non-decreasing by EM construction), and ``n_iter``.
    """
    returns = np.asarray(returns, dtype=float)
    mu, sigma, P, pi = _initial_params(returns)

    history: list[float] = []
    prev_ll = -np.inf
    for _ in range(max_iter):
        alpha, beta, gamma, loglik = _forward_backward(returns, mu, sigma, P, pi)
        history.append(loglik)

        # M-step
        xi = _xi_sum(returns, mu, sigma, P, alpha, beta)
        P_new = xi / xi.sum(axis=1, keepdims=True)
        w = gamma.sum(axis=0)
        mu_new = (gamma * returns[:, None]).sum(axis=0) / w
        var_new = (gamma * (returns[:, None] - mu_new[None, :]) ** 2).sum(axis=0) / w
        sigma_new = np.sqrt(np.maximum(var_new, 1e-12))
        pi_new = gamma[0]

        mu, sigma, P, pi = mu_new, sigma_new, P_new, pi_new
        if abs(loglik - prev_ll) < tol * (1.0 + abs(loglik)):
            prev_ll = loglik
            break
        prev_ll = loglik

    # Final E-step so the reported loglik matches the reported parameters.
    _, _, _, loglik = _forward_backward(returns, mu, sigma, P, pi)
    if not history or abs(history[-1] - loglik) > 1e-6:
        history.append(loglik)

    # Order states by volatility: 0 = low-vol, 1 = high-vol.
    if sigma[0] > sigma[1]:
        order = [1, 0]
        mu, sigma = mu[order], sigma[order]
        P = P[np.ix_(order, order)]
        pi = pi[order]

    return {
        "mu": mu,
        "sigma": sigma,
        "P": P,
        "pi": pi,
        "loglik": float(loglik),
        "loglik_history": history,
        "n_iter": len(history),
    }


def filtered_probs(returns: np.ndarray, params: dict) -> np.ndarray:
    """Hamilton filter: P(s_t | r_1..r_t), shape (T, 2), rows sum to 1."""
    returns = np.asarray(returns, dtype=float)
    mu, sigma, P, pi = params["mu"], params["sigma"], params["P"], params["pi"]
    log_b = _log_emission(returns, mu, sigma)
    b = np.exp(log_b - log_b.max(axis=1, keepdims=True))

    T = len(returns)
    filt = np.empty((T, 2))
    f = pi * b[0]
    filt[0] = f / f.sum()
    for t in range(1, T):
        f = (filt[t - 1] @ P) * b[t]
        filt[t] = f / f.sum()
    return filt


def smoothed_probs(returns: np.ndarray, params: dict) -> np.ndarray:
    """P(s_t | all data) via forward-backward, shape (T, 2)."""
    returns = np.asarray(returns, dtype=float)
    _, _, gamma, _ = _forward_backward(
        returns, params["mu"], params["sigma"], params["P"], params["pi"]
    )
    return gamma


def viterbi_path(returns: np.ndarray, params: dict) -> np.ndarray:
    """Most likely joint regime path (log-space Viterbi), shape (T,)."""
    returns = np.asarray(returns, dtype=float)
    mu, sigma, P, pi = params["mu"], params["sigma"], params["P"], params["pi"]
    log_b = _log_emission(returns, mu, sigma)
    log_P = np.log(np.maximum(P, 1e-300))
    log_pi = np.log(np.maximum(pi, 1e-300))

    T = len(returns)
    delta = np.empty((T, 2))
    psi = np.zeros((T, 2), dtype=int)
    delta[0] = log_pi + log_b[0]
    for t in range(1, T):
        cand = delta[t - 1][:, None] + log_P  # cand[i, j]
        psi[t] = np.argmax(cand, axis=0)
        delta[t] = cand[psi[t], np.arange(2)] + log_b[t]

    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


def expected_durations(params: dict) -> np.ndarray:
    """Expected duration (in observations) of each regime: 1 / (1 - P_ii)."""
    P = params["P"]
    return 1.0 / (1.0 - np.diag(P))


def simulate(
    n: int,
    mu: np.ndarray,
    sigma: np.ndarray,
    P: np.ndarray,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate (returns, states) from a known 2-regime process (for tests)."""
    rng = np.random.default_rng(seed)
    states = np.empty(n, dtype=int)
    states[0] = 0
    u = rng.random(n)
    for t in range(1, n):
        states[t] = 0 if u[t] < P[states[t - 1], 0] else 1
    returns = rng.normal(mu[states], sigma[states])
    return returns, states


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rets, _ = simulate(
        2000,
        mu=np.array([0.0005, -0.001]),
        sigma=np.array([0.006, 0.02]),
        P=np.array([[0.97, 0.03], [0.05, 0.95]]),
        seed=3,
    )
    out = fit(rets)
    print(
        "regimes: OK",
        "mu=", np.round(out["mu"], 5),
        "sigma=", np.round(out["sigma"], 5),
        "loglik=", round(out["loglik"], 2),
    )
    assert out["sigma"][1] > out["sigma"][0]
