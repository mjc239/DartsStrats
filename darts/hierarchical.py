"""
Seventeen players, or one player seventeen times?

Every fit in this project so far has been **per player and independent**. Notebook
21 fitted each professional separately, notebook 24 did it again for two more
families, and nothing any player's darts said was ever allowed to inform anyone
else's estimate. That is the right default -- it is the only way to find out that
players differ -- but it is not obviously the best estimator, and for a parameter
a single player's data barely constrains it is close to the worst.

The alternative is to say that players are drawn from a population:

    theta_hat_p ~ N(theta_p, s_p^2)        what one player's darts measure
    theta_p     ~ N(mu, tau^2)             what players look like

and estimate ``mu`` and ``tau`` alongside the players. ``tau`` is the interesting
one: it is how much players actually differ, *after* subtracting the noise in
measuring them. At ``tau = 0`` everyone is the same and the right estimate for
each player is the grand mean; at ``tau -> infinity`` nothing is shared and the
right estimate is their own. In between each player gets

    theta_tilde_p = (theta_hat_p / s_p^2 + mu / tau^2) / (1 / s_p^2 + 1 / tau^2)

which is their own estimate pulled toward the population by an amount set by how
badly measured they were. A player with 1,300 visits barely moves; a player whose
data says almost nothing about a parameter moves almost all the way.

Three things this module keeps separate, because conflating them is the usual way
this goes wrong:

* **How well a player is measured** (``s_p``) comes from the observed information
  -- the curvature of that player's own likelihood -- computed by finite
  differences and checked at two step sizes, because a Hessian that depends on the
  step is not a Hessian.
* **How much players differ** (``tau``) is estimated by REML, which unlike the
  method-of-moments version does not have to be told the answer is positive.
* **Whether they differ at all** is a separate question with its own test
  (:func:`heterogeneity`), and it is worth asking first: if `I^2` is near zero
  the shrinkage will be total and the "hierarchical model" is just pooling.

Whether any of it is an improvement is not something this module asserts. It is
measured on held-out legs, the same split everything else in the project uses,
against both of the things it sits between: no pooling and complete pooling.
"""

import numpy as np


def observed_information(log_likelihood, theta, rel_step=1e-3, min_step=1e-4,
                         free=None):
    """
    The Hessian of the negative log-likelihood at ``theta``, by central
    differences.

    The parameters are on wildly different scales -- a log scale near 1.8, a bias
    near 0, an ellipse coordinate near 0.1, a logit near -3 -- so the step is
    relative with an absolute floor rather than one number for all of them.

    Args:
        log_likelihood (callable): maps a parameter vector to a log-likelihood.
        theta (array): where to evaluate, normally the maximum.
        rel_step, min_step (float): step is ``max(rel_step * |theta_i|, min_step)``.
        free (sequence): indices to differentiate, the rest held fixed. Use this
            when a parameter has run to a constraint: the likelihood is then flat
            in its direction, its numerical curvature is noise, and it drags the
            whole matrix down with it. Holding it where the constraint put it
            gives the others an honest *conditional* standard error, which is
            what they are.

    Returns:
        np.ndarray: (k, k) observed information over the free coordinates,
        symmetric.
    """
    theta = np.asarray(theta, float)
    if free is not None:
        free = np.asarray(free, int)
        full = theta.copy()

        def sub_ll(x):
            full[free] = x
            return log_likelihood(full)
        return observed_information(sub_ll, theta[free], rel_step, min_step)
    k = len(theta)
    h = np.maximum(rel_step * np.abs(theta), min_step)
    f0 = log_likelihood(theta)
    H = np.zeros((k, k))

    for i in range(k):
        e = np.zeros(k)
        e[i] = h[i]
        H[i, i] = (log_likelihood(theta + e) - 2 * f0
                   + log_likelihood(theta - e)) / h[i] ** 2
    for i in range(k):
        for j in range(i + 1, k):
            ei, ej = np.zeros(k), np.zeros(k)
            ei[i], ej[j] = h[i], h[j]
            H[i, j] = H[j, i] = (
                log_likelihood(theta + ei + ej) - log_likelihood(theta + ei - ej)
                - log_likelihood(theta - ei + ej) + log_likelihood(theta - ei - ej)
            ) / (4 * h[i] * h[j])
    return -0.5 * (H + H.T)          # information is minus the Hessian of logL


def parameter_covariance(log_likelihood, theta, check_step=True, free=None,
                         rcond=1e-10):
    """
    Asymptotic covariance of the estimate, and whether it can be trusted.

    Returns ``None`` for the covariance when the information matrix cannot be
    inverted meaningfully, which is the honest answer when a parameter is not
    identified rather than a large standard error that looks like a measurement.

    "Cannot be inverted meaningfully" is a condition-number test, not a sign
    test. A matrix whose smallest eigenvalue is positive but a billionth of its
    largest is positive definite and still numerically singular -- checking only
    the sign let one through and it reached ``inv`` as a crash.

    Returns:
        dict: ``cov`` (k, k) or None, ``se`` (k,) or None, ``pd`` bool,
        ``step_sensitivity`` -- the largest relative change in a standard error
        when the finite-difference step is increased tenfold. Anything above a
        per cent means the curvature is being read off numerical noise.
    """
    def invert(matrix):
        w = np.linalg.eigvalsh(matrix)
        if w.min() <= 0 or w.min() < rcond * w.max():
            return None
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return None

    info = observed_information(log_likelihood, theta, free=free)
    cov = invert(info)
    out = {"pd": cov is not None, "step_sensitivity": np.nan}
    if cov is None:
        out["cov"] = out["se"] = None
        return out
    out["cov"], out["se"] = cov, np.sqrt(np.abs(np.diag(cov)))
    if check_step:
        coarse = invert(observed_information(log_likelihood, theta, rel_step=1e-2,
                                             min_step=1e-3, free=free))
        if coarse is not None:
            se2 = np.sqrt(np.abs(np.diag(coarse)))
            out["step_sensitivity"] = float(
                np.max(np.abs(se2 - out["se"]) / np.maximum(out["se"], 1e-300)))
    return out


def heterogeneity(estimates, ses):
    """
    Do these players differ at all, beyond the noise in measuring them?

    Cochran's ``Q`` against a chi-square, and ``I^2``, the share of the observed
    spread that is not measurement error. This is worth running before any
    shrinkage: at ``I^2 = 0`` the hierarchical model has nothing to model.
    """
    from scipy import stats

    est, se = np.asarray(estimates, float), np.asarray(ses, float)
    w = 1.0 / se ** 2
    mean = float((w * est).sum() / w.sum())
    Q = float((w * (est - mean) ** 2).sum())
    df = len(est) - 1
    return {"Q": Q, "df": df, "p": float(stats.chi2.sf(Q, df)),
            "I2": float(max(0.0, (Q - df) / Q)) if Q > 0 else 0.0,
            "fixed_effect": mean}


def random_effects(estimates, ses, max_iter=500, tol=1e-14):
    """
    REML estimate of the population mean and spread.

    The method-of-moments estimator of ``tau^2`` can come out negative and is
    then set to zero, which hides how uncertain it was. REML iterates

        tau^2 <- sum w_p^2 [ (theta_p - mu)^2 + 1/sum(w) - s_p^2 ] / sum w_p^2

    with ``w_p = 1 / (s_p^2 + tau^2)``, to a fixed point.

    Returns:
        dict: ``mu``, ``se_mu``, ``tau``, and ``n``.
    """
    est, se = np.asarray(estimates, float), np.asarray(ses, float)
    tau2 = max(float(np.var(est, ddof=1) - np.mean(se ** 2)), 0.0)
    mu = float(est.mean())
    for _ in range(max_iter):
        w = 1.0 / (se ** 2 + tau2)
        mu = float((w * est).sum() / w.sum())
        num = (w ** 2 * ((est - mu) ** 2 + 1.0 / w.sum() - se ** 2)).sum()
        new = max(float(num / (w ** 2).sum()), 0.0)
        if abs(new - tau2) <= tol * max(1.0, tau2):
            tau2 = new
            break
        tau2 = new
    w = 1.0 / (se ** 2 + tau2)
    return {"mu": mu, "se_mu": float(np.sqrt(1.0 / w.sum())),
            "tau": float(np.sqrt(tau2)), "n": len(est)}


def shrink(estimates, ses, mu, tau):
    """
    Each player's own estimate, pulled toward the population.

    Also returns ``weight``, the share of the shrunk estimate that is still the
    player's own data -- 1 means they were measured well enough to keep, 0 means
    the population knows more about them than their own darts do.
    """
    est, se = np.asarray(estimates, float), np.asarray(ses, float)
    tau2 = float(tau) ** 2
    if tau2 <= 0:
        w = np.zeros_like(est)
    else:
        w = (1.0 / se ** 2) / (1.0 / se ** 2 + 1.0 / tau2)
    return {"posterior_mean": w * est + (1 - w) * mu,
            "posterior_sd": np.sqrt(1.0 / (1.0 / se ** 2 + 1.0 / max(tau2, 1e-300))),
            "weight": w}


def delta_interval(fn, theta, cov, n_draw=20000, seed=0, quantiles=(2.5, 97.5)):
    """
    An interval for a nonlinear function of the parameters, by sampling.

    The delta method would linearise, which is wrong for anything bounded --
    ``rho`` lives in (-1, 1) and its estimate is often near an edge of what the
    data can distinguish. Drawing from the asymptotic normal in the model's own
    unbounded coordinates and mapping each draw through ``fn`` keeps the bound
    without pretending to a precision the linearisation would invent.
    """
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(np.asarray(theta, float),
                                    np.asarray(cov, float), n_draw)
    vals = np.array([fn(d) for d in draws])
    lo, hi = np.percentile(vals, quantiles)
    return {"point": float(fn(np.asarray(theta, float))),
            "lo": float(lo), "hi": float(hi), "sd": float(vals.std(ddof=1)),
            "draws": vals}


def random_effects_mv(estimates, covs, max_iter=2000, tol=1e-10, ridge=1e-10):
    """
    The same model with the parameters taken together rather than one at a time.

    Shrinking each coordinate toward its own mean quietly assumes the parameters
    are independent, and in this project they are conspicuously not: across
    players the fitted core scale and ``log(nu - 2)`` correlate at +0.52, and the
    core scale and the elongation at -0.74. A player who is above the population
    mean on one is systematically above it on the others, so pulling each
    coordinate toward its own average drags the whole vector *off* the ridge the
    players actually lie on -- to a place no player is.

        theta_hat_p ~ N(theta_p, V_p),      theta_p ~ N(mu, T)

    with ``V_p`` the player's own covariance and ``T`` the population's, both
    full matrices. Fitted by EM, which for this model is two lines:

        posterior:  C_p = (V_p^-1 + T^-1)^-1,  m_p = C_p (V_p^-1 theta_hat_p + T^-1 mu)
        update:     mu = mean(m_p),  T = mean(C_p + (m_p - mu)(m_p - mu)')

    Whether the extra structure is worth its parameters is a real question -- a
    7x7 ``T`` is 28 numbers from 17 players -- and is not settled here. It is
    settled on held-out legs.

    Args:
        estimates (array): (n, k) per-unit estimates.
        covs (array): (n, k, k) per-unit covariances.

    Returns:
        dict: ``mu`` (k,), ``T`` (k, k), ``posterior_mean`` (n, k), ``n_iter``.
    """
    est = np.asarray(estimates, float)
    V = np.asarray(covs, float)
    n, k = est.shape
    eye = np.eye(k)
    mu = est.mean(axis=0)
    T = np.cov(est.T, bias=False) + ridge * eye

    for it in range(max_iter):
        Tinv = np.linalg.inv(T + ridge * eye)
        m = np.empty_like(est)
        C = np.empty_like(V)
        for p in range(n):
            Vinv = np.linalg.inv(V[p] + ridge * eye)
            C[p] = np.linalg.inv(Vinv + Tinv)
            m[p] = C[p] @ (Vinv @ est[p] + Tinv @ mu)
        new_mu = m.mean(axis=0)
        d = m - new_mu
        new_T = (C.mean(axis=0)
                 + np.einsum("pi,pj->ij", d, d) / n)
        moved = max(float(np.abs(new_mu - mu).max()),
                    float(np.abs(new_T - T).max()))
        mu, T = new_mu, 0.5 * (new_T + new_T.T)
        if moved <= tol:
            break
    return {"mu": mu, "T": T, "posterior_mean": m, "n_iter": it + 1}
