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


def _pack_population(mu, T):
    """
    Pack ``(mu, T)`` into one unconstrained vector, ``T`` as a log-Cholesky.

    The acceleration below extrapolates in this space, and an extrapolated step
    is under no obligation to land on a positive definite matrix. Carrying ``T``
    as ``L L'`` with ``log`` on the diagonal makes that impossible to get wrong:
    *every* vector unpacks to a covariance, so the jump can be as wild as it
    likes and the population it names is still a population.
    """
    mu = np.asarray(mu, float)
    k = len(mu)
    L = np.linalg.cholesky(np.asarray(T, float))
    return np.concatenate([mu, np.log(np.diag(L)), L[np.tril_indices(k, -1)]])


def _unpack_population(v, k):
    """Inverse of :func:`_pack_population`."""
    L = np.zeros((k, k))
    L[np.diag_indices(k)] = np.exp(v[k:2 * k])
    L[np.tril_indices(k, -1)] = v[2 * k:]
    return v[:k].copy(), L @ L.T


def joint_hierarchical(log_likelihoods, theta_init, mu=None, T=None,
                       max_iter=200, tol=1e-7, inner_maxiter=400, t_ridge=1e-8,
                       inner_restarts=0, min_simplex=1e-4, accelerate=True,
                       checkpoint=None, verbose=False, pool=None):
    """
    The hierarchical model fitted properly, instead of from summaries.

    Everything above works in two stages: each unit's likelihood is collapsed to
    a maximum and a curvature, and the population is then fitted to those
    summaries as though they were data. That is standard and it is an
    approximation twice over -- it treats each unit's likelihood as Gaussian
    around its *unpenalised* maximum, and it treats the curvature there as known.

    This fits the thing itself,

        L(mu, T) = sum_p log integral exp(L_p(theta)) N(theta; mu, T) dtheta

    by EM with a Laplace approximation to the inner integral. Each iteration:

        theta*_p = argmax [ L_p(theta) + log N(theta; mu, T) ]     penalised mode
        C_p      = [ -grad^2 L_p(theta*_p) + T^-1 ]^-1             its curvature
        mu       = mean(theta*_p)
        T        = mean(C_p) + sum (theta*_p - mu)(theta*_p - mu)' / n

    Two things follow that the two-stage version cannot do.

    The mode moves. ``theta*_p`` is where the unit's likelihood and the population
    agree, not where the unit's likelihood peaks, so a unit whose own data is
    weak is *fitted* differently rather than merely averaged afterwards.

    And a unit sitting on a constraint stops being a problem. The Hessian at the
    penalised mode is ``-grad^2 L_p + T^-1``, which is positive definite whenever
    ``T`` is -- so the five players whose ``nu`` ran to the ``nu > 2`` clip, and
    whose observed information was unusable, are ordinary here. The population
    pulls them off the boundary instead of the boundary poisoning them.

    Args:
        log_likelihoods (sequence): one callable per unit, mapping a parameter
            vector to a log-likelihood.
        theta_init (array): (n, k) starting parameters, normally the per-unit
            maximum likelihood fits.
        mu, T: starting population; defaults to the mean and covariance of
            ``theta_init``.
        max_iter, tol: outer iterations, and the relative change in the Laplace
            marginal log-likelihood to stop at. Plain EM here crawls -- on a
            synthetic case with a known answer it needed **800** iterations to
            settle ``T`` to five figures, moving by less than 0.01 log units a
            step for the last seven hundred of them -- so a loose tolerance on
            this quantity does not mean the parameters have arrived. That is why
            the default is tight and the acceleration is on.
        inner_maxiter (int): cap on each unit's inner optimisation, which is warm
            started from the previous iteration and so has little to do.
        inner_restarts (int): times to restart Nelder-Mead from its own answer.
            On a real seven-parameter fit the first pass exhausts
            ``inner_maxiter`` rather than meeting its tolerance -- the same
            answer comes back for ``xatol`` of 1e-5 and 1e-8 -- so the restart is
            not insurance, it is the second half of the optimisation.
        min_simplex (float): floor on the initial simplex, which is otherwise
            sized to how far each unit's mode moved on the previous iteration.
            Scipy's default is 5% of each coordinate, which once the fit is warm
            is a hundred times further than the mode is going to move.
        t_ridge (float): added to the diagonal of ``T`` for conditioning.
        accelerate (bool): SQUAREM (:func:`darts.fitting._squarem`). The
            extrapolation is safeguarded on the marginal, so it cannot make
            things worse than the two plain steps it replaces.
        checkpoint (callable): called as ``(mu, T, theta, marginal)`` whenever
            the fit reaches a better population than it has seen, so that a run
            interrupted after hours can be resumed rather than repeated.
        pool: something with ``.map``, to run the units in parallel. The work
            item is a closure over the unit's likelihood, so a process pool needs
            to be able to pickle one.

    Returns:
        dict: ``mu``, ``T``, ``theta`` (n, k) the penalised modes, ``cov``
        (n, k, k), ``marginal`` the Laplace log-likelihood, ``history``,
        ``n_iter``, ``n_em_steps``, ``converged``.
    """
    from scipy import optimize

    from .fitting import _squarem

    theta0 = np.array(theta_init, float)
    n, k = theta0.shape
    eye = np.eye(k)
    if mu is None:
        mu = theta0.mean(axis=0)
    if T is None:
        T = np.cov(theta0.T, bias=False) + t_ridge * eye

    warm = {"theta": theta0, "step": None}
    cache = {}
    n_e = [0]
    best = [-np.inf]

    def e_step(v):
        """Penalised modes and curvatures under the population ``v`` names."""
        key = v.tobytes()
        if key in cache:
            return cache[key]
        mu_v, T_v = _unpack_population(v, k)
        T_v = T_v + t_ridge * eye
        Tinv = np.linalg.inv(T_v)
        logdet = np.linalg.slogdet(T_v)[1]
        if warm["step"] is None:
            # nothing measured yet, so start from the scale the *population*
            # says a mode could plausibly be away from where it is. This is also
            # what makes stopping and resuming cheap: a resumed fit gets the
            # right simplex on its first pass instead of paying for a cold one.
            warm["step"] = np.tile(np.maximum(0.05 * np.sqrt(np.diag(T_v)),
                                    min_simplex), (n, 1))

        def one(args):
            p, start, step = args
            ll = log_likelihoods[p]

            def penalised(x):
                d = x - mu_v
                return -(ll(x) - 0.5 * float(d @ Tinv @ d))

            # fatol is absolute, so it has to be set against the size of the
            # likelihood being maximised. 1e-10 on a real fit worth -22,700 log
            # units is a relative 5e-15 -- unreachable, so Nelder-Mead ran to
            # `maxiter` every time and the tolerance was doing nothing at all.
            opts = dict(maxiter=inner_maxiter, xatol=1e-6,
                        fatol=1e-9 * max(1.0, abs(ll(start))))
            # Nelder-Mead's default simplex is 5% of each coordinate, which after
            # the first iteration is a hundred times further than the mode is
            # going to move -- and it spends its whole budget walking back. So
            # the simplex is sized to how far this unit moved last time. It can
            # still expand, so an under-sized guess costs reflections rather than
            # the answer.
            for _ in range(inner_restarts + 1):
                res = optimize.minimize(
                    penalised, start, method="Nelder-Mead",
                    options=dict(opts, initial_simplex=np.vstack(
                        [start, start + np.diag(step)])))
                start = res.x
            mode = res.x
            step_used = np.maximum(np.abs(mode - args[1]) * 4.0, min_simplex)
            info = observed_information(lambda x: -penalised(x), mode)
            # -grad^2 (L_p + log N) is positive definite whenever T is, so this
            # inverse exists even for a unit pinned against a constraint
            w = np.linalg.eigvalsh(info)
            C = np.linalg.inv(info) if w.min() > 1e-10 * max(w.max(), 1.0) \
                else np.linalg.inv(info + 1e-6 * np.trace(info) / k * eye)
            # Laplace: h(mode) + (k/2) log 2pi - (1/2) log|H|, where h is the
            # penalised objective. The 2pi from the Gaussian prior and the 2pi
            # from the Laplace volume cancel, which is why neither appears.
            lap = -res.fun - 0.5 * logdet - 0.5 * np.linalg.slogdet(info)[1]
            return mode, C, lap, step_used

        args = [(p, warm["theta"][p], warm["step"][p]) for p in range(n)]
        out = (list(pool.map(one, args)) if pool is not None
               else [one(a) for a in args])
        got = (np.stack([o[0] for o in out]), np.stack([o[1] for o in out]),
               float(sum(o[2] for o in out)))
        warm["theta"] = got[0]
        warm["step"] = np.stack([o[3] for o in out])
        cache[key] = got
        n_e[0] += 1
        if checkpoint is not None and got[2] > best[0]:
            # the best population seen, not the latest: SQUAREM evaluates trial
            # points that it may well reject, and a fit resumed from one of
            # those would be resuming from a step that was not taken
            best[0] = got[2]
            checkpoint(mu_v, T_v, got[0], got[2])
        if verbose:
            moved = np.abs(got[0] - theta0).mean()
            print(f"   E step {n_e[0]:>3}  marginal logL {got[2]:12.3f}  "
                  f"modes moved {moved:.4f} from the per-unit fits", flush=True)
        return got

    def em_step(v):
        modes, C, _ = e_step(v)
        new_mu = modes.mean(axis=0)
        d = modes - new_mu
        new_T = C.mean(axis=0) + np.einsum("pi,pj->ij", d, d) / n
        return _pack_population(new_mu, 0.5 * (new_T + new_T.T))

    def marginal_of(v):
        return e_step(v)[2]

    v, history, converged, n_steps = _squarem(
        em_step, marginal_of, _pack_population(mu, T), tol, max_iter,
        accelerate=accelerate, verbose=False)
    if verbose:
        print(f"   {len(history) - 1} iterations, {n_steps} EM steps, "
              f"marginal logL {history[-1]:.3f}, converged {converged}",
              flush=True)

    mu, T = _unpack_population(v, k)
    modes, C, marginal = e_step(v)
    return {"mu": mu, "T": T, "theta": modes, "cov": C, "marginal": marginal,
            "history": history, "n_iter": len(history) - 1,
            "n_em_steps": n_steps, "converged": converged}
