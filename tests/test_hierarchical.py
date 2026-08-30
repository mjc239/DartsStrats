"""Per-player uncertainty, and pooling across players."""
import numpy as np
import pytest

from darts.hierarchical import (_pack_population, _unpack_population,
                                delta_interval, heterogeneity,
                                joint_hierarchical, observed_information,
                                parameter_covariance, random_effects,
                                random_effects_mv, shrink)


def _gaussian_loglik(mean, cov):
    """A log-likelihood whose Hessian is known exactly: -inv(cov)."""
    inv = np.linalg.inv(cov)

    def ll(theta):
        d = np.asarray(theta, float) - mean
        return -0.5 * float(d @ inv @ d)
    return ll


def test_observed_information_matches_an_exact_hessian():
    """
    Finite differences against a case where the answer is known in closed form.
    A quadratic log-likelihood has constant curvature, so any disagreement is the
    differencing and nothing else.
    """
    mean = np.array([1.8, 0.0, 0.12, -3.0, -1.0])
    A = np.diag([4.0, 25.0, 900.0, 2.0, 3.0])
    A[0, 2] = A[2, 0] = 8.0
    cov = np.linalg.inv(A)
    info = observed_information(_gaussian_loglik(mean, cov), mean)
    assert info == pytest.approx(A, rel=1e-5, abs=1e-6)
    assert np.allclose(info, info.T)


def test_the_covariance_reports_when_it_cannot_be_trusted():
    """
    A flat direction is not a large standard error, it is no standard error. The
    honest return is None, because a number there would look like a measurement.
    """
    mean = np.zeros(3)
    cov_ok = np.diag([0.25, 0.25, 0.25])
    out = parameter_covariance(_gaussian_loglik(mean, cov_ok), mean)
    assert out["pd"]
    assert out["se"] == pytest.approx(np.full(3, 0.5), rel=1e-4)
    assert out["step_sensitivity"] < 1e-3

    # a likelihood that does not curve in the third direction at all
    def flat(theta):
        d = np.asarray(theta, float)
        return -0.5 * (4 * d[0] ** 2 + 4 * d[1] ** 2)
    out = parameter_covariance(flat, mean)
    assert not out["pd"]
    assert out["cov"] is None and out["se"] is None


@pytest.mark.parametrize("true_tau", [0.0, 0.3, 1.0])
def test_reml_recovers_how_much_players_differ(true_tau):
    """
    The whole hierarchical model turns on tau, so it has to be recovered from
    data where tau is known -- including tau = 0, the case a method-of-moments
    estimator handles by clipping a negative number to zero.

    Averaged over replicates, not asserted on one. With seventeen players a
    single tau-hat carries enormous sampling error -- an earlier version of this
    test asserted on one draw and was measuring noise -- so what is checked is
    that the estimator is near-unbiased and that the heterogeneity test has
    roughly the right size and real power.
    """
    rng = np.random.default_rng(0)
    n, mu_true, reps = 17, 0.5, 300
    taus, mus, rejects = [], [], []
    for _ in range(reps):
        se = rng.uniform(0.15, 0.6, n)
        theta = mu_true + true_tau * rng.standard_normal(n)
        est = theta + se * rng.standard_normal(n)
        r = random_effects(est, se)
        taus.append(r["tau"])
        mus.append(r["mu"])
        rejects.append(heterogeneity(est, se)["p"] < 0.05)
    taus, mus = np.array(taus), np.array(mus)

    assert abs(mus.mean() - mu_true) < 0.02
    if true_tau == 0.0:
        # cannot be unbiased at a boundary -- tau >= 0 -- so what matters is that
        # it stays small (0.063 here) and that the test keeps its nominal size,
        # which it does almost exactly: 5.7% against 5%
        assert taus.mean() < 0.12
        assert 0.02 < np.mean(rejects) < 0.09
    else:
        assert abs(taus.mean() - true_tau) < 0.12 * true_tau
        # Power is the thing to be honest about, and it is recorded here rather
        # than assumed: seventeen players give only ~62% power against a moderate
        # tau = 0.3, and certainty against tau = 1. So a null result on real
        # players will mean "no large differences", not "no differences".
        assert np.mean(rejects) > (0.5 if true_tau < 0.5 else 0.95)


def test_shrinkage_goes_the_right_way_at_both_extremes():
    # all three the same distance from the population mean, so the only thing
    # that can order how far they move is how well they were measured
    est = np.array([2.0, 2.0, 2.0])
    se = np.array([0.1, 0.5, 1.0])
    mu = 1.0
    # players who cannot be told apart from the population lose their estimate
    total = shrink(est, se, mu, tau=1e-8)
    assert total["posterior_mean"] == pytest.approx(np.full(3, mu), abs=1e-6)
    assert total["weight"] == pytest.approx(np.zeros(3), abs=1e-6)
    # players from a wildly varied population keep it
    none_ = shrink(est, se, mu, tau=1e6)
    assert none_["posterior_mean"] == pytest.approx(est, rel=1e-8)
    # and in between, the badly measured player moves furthest
    part = shrink(est, se, mu, tau=0.5)
    moved = np.abs(part["posterior_mean"] - est)
    assert moved[2] > moved[1] > moved[0] > 0
    assert (part["weight"] > 0).all() and (part["weight"] < 1).all()
    assert (part["posterior_sd"] < se).all()


def test_an_interval_for_a_bounded_quantity_stays_in_bounds():
    """
    rho lives in (-1, 1) and its estimate often sits near the edge of what the
    data can resolve. A delta-method interval would run outside; sampling in the
    model's own unbounded coordinates and mapping through cannot.
    """
    theta = np.array([0.05, 0.02])
    cov = np.array([[0.09, 0.01], [0.01, 0.09]])

    def rho_of(e):
        log_ratio = float(np.hypot(e[0], e[1]))
        tilt = 0.5 * np.arctan2(e[1], e[0])
        k = np.exp(2 * log_ratio)
        c, s = np.cos(tilt), np.sin(tilt)
        num = c * s * (k - 1)
        return float(num / np.sqrt((c ** 2 + k * s ** 2) * (s ** 2 + k * c ** 2)))

    out = delta_interval(rho_of, theta, cov, n_draw=8000, seed=1)
    assert -1.0 < out["lo"] < out["hi"] < 1.0
    assert out["lo"] <= out["point"] <= out["hi"]
    # a linear function is where sampling and the delta method must agree
    lin = delta_interval(lambda e: 3 * e[0] - e[1], theta, cov, n_draw=40000, seed=2)
    assert lin["sd"] == pytest.approx(np.sqrt(9 * 0.09 + 0.09 - 2 * 3 * 0.01), rel=0.03)


def test_the_multivariate_version_recovers_a_correlated_population():
    """
    The point of doing it jointly is the off-diagonal, so that is what has to be
    recovered. Coordinate-wise shrinkage cannot see it at all.
    """
    from darts.hierarchical import random_effects_mv

    rng = np.random.default_rng(3)
    n, k = 200, 2
    T_true = np.array([[0.40, 0.28], [0.28, 0.30]])       # correlation +0.81
    mu_true = np.array([1.5, -0.5])
    theta = rng.multivariate_normal(mu_true, T_true, n)
    V = np.stack([np.diag(rng.uniform(0.02, 0.15, k)) for _ in range(n)])
    est = np.stack([rng.multivariate_normal(theta[i], V[i]) for i in range(n)])

    out = random_effects_mv(est, V)
    assert out["mu"] == pytest.approx(mu_true, abs=0.1)
    assert out["T"] == pytest.approx(T_true, abs=0.12)
    r_true = T_true[0, 1] / np.sqrt(T_true[0, 0] * T_true[1, 1])
    r_got = out["T"][0, 1] / np.sqrt(out["T"][0, 0] * out["T"][1, 1])
    assert abs(r_got - r_true) < 0.1
    # What it buys: lower error against the truth than the raw estimates.
    #
    # Note what it does *not* buy, because it is counterintuitive and an earlier
    # version of this test asserted it. A coordinate of the posterior mean need
    # not lie between its own estimate and its own population mean -- it does not
    # for 109 of these 200 units. Correlated information from the other
    # coordinates can push it the other way, and that is the whole point of
    # doing it jointly rather than one at a time.
    raw = np.mean((est - theta) ** 2)
    shrunk_mse = np.mean((out["posterior_mean"] - theta) ** 2)
    assert shrunk_mse < raw
    between = np.abs(out["posterior_mean"] - out["mu"]) <= np.abs(est - out["mu"])
    assert not between.all()


def test_the_multivariate_version_agrees_with_the_scalar_one_when_uncorrelated():
    """With a diagonal population and diagonal measurement error the two must
    give the same shrinkage, or one of them is wrong."""
    from darts.hierarchical import random_effects_mv

    rng = np.random.default_rng(4)
    n = 400
    tau = np.array([0.5, 0.3])
    mu_true = np.array([1.0, -1.0])
    theta = mu_true + tau * rng.standard_normal((n, 2))
    se = rng.uniform(0.1, 0.4, (n, 2))
    est = theta + se * rng.standard_normal((n, 2))
    V = np.stack([np.diag(se[i] ** 2) for i in range(n)])

    mv = random_effects_mv(est, V)
    for j in range(2):
        uni = random_effects(est[:, j], se[:, j])
        assert mv["mu"][j] == pytest.approx(uni["mu"], abs=0.05)
        assert np.sqrt(mv["T"][j, j]) == pytest.approx(uni["tau"], abs=0.06)
    assert abs(mv["T"][0, 1]) < 0.05


def test_a_barely_positive_direction_is_refused_like_a_flat_one():
    """
    A sign test is not enough. An information matrix whose smallest eigenvalue is
    positive but a billionth of its largest is positive definite and numerically
    singular -- checking only the sign let one through on a second data split and
    it reached `inv` as a crash rather than as a None.
    """
    mean = np.zeros(3)

    def nearly_flat(theta):
        d = np.asarray(theta, float)
        return -0.5 * (1e6 * d[0] ** 2 + 1e6 * d[1] ** 2 + 1e-9 * d[2] ** 2)

    out = parameter_covariance(nearly_flat, mean)
    assert not out["pd"]
    assert out["cov"] is None and out["se"] is None


def _quadratic_units(seed, n, k=2, flat=()):
    """
    Units whose log-likelihoods are exactly quadratic.

    This is the case where the answer is known: if every unit's likelihood is
    Gaussian then so is the integral over the population, the two-stage
    estimator is *exact* rather than an approximation, and the joint fit has
    something to be right about. Units listed in ``flat`` say nothing at all
    about their last coordinate.
    """
    rng = np.random.default_rng(seed)
    mu_true = np.array([1.5, -0.5] + [0.0] * (k - 2))
    theta = rng.multivariate_normal(mu_true, 0.3 * np.eye(k), size=n)
    covs, hats = [], []
    for p in range(n):
        A = rng.normal(size=(k, k)) * 0.35
        covs.append(A @ A.T + 0.05 * np.eye(k))
        hats.append(rng.multivariate_normal(theta[p], covs[-1]))
    covs, hats = np.stack(covs), np.stack(hats)

    lls = []
    for p in range(n):
        info = np.linalg.inv(covs[p])
        if p in flat:
            info[-1, :] = info[:, -1] = 0.0
        lls.append(lambda x, h=hats[p], A=info:
                   -0.5 * float((x - h) @ A @ (x - h)))
    return hats, covs, lls


def test_every_packed_vector_names_a_covariance():
    """
    The acceleration extrapolates in the packed space and is under no obligation
    to land anywhere sensible. Carrying T as a log-Cholesky is what makes that
    safe: there is no vector that unpacks to a non-covariance, so the jump cannot
    produce a population that is not one.
    """
    T = np.array([[0.4, 0.28, 0.05], [0.28, 0.30, -0.02], [0.05, -0.02, 0.9]])
    mu = np.array([1.0, -2.0, 0.5])
    back_mu, back_T = _unpack_population(_pack_population(mu, T), 3)
    assert back_mu == pytest.approx(mu)
    assert back_T == pytest.approx(T)

    rng = np.random.default_rng(0)
    for _ in range(200):
        v = rng.normal(scale=2.0, size=3 + 6)
        _, T_v = _unpack_population(v, 3)
        assert np.allclose(T_v, T_v.T)
        # the property is that it factors -- asserting on the smallest
        # eigenvalue instead would be asserting on the eigensolver, which
        # returns -1e-19 for a matrix a wild draw has conditioned at 1e5
        np.linalg.cholesky(T_v)


def test_the_joint_fit_reproduces_the_exact_answer():
    """
    The load-bearing test. With quadratic unit likelihoods the two-stage
    estimator is exact, so the Laplace-EM must land on it -- any disagreement is
    a bug in the joint model rather than the approximation it is entitled to
    elsewhere.
    """
    hats, covs, lls = _quadratic_units(seed=3, n=15)
    exact = random_effects_mv(hats, covs)
    out = joint_hierarchical(lls, hats)

    assert out["converged"]
    assert out["mu"] == pytest.approx(exact["mu"], abs=5e-3)
    assert out["T"] == pytest.approx(exact["T"], abs=5e-3)

    # and the penalised modes are where the two densities agree, which for this
    # case is the posterior mean in closed form
    Tinv = np.linalg.inv(out["T"])
    for p in range(len(hats)):
        Vinv = np.linalg.inv(covs[p])
        want = np.linalg.solve(Vinv + Tinv, Vinv @ hats[p] + Tinv @ out["mu"])
        assert out["theta"][p] == pytest.approx(want, abs=1e-4)


def test_a_unit_its_own_data_cannot_identify_is_ordinary_here():
    """
    The reason for fitting jointly rather than in two stages. A unit whose
    likelihood is flat in a direction has no covariance to summarise -- the
    two-stage route has to drop it -- but the *penalised* Hessian is
    ``-grad^2 L_p + T^-1``, which is positive definite whenever T is. The
    population supplies the missing curvature, and the unit is fitted like any
    other.
    """
    hats, covs, lls = _quadratic_units(seed=5, n=14, flat=(0, 4, 9))

    for p in (0, 4, 9):
        assert not parameter_covariance(lls[p], hats[p])["pd"]

    out = joint_hierarchical(lls, hats)
    assert out["converged"]
    assert len(out["theta"]) == len(hats)
    assert np.isfinite(out["theta"]).all() and np.isfinite(out["cov"]).all()
    slope = out["T"][1, 0] / out["T"][0, 0]
    for p in (0, 4, 9):
        assert np.linalg.eigvalsh(out["cov"][p]).min() > 0
        # And what they are given is better than the population mean. With
        # nothing of their own to say about the second coordinate they get the
        # population's *conditional* prediction from the first -- the ridge, not
        # the centre of it, which is the whole reason for a full T.
        conditional = out["mu"][1] + slope * (out["theta"][p][0] - out["mu"][0])
        assert out["theta"][p][1] == pytest.approx(conditional, abs=1e-3)
        # while the coordinate they *do* measure stays close to their own
        assert abs(out["theta"][p][0] - hats[p][0]) < abs(out["mu"][0] - hats[p][0])


def test_acceleration_changes_the_cost_and_not_the_answer():
    """
    SQUAREM is safeguarded on the marginal, so it is meant to be a pure saving.
    Plain EM here is slow for a reason worth recording: it converges linearly,
    and on this case it crawls for tens of iterations at a rate near 0.9, which
    is why the stopping rule is tight.
    """
    hats, _, lls = _quadratic_units(seed=7, n=12)
    fast = joint_hierarchical(lls, hats)
    slow = joint_hierarchical(lls, hats, accelerate=False)

    assert fast["converged"] and slow["converged"]
    assert fast["mu"] == pytest.approx(slow["mu"], abs=5e-3)
    assert fast["T"] == pytest.approx(slow["T"], abs=5e-3)
    assert fast["n_iter"] < slow["n_iter"]

    # the safeguard: the marginal never goes backwards
    h = np.array(fast["history"])
    assert (np.diff(h) > -1e-6).all()
