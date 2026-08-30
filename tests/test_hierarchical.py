"""Per-player uncertainty, and pooling across players."""
import numpy as np
import pytest

from darts.hierarchical import (delta_interval, heterogeneity,
                                observed_information, parameter_covariance,
                                random_effects, shrink)


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
