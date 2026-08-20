"""Notebook 20's claims about what couples the darts of a visit."""
import numpy as np
import pytest

from darts.dependence import (BedGrid, VisitModel, bed_geometry, bed_labels,
                              encode_visits, signatures, treble_centre_mm,
                              TARGETS)
from darts.utils import region_label


@pytest.fixture(scope="module")
def grid():
    return BedGrid(512)


def test_bed_labels_agree_with_the_projects_own_region_labels(grid):
    """The bed board must name the same regions darts.utils does."""
    for number in (20, 19, 5, 1, 6, 11):
        centre = treble_centre_mm(number)
        pixel = [512 // 2 + centre[1] / grid.scale, 512 // 2 + centre[0] / grid.scale]
        assert region_label(pixel, 512) == f"T{number}"
        code = grid.codes[int(round(pixel[0])), int(round(pixel[1]))]
        assert grid.names[code] == f"T{number}"


def test_the_windowed_bed_pmf_matches_summing_the_whole_board(grid):
    """The window is an optimisation, so it must change no answer."""
    codes, _ = bed_labels(512)
    coords = (np.arange(512) - 256) * grid.scale
    x, y = np.meshgrid(coords, coords)
    centre = treble_centre_mm(20)
    for sigma in (5.0, 8.0):
        w = np.exp(-0.5 * (((x - centre[0]) / sigma) ** 2
                           + ((y - centre[1]) / sigma) ** 2))
        w /= w.sum()
        brute = np.bincount(codes.ravel(), weights=w.ravel(), minlength=grid.n_beds)
        assert np.abs(brute - grid.bed_pmf(20, np.zeros(2), sigma)).max() < 1e-9
    assert abs(grid.bed_pmf(20, np.zeros(2), 7.0).sum() - 1.0) < 1e-12


def test_a_radial_pull_is_far_harder_to_see_than_a_sideways_one(grid):
    """Both single 20 beds carry one label, so "throwing low" barely shows.

    This is why VisitModel leaves the radial bias out by default: from beds
    alone, pushing the aim a millimetre above the treble and a millimetre below
    it look almost the same.
    """
    base = grid.bed_pmf(20, np.zeros(2), 7.0)
    fisher = {}
    for axis in (0, 1):
        step = np.zeros(2)
        step[axis] = 0.5
        grad = grid.bed_pmf(20, step, 7.0) - grid.bed_pmf(20, -step, 7.0)
        fisher[axis] = np.sum(grad ** 2 / np.maximum(base, 1e-12))
    assert fisher[0] > 10 * fisher[1], "sideways should be far better identified"


def test_the_aim_rule_alone_manufactures_a_treble_20_lift(grid):
    """The correction to notebook 19, as an assertion.

    With darts that are *exactly* independent, simply moving to the treble 19
    after a miss makes the treble 20 look correlated within a visit -- because
    after a miss the next dart is often not aimed at the treble 20 at all. Any
    measurement of within-visit dependence has to model the aim rule first.
    """
    model = VisitModel(grid, switching=True)
    theta = np.array([np.log(8.6), 0.0, -3.5, -1.0])       # s_hit .03, s_miss .27
    beds, hit = model.simulate(theta, 6000, rng=np.random.default_rng(0))
    s = signatures(beds, hit, grid)
    assert s["t20_lift_12"] > 4.0, "switching should show up as a T20 lift"
    # ...while a target-invariant statistic sees nothing, because nothing is there
    assert abs(s["treble_lift_12"]) < 3.0
    assert abs(s["dir_corr"]) < 0.02 and abs(s["mag_corr"]) < 0.02


def test_direction_and_magnitude_tell_the_two_couplings_apart(grid):
    """A shared offset moves a visit one way; a shared scale only spreads it."""
    off = VisitModel(grid, shared_offset=True, switching=True)
    sca = VisitModel(grid, shared_scale=True, switching=True)
    rng = np.random.default_rng(1)
    b1, h1 = off.simulate(np.array([np.log(7.0), 0.0, np.log(5.0), -3.5, -1.0]),
                          15000, rng=rng)
    b2, h2 = sca.simulate(np.array([np.log(8.35), 0.0, np.log(0.42), -3.5, -1.0]),
                          15000, rng=rng)
    s_off, s_sca = signatures(b1, h1, grid), signatures(b2, h2, grid)
    assert s_off["dir_corr"] > 0.03, "a shared offset must correlate directions"
    assert s_sca["mag_corr"] > s_off["mag_corr"] > 0.0
    # The discriminator is the ratio, not either number alone: a shared offset
    # couples direction and magnitude about equally, while a shared scale couples
    # magnitude only. (The signed correlation is a noisier estimate than its
    # sample size suggests when the magnitudes are themselves correlated, so an
    # absolute bound on it would be a flaky test rather than a sharper one.)
    assert s_off["mag_corr"] < 3 * s_off["dir_corr"]
    assert s_sca["mag_corr"] > 4 * abs(s_sca["dir_corr"])


def test_the_shared_offset_model_contains_the_independent_one(grid):
    """Nesting: at tau = 0 the coupled model must reproduce the plain one."""
    plain = VisitModel(grid, switching=True)
    coupled = VisitModel(grid, shared_offset=True, switching=True)
    theta = np.array([np.log(8.0), 0.5, -3.0, -1.0])
    beds, hit = plain.simulate(theta, 800, rng=np.random.default_rng(2))
    nested = np.array([theta[0], theta[1], np.log(1e-6), theta[2], theta[3]])
    assert abs(coupled.log_likelihood(nested, beds, hit)
               - plain.log_likelihood(theta, beds, hit)) < 1e-6


def test_the_fit_recovers_a_known_coupling(grid):
    """Simulate a player with a shared offset, and measure it back."""
    model = VisitModel(grid, shared_offset=True, switching=True)
    truth = np.array([np.log(7.0), 0.0, np.log(5.0), -3.5, -1.0])
    beds, hit = model.simulate(truth, 4000, rng=np.random.default_rng(3))
    got = model.unpack(model.fit(beds, hit).x)
    assert abs(got["sigma"] - 7.0) < 0.8
    assert abs(got["tau"] - 5.0) < 1.2
    assert abs(got["s_miss"] - 0.269) < 0.05


def test_a_gaussian_alone_cannot_reach_the_far_beds(grid):
    """Why the model needs a contaminating component at all.

    A throw tight enough to hit the treble 20 at a professional rate puts
    essentially nothing in the double 20 or off the board, but real players
    land there percents of the time.
    """
    tight = grid.bed_pmf(20, np.zeros(2), 7.0)
    assert tight[grid.names.index("T20")] > 0.35
    assert tight[grid.names.index("D20")] < 1e-4
    contaminated = 0.87 * tight + 0.13 * grid.wide_pmf(20, 7.0 * 6.0)
    assert contaminated[grid.names.index("D20")] > 1e-3
    assert contaminated[grid.names.index("T20")] > 0.30


def test_the_aim_steps_down_the_board_and_never_back_up(grid):
    """Professionals work 20 -> 19 -> 18 -> 17 after misses, and do not return.

    The chain is one-directional by construction, so a simulated player should
    show the 19 appearing only from dart 2 and the 18 and 17 only from dart 3 --
    which is the pattern the real data shows.
    """
    model = VisitModel(grid, switching=True)
    theta = np.array([np.log(8.0), 0.0, -3.5, -0.9])
    _, _, target = model.simulate(theta, 6000, rng=np.random.default_rng(4),
                                  return_targets=True)

    share = [(target == i).mean(axis=0) for i in range(len(TARGETS))]
    assert share[0][0] == 1.0, "dart 1 is always thrown at the 20"
    for i in range(1, len(TARGETS)):
        assert share[i][0] == 0.0, f"target {TARGETS[i]} cannot appear on dart 1"
        # each step down needs one more dart than the last to become reachable
        assert share[i][2] >= share[i][1] - 1e-9
    steps = target[:, 1:] - target[:, :-1]
    assert steps.min() >= 0, "the aim never steps back up"
    assert steps.max() <= 1, "and never skips a target"


def test_encode_visits_pools_beds_no_throw_can_reach(grid):
    """Far doubles collapse into the catch-all, so no observation is impossible."""
    beds, hit = encode_visits([["T20", "D16", "MISS"]], grid)
    assert grid.names[beds[0, 0]] == "T20"
    assert grid.names[beds[0, 1]] == "MISS", "D16 is unreachable, so it pools"
    assert hit[0, 0] and not hit[0, 1]
    assert grid.reachable[grid.names.index(f"T{TARGETS[0]}")]
