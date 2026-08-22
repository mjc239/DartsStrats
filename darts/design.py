"""
Where should a player aim, if the point of the session is to *measure* them?

:mod:`darts.fitting` answers "given that these darts were aimed at one spot,
what was the spread?". It never asks where that spot should be, and it assumes
there was only one. Both are choices, and both are worth optimising: a session
is a fixed budget of darts, and how you spend it decides how precisely sigma
comes back.

The model for a measurement session
-----------------------------------
The player is asked to aim at targets ``t_1 ... t_k`` and throws ``n_i`` darts
at each. A dart aimed at ``t_i`` lands at

    Z ~ N(t_i + b, Sigma)

where ``b`` is a systematic bias (the player pulls low and left, say) and
``Sigma`` the spread. Only the *score* is recorded. So the parameter vector is

    theta = (b_x, b_y, Sigma_xx, Sigma_xy, Sigma_yy)          -- five numbers

and it is the same five numbers whatever ``k`` is. That matters: it makes
"200 darts at one target" and "100 darts at each of two" a fair comparison,
because both estimate exactly five parameters. Only the information differs.

Why a mixture of targets should win
-----------------------------------
The observation at a target is multinomial over the board scores, so a single
throw at ``t`` carries Fisher information

    I(t) = sum_s (1 / p_s(t)) * grad p_s(t) grad p_s(t)^T

and information is *additive over independent throws*. A design putting weight
``w_i`` on target ``t_i`` therefore has per-throw information

    M(w) = sum_i w_i I(t_i)

which is a convex combination of the ``I(t)``. The achievable set is the convex
hull of ``{I(t)}``, and single-target designs are its extreme points. Since we
are minimising a convex functional of ``M^-1``, the optimum is generically in
the interior of the hull -- i.e. a mixture. So the hypothesis that splitting
the darts helps is not just plausible, it is close to guaranteed; the open
questions are *which* points and *how much* it is worth, and those are what
this module computes.

There is a concrete reason for it here too. One target constrains the spread
mostly along one direction: aim at T20 and the scores 20/5/1/60 tell you a lot
about vertical scatter (which ring you are in) and much less about horizontal.
A second target in a differently-oriented part of the board fills that in.

What is being optimised
-----------------------
The number every other model in this project consumes is a single ``sigma_mm``,
so the criterion is the asymptotic variance of that one number,

    Var(sigma_hat) ~ c^T M(w)^-1 c / n,     c = d sigma_mm / d theta

which is classical *c-optimality*. ``b`` is a nuisance parameter, and using
``M^-1`` rather than ``1 / I_sigma,sigma`` correctly charges the design for
having to estimate it. :func:`d_criterion` offers D-optimality (all five
parameters) as an alternative.

Certifying the answer
---------------------
Search alone cannot tell you whether a better design exists. The general
equivalence theorem can: a design ``w`` is optimal exactly when

    d(t) := c^T M^-1 I(t) M^-1 c  <=  c^T M^-1 c    for every candidate t

with equality on the support of ``w``. :func:`equivalence_certificate` returns
the ratio ``max_t d(t) / c^T M^-1 c``, which is 1 at the optimum. So the
continuous optimum here is proved, not merely searched for, and the practical
equal-allocation designs can be scored against it.

Locality
--------
Fisher information depends on the true parameters, so the optimal design
depends on the sigma you are trying to measure -- the usual chicken and egg of
optimal design. :func:`design_efficiency` evaluates a design built for one
sigma at another, so the cost of guessing wrong can be seen rather than
assumed.
"""

import numpy as np

from darts.dartboards import DARTBOARD_CONSTANTS, generate_dartboard
from darts.utils import mm_per_pixel

FULL_PARAMS = ("b_x", "b_y", "S_xx", "S_xy", "S_yy")
ISO_PARAMS = ("b_x", "b_y", "sigma")


# --------------------------------------------------------------------------
# Exact derivatives of the throw kernel
# --------------------------------------------------------------------------

def kernel_derivatives(pixels, Sigma_mm, mm_per_px, params="full", nu=None):
    """
    The normalised Gaussian throw kernel and its exact derivatives with respect
    to the session parameters.

    The kernel actually used is the *discrete* one -- the density sampled on
    the pixel grid and renormalised to sum to 1 -- so it is that object which is
    differentiated, not the continuous density. Writing ``K = W / sum(W)`` with
    ``W = exp(-q/2)``,

        dK/dtheta = K * (G - E_K[G]),     G = d(-q/2)/dtheta

    the second term being the derivative of the normalising sum. Doing this
    analytically rather than by finite differences avoids choosing a step size
    against FFT round-off.

    Note that ``q = u^T Sigma^-1 u`` is scale free: computing it in pixels with
    ``Sigma`` in pixels gives the same number as in millimetres. Only the
    derivatives carry the unit conversion.

    A Student-t changes this in exactly one place, and the place is worth
    naming. Its kernel is ``W = (1 + q/nu)^{-(nu+2)/2}``, so

        d log W / dtheta = ((nu+2)/(nu+q)) * d(-q/2)/dtheta

    -- the Gaussian score function multiplied pointwise by ``(nu+2)/(nu+q)``,
    which is the same weight :meth:`darts.fitting.ScoreLikelihood.mixture_weight`
    puts on a dart in the E step, and for the same reason. Under a Gaussian a
    dart far from the aim point dominates the score function, because the only
    way to explain it is a wide player. Under a t it is discounted, because it
    can be explained as a wide *dart*. That is a change of character rather than
    of magnitude, and it is why the design results have to be recomputed rather
    than adjusted.

The normalisation changes with it, for the reason
    :mod:`darts.transitions` had to change: a t leaves real mass off the board,
    and dividing by the window sum would put it back. The t kernel is normalised
    analytically instead, so ``K`` sums to less than one and the derivative of
    the normaliser is ``d log(2 pi sqrt(det Sigma)) / dtheta`` rather than an
    expectation under ``K``. :func:`information_maps` then books the deficit as
    a miss. The Gaussian path is untouched and still normalises over the window,
    where the two are the same number to 1e-15.

    Args:
        pixels (int): board resolution.
        Sigma_mm (np.ndarray): 2x2 covariance in mm^2, in (x, y) order -- or the
            scale matrix, when ``nu`` is given.
        mm_per_px (float): board scale.
        params (str): ``"full"`` for ``(b_x, b_y, S_xx, S_xy, S_yy)``, or
            ``"isotropic"`` for ``(b_x, b_y, sigma)`` with ``Sigma = sigma^2 I``.
        nu (float): degrees of freedom of a Student-t throw. ``None`` is the
            Gaussian and is the path every existing result took.

    Returns:
        tuple: ``(K, dK)`` with ``K`` of shape (pixels, pixels) and ``dK`` of
        shape (n_params, pixels, pixels).
    """
    if nu is not None and nu <= 0:
        raise ValueError("nu must be positive")
    S = np.asarray(Sigma_mm, dtype=float)
    S_px = S / mm_per_px ** 2
    det = S_px[0, 0] * S_px[1, 1] - S_px[0, 1] ** 2
    inv = np.array([[S_px[1, 1], -S_px[0, 1]], [-S_px[0, 1], S_px[0, 0]]]) / det

    offs = (np.arange(pixels) - pixels // 2).astype(float)
    x = offs[None, :]                      # column offset, the x direction
    y = offs[:, None]                      # row offset, the y direction
    q = inv[0, 0] * x * x + 2 * inv[0, 1] * x * y + inv[1, 1] * y * y
    if nu is None or np.isinf(nu):
        W = np.exp(-q / 2.0)
        weight = 1.0
    else:
        W = np.exp(-0.5 * (nu + 2.0) * np.log1p(q / nu))
        weight = (nu + 2.0) / (nu + q)
    K = W / W.sum() if nu is None else W / (2 * np.pi * np.sqrt(det))

    # (Sigma^-1 u) in pixel units; dividing by mm_per_px puts it in mm units
    a = inv[0, 0] * x + inv[0, 1] * y
    c = inv[0, 1] * x + inv[1, 1] * y
    a, c = np.broadcast_to(a, q.shape), np.broadcast_to(c, q.shape)

    # d(-q/2)/d mu = Sigma^-1 u, but note the sign: the FFT helper these
    # kernels are used with convolves rather than correlates, so it evaluates
    # the mask at ``t - v`` and the effective mean is ``t - b``. The kernel is
    # symmetric, so this is invisible in the probabilities themselves and in
    # the even-order (Sigma) derivatives -- only the odd-order bias
    # derivatives pick up the minus sign.
    if params == "full":
        G = [-a / mm_per_px, -c / mm_per_px]
        # d(-q/2)/d Sigma_ab = 0.5 (Sigma^-1 u u^T Sigma^-1)_ab, and the
        # off-diagonal parameter moves both (x,y) and (y,x), hence no 0.5
        s = 1.0 / mm_per_px ** 2
        G += [0.5 * a * a * s, a * c * s, 0.5 * c * c * s]
    elif params == "isotropic":
        G = [-a / mm_per_px, -c / mm_per_px]
        sigma = float(np.sqrt(np.trace(S) / 2))
        G += [q / sigma]                   # d(-q/2)/d sigma = |u|^2 / sigma^3
    else:
        raise ValueError("params must be 'full' or 'isotropic'")

    # The t's score function is the Gaussian's, discounted where a dart would
    # have had to travel. At nu = inf the weight is 1 and this line is a no-op.
    G = np.stack([np.broadcast_to(g, q.shape) * weight for g in G])
    if nu is None:
        # K is normalised by its own sum, so the correction is an expectation
        # under K -- which is what makes the discrete kernel's probabilities sum
        # to one exactly, whatever the quadrature error.
        correction = (K[None] * G).sum(axis=(1, 2))
    else:
        # K is normalised analytically, so the correction is the derivative of
        # log(2 pi sqrt(det Sigma)). It does not depend on the pixel, and it is
        # zero for the bias, which does not move the determinant.
        inv_mm = inv / mm_per_px ** 2          # Sigma^-1 in mm^-2
        if params == "full":
            correction = np.array([0.0, 0.0, 0.5 * inv_mm[0, 0],
                                   inv_mm[0, 1], 0.5 * inv_mm[1, 1]])
        else:
            correction = np.array([0.0, 0.0, 2.0 / float(np.sqrt(np.trace(S) / 2))])
    dK = K[None] * (G - correction[:, None, None])
    return K, dK


def information_maps(board_pixels, Sigma_mm, params="full", board=None,
                     floor=1e-12, nu=None):
    """
    Per-throw Fisher information for every pixel of the board, treated as a
    target.

    Computed with one FFT of each score mask, reused against the kernel and
    each of its derivatives, so the whole board costs a handful of transforms
    rather than one evaluation per candidate target.

A caveat inherited from :mod:`darts.transitions`: for a Gaussian the FFT wraps
    the kernel round the array edge. The board array extends to 225.5mm and
    targets are inside the 170mm double ring, so for a 28mm player the nearest
    edge is about two sigma away; the wrapped mass lands on the far edge, which
    is also zero-scoring, so it mostly cancels within the ``0`` category. For a
    Student-t it would not cancel and is not small, so that path zero-pads the
    transform instead and books what leaves the array as a miss -- the same
    treatment, and the same reason, as the transition builder.

    Args:
        board_pixels (int): resolution.
        Sigma_mm (np.ndarray or float): covariance in mm^2, or a scalar sigma
            in mm for the isotropic case.
        params (str): parameterisation, see :func:`kernel_derivatives`.
        board (np.ndarray): prebuilt board array.
        floor (float): scores rarer than this at a given target are dropped
            from the sum there, to keep ``1/p`` finite.
        nu (float): degrees of freedom of a Student-t throw. ``None`` is the
            Gaussian. Note what is then being measured: ``sigma_mm`` is the t's
            **core scale**, so a standard error on it is a standard error on the
            core and not on a spread.

    Returns:
        dict: ``info`` (pixels, pixels, n_params, n_params), ``probs``
        (n_scores, pixels, pixels), ``allowed_scores``, ``params``,
        ``Sigma_mm``, ``mm_per_pixel``.
    """
    if np.isscalar(Sigma_mm):
        Sigma_mm = float(Sigma_mm) ** 2 * np.eye(2)
    Sigma_mm = np.asarray(Sigma_mm, float)

    if board is None:
        board, _ = generate_dartboard(board_pixels)
    n = board.shape[0]
    mm_px = mm_per_pixel(n)

    K, dK = kernel_derivatives(n, Sigma_mm, mm_px, params=params, nu=nu)
    npar = dK.shape[0]

    pad = nu is not None
    m = 2 * n if pad else n

    def centred_ft(a):
        if not pad:
            return np.fft.fft2(np.fft.ifftshift(a))
        # ifftshift would fold the negative offsets into the middle of a padded
        # array rather than its end; roll them into place after padding instead.
        buf = np.zeros((m, m))
        buf[:n, :n] = a
        return np.fft.fft2(np.roll(buf, (-(n // 2), -(n // 2)), axis=(0, 1)))

    K_ft = centred_ft(K)
    dK_ft = np.stack([centred_ft(d) for d in dK])

    allowed = np.unique(board).astype(np.int32)
    info = np.zeros((n, n, npar, npar))
    probs = np.empty((len(allowed), n, n))
    def accumulate(p, g):
        good = p > floor
        w = np.where(good, 1.0 / np.where(good, p, 1.0), 0.0)
        # info += (1/p) * outer(g, g), accumulated in place over scores
        info[...] += w[:, :, None, None] * (g.transpose(1, 2, 0)[:, :, :, None]
                                            * g.transpose(1, 2, 0)[:, :, None, :])

    # The masks tile the array, so what the scores do not account for is the mass
    # that left it -- and how much that is depends on where the dart was aimed,
    # not only on the kernel. So score 0 is held back until the rest are summed.
    zero_k = int(np.flatnonzero(allowed == 0)[0]) if pad else -1
    sum_p = np.zeros((n, n))
    sum_g = np.zeros((npar, n, n))
    held = None

    for k, s in enumerate(allowed):
        mask_ft = np.fft.fft2((board == s).astype(np.float64), s=(m, m))
        p = np.real(np.fft.ifft2(mask_ft * K_ft))[:n, :n]
        g = np.stack([np.real(np.fft.ifft2(mask_ft * f))[:n, :n] for f in dK_ft])
        if k == zero_k:
            held = True
            continue
        probs[k] = np.clip(p, 0.0, None)
        sum_p += p
        sum_g += g
        accumulate(p, g)

    if held is not None:
        # Everything the other scores did not claim is a dart that scored
        # nothing -- whether it landed on the board's black, on the wire, or on
        # the floor behind the oche. So P(0) is one minus the rest, and its
        # derivative is minus the rest's, which is also what makes the
        # probabilities sum to one and their derivatives to zero by
        # construction.
        p = 1.0 - sum_p
        g = -sum_g
        probs[zero_k] = np.clip(p, 0.0, None)
        accumulate(p, g)

    return {"info": info, "probs": probs, "allowed_scores": allowed,
            "params": params, "Sigma_mm": Sigma_mm, "mm_per_pixel": mm_px,
            "board": board, "nu": nu}


def information_at_points(maps, points):
    """Pull the per-throw information matrices at a set of pixel targets."""
    points = np.asarray(points, dtype=int)
    return maps["info"][points[:, 0], points[:, 1]]


def candidate_targets(board_pixels, point_stride=4, margin_mm=0.0):
    """
    Candidate targets: a grid inside the double ring. Unlike the MDP's aiming
    grid there is no reason to allow points beyond the board, since a target
    you cannot see is not one you can ask a player to aim at.
    """
    centre = board_pixels // 2
    radius_px = (DARTBOARD_CONSTANTS["DOUBLE_OUTER_RADIUS"] + margin_mm) \
        / mm_per_pixel(board_pixels)
    idx = np.arange(0, board_pixels, point_stride)
    ii, jj = np.meshgrid(idx, idx, indexing="ij")
    r = np.hypot(ii - centre, jj - centre)
    keep = r <= radius_px
    return np.stack([ii[keep], jj[keep]], axis=1).astype(np.int32)


# --------------------------------------------------------------------------
# Design criteria
# --------------------------------------------------------------------------

def sigma_gradient(Sigma_mm, params="full"):
    """
    ``c = d sigma_mm / d theta``, for the delta method.

    ``sigma_mm`` is defined throughout this project as ``sqrt(tr(Sigma) / 2)``,
    the isotropic equivalent of a possibly anisotropic throw.
    """
    if np.isscalar(Sigma_mm):
        Sigma_mm = float(Sigma_mm) ** 2 * np.eye(2)
    sigma = float(np.sqrt(np.trace(Sigma_mm) / 2))
    if params == "isotropic":
        return np.array([0.0, 0.0, 1.0])
    c = np.zeros(5)
    c[2] = c[4] = 1.0 / (4.0 * sigma)
    return c


def c_criterion(M, c, ridge=1e-12):
    """
    Per-throw asymptotic variance of ``c^T theta``, i.e. ``c^T M^-1 c``.

    Accepts a single matrix or a stack of them.
    """
    M = np.asarray(M, float)
    single = M.ndim == 2
    Ms = M[None] if single else M
    eye = np.eye(Ms.shape[-1])
    sol = np.linalg.solve(Ms + ridge * np.trace(Ms, axis1=-2, axis2=-1)[
        ..., None, None] * eye, np.broadcast_to(c, Ms.shape[:-1])[..., None])
    out = (np.broadcast_to(c, Ms.shape[:-1]) * sol[..., 0]).sum(axis=-1)
    return float(out[0]) if single else out


def d_criterion(M, ridge=1e-12):
    """``-log det M``: D-optimality, treating all parameters as of interest."""
    M = np.asarray(M, float)
    single = M.ndim == 2
    Ms = M[None] if single else M
    eye = np.eye(Ms.shape[-1])
    sign, logdet = np.linalg.slogdet(
        Ms + ridge * np.trace(Ms, axis1=-2, axis2=-1)[..., None, None] * eye)
    out = np.where(sign > 0, -logdet, np.inf)
    return float(out[0]) if single else out


def sigma_standard_error(M, n, Sigma_mm, params="full"):
    """Asymptotic standard error of ``sigma_hat`` from ``n`` throws."""
    c = sigma_gradient(Sigma_mm, params)
    return float(np.sqrt(c_criterion(M, c) / n))


# --------------------------------------------------------------------------
# Designs
# --------------------------------------------------------------------------

def design_information(I_pts, weights):
    """``M(w) = sum_i w_i I(t_i)``, the per-throw information of a design."""
    w = np.asarray(weights, float)
    return np.tensordot(w / w.sum(), I_pts, axes=(0, 0))


def best_single_target(I_pts, c):
    """The best one-target design, by exhaustive evaluation."""
    vals = c_criterion(I_pts, c)
    i = int(np.argmin(vals))
    return i, float(vals[i]), vals


def best_pair(I_pts, c, chunk=None, max_bytes=64 << 20):
    """
    The best equally-weighted two-target design, exhaustively.

    Every pair is evaluated -- with a few thousand candidates that is millions
    of 5x5 solves, which is fine batched, and it removes any doubt about a
    greedy search having missed something.

    Args:
        I_pts (np.ndarray): (n_candidates, p, p) per-throw information.
        c (np.ndarray): criterion vector.
        chunk (int): rows of the pair matrix to build at once. By default it is
            chosen so each block stays near ``max_bytes``, since the block is
            ``chunk * n * p * p`` doubles and a fine candidate grid would
            otherwise allocate gigabytes.
        max_bytes (int): target size of one block.
    """
    n = len(I_pts)
    if chunk is None:
        per_row = n * I_pts.shape[1] * I_pts.shape[2] * 8
        chunk = max(1, min(n, int(max_bytes // max(per_row, 1))))
    best = (np.inf, -1, -1)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        M = 0.5 * (I_pts[lo:hi, None] + I_pts[None, :])       # (chunk, n, p, p)
        vals = c_criterion(M.reshape(-1, *I_pts.shape[1:]), c).reshape(hi - lo, n)
        k = int(np.argmin(vals))
        v = float(vals.flat[k])
        if v < best[0]:
            best = (v, lo + k // n, k % n)
    return (best[1], best[2]), best[0]


def greedy_design(I_pts, c, k, seed_points=(), refine=True, max_swaps=50):
    """
    An equally-weighted ``k``-target design, by greedy forward selection
    followed by exchange refinement.

    Args:
        I_pts (np.ndarray): (n_candidates, p, p) per-throw information.
        c (np.ndarray): criterion vector from :func:`sigma_gradient`.
        k (int): number of targets.
        seed_points (sequence[int]): indices to start from.
        refine (bool): run swap refinement after the greedy pass.
        max_swaps (int): cap on refinement sweeps.

    Returns:
        tuple: ``(indices, value)`` with ``value`` the per-throw variance.
    """
    chosen = list(seed_points)
    while len(chosen) < k:
        if chosen:
            base = I_pts[chosen].sum(axis=0)
            M = (base[None] + I_pts) / (len(chosen) + 1)
        else:
            M = I_pts
        vals = c_criterion(M, c)
        vals[chosen] = np.inf
        chosen.append(int(np.argmin(vals)))

    value = c_criterion(design_information(I_pts[chosen], np.ones(k)), c)
    if not refine or k == 1:
        return chosen, float(value)

    for _ in range(max_swaps):
        improved = False
        for slot in range(k):
            others = [j for i, j in enumerate(chosen) if i != slot]
            base = I_pts[others].sum(axis=0)
            vals = c_criterion((base[None] + I_pts) / k, c)
            j = int(np.argmin(vals))
            if vals[j] < value - 1e-15:
                chosen[slot], value, improved = j, float(vals[j]), True
        if not improved:
            break
    return chosen, float(value)


def l_criterion(M, W, ridge=1e-12):
    """
    ``tr(W M^-1)``: the general linear-criterion family.

    Everything else in this module is a special case. ``W = c c^T`` gives
    c-optimality (the variance of one scalar, as in notebook 09); ``W = I``
    gives A-optimality (the total variance of all the parameters); and setting
    ``W`` to the Hessian of the *decision* loss gives a design that minimises
    expected visits lost rather than expected estimation error -- which is a
    different thing and generally a different design.

    Accepts a single matrix or a stack of them.
    """
    M = np.asarray(M, float)
    single = M.ndim == 2
    Ms = M[None] if single else M
    eye = np.eye(Ms.shape[-1])
    reg = Ms + ridge * np.trace(Ms, axis1=-2, axis2=-1)[..., None, None] * eye
    out = np.einsum("ij,pji->p", np.asarray(W, float), np.linalg.inv(reg))
    return float(out[0]) if single else out


def _criterion_pieces(M, kind, c=None, W=None, ridge=1e-12):
    """
    ``(phi, d_weights)`` for the multiplicative algorithm and the equivalence
    certificate, for whichever criterion is in use.

    The general equivalence theorem takes the same shape for all of them: a
    design is optimal exactly when the directional derivative ``d(t)`` toward
    every candidate is no greater than the criterion's own scale.
    """
    p = M.shape[-1]
    eye = np.eye(p)
    reg = M + ridge * np.trace(M) * eye
    Minv = np.linalg.inv(reg)
    if kind == "D":
        # phi = -log det M; d(t) = tr(M^-1 I(t)), optimal when max d <= p
        sign, logdet = np.linalg.slogdet(reg)
        return -logdet, Minv, float(p)
    if kind == "c":
        W = np.outer(c, c)
    elif kind != "L":
        raise ValueError("kind must be 'c', 'D' or 'L'")
    A = Minv @ np.asarray(W, float) @ Minv
    return float(np.trace(np.asarray(W, float) @ Minv)), A, None


def optimal_design_general(I_pts, kind="D", c=None, W=None, max_iter=4000,
                           lam=0.6, tol=1e-12, prune=1e-9):
    """
    The continuous optimal design for any of the criteria, by the multiplicative
    algorithm, with the equivalence-theorem certificate.

    Args:
        I_pts (np.ndarray): (n_candidates, p, p) per-throw information.
        kind (str): ``"D"`` for D-optimality (all parameters, best overall fit),
            ``"c"`` for one scalar (pass ``c``), ``"L"`` for a weighted
            criterion (pass ``W``).
        c (np.ndarray): criterion vector, for ``kind="c"``.
        W (np.ndarray): weight matrix, for ``kind="L"``.

    Returns:
        dict: ``weights``, ``support``, ``value``, ``certificate`` (1 at the
        optimum), ``n_iter``.
    """
    n = len(I_pts)
    w = np.full(n, 1.0 / n)
    value, d, scale = np.inf, None, None
    for it in range(max_iter):
        M = np.tensordot(w, I_pts, axes=(0, 0))
        phi, A, fixed_scale = _criterion_pieces(M, kind, c, W)
        d = np.einsum("ij,pji->p", A, I_pts)
        scale = fixed_scale if fixed_scale is not None else phi
        if abs(value - phi) < tol * max(abs(phi), 1e-30) and it > 20:
            value = phi
            break
        value = phi
        w = w * np.power(np.maximum(d / scale, 1e-300), lam)
        w /= w.sum()

    return {"weights": w, "support": np.flatnonzero(w > prune),
            "value": float(value), "certificate": float(d.max() / scale),
            "n_iter": it + 1}


def certificate_general(I_pts, M, kind="D", c=None, W=None):
    """Equivalence-theorem certificate for any design and criterion: >= 1
    always, and 1 only at the optimum."""
    phi, A, fixed_scale = _criterion_pieces(np.asarray(M, float), kind, c, W)
    d = np.einsum("ij,pji->p", A, I_pts)
    return float(d.max() / (fixed_scale if fixed_scale is not None else phi))


def optimal_design(I_pts, c, max_iter=4000, lam=0.6, tol=1e-12, prune=1e-9):
    """
    The continuous c-optimal design over the candidate set, by the standard
    multiplicative algorithm.

    Weights are updated by ``w_i <- w_i (d_i / phi)^lam`` where ``d_i`` is the
    directional derivative toward target ``i``. Unlike the equal-allocation
    designs this is allowed any weights at all, so it is the benchmark the
    practical designs are scored against.

    Returns:
        dict: ``weights``, ``support`` (indices with non-negligible weight),
        ``value`` (per-throw variance), ``certificate`` (see
        :func:`equivalence_certificate`), ``n_iter``.
    """
    n = len(I_pts)
    w = np.full(n, 1.0 / n)
    value = np.inf
    for it in range(max_iter):
        M = np.tensordot(w, I_pts, axes=(0, 0))
        u = np.linalg.solve(M, c)
        phi = float(c @ u)
        d = np.einsum("i,pij,j->p", u, I_pts, u)
        if abs(value - phi) < tol * max(phi, 1e-30) and it > 20:
            value = phi
            break
        value = phi
        w = w * np.power(np.maximum(d / phi, 1e-300), lam)
        w /= w.sum()

    support = np.flatnonzero(w > prune)
    return {"weights": w, "support": support, "value": float(value),
            "certificate": float(d.max() / phi), "n_iter": it + 1}


def equivalence_certificate(I_pts, c, M):
    """
    The general equivalence theorem check for c-optimality.

    Returns ``max_t d(t) / phi``, which is >= 1 always and equals 1 exactly at
    the optimum. A value of 1.02 means no design over this candidate set can be
    much better; a value of 3 means the search has not finished.
    """
    u = np.linalg.solve(M, c)
    phi = float(c @ u)
    d = np.einsum("i,pij,j->p", u, I_pts, u)
    return float(d.max() / phi)


def robust_design(scenarios, k, max_swaps=50, seed_points=(), n_restarts=8,
                  rng=None):
    """
    An equally-weighted ``k``-target design that is good across a *range* of
    abilities, rather than optimal for one.

    The optimal design depends on the sigma you are trying to measure, which is
    circular: you pick the targets before you know the answer. This picks the
    design maximising the worst-case efficiency over a set of scenarios, so a
    coach can hand the same routine to any player.

    Efficiency is measured against the local optimum for each scenario, so 0.8
    means "needs 25% more darts than a design built knowing the answer".

    Unlike the single-ability criterion this objective is a *minimum* over
    scenarios, so it is not smooth and a single greedy pass lands in local
    optima readily -- greedily building up to four targets can end up worse
    than the three-target answer, which is a property of the search and not of
    the designs. Several restarts are run and the best kept.

    Args:
        scenarios (sequence): one ``(I_pts, c, best_value)`` per ability, where
            ``best_value`` is the locally optimal per-throw variance (from
            :func:`optimal_design`).
        k (int): number of targets.
        max_swaps (int): cap on exchange sweeps.
        seed_points (sequence[int]): indices to start every restart from.
        n_restarts (int): random restarts in addition to the greedy start.
        rng: numpy generator, for reproducible restarts.

    Returns:
        tuple: ``(indices, worst_efficiency, per_scenario_efficiency)``.
    """
    n_cand = len(scenarios[0][0])
    rng = np.random.default_rng(0) if rng is None else rng

    def worst(idx):
        effs = [best / c_criterion(design_information(I[list(idx)],
                                                      np.ones(len(idx))), c)
                for I, c, best in scenarios]
        return min(effs), effs

    def worst_all_extensions(chosen):
        """Worst-case efficiency of adding each candidate to ``chosen``."""
        out = np.full(n_cand, -np.inf)
        stack = []
        for I, c, best in scenarios:
            if chosen:
                M = (I[chosen].sum(axis=0)[None] + I) / (len(chosen) + 1)
            else:
                M = I
            stack.append(best / c_criterion(M, c))
        return np.min(np.stack(stack), axis=0)

    def refine(chosen):
        """Exchange refinement: repeatedly replace the worst-placed target."""
        chosen = list(chosen)
        value, effs = worst(chosen)
        for _ in range(max_swaps):
            improved = False
            for slot in range(len(chosen)):
                others = [j for i, j in enumerate(chosen) if i != slot]
                stack = []
                for I, c, best in scenarios:
                    M = (I[others].sum(axis=0)[None] + I) / len(chosen)
                    stack.append(best / c_criterion(M, c))
                vals = np.min(np.stack(stack), axis=0)
                j = int(np.argmax(vals))
                if vals[j] > value + 1e-15:
                    chosen[slot] = j
                    value, effs = worst(chosen)
                    improved = True
            if not improved:
                break
        return chosen, value, effs

    starts = []
    greedy = list(seed_points)
    while len(greedy) < k:
        vals = worst_all_extensions(greedy)
        vals[greedy] = -np.inf
        greedy.append(int(np.argmax(vals)))
    starts.append(greedy)
    for _ in range(n_restarts):
        extra = list(rng.choice(n_cand, size=k - len(seed_points), replace=False))
        starts.append(list(seed_points) + [int(j) for j in extra])

    best = None
    for start in starts:
        chosen, value, effs = refine(start)
        if best is None or value > best[1]:
            best = (chosen, value, effs)
    return best[0], float(best[1]), best[2]


def darts_to_detect(per_throw_sd, delta, alpha=0.05, power=0.8, sessions=2):
    """
    Darts per session needed to detect an improvement of ``delta`` mm.

    The test is a two-sided z-test on the difference of two session estimates
    (or one estimate against a known baseline, with ``sessions=1``). With a
    per-throw standard deviation ``S`` -- so a session of ``n`` darts measures
    sigma to ``S / sqrt(n)`` -- the difference of two sessions has standard
    error ``S * sqrt(2 / n)`` and the requirement is

        n = sessions * S^2 * (z_{1-alpha/2} + z_{power})^2 / delta^2

    ``S`` is what the design work computes: ``sqrt(c^T M^-1 c)`` at the chosen
    target, i.e. ``se_from_200_darts * sqrt(200)`` read off the manifests.

    This is the asymptotic answer. The simulation studies show the Fisher
    prediction is accurate at the well-chosen targets from a few hundred darts
    -- and the n this returns is comfortably past that -- but a single-target
    T20 session stays above its bound far longer, so treat the T20 column as
    optimistic.

    Args:
        per_throw_sd (float or array): ``S`` in mm per sqrt(dart).
        delta (float): the improvement to detect, in mm of sigma.
        alpha (float): two-sided false-positive rate.
        power (float): probability of detecting a real improvement of delta.
        sessions (int): 2 for before-and-after, 1 against a known baseline.

    Returns:
        float or np.ndarray: darts per session.
    """
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    S = np.asarray(per_throw_sd, dtype=float)
    n = sessions * (S * z / delta) ** 2
    return float(n) if np.isscalar(per_throw_sd) else n


def design_efficiency(I_pts_at_truth, c_at_truth, design_idx, weights=None):
    """
    How well a design does at a sigma other than the one it was built for,
    relative to the best design *for that* sigma.

    Returns a number in (0, 1]: 0.8 means the transplanted design needs 25%
    more darts than the design built for the true sigma.
    """
    if weights is None:
        weights = np.ones(len(design_idx))
    M = design_information(I_pts_at_truth[list(design_idx)], weights)
    got = c_criterion(M, c_at_truth)
    best = optimal_design(I_pts_at_truth, c_at_truth)["value"]
    return float(best / got)
