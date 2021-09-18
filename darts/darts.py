"""Main module."""
import numpy as np
import functools
from scipy.stats import multivariate_normal
from collections import defaultdict


class Dartboard:

    def __init__(self, pixels, board_padding=None):
        self.mm_per_pixel = None
        self.db_score_map = None
        self.dartboard_dims = None
        self.generate_dartboard(pixels)
        if board_padding:
            self._pad_score_map(board_padding)

        self.board_padding = board_padding

    def generate_dartboard(self, pixels):
        assert pixels % 2 == 1
        # width of dartboard score area = 340mm
        self.mm_per_pixel = 340/(pixels-1)

        x, y = np.meshgrid(np.linspace(-170, 170, pixels),
                           np.linspace(-170, 170, pixels))

        r = np.sqrt(x*x + y*y)
        theta = np.arctan2(y, x)

        single_rings = (((r >= 15.9) & (r < 99)) | ((r >= 107) & (r < 162))).astype(int)
        double_rings = ((r >= 162) & (r < 170)).astype(int)
        triple_rings = ((r >= 99) & (r < 107)).astype(int)
        outer_bull = ((r >= 6.35) & (r < 15.9)).astype(int)
        inner_bull = (r < 6.35).astype(int)
        ring_filter = single_rings + 2*double_rings + 3*triple_rings

        clockwise_after_11 = [14, 9, 12, 5, 20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8]
        start_angle = np.pi * (1 - 1./20)

        score_arr = np.zeros([pixels, pixels])

        # Loop around dartboard in slices
        angle = start_angle
        for score in clockwise_after_11:
            #print(f'Working on slice {score}')
            score_slice = ((theta < angle) & (theta >= angle - np.pi/10)).astype(int)
            angle -= np.pi/10

            score_arr += score * score_slice * ring_filter

        # Do 11 by hand (straddles jump in theta)
        score_slice = ((theta >= start_angle) | (theta < angle)).astype(int)
        score_arr += 11 * score_slice * ring_filter

        # Bullseyes
        score_arr += 25*outer_bull + 50*inner_bull

        # Save dartboard
        self.db_score_map = score_arr
        self.dartboard_dims = score_arr.shape

    def _pad_score_map(self, board_padding):
        self.db_score_map = np.pad(self.db_score_map, board_padding)
        self.dartboard_dims = self.db_score_map.shape
        self.board_padding = board_padding

    #@functools.cached_property
    def _segment_dict(self):
        segment_vals = [11, 14, 9, 12, 5, 20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11]
        segment_dict = {k: v for k, v in zip(np.arange(-1, 20), segment_vals)}
        return segment_dict

    def score_at_point(self, point):

        x, y = point
        r = np.sqrt(x*x + y*y)
        theta = np.arctan2(-y, x)

        if r < 6.35:
            return 50
        elif 6.35 <= r < 15.9:
            return 25
        elif 15.9 <= r < 99:
            ring_multiplier = 1
        elif 99 <= r < 107:
            ring_multiplier = 3
        elif 107 <= r < 162:
            ring_multiplier = 1
        elif 162 < r < 170:
            ring_multiplier = 2
        else:
            return 0

        segment = np.floor(10 * (1 + theta / np.pi) - 0.5).astype(int)
        segment_base = self._segment_dict[segment]
        return ring_multiplier * segment_base

    def pixel_to_point(self, pixel):
        centre_coords = ((self.dartboard_dims[0]-1)/2, (self.dartboard_dims[1]-1)/2)
        return self.mm_per_pixel * np.array([pixel[1] - centre_coords[1], pixel[0] - centre_coords[0]])

    def point_to_pixel(self, point):
        centre_coords = np.array(((self.dartboard_dims[0]-1)/2, (self.dartboard_dims[1]-1)/2))
        return (centre_coords + np.array(point)/self.mm_per_pixel)[::-1]

    def _remove_padding(self, arr):
        return arr[self.board_padding:-self.board_padding, self.board_padding:-self.board_padding]

    def gaussian_filter(self, mu, Sigma):
        assert len(mu) == 2
        assert Sigma.shape == (2, 2)

        half_width = 0.5 * self.mm_per_pixel * self.dartboard_dims[0]
        half_height = 0.5 * self.mm_per_pixel * self.dartboard_dims[1]

        x, y = np.meshgrid(np.linspace(-half_width, half_width, self.dartboard_dims[0]),
                           np.linspace(-half_height, half_height, self.dartboard_dims[1]))

        det = Sigma[0, 0] * Sigma[1, 1] - Sigma[0, 1] * Sigma[1, 0]
        d = 1./det * (Sigma[1, 1]*(x - mu[0])**2
                      - (Sigma[0, 1] + Sigma[1, 0])*(x - mu[0])*(y - mu[1])
                      + Sigma[0, 0]*(y - mu[1])**2)

        g = np.exp(- d / 2.0)
        g /= (2 * np.pi * np.sqrt(det))

        return g

    def exp_score_map(self, mu, Sigma, padding=False, score_function=None):
        # Add padding if required, alters db_score_map attribute
        if padding:
            self._pad_score_map(padding)

        g = self.gaussian_filter(-mu, Sigma)
        # Why is this necessary?
        g /= np.sum(g)

        if score_function:
            score_map = score_function(self.db_score_map)
        else:
            score_map = self.db_score_map

        db_ft = np.fft.fft2(score_map)
        g_slice_ft = np.fft.fft2(np.fft.ifftshift(g))

        prod_ft = db_ft * g_slice_ft
        exp_map = np.real(np.fft.ifft2(prod_ft))

        # Remove padding
        if padding:
            self.db_score_map = self._remove_padding(self.db_score_map)
            self.dartboard_dims = self.db_score_map.shape
            exp_map = self._remove_padding(exp_map)

        return exp_map

    def var_score_map(self, mu, Sigma, padding=False):
        sq_map = self.exp_score_map(mu, Sigma, padding=padding, score_function=lambda x: x*x)
        exp_map = self.exp_score_map(mu, Sigma, padding=padding)

        var_map = sq_map - exp_map*exp_map

        return var_map

    def std_score_map(self, mu, Sigma, padding=False):
        var_map = self.var_score_map(mu, Sigma, padding=padding)

        return np.sqrt(np.abs(var_map))

    def optimal_aim(self, exp_map, unit='mm'):
        assert unit in ['mm', 'pixel']

        max_score = np.max(exp_map)
        args = np.unravel_index(exp_map.argmax(), exp_map.shape)

        if unit == 'pixel':
            return np.array(args), max_score
        elif unit == 'mm':
            return self.pixel_to_point(args), max_score


class ThrowingDistribution(Dartboard):

    def __init__(self):
        super().__init__()
        self.mu = None
        self.Sigma = None
        self._choices_dict = defaultdict(list)

    def estimate_distribution(self,
                              throws,
                              tol=1e-10,
                              mu_initial=np.zeros(2),
                              Sigma_initial=20*np.eye(2),
                              return_params=False,
                              N_max=100_000):
        assert isinstance(throws, (list, tuple))

        mu = mu_initial
        Sigma = Sigma_initial
        likelihood = -1e20
        likelihoods = []
        n = len(throws)
        unique_throws = np.unique(throws)

        convergence = False

        N = 100
        while (N < N_max) or (not convergence):
            # E step
            # Burn-in period
            N = int(min(N_max, 2*N))

            #print('creating caches')
            ezs_cache = {}
            ezzs_cache = {}
            for throw in unique_throws:
                ezs_cache[throw] = self._moment_z_given_x(mu, Sigma, 1, throw, N)
                ezzs_cache[throw] = self._moment_z_given_x(mu, Sigma, 2, throw, N)
            #print('caches created')

            #print('Using caches')
            ezs = [ezs_cache[throw] for throw in throws]
            ezzs = [ezzs_cache[throw] for throw in throws]

            # M step
            mu = 1./n * np.sum(ezs, axis=0)
            Sigma = 1./n * np.sum(ezzs, axis=0) - np.outer(mu, mu)

            # Stopping criterion
            likelihood = self._Q(mu, Sigma, mu, Sigma, n)
            recent_likelihood_mean = np.mean(likelihoods[-5:])
            likelihoods.append(likelihood)

            likelihood_eval = np.abs((likelihood - recent_likelihood_mean)/recent_likelihood_mean)
            print(mu, likelihood, recent_likelihood_mean, likelihood_eval)
            convergence = (likelihood_eval < tol)

        print(N)
        # Save optimal mu, Sigma
        self.mu = mu
        self.Sigma = Sigma
        # clear choices cache
        self._choices_dict = defaultdict(list)

        if return_params:
            return mu, Sigma, likelihoods

    def _moment_z_given_x(self, mu, Sigma, moment, score, N):
        assert moment in [1, 2]

        # Used cached values first
        n_cached_values = len(self._choices_dict[score])
        if N <= n_cached_values:
            choices = self._choices_dict[score][:N]
        else:
            n_new_choices = N - n_cached_values
            #print('isolating relevant pixels')
            allowed_pixels = np.array(np.where(self.db_score_map == score)).T
            n_allowed_pixels = allowed_pixels.shape[0]
            #print('choices')
            choices_idx = np.random.choice(np.arange(n_allowed_pixels), n_new_choices)
            choices = [self.pixel_to_point(allowed_pixels[i]) for i in choices_idx]
            # cache choices for next iteration
            self._choices_dict[score] += choices

        # Make into array
        choices = np.array(choices)

        #print('computing gaussians')
        weights = self.gaussian_2d(choices, mu, Sigma)
        norm = np.sum(weights)
        if norm == 0:
            return np.zeros(2)

        #print('calculating arg')
        if moment == 2:
            arg = np.array([np.outer(i, i) for i in choices])
        else:
            arg = choices

        #print('integral estimate')
        return np.sum(weights * arg.T, axis=moment) / norm

    def _Q(self, mu, Sigma, ez, ezz, n):
        log_lik = -0.5*n*np.log(np.linalg.det(Sigma))
        log_lik -= 0.5*n*np.trace(np.linalg.inv(Sigma) @ (ezz - np.outer(ez, ez)))
        return log_lik

    @staticmethod
    def gaussian_2d(point, mu, Sigma):
        x, y = point.T

        det = Sigma[0, 0] * Sigma[1, 1] - Sigma[0, 1] * Sigma[1, 0]
        d = 1. / det * (Sigma[1, 1] * (x - mu[0]) ** 2
                        - (Sigma[0, 1] + Sigma[1, 0]) * (x - mu[0]) * (y - mu[1])
                        + Sigma[0, 0] * (y - mu[1]) ** 2)

        g = np.exp(- d / 2.0)
        g /= (2 * np.pi * np.sqrt(det))

        return g
