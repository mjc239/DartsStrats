import numpy as np
from darts_refactored import DARTBOARD_CONSTANTS

def mm_per_pixel(pixels):
    return 2*DARTBOARD_CONSTANTS['TRIPLE_OUTER_RADIUS']/(pixels-1)
