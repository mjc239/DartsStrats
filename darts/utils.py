import numpy as np
from dartboards import DARTBOARD_CONSTANTS

def mm_per_pixel(pixels):
    return 2*DARTBOARD_CONSTANTS['DARTBOARD_RADIUS_MM']/pixels

