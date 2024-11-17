import numpy as np
from dartboards import DARTBOARD_CONSTANTS

def mm_per_pixel(pixels):
    """Converts pixels to millimeters

    Args:
        pixels (int): Number of pixels

    Returns:
        float: Number of millimeters
    """
    return 2*DARTBOARD_CONSTANTS['DARTBOARD_RADIUS_MM']/pixels

