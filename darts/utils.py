import numpy as np
from darts.dartboards import DARTBOARD_CONSTANTS


def mm_per_pixel(pixels):
    """Converts pixels to millimeters

    Args:
        pixels (int): Number of pixels

    Returns:
        float: Number of millimeters
    """
    return 2 * DARTBOARD_CONSTANTS["DARTBOARD_RADIUS_MM"] / pixels


def region_label(point, board_pixels, quadro=False):
    """
    Describe an aiming point in darts language, e.g. ``"T20"``, ``"D16"``,
    ``"19"``, ``"25"``, ``"BULL"`` or ``"miss"``.

    Args:
        point (array-like): [row, column] pixel coordinates of the aim point.
        board_pixels (int): resolution of the board array the point refers to.
        quadro (bool): label the Quadro ring as ``"Q<n>"``.

    Returns:
        str: label of the board region containing the point.
    """
    c = DARTBOARD_CONSTANTS
    centre = board_pixels // 2
    scale = mm_per_pixel(board_pixels)
    # generate_dartboard builds the board with meshgrid(x, y), so the first
    # array index runs over y and the second over x.
    y = (point[0] - centre) * scale
    x = (point[1] - centre) * scale
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)

    if r < c["INNER_BULLSEYE_RADIUS_MM"]:
        return "BULL"
    if r < c["OUTER_BULLSEYE_RADIUS_MM"]:
        return "25"
    if r >= c["DOUBLE_OUTER_RADIUS"]:
        return "miss"

    number = None
    for score, intervals in c["SEGMENTS"].items():
        for lo, hi in intervals:
            if lo * np.pi <= theta < hi * np.pi:
                number = score
    if number is None:
        return "miss"

    if c["TRIPLE_INNER_RADIUS"] <= r < c["TRIPLE_OUTER_RADIUS"]:
        return f"T{number}"
    if c["DOUBLE_INNER_RADIUS"] <= r < c["DOUBLE_OUTER_RADIUS"]:
        return f"D{number}"
    if quadro and 56.6 <= r < 64.6:
        return f"Q{number}"
    return f"{number}"
