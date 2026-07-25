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
    # Wrap into [-pi, pi): the segment intervals are half-open, so a point
    # exactly on the negative x-axis would otherwise match no segment and be
    # reported as a miss even though it is on the board.
    theta = np.mod(np.arctan2(y, x) + np.pi, 2 * np.pi) - np.pi

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


def aim_description(point, board_pixels, quadro=False):
    """
    Describe an aiming point the way you would say it out loud.

    :func:`region_label` returns ``"miss"`` for anything past the double ring,
    which is unhelpful when the model deliberately aims at the outer edge of a
    double to protect the number -- a real and common recommendation for weaker
    players. This names those points by the bed they sit outside.

    Args:
        point (array-like): [row, column] pixel coordinates.
        board_pixels (int): resolution of the board array.
        quadro (bool): label the Quadro ring.

    Returns:
        str: e.g. ``"T20"``, ``"D16"``, ``"outside D5"``, ``"off the board"``.
    """
    c = DARTBOARD_CONSTANTS
    label = region_label(point, board_pixels, quadro=quadro)
    if label != "miss":
        return label

    centre = board_pixels // 2
    scale = mm_per_pixel(board_pixels)
    y = (point[0] - centre) * scale
    x = (point[1] - centre) * scale
    r = np.hypot(x, y)
    # Wrap into [-pi, pi): the segment intervals are half-open, so a point
    # exactly on the negative x-axis would otherwise match no segment.
    theta = np.mod(np.arctan2(y, x) + np.pi, 2 * np.pi) - np.pi

    # Far enough out that no bed is meaningfully being aimed at.
    if r > c["DOUBLE_OUTER_RADIUS"] + 25:
        return "off the board"

    number = None
    for score, intervals in c["SEGMENTS"].items():
        for lo, hi in intervals:
            if lo * np.pi <= theta < hi * np.pi:
                number = score
    if number is None:
        return "off the board"
    return f"outside D{number}"
