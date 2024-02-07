import numpy as np

DARTBOARD_CONSTANTS = {
    "DARTBOARD_RADIUS_MM": 225.5,
    "INNER_BULLSEYE_RADIUS_MM": 6.35,
    "OUTER_BULLSEYE_RADIUS_MM": 15.9,
    "TRIPLE_INNER_RADIUS": 99,
    "TRIPLE_OUTER_RADIUS": 107,
    "DOUBLE_INNER_RADIUS": 162,
    "DOUBLE_OUTER_RADIUS": 170,
    "SEGMENTS": {
        20: [(9 / 20, 11 / 20)],
        1: [(7 / 20, 9 / 20)],
        18: [(5 / 20, 7 / 20)],
        4: [(3 / 20, 5 / 20)],
        13: [(1 / 20, 3 / 20)],
        6: [(-1 / 20, 1 / 20)],
        10: [(-3 / 20, -1 / 20)],
        15: [(-5 / 20, -3 / 20)],
        2: [(-7 / 20, -5 / 20)],
        17: [(-9 / 20, -7 / 20)],
        3: [(-11 / 20, -9 / 20)],
        19: [(-13 / 20, -11 / 20)],
        7: [(-15 / 20, -13 / 20)],
        16: [(-17 / 20, -15 / 20)],
        8: [(-19 / 20, -17 / 20)],
        11: [(19 / 20, 1), (-1, -19 / 20)],
        14: [(17 / 20, 19 / 20)],
        9: [(15 / 20, 17 / 20)],
        12: [(13 / 20, 15 / 20)],
        5: [(11 / 20, 13 / 20)],
    },
}

QUADRO_CONSTANTS = {
    **DARTBOARD_CONSTANTS,
    "QUAD_OUTER_RADIUS": 64.6,
    "QUAD_INNER_RADIUS": 56.6,
}


def generate_dartboard(pixels: int, quadro: bool = False) -> np.ndarray:
    """Generate a dartboard as a numpy array, with each entry indicating the
    score at that pixel.

    Args:
        pixels (int): Dimension of the square array, indicating how pixels
        each side of the image should have.

    Returns:
        np.ndarray: numpy array, with each entry indicating the score at
        that pixel.
    """
    radius = DARTBOARD_CONSTANTS["DARTBOARD_RADIUS_MM"]

    # Create grid covering full dartboard area, with the
    # correct number of pixels
    x, y = np.meshgrid(
        np.linspace(-radius, radius, pixels), np.linspace(-radius, radius, pixels)
    )

    # Convert to polar coordinates
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)

    # Score multiplier mask
    bi = DARTBOARD_CONSTANTS["INNER_BULLSEYE_RADIUS_MM"]
    bo = DARTBOARD_CONSTANTS["OUTER_BULLSEYE_RADIUS_MM"]
    di = DARTBOARD_CONSTANTS["DOUBLE_INNER_RADIUS"]
    do = DARTBOARD_CONSTANTS["DOUBLE_OUTER_RADIUS"]
    ti = DARTBOARD_CONSTANTS["TRIPLE_INNER_RADIUS"]
    to = DARTBOARD_CONSTANTS["TRIPLE_OUTER_RADIUS"]

    single_rings = (((r >= bo) & (r < ti)) | ((r >= to) & (r < di))).astype(int)
    double_rings = ((r >= di) & (r < do)).astype(int)
    triple_rings = ((r >= ti) & (r < to)).astype(int)
    outer_bull = ((r >= bi) & (r < bo)).astype(int)
    inner_bull = (r < bi).astype(int)
    multiplier_mask = single_rings + 2 * double_rings + 3 * triple_rings
    checkouts = np.logical_or(double_rings, inner_bull)

    if quadro:
        qi = QUADRO_CONSTANTS["QUAD_INNER_RADIUS"]
        qo = QUADRO_CONSTANTS["QUAD_OUTER_RADIUS"]
        quad_rings = ((r >= qi) & (r < qo)).astype(int)
        # Only 3, as single rings has already added 1 multiple
        multiplier_mask += 3 * quad_rings

    score_arr = np.zeros([pixels, pixels])

    # Loop around dartboard in segments
    for score, intervals in DARTBOARD_CONSTANTS["SEGMENTS"].items():
        score_segment = np.zeros([pixels, pixels])
        for interval in intervals:
            low_angle, high_angle = interval[0] * np.pi, interval[1] * np.pi
            score_segment = np.logical_or(
                score_segment, ((theta < high_angle) & (theta >= low_angle)).astype(int)
            )

        score_arr += score * score_segment * multiplier_mask

    # Bullseyes
    score_arr += 25 * outer_bull + 50 * inner_bull

    return (score_arr, checkouts)
