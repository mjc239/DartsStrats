"""
Ability-tailored checkout charts.

Every checkout chart in every pub is the same chart. It is computed for a
notional player who hits what they aim at, and it tells you a *route*: "170:
T20, T20, bull". That format carries two assumptions -- that you will hit the
first dart, and that everyone should play the same way -- and both are wrong
for most people holding the chart.

This builds the equivalent chart for a given ability, from the solved MDP.

Which model?
------------
A conventional chart is a **route for one visit**, so the model has to be the
3-dart one: it is the only one that knows a bust on dart 1 forfeits the other
two, and that the right aim for the same score depends on which dart it is.

The objective is **minimum visits**, not minimum darts. A chart is used in a
match, where what you are racing is your opponent's visits; and under the
per-dart objective a bust on dart 1 costs only the one dart that was actually
thrown, which under-penalises early busts.

The opponent is deliberately *not* modelled. A chart cannot be a function of
the opponent's score without becoming a two-dimensional table, and the
two-player solve shows the cost of ignoring them is at most ~1.4 points of win
probability. :func:`pressure_adjustments` lists the scores where that
approximation is worst, which is the honest way to present the gap.

Two chart styles are produced:

``route_chart``   the conventional one -- the sequence you would throw if every
                  dart lands where you aimed. Directly comparable to a printed
                  chart.
``policy_chart``  what the model actually says: the aim for each dart of the
                  visit, which is not the same thing once you have missed.
"""

import numpy as np

from darts.utils import aim_description, region_label


def _label(points, idx, board_pixels):
    return aim_description(points[idx], board_pixels)


def _label_to_score(label):
    """Score of a region label such as 'T20', 'D16', '19', 'BULL', 'miss'."""
    if label == "BULL":
        return 50
    if label == "25":
        return 25
    if label == "miss":
        return 0
    if label.startswith("outside") or label == "off the board":
        return 0
    if label[0] == "T":
        return 3 * int(label[1:])
    if label[0] == "D":
        return 2 * int(label[1:])
    return int(label)


def route_chart(model, points, board_pixels, scores=range(2, 171)):
    """
    The conventional chart: the route you would throw if each dart landed in
    the region you aimed at.

    Returns:
        list[dict]: one row per score with the up-to-three-dart route, the
        number of darts it takes, and whether it actually finishes.
    """
    rows = []
    for score in scores:
        u, route, start = score, [], score
        finished = False
        for dart in (1, 2, 3):
            idx = model.policy(u, dart, start)
            lab = _label(points, idx, board_pixels)
            route.append(lab)
            gained = _label_to_score(lab)
            if gained == u:
                finished = True
                break
            if gained > u - 2:
                break          # the model is not going for it this dart
            u -= gained
        rows.append({"score": score, "route": " ".join(route),
                     "darts": len(route), "route finishes": finished})
    return rows


def policy_chart(model, points, board_pixels, scores=range(2, 171)):
    """
    What the model actually recommends: the aim for each dart of the visit,
    given the score you are on when you throw it.

    This is the chart a printed one cannot express -- the dart-3 column is what
    to do when the visit has not gone to plan.
    """
    rows = []
    for score in scores:
        row = {"score": score}
        for dart in (1, 2, 3):
            row[f"dart {dart}"] = _label(points, model.policy(score, dart, score),
                                         board_pixels)
        rows.append(row)
    return rows


def chart_with_odds(model, points, board_pixels, checkout_pct,
                    scores=range(2, 171)):
    """
    A route chart annotated with the two numbers a player actually wants: the
    chance of finishing this visit, and the expected visits from here.
    """
    rows = route_chart(model, points, board_pixels, scores)
    for r in rows:
        s = r["score"]
        r["P(finish this visit)"] = round(float(checkout_pct[s]), 4)
        r["expected visits"] = round(float(-model.V1[s]), 3)
    return rows


def compare_bands(models, points, board_pixels, scores=range(2, 171)):
    """
    One row per score, one column per ability band, holding the recommended
    first dart. Shows where the standard chart is wrong for weaker players.

    Args:
        models (dict): band name -> solved ThreeDartMDP.
        points (dict): band name -> aiming grid for that model.
    """
    rows = []
    for score in scores:
        row = {"score": score}
        for band, m in models.items():
            row[band] = _label(points[band], m.policy(score, 1, score), board_pixels)
        rows.append(row)
    return rows


def disagreements(models, points, board_pixels, reference, scores=range(2, 171)):
    """
    Scores where a band's recommended first dart differs from the reference
    band's, with what each would throw.
    """
    out = []
    for score in scores:
        ref = _label(points[reference], models[reference].policy(score, 1, score),
                     board_pixels)
        for band, m in models.items():
            if band == reference:
                continue
            lab = _label(points[band], m.policy(score, 1, score), board_pixels)
            if lab != ref:
                out.append({"score": score, "band": band,
                            f"{reference} throws": ref, "band throws": lab})
    return out


def value_of_following_the_standard_chart(model, points, board_pixels,
                                          standard_first_dart,
                                          scores=range(2, 171)):
    """
    What it costs a player to follow a one-size-fits-all chart instead of their
    own, in expected visits.

    Args:
        standard_first_dart (dict): score -> region label the printed chart
            tells you to throw at.

    Returns:
        list[dict]: rows for the scores where the standard chart is not what
        this player should throw, with the cost in expected visits.
    """
    labels = {i: region_label(p, board_pixels) for i, p in enumerate(points)}
    by_label = {}
    for i, lab in labels.items():
        by_label.setdefault(lab, []).append(i)

    rows = []
    for score in scores:
        want = standard_first_dart.get(score)
        if want is None or want not in by_label:
            continue
        q = model.q_values(score, 1, score)
        best = q.max()
        # best point within the region the standard chart names
        alt = max(by_label[want], key=lambda i: q[i])
        cost = best - q[alt]
        if cost > 1e-9:
            rows.append({"score": score,
                         "your best": _label(points, int(q.argmax()), board_pixels),
                         "standard chart": want,
                         "cost (visits)": round(float(cost), 4)})
    return rows
