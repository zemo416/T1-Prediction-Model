"""Convert constant independent map win probabilities into series probabilities."""

from math import comb, isfinite
from numbers import Real


def series_win_probability(p: float, best_of: int) -> float:
    """Return the chance of winning BO1, BO3, or BO5 under an iid map model.

    Summing a full ``best_of`` binomial experiment is equivalent to stopping
    when either team reaches the required number of wins. The assumption that
    every map has the same independent probability is a simplification: side
    selection, drafts, roster changes, and information revealed during the
    series can change the probability of later maps.
    """
    if isinstance(p, bool) or not isinstance(p, Real):
        raise ValueError("Map win probability must be a finite number in [0, 1].")
    if not 0.0 <= p <= 1.0 or not isfinite(p):
        raise ValueError("Map win probability must be a finite number in [0, 1].")
    if (
        isinstance(best_of, bool)
        or not isinstance(best_of, int)
        or best_of not in (1, 3, 5)
    ):
        raise ValueError("best_of must be an integer: 1, 3, or 5.")
    required_wins = best_of // 2 + 1
    return float(sum(
        comb(best_of, wins) * p ** wins * (1.0 - p) ** (best_of - wins)
        for wins in range(required_wins, best_of + 1)
    ))
