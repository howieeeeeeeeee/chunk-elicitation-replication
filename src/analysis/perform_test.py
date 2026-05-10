import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance, chi2_contingency
from collections import Counter

from games.instructions import GAME_DECISION_ARRAY_CONFIG

import logging

logger = logging.getLogger(__name__)


def standardize(arr: np.ndarray) -> np.ndarray:
    """Standardize array to mean=0, std=1."""
    arr = np.asarray(arr, dtype=float)
    std = arr.std()
    if std == 0 or np.isnan(std):
        return arr - arr.mean()
    return (arr - arr.mean()) / std


def perform_ks_test(sample1, sample2, alpha=0.05):
    statistic, p_value = stats.ks_2samp(sample1, sample2)

    # Determine if we reject the null hypothesis
    reject_null = float(p_value) <= alpha

    # Prepare the interpretation
    if reject_null:
        interpretation = "Reject the null hypothesis: The two samples likely come from different distributions."
    else:
        interpretation = "Fail to reject the null hypothesis: There's not enough evidence to conclude the samples come from different distributions."

    # Return a dictionary with all relevant information
    return {
        "Statistic": statistic,
        "P Value": p_value,
        "Reject Null?": reject_null,
        "Interpretation": interpretation,
        "Alpha": alpha,
    }


def perform_ks_test_standardized(sample1, sample2, alpha=0.05):
    """Perform KS test on standardized samples to isolate shape differences.

    Standardizing (mean=0, std=1) removes location and scale effects,
    allowing the KS test to focus purely on distributional shape.
    """
    s1 = standardize(np.asarray(sample1))
    s2 = standardize(np.asarray(sample2))
    statistic, p_value = stats.ks_2samp(s1, s2)
    reject_null = float(p_value) <= alpha

    return {
        "Statistic": statistic,
        "P Value": p_value,
        "Reject Null?": reject_null,
        "Alpha": alpha,
    }


def perform_wasserstein(sample1, sample2):
    """Compute Wasserstein distance (Earth Mover's Distance) on raw samples.

    Sensitive to differences in location (mean), scale (spread), and shape.
    """
    return wasserstein_distance(np.asarray(sample1), np.asarray(sample2))


def perform_wasserstein_standardized(sample1, sample2):
    """Compute Wasserstein distance on standardized samples.

    Also known as Earth Mover's Distance. Unlike KS which only captures the
    maximum pointwise CDF difference, Wasserstein considers the entire
    distribution, making it more robust for ranking samples by shape similarity.
    """
    s1 = standardize(np.asarray(sample1))
    s2 = standardize(np.asarray(sample2))
    return wasserstein_distance(s1, s2)


def perform_tvd(sample1, sample2): ...


def perform_chi_square_test(sample1, sample2, alpha=0.05):
    """Perform Chi-Square test for comparing two discrete distributions.

    Appropriate for categorical/discrete data where KS test is invalid.
    Compares observed frequencies between simulation and benchmark data.

    Args:
        sample1: Simulation decisions (discrete values)
        sample2: Benchmark decisions (discrete values)
        alpha: Significance level

    Returns:
        dict with Statistic, P Value, Reject Null?, Alpha
    """
    # Handle edge cases
    if not sample1 or not sample2:
        return {
            "Statistic": None,
            "P Value": None,
            "Reject Null?": None,
            "Alpha": alpha,
        }

    # Get all unique categories from both samples
    all_categories = sorted(set(sample1) | set(sample2))

    # If only one category exists, chi-square test is not meaningful
    if len(all_categories) < 2:
        return {
            "Statistic": 0.0,
            "P Value": 1.0,
            "Reject Null?": False,
            "Alpha": alpha,
        }

    # Count frequencies for each category
    counts1 = Counter(sample1)
    counts2 = Counter(sample2)

    # Build contingency table (categories x samples)
    observed = []
    for cat in all_categories:
        observed.append([counts1.get(cat, 0), counts2.get(cat, 0)])

    # Perform chi-square test
    try:
        chi2, p_value, dof, expected = chi2_contingency(observed)
        reject_null = float(p_value) <= alpha

        return {
            "Statistic": chi2,
            "P Value": p_value,
            "Reject Null?": reject_null,
            "Alpha": alpha,
        }
    except ValueError:
        # Handle edge cases where chi2_contingency fails
        return {
            "Statistic": None,
            "P Value": None,
            "Reject Null?": None,
            "Alpha": alpha,
        }


def perform_wasserstein_1(
    sample1: list,
    sample2: list,
    game_type: str,
    decision_index: int = 0,
) -> float:
    """
    Compute unified Wasserstein-1 distance for any game type.

    Automatically selects calculation method based on GAME_DECISION_ARRAY_CONFIG:
    - Ordered data: Normalized Wasserstein (EMD / max_range)
    - Categorical data: Total Variation Distance

    Args:
        sample1: Simulation decisions
        sample2: Benchmark decisions
        game_type: Game name to look up config
        decision_index: Which decision element to compare (default 0)

    Returns:
        float: Distance in [0, 1] range. Lower = more similar. None if invalid input.
    """
    if not sample1 or not sample2:
        return None

    # Get config for this game
    config = GAME_DECISION_ARRAY_CONFIG.get(game_type, [])
    if not config:
        logger.warning(
            f"Game type '{game_type}' not found in GAME_DECISION_ARRAY_CONFIG"
        )
        return None
    if len(config) <= decision_index:
        raise ValueError(
            f"Decision index {decision_index} out of range for game '{game_type}' "
            f"(config has {len(config)} decision(s))"
        )
    else:
        value_range, is_categorical = config[decision_index]

    # Convert to probability distributions
    if is_categorical:
        # Categorical: use TVD
        return _calc_tvd(sample1, sample2, value_range)
    else:
        # Ordered: use normalized Wasserstein
        return _calc_ordered_wasserstein(sample1, sample2, value_range)


def _calc_tvd(sample1: list, sample2: list, categories) -> float:
    """Total Variation Distance for categorical data.

    For unordered data, a miss is a miss (no geometric distance).
    TVD = 0.5 * sum(|P - Q|)
    Maximum value is 1.0 (completely disjoint distributions).
    """
    # Get all possible categories
    if isinstance(categories, tuple):
        all_cats = list(range(categories[0], categories[1] + 1))
    else:
        all_cats = list(categories)

    # Convert samples to probability distributions
    n1, n2 = len(sample1), len(sample2)

    prob1 = {cat: 0.0 for cat in all_cats}
    prob2 = {cat: 0.0 for cat in all_cats}

    for val in sample1:
        if val in prob1:
            prob1[val] += 1 / n1
    for val in sample2:
        if val in prob2:
            prob2[val] += 1 / n2

    # TVD = 0.5 * sum(|P - Q|)
    tvd = 0.5 * sum(abs(prob1[cat] - prob2[cat]) for cat in all_cats)
    return tvd


def _calc_ordered_wasserstein(
    sample1: list, sample2: list, value_range: tuple
) -> float:
    """Normalized Wasserstein-1 for ordered data.

    For ordered data, distance between categories matters.
    Normalizes by max possible distance to get [0, 1] range.
    """
    s1 = np.array(sample1)
    s2 = np.array(sample2)

    # Raw Wasserstein distance
    raw_w1 = wasserstein_distance(s1, s2)

    # Normalize by max possible distance
    min_val, max_val = value_range
    max_dist = max_val - min_val

    if max_dist == 0:
        return 0.0

    return raw_w1 / max_dist
