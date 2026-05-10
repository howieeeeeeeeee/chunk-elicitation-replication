from games.treatments import (
    THEORETICAL_PREDICTIONS,
    FOCAL_POINTS,
    INCENTIVE_SIZES,
)
from games.instructions import GAME_DESCRIPTION
import logging

# Set up the logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a handler and set its level
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Create a formatter and set it for the handler
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

# Add the handler to the logger if it doesn't already have one
if not logger.handlers:
    logger.addHandler(handler)


## return true if it exist and is non empty string
def validate_treatment(simulation_config, instruction_config):
    game_type = simulation_config.get("game_type", "")
    theoretical_prediction = instruction_config.get("theoretical_prediction", False)
    focal_point = instruction_config.get("focal_point", False)
    context = instruction_config.get("context", "")
    incentive_size = instruction_config.get("incentive_size", "")

    if focal_point:
        logger.info(f"Focal point is no longer supported")
        return False

    if game_type and not GAME_DESCRIPTION.get(game_type, ""):
        logger.info(f"Game type '{game_type}' not found in GAME_DESCRIPTION")
        return False
    if context and incentive_size and context not in INCENTIVE_SIZES[incentive_size]:
        logger.info(
            f"Invalid combination of incentive size '{incentive_size}' and context '{context}'"
        )
        return False
    if game_type and focal_point and not FOCAL_POINTS[focal_point].get(game_type, ""):
        logger.info(f"Focal point not defined for game type '{game_type}'")
        return False
    if (
        game_type
        and theoretical_prediction
        and not THEORETICAL_PREDICTIONS[theoretical_prediction].get(game_type, "")
    ):
        logger.info(f"Theoretical prediction not defined for game type '{game_type}'")
        return False

    explain_reasoning_mode = instruction_config.get("explain_reasoning_mode", "basic")
    if explain_reasoning_mode == "split":
        if not instruction_config.get("explain_reasoning", False):
            logger.info("Split reasoning mode requires explain_reasoning=True")
            return False
        if simulation_config.get("simulation_mode") != "BatchSimulate":
            logger.info("Split reasoning mode only supports BatchSimulate")
            return False
        if simulation_config.get("batch_mode", "Independent") != "Independent":
            logger.info("Split reasoning mode requires batch_mode='Independent'")
            return False
        split_n = instruction_config.get("split_n")
        batch_simulation_n = simulation_config.get("batch_simulation_n")
        if not isinstance(split_n, int) or split_n <= 0:
            logger.info("split_n must be a positive integer when mode is 'split'")
            return False
        if not isinstance(batch_simulation_n, int) or batch_simulation_n <= 0:
            logger.info("batch_simulation_n must be a positive integer in split mode")
            return False
        if batch_simulation_n % split_n != 0:
            logger.info(
                f"split_n ({split_n}) must evenly divide batch_simulation_n ({batch_simulation_n})"
            )
            return False
    elif explain_reasoning_mode != "basic":
        logger.info(
            f"Unknown explain_reasoning_mode '{explain_reasoning_mode}' (expected 'basic' or 'split')"
        )
        return False

    return True
