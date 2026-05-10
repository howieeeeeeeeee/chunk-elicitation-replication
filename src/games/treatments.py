from .instructions import GAME_THEORETICAL_PREDICTIONS, GAME_FOCAL_POINTS

CONTEXT_TREATMENTS = {
    "Not Specified": "",
    "Classroom": "This is part of a classroom experiment.",
    "Lab": "This is a laboratory experiment.",
}

INCENTIVE_SIZES = {
    "Not Specified": {"Not Specified": "", "Classroom": "", "Lab": ""},
    "Standard": {
        "Classroom": "The impact of this experiment on your final semester grade is low.",
        "Lab": "Your earnings from this experiment are low compared to your monthly income outside of it.",
    },
    "High": {
        "Classroom": "The impact of this experiment on your final semester grade is high.",
        "Lab": "Your earnings from this experiment are high compared to your monthly income outside of it.",
    },
}


EXPLAIN_REASONING = {True: "", False: ""}

PRIVACY_TREATMENTS = {
    "Not Specified": "",
    "Public": "Please note that your decisions may be shared with other participants and may be subject to discussion.",
    "Private": "Your decisions will remain strictly confidential. No other participant will know your choices.",
}

THEORETICAL_PREDICTIONS = {
    True: GAME_THEORETICAL_PREDICTIONS,
    False: {},
}


FOCAL_POINTS = {
    True: GAME_FOCAL_POINTS,
    False: {},
}
