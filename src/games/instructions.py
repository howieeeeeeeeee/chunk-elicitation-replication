BEHAVIOR_GAMES = [
    "Dictator",
    "Ultimatum Strategy (Proposer)",
    "Ultimatum Strategy (Responder)",
    "Linear Public Good",
    "Prisoner's Dilemma",
    "Bomb Risk",
    "Trust in CC09 (trustor)",
    "Trust in CC09 (trustee)",
    "BoS in CDJFR89",
    "Stag Hunt in CDFR92",
]


GAME_DESCRIPTION = {
    "Dictator": """This is a one-shot game. You are paired with another player and must decide how to split $100. The other player will receive whatever amount you allocate to them. Please choose how much to give to the other player (an integer in dollars) and return your decision as an array, e.g.: THE_DECISION_ARRAY = [x]. The other player will receive $x and you will receive $100 - x.""",
    "Ultimatum Strategy (Proposer)": """This is a one-shot game. You are paired with another player. As the Proposer, you decide how to split $100 between yourself and the other player. The other player, as the Responder, has set a minimum amount they are willing to accept. If your proposal meets or exceeds their minimum, it is accepted. Otherwise, both players receive $0. Submit your decision as an array, e.g.: THE_DECISION_ARRAY = [x]. The value should be an integer between 0 and 100, representing how much you give to the other player.""",
    "Ultimatum Strategy (Responder)": """This is a one-shot game. You are paired with another player. As the Responder, you set the minimum amount you are willing to accept from the Proposer's offer. If the Proposer offers at least this amount, the offer is accepted. Otherwise, both players receive $0. Submit your decision as an array, e.g.: THE_DECISION_ARRAY = [x]. The value should be an integer between 0 and 100, representing the minimum amount you are willing to accept.""",
    "Linear Public Good": """This is a one-shot game with 4 players. Each player starts with an endowment of $20 and decides how much to contribute to a public pot. The total contributions will be multiplied by 1.5, and all players share the benefits. Your final earnings consist of the amount you keep from your initial endowment plus an equal share of the multiplied total contributions. Submit your contribution as an array, e.g.: THE_DECISION_ARRAY = [x]. The value should be an integer between 0 and 20, representing how much you contribute.""",
    "Prisoner's Dilemma": """This is a one-shot game. You are paired with another player, and both of you decide simultaneously to push or pull without knowing the other's choice. If both players push, you each earn $400. If one player pulls while the other pushes, the player who pulls earns $700, while the player who pushes gets $0. If both players pull, you each earn $300. Submit your decision as an array, e.g.: THE_DECISION_ARRAY = [x]. The value should be either 0 (for push) or 1 (for pull).""",
    "Bomb Risk": """This is a one-shot game. You are presented with 100 boxes, one of which contains a bomb. Your task is to decide how many boxes to collect. Each box collected increases your earnings by $1, but if you collect the box with the bomb, your earnings drop to $0. Submit your decision as an array, e.g.: THE_DECISION_ARRAY = [x]. The value should be an integer between 0 and 100, representing the number of boxes you choose to collect.""",
    "Random Number Generation": """This is a one-shot game. You are asked to randomly pick an integer from 1 to 100. Please choose an integer that you consider to be random. Submit your decision as an array, e.g.: THE_DECISION_ARRAY = [x]. The value should be a random integer between 1 and 100, representing your randomly chosen number.""",
    "Trust in CC09 (trustor)": """This is a one-shot trust game. You are paired with another player. As the Trustor, you start with $10. You make one choice: **Trust** (send your $10) or **Not Trust** (keep your $10). If you send $10, it is multiplied by 5 (becoming $50), and the other player (Trustee) will decide how much of the total $60 (their $10 endowment + $50) to return to you. If you Not Trust, both players keep $10. Submit your decision as an array, e.g.: THE_DECISION_ARRAY = [x]. The value should be 0 (Not Trust) or 1 (Trust).""",
    "Trust in CC09 (trustee)": """This is a one-shot trust game. You are paired with another player. As the Trustee, you start with $10. You must decide how much you *will* return *if* the Trustor trusts you (sends you their $10). If they trust you, their $10 is multiplied by 5 (becoming $50), giving you a total of $60 (your $10 + $50). You make this choice without knowing the Trustor's decision. Your choice is implemented only if they trust you. Submit your decision as an array, e.g.: THE_DECISION_ARRAY = [x]. The value should be an integer between 0 and 60, representing the amount you return.""",
    "BoS in CDJFR89": """This is a one-shot game. You are **Player B** and you are playing against **Player S**. You both must simultaneously choose either **Action 1** or **Action 2** without knowing the other's choice .
**Payoff Structure:**
Your earnings depend on the combination of actions chosen by you and Player S:
* If you choose Action 1 and Player S chooses Action 1: You both get 0 points .
* If you choose Action 2 and Player S chooses Action 2: You both get 0 points .
* If you choose Action 2 and Player S chooses Action 1: You get 600 points; Player S gets 200 points .
* If you choose Action 1 and Player S chooses Action 2: You get 200 points; Player S gets 600 points .
**Decision Output:**
Submit your decision as an array. The value should be 1 (for Action 1) or 2 (for Action 2).
`THE_DECISION_ARRAY = [x]`""",
    "Stag Hunt in CDFR92": """This is a one-shot game. You are Player B and you are playing against Player S. You both must simultaneously choose either Action 1 or Action 2 without knowing the other's choice.
Payoff Structure: The points you earn depend on the actions chosen by you and the player you are paired with:
* If you choose Action 1 and Player S chooses Action 1: You get 800 points; Player S gets 800 points.
* If you choose Action 1 and Player S chooses Action 2: You get 800 points; Player S gets 0 points.
* If you choose Action 2 and Player S chooses Action 1: You get 0 points; Player S gets 800 points.
* If you choose Action 2 and Player S chooses Action 2: You get 1000 points; Player S gets 1000 points.
**Decision Output:**
Submit your decision as an array. The value should be 1 (for Action 1) or 2 (for Action 2).
`THE_DECISION_ARRAY = [x]`""",
    "Arithmetic Verification": """This is a one-shot task. You are a participant in a basic numeracy test. You must strictly calculate the sum of 23 + 18. Submit your calculated answer as an array, e.g.: THE_DECISION_ARRAY = [x]. The value x must be the integer result of the addition.""",
    "Trivial Dominance": """This is a one-shot decision. You are offered a choice between two options to maximize your payoff. Option A pays you $100. Option B pays you $0. There is no trick; you simply choose the reward. Return 1 to choose Option A ($100) or 2 to choose Option B ($0). Submit your decision as an array, e.g.: THE_DECISION_ARRAY = [x].""",
    "TicTacToe Logic": """This is a one-shot game. You are playing Tic-Tac-Toe as Player X. The board positions are numbered 0 to 8 (from left to right, top to bottom). 0-2 is top row, 3-5 is middle row, 6-8 is bottom row. Currently, you (X) have pieces at positions 0 and 1. Your opponent (O) has pieces at positions 3 and 4. It is your turn. You must choose the specific move that wins the game immediately. Submit your decision as an array containing the position index, e.g.: THE_DECISION_ARRAY = [x].""",
    "TicTacToe Logic - L2": """This is a one-shot game. You are playing Tic-Tac-Toe as Player X. The board positions are numbered 0 to 8 (from left to right, top to bottom). 0-2 is top row, 3-5 is middle row, 6-8 is bottom row. Currently, you (X) have pieces at positions 2, 4, and 8. Your opponent (O) has pieces at positions 0, 1, 5, and 7. It is your turn. You must choose the specific move that wins the game immediately. Submit your decision as an array containing the position index, e.g.: THE_DECISION_ARRAY = [x].""",
}

# "TicTacToe Logic - L2": """This is a one-shot game. You are playing Tic-Tac-Toe as Player X. The board positions are numbered 0 to 8.

#     Current Board State:
#     - You (X) have pieces at positions 2, 4, and 8.
#     - Your opponent (O) has pieces at positions 0, 1, 5, and 7.

#     It is your turn. Positions 3 and 6 are empty. Analyze the board to identify which line is blocked and which line is open for a win. You must choose the specific move that wins the game immediately. Submit your decision as an array containing the position index, e.g.: THE_DECISION_ARRAY = [x].""",
# }

# "Trust (Investor)": """This is a one-shot trust game. You are paired with another player. As the investor, you decide how much of your $100 endowment to send to the other player. The amount you send will be tripled before reaching them. The other player will then decide how much to return to you. Submit your decision as an array, e.g.: THE_DECISION_ARRAY = [x]. The value should be an integer between 0 and 100, representing how much you send to the other player.""",
#     "Trust (Banker)": """This is a one-shot trust game. You are paired with another player. The other player, as the investor, has sent you an amount that was tripled before reaching you. Now, as the banker, you decide what percentage of this total amount to return to the investor. This percentage can range from 0 (returning nothing) to 100 (returning the entire amount, including the original investment and twice that as profit). Submit your decision as an array, e.g.: THE_DECISION_ARRAY = [x]. The value should be an integer between 0 and 100, representing the percentage of the total amount you choose to return to the investor.""",

GAME_DECISION_ARRAY = {
    "Dictator": "An array : [x (where x is an integer, specifying the amount of money given to the other party)]",
    "Ultimatum Strategy (Proposer)": "An array : [x (where x is an integer between 0 and 100, specifying the amount of money offered to the other player)]",
    "Ultimatum Strategy (Responder)": "An array : [x (where x is an integer between 0 and 100, specifying the minimum amount you would accept)]",
    "Linear Public Good": "An array : [x (where x is an integer, specifying the amount of money contributed to the public pot)]",
    "Trust (Investor)": "An array : [x (where x is an integer, specifying the amount of money sent to the other player)]",
    "Trust (Banker)": "An array : [x (where x is an integer between 0 and 100, specifying the percentage of the tripled amount to return)]",
    "Prisoner's Dilemma": "An array : [x (where x is either 0 or 1, specifying the decision to push (0) or pull (1))]",
    "Bomb Risk": "An array : [x (where x is an integer between 0 and 100, specifying the number of boxes to collect)]",
    "Random Number Generation": "An array : [x (where x is an integer between 1 and 100, specifying the randomly chosen number)]",
    "Trust in CC09 (trustor)": "An array : [x (where x is either 0 or 1, specifying the decision to Not Trust (0) or Trust (1))]",
    "Trust in CC09 (trustee)": "An array : [x (where x is an integer between 0 and 60, specifying the amount to return if trusted)]",
    "BoS in CDJFR89": "An array : [x (where x is either 1 or 2, specifying the decision to choose Action 1 (1) or Action 2 (2))]",
    "Stag Hunt in CDFR92": "An array : [x (where x is either 1 or 2, specifying the decision to choose Action 1 (1) or Action 2 (2))]",
    "Arithmetic Verification": "An array : [x (where x is the integer sum of 23 + 18)]",
    "Trivial Dominance": "An array : [x (where x is 1 for Option A, or 0 for Option B)]",
    "TicTacToe Logic": "An array : [x (where x is the integer index of the winning board position)]",
    "TicTacToe Logic - L2": "An array : [x (where x is the integer index of the winning board position)]",
}

GAME_DECISION_ARRAY_DESCRIPTION = {
    "Dictator": ["Amount to give"],
    "Ultimatum Strategy (Proposer)": ["Amount to offer"],
    "Ultimatum Strategy (Responder)": ["Min accept"],
    "Linear Public Good": ["Amount to contribute"],
    "Trust (Investor)": ["Amount to send"],
    "Trust (Banker)": ["Percentage to return"],
    "Prisoner's Dilemma": ["=1 if Pull/Defect"],
    "Bomb Risk": ["Boxes to collect"],
    "Random Number Generation": ["Random number"],
    "Trust in CC09 (trustor)": ["=1 if Trust"],
    "Trust in CC09 (trustee)": ["Amount to return if trusted"],
    "BoS in CDJFR89": [
        "=2 if Action 2 (the action preferred by the player choosing it)"
    ],
    "Stag Hunt in CDFR92": ["=1 if Action 1 (Safe Choice)"],
    "Arithmetic Verification": ["Calculated Sum"],
    "Trivial Dominance": ["Choice (1=$100, 0=$0)"],
    "TicTacToe Logic": ["Winning Move Index"],
    "TicTacToe Logic - L2": ["Winning Move Index"],
}


GAME_DECISION_ARRAY_CONFIG = {
    "Dictator": [[(0, 100), False]],
    "Ultimatum Strategy (Proposer)": [[(0, 100), False]],
    "Ultimatum Strategy (Responder)": [[(0, 100), False]],
    "Linear Public Good": [[(0, 20), False]],
    "Trust (Investor)": [[(0, 100), False]],
    "Trust (Banker)": [[(0, 100), False]],
    "Prisoner's Dilemma": [[[0, 1], True]],
    "Bomb Risk": [[(0, 100), False]],
    "Random Number Generation": [[(1, 100), False]],
    "Trust in CC09 (trustor)": [[[0, 1], True]],
    "Trust in CC09 (trustee)": [[(0, 60), False]],
    "BoS in CDJFR89": [[[1, 2], True]],
    "Stag Hunt in CDFR92": [[[1, 2], True]],
    "Arithmetic Verification": [[(0, 100), False]],
    "Trivial Dominance": [[[0, 1], True]],
    "TicTacToe Logic": [[[0, 1, 2, 3, 4, 5, 6, 7, 8], True]],
    "TicTacToe Logic - L2": [[[0, 1, 2, 3, 4, 5, 6, 7, 8], True]],
}  ## [[range],non-ordered/categorical], ordered is important as it affects the choice of statistical test


GAME_THEORETICAL_PREDICTIONS = {
    "Dictator": "In a purely rational scenario, players would give 0 to maximize their own payoff.",
    "Ultimatum Strategy (Proposer)": "In a purely rational scenario, players would either give 0 or 1.",
    "Ultimatum Strategy (Responder)": "In a purely rational scenario, players would either choose 0 or 1 as the minimum acceptance amount.",
    "Linear Public Good": "In a purely rational scenario, players would contribute 0 to the public pot to maximize their own payoff.",
    "Trust (Investor)": "In a purely rational scenario, a rational investor would anticipate the trustee will return 0, and so would send 0.",
    "Trust (Banker)": "In a purely rational scenario, a rational banker would return 0.",
    "Prisoner's Dilemma": "In a purely rational scenario, player would choose 1 (pull) as it is the dominant strategy, regardless of the other player's choice.",
    "Bomb Risk": "A risk-neutral rational player would choose to collect 50 boxes, as this maximizes the expected value of their earnings. Risk-averse players would choose fewer boxes, while risk-seeking players would choose more.",
    "Random Number Generation": "There is no theoretical prediction for this game.",
    "Trust in CC09 (trustor)": "In a purely rational scenario, a rational trustor would choose not to trust as it is rational for the trustee to return 0.",
    "Trust in CC09 (trustee)": "In a purely rational scenario, a rational trustee would return 0.",
    "BoS in CDJFR89": "Pure Strategy Nash Equilibria: There are two asymmetric pure equilibria. 1) Player B chooses Action 2 and Player S chooses Action 1 (Player B's preferred outcome yielding 600 points). 2) Player B chooses Action 1 and Player S chooses Action 2 (Player S's preferred outcome yielding 200 points). Mixed Strategy Nash Equilibrium: Player B should choose Action 2 (the aggressive strategy) 75 percent of the time and Action 1 (the accommodating strategy) 25 percent of the time.",
    "Stag Hunt in CDFR92": "Pure Strategy Nash Equilibria: There are two symmetric pure equilibria. 1) Both players choose Action 1 (the 'Risk Dominant' or safe equilibrium yielding 800 points). 2) Both players choose Action 2 (the 'Pareto Dominant' or risky equilibrium yielding 1000 points). Mixed Strategy Nash Equilibrium: Player B should choose Action 2 80 percent of the time and Action 1 20 percent of the time.",
    "Arithmetic Verification": "The correct answer is 41 (23 + 18 = 41).",
    "Trivial Dominance": "A rational player would choose Option A (1) to maximize payoff, as it strictly dominates Option B.",
    "TicTacToe Logic": "The winning move is position 2, which completes the top row (positions 0, 1, 2) for an immediate win.",
    "TicTacToe Logic - L2": "The winning move is position 6, which completes the diagonal (positions 2, 4, 6) for an immediate win.",
}


GAME_FOCAL_POINTS = {
    "Dictator": "The focal point is often to keep everything (giving nothing) or to split the money equally (giving half), with equal split emerging as a social norm.",
    "Ultimatum Strategy (Proposer)": "The focal point is to offer half (a fair split) or the minimum amount they believe the responder will accept.",
    "Ultimatum Strategy (Responder)": "The focal point is to accept any positive offer (economic rationality) or reject offers below half (fairness norm).",
    "Linear Public Good": "The focal points are to contribute nothing (free-riding), contribute half of the endowment, or contribute all of the endowment to the public pot.",
    "Trust (Investor)": "The focal points are to send nothing (complete distrust), send half (balanced trust), or send everything (complete trust).",
    "Trust (Banker)": "The focal point is to return a fair portion of the received amount, often around half, to reciprocate the investor's trust.",
    "Prisoner's Dilemma": "The focal point is to defect (pull), as it is the dominant strategy, although mutual cooperation leads to a better collective outcome.",
    "Bomb Risk": "The focal points typically emerge around collecting a small number of boxes (1-5) representing a cautious approach, or around 50 boxes, representing a risk-neutral choice.",
    "Random Number Generation": "There is no focal point for this game.",
}
