BLUNT_NEGATIVE_PHRASES = [
    # Direct negatives without context
    "nothing we can do",
    "nothing i can do",
    "can't do that",
    "cannot do that",
    "it's not possible",
    "that's not possible",
    "unable to help",
    "can't help you with that",
    "i can't do anything about that",
    "we don't do that",
    "that option isn't available",

    # Strongly negative with finality
    "it's against policy",
    "our policy doesn't allow that",
    "that's just how it is",
    "no way to do that",
    "there are no options"
]

# Contextual negative phrases that are acceptable in certain situations
# These should not be penalized when properly explained or followed by alternatives
CONTEXTUAL_NEGATIVES = [
    "unfortunately",
    "unable to process",
    "cannot approve",
    "doesn't meet our criteria",
    "not eligible for",
    "outside of our policy",
    "exceeds our limits"
] 