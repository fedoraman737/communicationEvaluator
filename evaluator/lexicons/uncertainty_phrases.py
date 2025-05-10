UNCERTAINTY_PHRASES = [
    # Phrases that indicate lack of confidence or knowledge
    "i think so",
    "i guess",
    "i'm not sure",
    "i don't know for certain",
    "to be honest, i'm not certain",
    "i'm not too familiar with that"
]

# Phrases that indicate uncertainty but may be appropriate when offering flexible solutions
FLEXIBILITY_PHRASES = [
    "maybe",
    "perhaps",
    "it might be",
    "possibly",
    "i believe",
    "may be able to"
]

# Phrases that are slightly better than direct uncertainty but still not fully confident
# Could be used to suggest improvement towards more confident phrasing.
LESS_CONFIDENT_PHRASES = [
    "i'll have to check", # Good that they will check, but initial lack of knowledge
    "let me see if i can find that out",
    "that's a good question, let me verify"
]

# Phrases that indicate appropriate caution rather than uncertainty
APPROPRIATE_CAUTION_PHRASES = [
    "typically",
    "generally",
    "in most cases",
    "usually",
    "often"
] 