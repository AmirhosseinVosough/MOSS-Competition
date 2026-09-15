"""Deciding whether a sentence may be spoken aloud.

Split out because this is the one piece of the agent with no dependency on
LiveKit, Moss, or any conversation state -- it is pure text handling, and it
is the part most worth being able to read on its own.
"""

import re

# ---------------------------------------------------------------------------
# Grounding gate
#
# The agent must never speak a dose, a time or a phone number that is not in
# Bill's notes. Checking every sentence before it is spoken costs one retrieval
# per sentence -- affordable only because retrieval is ~5ms. At a hosted vector
# database's 200-500ms this would add seconds to every reply.
#
# The check is deliberately narrow. Only sentences carrying a specific factual
# claim are verified; conversation passes untouched. An over-eager gate that
# hedges ordinary speech would make the agent useless, which is a worse failure
# than the one it is guarding against.
# ---------------------------------------------------------------------------

# Spoken numbers, because the agent is instructed to say "five hundred" rather
# than "500". Ordinals are included because dates are spoken as "the fifteenth".
# These are parsed as phrases, not as separate words: "five hundred" is 500, and
# reading it as {5, 100} once caused the gate to reject a correct dose.
_NUMBER_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
    # Ordinals, for spoken dates.
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20, "thirtieth": 30,
}
_NUMBER_SCALES = {"hundred": 100, "thousand": 1000}

# A sentence is only worth checking if it asserts something specific.
_CLAIM_HINTS = (
    "mg", "milligram", "microgram", "tablet", "capsule", "dose", "doses",
    "pill", "pills", "take", "takes", "taking", "appointment", "allergic",
    "allergy", "o'clock", "am", "pm", "morning", "afternoon", "evening",
    "bedtime", "daily", "twice", "once", "number", "call",
)


def numbers_in(text: str) -> set:
    """Every number in a piece of text, as digit strings.

    Handles digits ("500mg") and spoken phrases ("five hundred", "twenty
    two", "the fifteenth"). Phrases are accumulated rather than read word by
    word: "five hundred" is 500, not 5 and 100.
    """
    low = text.lower()
    found = set(re.findall(r"\d+", low))

    total = 0     # completed hundreds/thousands within the current phrase
    current = 0   # the part being accumulated
    seen = False

    def flush():
        nonlocal total, current, seen
        if seen:
            value = total + current
            if value:
                found.add(str(value))
        total, current, seen = 0, 0, False

    for word in re.findall(r"[a-z]+", low):
        if word in _NUMBER_UNITS:
            current += _NUMBER_UNITS[word]
            seen = True
        elif word in _NUMBER_SCALES:
            scale = _NUMBER_SCALES[word]
            # "five hundred" -> 500; a bare "hundred" -> 100.
            current = (current or 1) * scale
            if scale >= 1000:
                total += current
                current = 0
            seen = True
        elif word == "and" and seen:
            # "one hundred and twenty" keeps going.
            continue
        else:
            flush()
    flush()
    return found


def is_checkable(sentence: str) -> bool:
    """True if the sentence asserts something specific enough to verify.

    Narrow on purpose: a gate that hedges ordinary conversation is worse than
    no gate at all.
    """
    low = sentence.lower()
    if not any(h in low for h in _CLAIM_HINTS):
        return False
    # A claim worth checking names a number -- a dose, a time, a date, a
    # phone number. Advice without one has nothing to get factually wrong.
    return bool(numbers_in(sentence))
