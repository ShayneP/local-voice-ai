"""Inverse text normalisation for STT output — spoken numbers to digits.

Nemotron transcribes verbatim: "call me at four one five five five five zero
one nine eight" rather than "415-555-0198". Whisper applies this normalisation
inside the model; nemotron does not, and it was the only real accuracy gap
between the two.

Two passes, because they fail in different ways:

1. Digit *runs* ("four one five five...") are collapsed first, by hand. Passing
   them to a general number parser yields "4 1 5 5 5 5 01 9 8" — it groups
   "zero one" into "01" and leaves the rest scattered.
2. Everything else ("twenty three", "four thousand seventy") goes to
   ``text_to_num``, which handles compound numbers properly.

``text_to_num`` is optional; without it pass 1 still runs.
"""

from __future__ import annotations

import re

# "oh" is only read as zero inside a run, so the bare interjection survives.
_DIGIT_WORDS = {
    "zero": "0", "oh": "0", "o": "0", "nought": "0",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
}

# Below this, a sequence is more likely to be prose ("one two punch") than a
# spoken digit string, so leave it for the number parser.
_MIN_RUN = 3

_TOKEN_RE = re.compile(r"[A-Za-z']+|\d+|[^A-Za-z'\d]+")


def _digit_of(token: str) -> str | None:
    return _DIGIT_WORDS.get(token.lower().strip("'"))


def _collapse_digit_runs(text: str, min_run: int = _MIN_RUN) -> str:
    """Join runs of ``min_run``+ spoken digits into one number.

    "four one five five five five zero one nine eight" -> "4155550198"
    """
    tokens = _TOKEN_RE.findall(text)
    out: list[str] = []
    i = 0

    while i < len(tokens):
        digit = _digit_of(tokens[i])
        if digit is None:
            out.append(tokens[i])
            i += 1
            continue

        # Greedily take digit words separated only by whitespace.
        digits = [digit]
        j = i + 1
        while j + 1 < len(tokens) and tokens[j].isspace():
            nxt = _digit_of(tokens[j + 1])
            if nxt is None:
                break
            digits.append(nxt)
            j += 2

        if len(digits) >= min_run:
            out.append("".join(digits))
            i = j
        else:
            # Too short to be a digit string — leave the words untouched.
            out.append(tokens[i])
            i += 1

    return "".join(out)


def normalize_numbers(text: str, lang: str = "en") -> str:
    """Convert spoken numbers in ``text`` to digits. Never raises."""
    if not text:
        return text

    result = _collapse_digit_runs(text)

    try:
        from text_to_num import alpha2digit
    except ImportError:
        return result

    try:
        return alpha2digit(result, lang)
    except Exception:
        # A parser failure must not cost us the transcript.
        return result
