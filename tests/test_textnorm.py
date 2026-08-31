"""Tests for spoken-number normalisation of STT output."""

from local_voice_ai.textnorm import _collapse_digit_runs, normalize_numbers


class TestDigitRuns:
    def test_phone_number_becomes_one_number(self) -> None:
        # The case that motivated this: nemotron spells phone numbers out.
        assert (
            normalize_numbers("Call me at four one five five five five zero one nine eight.")
            == "Call me at 4155550198."
        )

    def test_oh_reads_as_zero_inside_a_run(self) -> None:
        assert normalize_numbers("Call four oh five five five one two.") == "Call 4055512."

    def test_bare_oh_is_left_alone(self) -> None:
        # "oh" only means zero inside a digit run, never as an interjection.
        assert normalize_numbers("Oh, I see.") == "Oh, I see."

    def test_short_runs_are_not_collapsed(self) -> None:
        # Two digits are more likely prose than a spoken number string.
        assert _collapse_digit_runs("one two") == "one two"

    def test_run_at_threshold_is_collapsed(self) -> None:
        assert _collapse_digit_runs("one two three") == "123"

    def test_punctuation_ends_a_run(self) -> None:
        assert _collapse_digit_runs("one two, three four five") == "one two, 345"

    def test_zip_code(self) -> None:
        assert normalize_numbers("My zip is nine four one zero five.") == "My zip is 94105."


class TestCompoundNumbers:
    def test_compound_number(self) -> None:
        assert normalize_numbers("an RTX four thousand seventy Super") == "an RTX 4070 Super"

    def test_tens(self) -> None:
        assert normalize_numbers("extension twenty three") == "extension 23"


class TestLeavesProseAlone:
    def test_plain_sentence_unchanged(self) -> None:
        text = "The quick brown fox jumps over the lazy dog."
        assert normalize_numbers(text) == text

    def test_empty_string(self) -> None:
        assert normalize_numbers("") == ""

    def test_no_digits_at_all(self) -> None:
        text = "Speech recognition latency matters more than raw accuracy."
        assert normalize_numbers(text) == text
