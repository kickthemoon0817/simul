"""The regular-expression guard that Blender's and Unreal's name searches share."""

from __future__ import annotations

from typing import Optional

import pytest

from simul.utils.regex_guard import exponential_regex_construct


@pytest.mark.parametrize(
    ("pattern", "refusal"),
    [
        # Repeated groups that can split the same text in several ways.
        (r"^(\w+)+_light$", "a quantifier inside a repeated group"),
        (r"(\w+_)*light", "a quantifier inside a repeated group"),
        (r"(a|ab)*c", "alternatives inside a repeated group"),
        # B1-17: quantifiers in a row over the same characters. The audit's
        # pattern froze Blender for 15.6 s over ten 63-character names; three
        # .* took 140 ms on one 255-character name, four took 6 s.
        (".*.*.*.*.*.*x", "6 quantifiers in a row"),
        (".*.*.*x", "3 quantifiers in a row"),
        (r"\w*\w*\w*x", "3 quantifiers in a row"),
        ("(.*)(.*)(.*)x", "3 quantifiers in a row"),
        (r"\d*.*[a-z]*x", "3 quantifiers in a row"),
        ("(ab)*(ab)*(ab)*c", "3 quantifiers in a row"),
        (".{0,100}.{0,100}.{0,100}x", "3 quantifiers in a row"),
        (".*.*.*$", "3 quantifiers in a row"),
        ("(?i)A*a*A*x", "3 quantifiers in a row"),
        # re.MULTILINE: ^ matches after each newline, so it anchors nothing.
        ("(?m)^.*.*.*x", "3 quantifiers in a row"),
        ("^.*.*.*.*x", "4 quantifiers in a row"),
        # Inside an optional group or an alternative.
        ("(a.*.*.*x)?", "3 quantifiers in a row"),
        ("(?:.*.*.*x|y)", "3 quantifiers in a row"),
        (".*a.*a.*a.*x", "4 quantifiers that can match the same characters"),
    ],
)
def test_patterns_whose_failing_search_backtracks_are_refused(
    pattern: str, refusal: str
) -> None:
    construct = exponential_regex_construct(pattern)
    assert construct is not None
    assert construct.startswith(refusal)


@pytest.mark.parametrize(
    "pattern",
    [
        r"^chair_\d+$",
        r"^(chair|table)$",
        r"(_\d+)?$",
        r".*leg.*",
        r"(ab{2})+",
        # Two in a row take milliseconds; anchored, three do.
        ".*.*x",
        "^.*.*.*x",
        r"\A.*.*.*x",
        # Characters between them pin the divisions on realistic names.
        ".*a.*a.*x",
        "foo.*bar.*baz",
        ".*foo.*bar.*baz",
        r"^SM_\w+_\w+_LOD\d+$",
        r"SM_\w+_\w+_LOD\d+$",
        # A character one run cannot match ends the run.
        r"\d+\.\d+\.\d+x",
        "[a-z]*[0-9]*[a-z]*x",
        # Nothing after the last quantifier can fail.
        ".*.*.*",
        # Possessive quantifiers never give back what they took.
        r"\w*+\w*+\w*+x",
        # Counts too narrow to matter.
        ".{0,8}.{0,8}.{0,8}x",
        # Case matters without re.IGNORECASE.
        "A*a*A*x",
    ],
)
def test_patterns_that_cannot_backtrack_for_long_are_allowed(pattern: str) -> None:
    found: Optional[str] = exponential_regex_construct(pattern)
    assert found is None, found
