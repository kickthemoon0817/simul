"""The regular-expression guard and linear-time search that Blender's and Unreal's name searches share."""

from __future__ import annotations

import random
import re
import time
from typing import List, Optional

import pytest

from simul.utils.regex_guard import (
    BLENDER_NAME_LENGTH,
    UNREAL_TAG_LENGTH,
    RegexRefused,
    _LinearSearch,
    _Unbounded,
    exponential_regex_construct,
    name_searcher,
)

#: What a failing search of one name of the longest length may take in a test,
#: about five times what the guard's own budget lets through on this machine.
_TEST_BUDGET_SECONDS = 0.1
#: The flags and longest names the two callers search with.
BLENDER = (0, BLENDER_NAME_LENGTH)
UNREAL = (re.IGNORECASE, UNREAL_TAG_LENGTH)


def _refused(pattern: str, caller: tuple = BLENDER) -> Optional[str]:
    flags, length = caller
    return exponential_regex_construct(pattern, flags, length)


@pytest.mark.parametrize(
    ("pattern", "refusal"),
    [
        # Repeated groups that can split the same text in several ways.
        (r"^(\w+)+_light$", "a quantifier inside a repeated group"),
        (r"(\w+_)*light", "a quantifier inside a repeated group"),
        (r"(a|ab)*c", "alternatives inside a repeated group"),
        # B1-17: quantifiers that can divide the same characters. The audit's
        # pattern froze Blender for 15.6 s over ten 63-character names.
        (".*.*.*.*.*.*x", "6 quantifiers that can divide the same characters"),
        (".*.*.*x", "3 quantifiers"),
        (r"\w*\w*\w*x", "3 quantifiers"),
        ("(.*)(.*)(.*)x", "3 quantifiers"),
        (r"\d*.*[a-z]*x", "3 quantifiers"),
        ("(ab)*(ab)*(ab)*c", "3 quantifiers"),
        (".{0,100}.{0,100}.{0,100}x", "3 quantifiers"),
        (".*.*.*$", "3 quantifiers"),
        ("(?i)A*a*A*x", "3 quantifiers"),
        ("(?m)^.*.*.*x", "3 quantifiers"),
        ("^.*.*.*.*x", "4 quantifiers"),
        ("(?:.*.*.*x|y)", "3 quantifiers"),
        (".*a.*a.*a.*x", "4 quantifiers"),
        # Review: an optional group was left out, and an empty-matching
        # quantifier between two others broke their link. 15.4 s and 0.36 s.
        (r".*(.*)?(.*)?.*x", "4 quantifiers"),
        (r".*(?:.*)?.*x", "3 quantifiers"),
        (r"[a-z]*[0-9]*[a-z]*[0-9]*[a-z]*x", "3 quantifiers"),
        # Pairs one after another multiply too: 131 ms on 255 characters.
        (r"\d*\d*x[a-z]*[a-z]*y", "3 quantifiers"),
        # Optional characters are choices: a?{25}a{25} takes 2**25 tries.
        ("a?" * 25 + "a" * 25, "25 quantifiers"),
    ],
)
def test_patterns_whose_failing_search_backtracks_are_refused(
    pattern: str, refusal: str
) -> None:
    construct = _refused(pattern)
    assert construct is not None
    assert construct.startswith(refusal), construct


@pytest.mark.parametrize(
    "pattern",
    [
        "A*a*A*x",
        "[A-Z]*[a-z]*[A-Z]*[a-z]*x",
        ".*.*x",
        ".*a.*a.*x",
        ".*left.*front.*wheel.*joint",
    ],
)
def test_unreal_tags_are_guarded_as_unreal_searches_them(pattern: str) -> None:
    """Review: Unreal compiles with IGNORECASE and searches 1023-character tags.

    Analysed case-sensitively, A*a*A*x looked safe and took 191 s on one tag;
    [A-Z]*[a-z]*[A-Z]*[a-z]*x took 82 s.
    """
    assert _refused(pattern, UNREAL) is not None


@pytest.mark.parametrize(
    ("pattern", "callers"),
    [
        (r"^chair_\d+$", [BLENDER, UNREAL]),
        (r"^(chair|table)$", [BLENDER, UNREAL]),
        (r"(_\d+)?$", [BLENDER, UNREAL]),
        (r".*leg.*", [BLENDER, UNREAL]),
        (r"(ab{2})+", [BLENDER, UNREAL]),
        (r"left.*front", [BLENDER, UNREAL]),
        (r"^.*.*x", [BLENDER, UNREAL]),
        (r"^SM_\w+_\w+_LOD\d+$", [BLENDER, UNREAL]),
        (r"\d+\.\d+\.\d+x", [BLENDER, UNREAL]),
        # Nothing after the last quantifiers can fail.
        (".*.*.*", [BLENDER, UNREAL]),
        # Possessive quantifiers never give back what they took.
        (r"\w*+\w*+\w*+x", [BLENDER, UNREAL]),
        # Counts too narrow to matter.
        (".{0,8}.{0,8}.{0,8}x", [BLENDER]),
        (r"^(?!.*Light).*$", [BLENDER, UNREAL]),
        (r".*(?<!_proxy)$", [BLENDER, UNREAL]),
        # It matches the empty name, so any search succeeds at once.
        ("(a.*.*.*x)?", [BLENDER, UNREAL]),
    ],
)
def test_patterns_that_cannot_backtrack_for_long_are_allowed(pattern: str, callers: List[tuple]) -> None:
    for caller in callers:
        found = _refused(pattern, caller)
        assert found is None, (caller, found)


# ---------------------------------------------------------------------------
# Every pattern the guard lets re search finishes in time (fuzzed)
# ---------------------------------------------------------------------------

_ATOMS = [".", "a", "A", "x", "_", r"\d", r"\w", "[a-z]", "[A-Z]", "[^_]", "[0-9]", "(?:ab)", "(?:a|b)", "[aA]"]
_QUANTIFIERS = ["", "", "*", "+", "?", "{0,20}", "{1,3}", "*?"]


def _random_pattern(rng: random.Random, longest: int = 8) -> str:
    parts = ["^"] if rng.random() < 0.2 else []
    for _ in range(rng.randint(2, longest)):
        if rng.random() < 0.15:
            inner = "".join(rng.choice(_ATOMS) + rng.choice(_QUANTIFIERS[:5]) for _ in range(rng.randint(1, 2)))
            parts.append(f"(?:{inner})?")
            continue
        atom, quantifier = rng.choice(_ATOMS), rng.choice(_QUANTIFIERS)
        if atom == "(?:a|b)" and quantifier not in ("", "?"):
            quantifier = "?"
        parts.append(atom + quantifier)
    parts.append(rng.choice(["x", r"\d", "$", "Z", ""]))
    return "".join(parts)


def _hard_names(length: int) -> List[str]:
    """Names made of what the generated patterns match, repeated: their slowest failures."""
    units = ["a", "A", "x", "_", "1", "ab", "aA", "a_", "a1", "Ab_"]
    return [(unit * (length // len(unit) + 1))[:length] for unit in units]


@pytest.mark.parametrize("caller", [BLENDER, (re.IGNORECASE, BLENDER_NAME_LENGTH), UNREAL], ids=str)
def test_every_pattern_the_guard_allows_searches_a_longest_name_in_time(caller: tuple) -> None:
    """Review: the guard has to be sound, since re holds the GIL while it matches.

    Generated patterns mix quantifiers, optional groups, alternatives, classes
    that overlap or not, and anchors; each one the guard lets re search must
    fail on every hard name of the caller's longest length within the budget.
    """
    flags, length = caller
    rng = random.Random(20261108 + flags + length)
    names = _hard_names(length)
    allowed = 0
    for _ in range(150):
        pattern = _random_pattern(rng)
        if exponential_regex_construct(pattern, flags, length) is not None:
            continue
        allowed += 1
        compiled = re.compile(pattern, flags)
        for name in names:
            start = time.perf_counter()
            compiled.search(name)
            elapsed = time.perf_counter() - start
            assert elapsed < _TEST_BUDGET_SECONDS, (pattern, name[:12], elapsed)
    assert allowed > 40


# ---------------------------------------------------------------------------
# The linear-time search
# ---------------------------------------------------------------------------


def test_the_linear_search_finds_what_re_finds() -> None:
    """Same answers as re.search, fuzzed over patterns and short names."""
    rng = random.Random(7)
    alphabet = "aAbx_1. \n"
    extras = [r"\b", r"\B", "$", r"\Z", "(?=a)", "(?!x)", "(?<=_)", "(?<!b)"]
    compared = 0
    for _ in range(400):
        pattern = _random_pattern(rng, 6)
        if rng.random() < 0.3:
            position = rng.randint(0, len(pattern))
            pattern = pattern[:position] + rng.choice(extras) + pattern[position:]
        if rng.random() < 0.2:
            pattern = "(?i)" + pattern
        try:
            compiled = re.compile(pattern)
            search = _LinearSearch(pattern, 0, BLENDER_NAME_LENGTH)
        except (re.error, _Unbounded):
            continue
        for _ in range(25):
            name = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 10)))
            assert search(name) == (compiled.search(name) is not None), (pattern, name)
            compared += 1
    assert compared > 5000


@pytest.mark.parametrize(
    ("pattern", "matching"),
    [
        (r".*left.*front.*wheel.*joint", "left_front_wheel_joint"),
        (r"robot.*arm.*left.*joint.*2", "robot_arm_left_joint_2"),
        (r".*.*.*.*.*.*x", "box"),
        (r"^(\w+)+_x$", "long_x"),
        (r".*(.*)?(.*)?.*x", "box"),
        (r"[a-z]*[0-9]*[a-z]*[0-9]*[a-z]*x", "box"),
        (r"^[A-Z][a-z]+(_[A-Z][a-z]+)*$", "Base_Link"),
        (r"(?i)[A-Z]*[a-z]*[A-Z]*[a-z]*x", "ABx"),
        ("a?" * 25 + "a" * 25, "a" * 30),
    ],
)
def test_blender_searches_backtracking_patterns_in_linear_time(pattern: str, matching: str) -> None:
    """Review: realistic patterns were refused; searched without backtracking they are cheap.

    Python's re takes from a tenth of a second to hours on these names.
    """
    assert exponential_regex_construct(pattern) is not None
    search = name_searcher(pattern)
    names = _hard_names(BLENDER_NAME_LENGTH)
    start = time.perf_counter()
    for name in names:
        search(name)
    assert time.perf_counter() - start < _TEST_BUDGET_SECONDS
    assert search(matching) is True
    assert search(matching.replace("x", "y").replace("2", "3").replace("_", "-").lower()[:2]) is False


def test_a_pattern_neither_search_can_bound_is_refused_with_its_reason() -> None:
    with pytest.raises(RegexRefused, match="a back-reference") as refused:
        name_searcher(r"(.*).*.*.*\1x", subject="name_pattern")
    assert str(refused.value).startswith("name_pattern has ")
    assert refused.value.error_type == "ValidationError"
    assert refused.value.details["code"] == "regex_backtracking"
