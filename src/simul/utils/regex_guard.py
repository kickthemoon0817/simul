"""Refuse regular expressions whose search can backtrack for hours.

Some read-only tools search names with a caller's regular expression inside
an editor, where nothing can interrupt the search: Unreal actor tags on the
editor's game thread, Blender object names on Blender's main thread. Python's
``re`` backtracks, so one pattern can freeze the editor. The Blender add-on
bundles this module, so it imports only the standard library.
"""

from __future__ import annotations

from typing import Any, Optional


def exponential_regex_construct(pattern: str) -> Optional[str]:
    """Name a construct that can make a search for ``pattern`` take exponential time.

    A repeated group that can match the same text in more than one way, as
    ``(\\w+)+`` or ``(a|ab)*`` can, makes a failing search try every way of
    splitting the searched text between its repetitions: hours for a
    40-character name. Such a group is one that repeats and holds a quantifier
    of variable length, or alternatives. A fixed count inside it
    (``(ab{2})+``) cannot split, and an optional group (``(_\\d+)?``) does not
    repeat, so both are allowed.

    Args:
        pattern: A valid regular expression.

    Returns:
        A description of the first such construct, or None.
    """
    # The stdlib's own parser, private since 3.11 (formerly sre_parse).
    from re import _constants as sre, _parser  # type: ignore[attr-defined]

    repeats = (sre.MAX_REPEAT, sre.MIN_REPEAT, sre.POSSESSIVE_REPEAT)

    def walk(items: Any, repeated: bool) -> Optional[str]:
        for op, av in items:
            found: Optional[str] = None
            if op in repeats:
                low, high, body = av
                if repeated and low != high:
                    return "a quantifier inside a repeated group, as in (\\w+)+"
                found = walk(body, repeated or high > 1)
            elif op in (sre.BRANCH, sre.GROUPREF_EXISTS):
                if repeated:
                    return "alternatives inside a repeated group, as in (a|ab)*"
                branches = av[1] if op == sre.BRANCH else [av[1], av[2] or []]
                for branch in branches:
                    found = found or walk(branch, repeated)
            elif op == sre.SUBPATTERN:
                found = walk(av[3], repeated)
            elif op in (sre.ASSERT, sre.ASSERT_NOT):
                found = walk(av[1], repeated)
            elif op == sre.ATOMIC_GROUP:
                found = walk(av, repeated)
            if found:
                return found
        return None

    return walk(_parser.parse(pattern), False)
