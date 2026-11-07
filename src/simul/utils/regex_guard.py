"""Refuse regular expressions whose search can backtrack for seconds to hours.

Some read-only tools search names with a caller's regular expression inside
an editor, where nothing can interrupt the search: Unreal actor tags on the
editor's game thread, Blender object names on Blender's main thread. Python's
``re`` backtracks, so one pattern can freeze the editor. The Blender add-on
bundles this module, so it imports only the standard library.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

#: Characters the overlap of two character classes is tested on, besides the
#: pattern's own literals: Latin-1 and Latin Extended, plus a few letters,
#: digits and spaces of other scripts.
_SAMPLE = [chr(code) for code in range(0x250)] + list("αΩЖж中한٣  　")
#: A quantifier whose count can vary by this much or more takes a stretch of a
#: name in many ways; ``x{0,8}`` cannot take enough for it to matter.
_WIDE = 16
#: Quantifiers in a row (nothing they cannot match between them) that a search
#: may have: two ``.*`` took 2 ms on a 255-character name (Blender 5's limit,
#: Python 3.11), three took 140 ms, four 6 s, each more about 50 times as long.
_ADJACENT = 2
#: Quantifiers that can match the same characters, with characters they match
#: too between them, as in ``.*a.*a.*x``: those characters pin the divisions,
#: so a name has to be made of them for the search to be slow (110 ms with
#: three on 255 a's), but four already take seconds.
_SEPARATED = 3

Predicate = Callable[[str], bool]


def exponential_regex_construct(pattern: str) -> Optional[str]:
    """Name a construct that can make a failing search for ``pattern`` take hours.

    A repeated group that can match the same text in more than one way, as
    ``(\\w+)+`` or ``(a|ab)*`` can, makes a failing search try every way of
    splitting the searched text between its repetitions: hours for a
    40-character name. Such a group is one that repeats and holds a quantifier
    of variable length, or alternatives. A fixed count inside it
    (``(ab{2})+``) cannot split, and an optional group (``(_\\d+)?``) does not
    repeat, so both are allowed.

    Several quantifiers in a row that can match the same characters, as in
    ``.*.*.*x``, are refused too: a failing search tries every way of dividing
    a stretch of the name among them, which grows with the name's length to the
    power of their number, plus one for the starting position, unless ``^`` or
    ``\\A`` anchors the pattern. Three such quantifiers are refused, or four with
    characters between them that they also match (``.*a.*a.*a.*x``); anchored,
    one more of either is allowed. Characters one of them cannot match
    (``\\d+\\.\\d+``), a quantifier ending the pattern (``.*leg.*``) and a
    possessive quantifier (``\\w*+``) do not count.

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

    parsed = _parser.parse(pattern)
    return walk(parsed, False) or _quantifier_run(parsed, sre)


def _quantifier_run(parsed: Any, sre: Any) -> Optional[str]:
    """Describe the first too-long run of quantifiers that match the same characters."""
    # The pattern's literals join the sample as they are met.
    sample = list(_SAMPLE)
    sequences: List[Tuple[Any, int, bool]] = [(parsed, parsed.state.flags, True)]
    while sequences:
        items, flags, top = sequences.pop(0)
        elements: List[Tuple[str, Optional[Predicate]]] = []
        _flatten(items, flags, elements, sequences, sample, sre)
        anchored = top and bool(elements) and elements[0] == ("anchor", None)
        if top and elements and elements[-1][0] == "run":
            # Whatever division the search tries, nothing after it can fail.
            elements.pop()
        found = _longest_runs(elements, sample, int(anchored))
        if found:
            return found
    return None


def _flatten(
    items: Any,
    flags: int,
    out: List[Tuple[str, Optional[Predicate]]],
    nested: List[Tuple[Any, int, bool]],
    sample: List[str],
    sre: Any,
) -> None:
    """List a sequence's elements: runs, characters, checks, anchors and barriers.

    Groups that do not repeat are inlined. What a run, an alternative or a
    lookaround holds is queued in ``nested`` to be checked on its own.
    """
    for op, av in items:
        if op in (sre.LITERAL, sre.NOT_LITERAL, sre.ANY, sre.IN):
            if op == sre.LITERAL and chr(av) not in sample:
                sample.append(chr(av))
            out.append(("char", _predicate(op, av, flags, sre)))
        elif op in (sre.MAX_REPEAT, sre.MIN_REPEAT):
            low, high, body = av
            if high == sre.MAXREPEAT or high - low >= _WIDE:
                out.append(("run", _union(body, flags, sample, sre)))
                nested.append((body, flags, False))
            elif low >= 1:
                _flatten(body, flags, out, nested, sample, sre)
            else:
                nested.append((body, flags, False))
        elif op == sre.SUBPATTERN:
            _group, add_flags, del_flags, body = av
            _flatten(body, (flags | add_flags) & ~del_flags, out, nested, sample, sre)
        elif op == sre.AT:
            # With re.MULTILINE, ^ also matches after every newline.
            start = av == sre.AT_BEGINNING_STRING or (
                av == sre.AT_BEGINNING and not flags & sre.SRE_FLAG_MULTILINE
            )
            out.append(("anchor" if start and not out else "check", None))
        elif op in (sre.ASSERT, sre.ASSERT_NOT):
            out.append(("check", None))
            nested.append((av[1], flags, False))
        elif op == sre.BRANCH:
            out.append(("barrier", None))
            nested.extend((branch, flags, False) for branch in av[1])
        elif op == sre.ATOMIC_GROUP:
            out.append(("barrier", None))
            nested.append((av, flags, False))
        elif op == sre.POSSESSIVE_REPEAT:
            # Never gives back what it took, so no division is tried through it.
            out.append(("barrier", None))
            nested.append((av[2], flags, False))
        else:
            # Back-references and conditionals: no division is guessed through them.
            out.append(("barrier", None))


def _longest_runs(
    elements: List[Tuple[str, Optional[Predicate]]], sample: List[str], extra: int
) -> Optional[str]:
    """Find a run longer than allowed; ``extra`` more are allowed when anchored."""
    previous: Optional[Predicate] = None
    adjacent = chained = most_adjacent = most_chained = 0
    between = False
    for kind, predicate in elements:
        if kind == "run":
            assert predicate is not None
            if previous is not None and _overlap(previous, predicate, sample):
                chained += 1
                adjacent = 1 if between else adjacent + 1
            else:
                chained = adjacent = 1
            previous, between = predicate, False
            most_adjacent = max(most_adjacent, adjacent)
            most_chained = max(most_chained, chained)
        elif kind == "char":
            assert predicate is not None
            if previous is not None and not _overlap(previous, predicate, sample):
                # The run before cannot take this character, so it pins the division.
                previous, adjacent, chained = None, 0, 0
            between = True
        elif kind == "barrier":
            previous, adjacent, chained = None, 0, 0
    if most_adjacent > _ADJACENT + extra:
        return (
            f"{most_adjacent} quantifiers in a row that can match the same characters, "
            "as in .*.*.*"
        )
    if most_chained > _SEPARATED + extra:
        return (
            f"{most_chained} quantifiers that can match the same characters, with only "
            "characters they match too between them, as in .*a.*a.*a.*"
        )
    return None


def _overlap(first: Predicate, second: Predicate, sample: List[str]) -> bool:
    return any(first(char) and second(char) for char in sample)


def _union(items: Any, flags: int, sample: List[str], sre: Any) -> Predicate:
    """Every character something in ``items`` can match; anything when unknown."""
    tests: List[Predicate] = []

    def collect(sequence: Any, scoped: int) -> bool:
        for op, av in sequence:
            if op in (sre.LITERAL, sre.NOT_LITERAL, sre.ANY, sre.IN):
                if op == sre.LITERAL and chr(av) not in sample:
                    sample.append(chr(av))
                tests.append(_predicate(op, av, scoped, sre))
            elif op in (sre.MAX_REPEAT, sre.MIN_REPEAT, sre.POSSESSIVE_REPEAT):
                if not collect(av[2], scoped):
                    return False
            elif op == sre.SUBPATTERN:
                if not collect(av[3], (scoped | av[1]) & ~av[2]):
                    return False
            elif op == sre.BRANCH:
                if not all(collect(branch, scoped) for branch in av[1]):
                    return False
            elif op == sre.ATOMIC_GROUP:
                if not collect(av, scoped):
                    return False
            elif op not in (sre.AT, sre.ASSERT, sre.ASSERT_NOT):
                return False
        return True

    if not collect(items, flags) or not tests:
        return lambda char: True
    return lambda char: any(test(char) for test in tests)


def _predicate(op: Any, av: Any, flags: int, sre: Any) -> Predicate:
    """Whether one character matches a LITERAL, NOT_LITERAL, ANY or IN item."""
    ignore = bool(flags & sre.SRE_FLAG_IGNORECASE)
    ascii_only = bool(flags & sre.SRE_FLAG_ASCII)
    if op == sre.ANY:
        if flags & sre.SRE_FLAG_DOTALL:
            return lambda char: True
        return lambda char: char != "\n"
    if op in (sre.LITERAL, sre.NOT_LITERAL):
        literal = _same_as(chr(av), ignore)
        if op == sre.LITERAL:
            return literal
        return lambda char: not literal(char)
    negate = any(item == sre.NEGATE for item, _ in av)
    tests = [_set_member(item, value, ignore, ascii_only, sre) for item, value in av if item != sre.NEGATE]
    return lambda char: any(test(char) for test in tests) != negate


def _same_as(literal: str, ignore: bool) -> Predicate:
    if ignore:
        folded = literal.lower()
        return lambda char: char.lower() == folded
    return lambda char: char == literal


def _set_member(item: Any, value: Any, ignore: bool, ascii_only: bool, sre: Any) -> Predicate:
    """Whether one character matches an item of a character class."""
    if item == sre.LITERAL:
        return _same_as(chr(value), ignore)
    if item == sre.RANGE:
        low, high = value

        def in_range(char: str) -> bool:
            variants = {char, char.lower(), char.upper()} if ignore else {char}
            return any(low <= ord(v) <= high for v in variants if len(v) == 1)

        return in_range
    if item == sre.CATEGORY:
        categories: dict[Any, Predicate] = {
            sre.CATEGORY_DIGIT: lambda c: c.isdecimal() and (c.isascii() or not ascii_only),
            sre.CATEGORY_SPACE: lambda c: c.isspace() and (c.isascii() or not ascii_only),
            sre.CATEGORY_WORD: lambda c: (c.isalnum() or c == "_") and (c.isascii() or not ascii_only),
            sre.CATEGORY_LINEBREAK: lambda c: c == "\n",
        }
        negated = {
            sre.CATEGORY_NOT_DIGIT: sre.CATEGORY_DIGIT,
            sre.CATEGORY_NOT_SPACE: sre.CATEGORY_SPACE,
            sre.CATEGORY_NOT_WORD: sre.CATEGORY_WORD,
            sre.CATEGORY_NOT_LINEBREAK: sre.CATEGORY_LINEBREAK,
        }
        if value in categories:
            return categories[value]
        if value in negated:
            positive = categories[negated[value]]
            return lambda char: not positive(char)
    # Anything else a class can hold: assume it matches every character.
    return lambda char: True
