"""Bound how long Python's ``re`` can search a name, and search in linear time when it cannot.

Some read-only tools search names with a caller's regular expression inside
an editor, where nothing can interrupt the search: Unreal actor tags on the
editor's game thread, Blender object names on Blender's main thread. Python's
``re`` backtracks and holds the GIL while it does, so one pattern can freeze
the editor and no timeout can stop it.

``exponential_regex_construct`` bounds, for the longest name the caller
accepts, how many steps a failing ``re`` search can take, and names the
construct that makes that too many. ``name_searcher`` uses ``re`` when the
bound allows it and otherwise searches with a Thompson-NFA simulation whose
work grows linearly with the name, so it accepts such patterns too.

The Blender add-on bundles this module, so it imports only the standard library.
"""

from __future__ import annotations

import copy
import re
from itertools import accumulate
from typing import Any, Callable, Dict, List, Optional, Tuple

#: Blender 5's longest object name, in characters.
BLENDER_NAME_LENGTH = 255
#: The longest Unreal actor tag (an FName) the semantic-label search expects.
UNREAL_TAG_LENGTH = 1023
#: Seconds a failing ``re`` search of one name of the longest length may take.
SEARCH_BUDGET_SECONDS = 0.02
#: Seconds per backtracking step: about twice the most Python 3.11 took per
#: counted step for ``.*.*\d``, ``\w*\w*\d``, ``[a-z]*[a-z]*\d`` and the like
#: on 255- and 1023-character names (6.2 ns, and 9.7 ns with IGNORECASE). A
#: literal after a quantifier lets ``re`` skip ahead, so ``.*.*x`` took 0.8 ns.
_STEP_SECONDS = 12e-9
_STEP_SECONDS_IGNORECASE = 20e-9
#: Alternatives multiply the sequences a search tries; more are not bounded.
_MAX_SEQUENCES = 256
#: Instructions the linear-time search compiles a pattern to, at most.
_MAX_PROGRAM = 5000
#: Cached steps of the linear-time search before the cache starts over.
_CACHE_LIMIT = 50_000
#: Characters the overlap of two character classes is tested on, besides the
#: pattern's own literals: Latin-1 and Latin Extended, plus a few letters,
#: digits and spaces of other scripts.
_SAMPLE = [chr(code) for code in range(0x250)] + list("αΩЖж中한٣  　")

Predicate = Callable[[str], bool]


class RegexRefused(ValueError):
    """A pattern refused because its search could run too long; ``details`` name why.

    Its ``error_type`` is what tool envelopes report.
    """

    error_type = "ValidationError"

    def __init__(self, message: str, construct: str, estimate_seconds: Optional[float] = None) -> None:
        super().__init__(message)
        self.construct = construct
        self.details: Dict[str, Any] = {"code": "regex_backtracking", "construct": construct}
        if estimate_seconds is not None:
            self.details["estimate_seconds"] = round(estimate_seconds, 3)


class _Unbounded(Exception):
    """The analysis cannot bound the pattern; the message names what it met."""


def exponential_regex_construct(
    pattern: str, flags: int = 0, max_length: int = BLENDER_NAME_LENGTH
) -> Optional[str]:
    """Name a construct that can make a failing ``re`` search for ``pattern`` too slow.

    A repeated group that can match the same text in more than one way, as
    ``(\\w+)+`` or ``(a|ab)*`` can, makes a failing search try every way of
    splitting the searched text between its repetitions: hours for a
    40-character name. Such a group is one that repeats and holds a quantifier
    of variable length, or alternatives. A fixed count inside it
    (``(ab{2})+``) cannot split, and an optional group (``(_\\d+)?``) does not
    repeat, so both are allowed.

    Otherwise the pattern is bounded: every quantifier, optional element and
    alternative is a choice, and quantifiers that can divide the same
    characters between them (``.*.*``, ``\\w+_\\w+``, ``[a-z]*[0-9]*[a-z]*``
    through the empty ``[0-9]*``) multiply each other's choices, as in Weber and
    Seidl's degree of ambiguity, as do further such pairs after them. An
    unanchored search adds one more choice, where it starts. The steps are
    counted for a name of ``max_length`` characters and refused above
    SEARCH_BUDGET_SECONDS. A quantifier ending the pattern (``.*leg.*``) never
    gives anything back and does not count; possessive quantifiers do not
    either. Back-references count as a quantifier over any characters.

    Args:
        pattern: A valid regular expression.
        flags: The flags the caller compiles it with; IGNORECASE makes more
            classes overlap and each step slower.
        max_length: The longest name the caller searches.

    Returns:
        A description of the first such construct, or None.
    """
    # The stdlib's own parser, private since 3.11 (formerly sre_parse).
    from re import _constants as sre, _parser  # type: ignore[attr-defined]

    parsed = _parser.parse(pattern, flags)
    found = _nested_repetition(parsed, sre)
    if found:
        return found
    try:
        seconds, factors = _bound(parsed, sre, max_length)
    except _Unbounded as exc:
        return str(exc)
    if seconds <= SEARCH_BUDGET_SECONDS:
        return None
    return (
        f"{factors} quantifiers that can divide the same characters between them, as in "
        f".*.*.* (a failing search of a {max_length}-character name can take about "
        f"{_readable(seconds)})"
    )


def name_searcher(
    pattern: str,
    flags: int = 0,
    max_length: int = BLENDER_NAME_LENGTH,
    subject: str = "The pattern",
) -> Callable[[str], bool]:
    """Return a function that says whether ``pattern`` is found in a name, as ``re.search``.

    Python's ``re`` searches when ``exponential_regex_construct`` bounds it;
    otherwise a Thompson-NFA simulation does, in time linear in the name.

    Args:
        pattern: A valid regular expression.
        flags: Its flags.
        max_length: The longest name searched, for the bound.
        subject: How a refusal names the pattern, e.g. ``name_pattern``.

    Returns:
        ``search(name) -> bool``.

    Raises:
        re.error: The pattern is not valid.
        RegexRefused: Neither way can search it in bounded time: a construct
            that ``re`` backtracks on, together with one the linear-time
            search does not support (a back-reference, an atomic group, a
            possessive quantifier, a conditional, or a lookaround that cannot
            be bounded itself).
    """
    compiled = re.compile(pattern, flags)
    construct = exponential_regex_construct(pattern, flags, max_length)
    if construct is None:
        return lambda name: compiled.search(name) is not None
    try:
        return _LinearSearch(pattern, flags, max_length)
    except _Unbounded as exc:
        raise RegexRefused(
            f"{subject} has {construct}, so Python's re could search for it for long, and "
            f"{exc}, which the linear-time search used instead does not support. Rewrite it "
            "without that, or with fewer quantifiers that can divide the same characters "
            "(.* rather than .*.*)",
            construct,
        ) from None


# ---------------------------------------------------------------------------
# Bounding the backtracking
# ---------------------------------------------------------------------------


def _nested_repetition(parsed: Any, sre: Any) -> Optional[str]:
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

    return walk(parsed, False)


class _Element:
    """One position of a pattern laid out as a sequence.

    Kinds: ``loop`` (a choice of ``width + 1`` lengths of characters
    ``predicate`` matches), ``char`` (one character, ``mandatory`` unless it
    can be skipped), ``check`` (zero-width, may fail) and ``anchor`` (``^`` or
    ``\\A`` at the start).
    """

    __slots__ = ("kind", "predicate", "mandatory", "width", "group", "inside")

    def __init__(
        self, kind: str, predicate: Optional[Predicate] = None, mandatory: bool = False, width: int = 0
    ) -> None:
        self.kind = kind
        self.predicate = predicate
        self.mandatory = mandatory
        self.width = width
        #: The optional group whose taking or skipping this choice is.
        self.group: Optional[int] = None
        #: The optional groups this element is inside.
        self.inside: Tuple[int, ...] = ()


class _Context:
    def __init__(self, sre: Any, max_length: int) -> None:
        self.sre = sre
        self.max_length = max_length
        self.sample = list(_SAMPLE)
        self.ignorecase = False
        self.inside: Tuple[int, ...] = ()
        self.groups = 0

    def element(
        self, kind: str, predicate: Optional[Predicate] = None, mandatory: bool = False, width: int = 0
    ) -> _Element:
        made = _Element(kind, predicate, mandatory, width)
        made.inside = self.inside
        return made

    def literal(self, char: str) -> None:
        if char not in self.sample:
            self.sample.append(char)


def _bound(parsed: Any, sre: Any, max_length: int, anchored: bool = False) -> Tuple[float, int]:
    """Steps' worth of seconds a failing search may take, and the factors behind it.

    ``anchored`` bounds one match attempt at one position instead of a search.
    """
    context = _Context(sre, max_length)
    context.ignorecase = bool(parsed.state.flags & sre.SRE_FLAG_IGNORECASE)
    sequences = _expand(parsed, parsed.state.flags, False, context)
    steps = 0
    factors = 0
    for sequence in sequences:
        count, width_factors = _sequence_steps(sequence, context, anchored)
        steps += count
        factors = max(factors, width_factors)
    step = _STEP_SECONDS_IGNORECASE if context.ignorecase else _STEP_SECONDS
    return steps * step, factors


def _expand(items: Any, flags: int, optional: bool, context: _Context) -> List[List[_Element]]:
    """Lay ``items`` out as the sequences their alternatives make."""
    sequences: List[List[_Element]] = [[]]
    for op, av in items:
        alternatives = _expand_item(op, av, flags, optional, context)
        sequences = [head + tail for head in sequences for tail in alternatives]
        if len(sequences) > _MAX_SEQUENCES:
            raise _Unbounded(f"more than {_MAX_SEQUENCES} combinations of alternatives")
    return sequences


def _expand_item(op: Any, av: Any, flags: int, optional: bool, context: _Context) -> List[List[_Element]]:
    sre = context.sre
    if flags & sre.SRE_FLAG_IGNORECASE:
        context.ignorecase = True
    if op in (sre.LITERAL, sre.NOT_LITERAL, sre.ANY, sre.IN):
        if op == sre.LITERAL:
            context.literal(chr(av))
        return [[context.element("char", _predicate(op, av, flags, sre), not optional)]]
    if op in (sre.MAX_REPEAT, sre.MIN_REPEAT):
        low, high, body = av
        width = context.max_length if high == sre.MAXREPEAT else min(high - low, context.max_length)
        if width == 0:
            return _expand(body, flags, optional or low == 0, context) if low else [[]]
        if high == 1 and not _single_character(body, sre):
            # An optional group: taking it or not is a choice, and what it holds counts.
            choice = context.element("loop", _union(body, flags, context), False, 1)
            context.groups += 1
            choice.group = context.groups
            context.inside += (choice.group,)
            try:
                inner = _expand(body, flags, True, context)
            finally:
                context.inside = context.inside[:-1]
            return [[choice] + sequence for sequence in inner]
        # The nested-repetition check left only bodies of fixed length here.
        predicate = _union(body, flags, context)
        elements = [context.element("char", predicate, not optional)] if low else []
        return [elements + [context.element("loop", predicate, False, width)]]
    if op == sre.POSSESSIVE_REPEAT:
        low, _high, body = av
        # It never gives back what it took, so no division is tried through it.
        predicate = _union(body, flags, context)
        elements = [context.element("char", predicate, not optional)] if low else []
        return [elements + [context.element("check")]]
    if op == sre.SUBPATTERN:
        _group, add_flags, del_flags, body = av
        return _expand(body, (flags | add_flags) & ~del_flags, optional, context)
    if op == sre.ATOMIC_GROUP:
        return _expand(av, flags, optional, context)
    if op == sre.BRANCH:
        sequences: List[List[_Element]] = []
        for branch in av[1]:
            sequences.extend(_expand(branch, flags, optional, context))
            if len(sequences) > _MAX_SEQUENCES:
                raise _Unbounded(f"more than {_MAX_SEQUENCES} combinations of alternatives")
        return sequences
    if op == sre.GROUPREF_EXISTS:
        _group, yes, no = av
        return _expand(yes, flags, optional, context) + _expand(no or [], flags, optional, context)
    if op == sre.AT:
        # With re.MULTILINE, ^ also matches after every newline.
        start = av == sre.AT_BEGINNING_STRING or (
            av == sre.AT_BEGINNING and not flags & sre.SRE_FLAG_MULTILINE
        )
        return [[context.element("anchor" if start else "check")]]
    if op in (sre.ASSERT, sre.ASSERT_NOT):
        # A lookaround searches at the positions it is reached from, so what it
        # holds counts where it stands; none of it has to match there.
        return [[context.element("check")] + sequence for sequence in _expand(av[1], flags, True, context)]
    if op == sre.GROUPREF:
        # It matches what its group took, so treat it as any number of any characters.
        return [[context.element("loop", lambda char: True, False, context.max_length)]]
    raise _Unbounded(f"the construct {op}")


def _single_character(body: Any, sre: Any) -> bool:
    items = list(body)
    return len(items) == 1 and items[0][0] in (sre.LITERAL, sre.NOT_LITERAL, sre.ANY, sre.IN)


def _sequence_steps(
    sequence: List[_Element], context: _Context, anchored: bool = False
) -> Tuple[int, int]:
    """Bound a failing search's steps for one sequence; also return its loop factors."""
    n = context.max_length
    anchored = anchored or (bool(sequence) and sequence[0].kind == "anchor")
    elements = list(sequence)
    # Whatever the loops and optional characters at the end take, nothing after
    # them can fail, so they never give anything back.
    while elements and (
        elements[-1].kind == "loop" or (elements[-1].kind == "char" and not elements[-1].mandatory)
    ):
        elements.pop()
    loops = [index for index, element in enumerate(elements) if element.kind == "loop"]
    sample = context.sample

    def linked(first: int, second: int) -> bool:
        """Both loops can take a stretch, and everything required between them, of one text."""
        if elements[first].group in elements[second].inside:
            # Skipping the group skips this loop too: no choice is made twice.
            return False
        a = elements[first].predicate
        b = elements[second].predicate
        assert a is not None and b is not None
        shared = [char for char in sample if a(char) and b(char)]
        if not shared:
            return False
        for element in elements[first + 1 : second]:
            if element.kind == "char" and element.mandatory:
                predicate = element.predicate
                assert predicate is not None
                if not any(predicate(char) for char in shared):
                    return False
        return True

    # A path's factors are the first loop of a linked pair and each loop a pair
    # leads to; pairs may follow each other along the sequence.
    target: Dict[int, List[int]] = {}
    best: List[int] = []
    for index, second in enumerate(loops):
        candidates: List[List[int]] = []
        for first in loops[:index]:
            if not linked(first, second):
                continue
            before = max(
                [[elements[first].width]] + [target[t] for t in loops if t <= first and t in target],
                key=_weight,
            )
            candidates.append(before + [elements[second].width])
        if candidates:
            target[second] = max(candidates, key=_weight)
            best = max(best, target[second], key=_weight)
        best = max(best, [elements[second].width], key=_weight)
    factors = list(best)
    if not anchored:
        factors = [n] + factors
    return _compositions(factors, n), len(best)


def _weight(widths: List[int]) -> float:
    total = 1.0
    for width in widths:
        total *= width + 1
    return total


def _compositions(widths: List[int], n: int) -> int:
    """Count the length tuples with each length at most its width and n in all."""
    ways = [1] + [0] * n
    for width in widths:
        prefix = list(accumulate(ways))
        ways = [prefix[s] - (prefix[s - width - 1] if s - width - 1 >= 0 else 0) for s in range(n + 1)]
    return sum(ways)


def _readable(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 3600:
        return f"{seconds:.0f} s"
    if seconds < 86400 * 365:
        return f"{seconds / 3600:.0f} hours"
    return "longer than a year"


# ---------------------------------------------------------------------------
# Character classes
# ---------------------------------------------------------------------------


def _union(items: Any, flags: int, context: _Context) -> Predicate:
    """Every character something in ``items`` can match; anything when unknown."""
    sre = context.sre
    tests: List[Predicate] = []

    def collect(sequence: Any, scoped: int) -> bool:
        for op, av in sequence:
            if op in (sre.LITERAL, sre.NOT_LITERAL, sre.ANY, sre.IN):
                if op == sre.LITERAL:
                    context.literal(chr(av))
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
        categories: Dict[Any, Predicate] = {
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


# ---------------------------------------------------------------------------
# The linear-time search
# ---------------------------------------------------------------------------

_CHAR, _SPLIT, _ASSERT, _MATCH = 0, 1, 2, 3


class _LinearSearch:
    """Search a name as ``re.search`` does, in time linear in its length.

    The pattern compiles to a Thompson NFA whose threads advance together, one
    character at a time, so no path is ever retried. Characters, anchors and
    lookarounds are tested by ``re`` itself, compiled one item at a time, so
    they mean exactly what they mean to ``re``; a lookaround is admitted only
    when ``re`` searches it in bounded time at every position. The steps are
    cached per state, character and position context.
    """

    def __init__(self, pattern: str, flags: int, max_length: int) -> None:
        from re import _compiler, _constants as sre, _parser  # type: ignore[attr-defined]

        self._sre = sre
        self._compiler = _compiler
        self._parser = _parser
        self._max_length = max_length
        parsed = _parser.parse(pattern, flags)
        self._state = parsed.state
        self._program: List[List[Any]] = []
        self._atoms: List[Any] = []
        self._atom_keys: Dict[Any, int] = {}
        self._checks: List[Any] = []
        self._match = self._add([_MATCH, 0, 0])
        self._start = self._emit(parsed, parsed.state.flags, self._match)
        self._atom_cache: Dict[Tuple[int, str], bool] = {}
        self._steps: Dict[Any, frozenset] = {}

    def __call__(self, name: str) -> bool:
        checks = self._checks
        contexts = (
            [tuple(check.match(name, i) is not None for check in checks) for i in range(len(name) + 1)]
            if checks
            else None
        )
        state = self._closure((self._start,), contexts[0] if contexts else ())
        if self._match in state:
            return True
        steps = self._steps
        for index, char in enumerate(name):
            context = contexts[index + 1] if contexts else ()
            key = (state, char, context)
            following = steps.get(key)
            if following is None:
                targets = [self._start]
                for pc in state:
                    instruction = self._program[pc]
                    if instruction[0] == _CHAR and self._matches(instruction[1], char):
                        targets.append(instruction[2])
                following = self._closure(tuple(targets), context)
                if len(steps) >= _CACHE_LIMIT:
                    steps.clear()
                steps[key] = following
            if self._match in following:
                return True
            state = following
        return False

    def _closure(self, pcs: Tuple[int, ...], context: Tuple[bool, ...]) -> frozenset:
        stack = list(pcs)
        seen = set()
        out = set()
        program = self._program
        while stack:
            pc = stack.pop()
            if pc in seen:
                continue
            seen.add(pc)
            kind, first, second = program[pc]
            if kind == _SPLIT:
                stack.append(first)
                stack.append(second)
            elif kind == _ASSERT:
                if context[first]:
                    stack.append(second)
            else:
                out.add(pc)
        return frozenset(out)

    def _matches(self, atom: int, char: str) -> bool:
        key = (atom, char)
        found = self._atom_cache.get(key)
        if found is None:
            found = self._atoms[atom].match(char) is not None
            if len(self._atom_cache) >= _CACHE_LIMIT:
                self._atom_cache.clear()
            self._atom_cache[key] = found
        return found

    def _add(self, instruction: List[Any]) -> int:
        if len(self._program) >= _MAX_PROGRAM:
            raise _Unbounded(f"more than {_MAX_PROGRAM} steps once its counted repeats are spelled out")
        self._program.append(instruction)
        return len(self._program) - 1

    def _emit(self, items: Any, flags: int, follow: int) -> int:
        for op, av in reversed(list(items)):
            follow = self._emit_item(op, av, flags, follow)
        return follow

    def _emit_item(self, op: Any, av: Any, flags: int, follow: int) -> int:
        sre = self._sre
        if op in (sre.LITERAL, sre.NOT_LITERAL, sre.ANY, sre.IN):
            return self._add([_CHAR, self._atom(op, av, flags), follow])
        if op in (sre.MAX_REPEAT, sre.MIN_REPEAT):
            low, high, body = av
            if high == sre.MAXREPEAT:
                loop = self._add([_SPLIT, None, follow])
                self._program[loop][1] = self._emit(body, flags, loop)
                entry = loop
            else:
                entry = follow
                for _ in range(high - low):
                    entry = self._add([_SPLIT, self._emit(body, flags, entry), follow])
            for _ in range(low):
                entry = self._emit(body, flags, entry)
            return entry
        if op == sre.SUBPATTERN:
            _group, add_flags, del_flags, body = av
            return self._emit(body, (flags | add_flags) & ~del_flags, follow)
        if op == sre.BRANCH:
            entries = [self._emit(branch, flags, follow) for branch in av[1]]
            entry = entries[-1]
            for other in reversed(entries[:-1]):
                entry = self._add([_SPLIT, other, entry])
            return entry
        if op == sre.AT:
            return self._add([_ASSERT, self._check(op, av, flags), follow])
        if op in (sre.ASSERT, sre.ASSERT_NOT):
            body = av[1]
            seconds, _factors = _bound(body, sre, self._max_length, anchored=True)
            # It runs at every position of the name.
            if seconds * (self._max_length + 1) > SEARCH_BUDGET_SECONDS:
                raise _Unbounded("a lookaround that can itself backtrack for long")
            return self._add([_ASSERT, self._check(op, av, flags), follow])
        names = {
            sre.GROUPREF: "a back-reference",
            sre.GROUPREF_EXISTS: "a conditional group",
            sre.ATOMIC_GROUP: "an atomic group",
            sre.POSSESSIVE_REPEAT: "a possessive quantifier",
        }
        raise _Unbounded(names.get(op, f"the construct {op}"))

    def _compile_item(self, op: Any, av: Any, flags: int) -> Any:
        state = copy.copy(self._state)
        state.flags = flags
        return self._compiler.compile(self._parser.SubPattern(state, [(op, av)]), flags)

    def _atom(self, op: Any, av: Any, flags: int) -> int:
        key = (op, repr(av), flags)
        index = self._atom_keys.get(key)
        if index is None:
            self._atoms.append(self._compile_item(op, av, flags))
            index = self._atom_keys[key] = len(self._atoms) - 1
        return index

    def _check(self, op: Any, av: Any, flags: int) -> int:
        self._checks.append(self._compile_item(op, av, flags))
        return len(self._checks) - 1
