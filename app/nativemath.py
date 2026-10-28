"""Word equations rebuilt from the text layer of born-digital PDFs.

Word's PDF export draws an equation (Cambria Math) as positioned glyphs: math letters
(𝑚 𝑣 ℎ), digits and operators on the baseline, scripts set smaller above or below it,
fraction bars as thin filled rectangles, and vector arrows as combining glyphs (U+20D7)
of no width at the right edge of their letter. build() reads that layout back as LaTeX,
which app/omml.py writes as an m:oMath; native.py keeps the formula's picture whenever
build() returns None.

Precision first: every glyph and stroke must find its place (fraction part, baseline,
script, accent, a text label before or after the equation) or the formula stays a
picture. Strokes other than fraction bars and absolute-value bars (radicals, braces,
underlines, highlight boxes), glyphs of another size off the baseline, letters it cannot
map and LaTeX the converter rejects all give None.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from itertools import pairwise
from statistics import median

from .omml import GREEK, latex_to_omml

SCRIPT = 0.85  # glyphs smaller than this share of the line's size are scripts (or a smaller line)
ON_BASE = 0.12  # baseline tolerance, in em
AXIS = 0.28  # height of the maths axis (fraction bars) over the baseline, in em (0.284 measured)
# A gap this wide (em) between two ordinary symbols is a typed space (0.18–0.22 em). Gaps after
# punctuation are Word's own (0.15–0.17 em after a dot before a script or fraction: "945. 10³").
SPACE = 0.16
EMPTY_BASE = 0.2  # a script this far (em) from the symbol before it has no base: "( ⁰C)", "80,6 ⁰F"
PUNCT_SPACE = 0.2  # after a dot or comma Word sets 0.15-0.17 em itself: a wider gap holds a typed space ("a. x")
EMPTY_BASE_WIDTH = 0.28  # em: the box Word keeps for that empty base, left of the script's glyphs

ARROW = "⃗"  # combining right arrow above: Word's \vec
COMBINING = {ARROW: "vec", "̂": "hat", "̃": "tilde", "̄": "bar", "̇": "dot", "̅": "bar"}
# Relations and binary operators: Word spaces them itself, so gaps beside them are not spaces.
OPERATORS = {
    "=": "=", "+": "+", "−": "-", "<": "<", ">": ">", "×": r"\times", "÷": r"\div", "⋅": r"\cdot",
    "·": r"\cdot", "±": r"\pm", "∓": r"\mp", "≤": r"\le", "≥": r"\ge", "≠": r"\ne", "≈": r"\approx",
    "≡": r"\equiv", "∼": r"\sim", "≃": r"\simeq", "∝": r"\propto", "→": r"\rightarrow",
    "←": r"\leftarrow", "↔": r"\leftrightarrow", "⇒": r"\Rightarrow", "⇐": r"\Leftarrow",
    "⇔": r"\Leftrightarrow", "⟹": r"\Longrightarrow", "∈": r"\in", "∉": r"\notin", "⊂": r"\subset",
    "⊃": r"\supset", "∪": r"\cup", "∩": r"\cap", "⊥": r"\perp", "∥": r"\parallel", "∗": r"\ast",
}
# ("!" and "?" are left out: in math fonts they are more often a glyph code read as text than punctuation)
PUNCT = {",": ",", ".": ".", ";": ";", ":": ":", "/": "/"}
OPENERS = {"(": "(", "[": "[", "{": r"\{"}
CLOSERS = {")": ")", "]": "]", "}": r"\}"}
# ∆ (U+2206) is kept as it is: Word sets it upright, \Delta (U+0394) italic. Upright Greek letters (Ω of
# ohms, μ of micro) are \mathrm (_atom).
SYMBOLS = {
    "∆": "∆", "∞": r"\infty", "∂": r"\partial", "∇": r"\nabla", "∠": r"\angle", "°": "°", "∘": r"\circ", "′": "'", "″": "''", "%": r"\%",
    "∎": "∎", "■": "∎", "|": "|", "‖": r"\|", "ℏ": r"\hbar", "ℓ": r"\ell", "…": r"\ldots", "∴": r"\therefore",
}
# Greek letters by their character (ε is \varepsilon, ϵ \epsilon, φ \varphi, ϕ \phi); capitals Word
# shares with Latin (Α Β Ε …) are Latin letters.
GREEK_NAMES = {ch: name for name, ch in GREEK.items()}
GREEK_CAPITAL_LATIN = dict(zip("ΑΒΕΖΗΙΚΜΝΟΡΤΧ", "ABEZHIKMNOPTX"))
GREEK_VARIANTS = {"EPSILON SYMBOL": "ϵ", "THETA SYMBOL": "ϑ", "PHI SYMBOL": "ϕ", "KAPPA SYMBOL": "ϰ", "RHO SYMBOL": "ϱ",
                  "PI SYMBOL": "ϖ"}
MATH_STYLES = {"ITALIC": "{}", "BOLD ITALIC": r"\boldsymbol{{{}}}", "BOLD": r"\mathbf{{{}}}"}
# punctuation a run of words in an equation (Word's "normal text") carries with it
TEXT_PUNCT = set("(),.:;+-/'")
COMMAND_END = re.compile(r"\\[A-Za-z]+$")


class Unsure(Exception):
    """The layout is not one this module reads with certainty: keep the picture."""


@dataclass(eq=False)
class Glyph:
    c: str
    x0: float
    y0: float
    x1: float
    y1: float
    oy: float  # baseline
    size: float
    font: str = ""
    bold: bool = False
    italic: bool = False
    color: str | None = None
    math_font: bool = True
    drawn: bool = False  # a combining mark drawn with a width, its box where it is drawn (not where the text layer puts it)
    accents: list[str] = field(default_factory=list)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass(eq=False)
class Bar:
    x0: float
    x1: float
    y: float


@dataclass(eq=False)
class _Frac:
    x0: float
    x1: float
    num: str
    den: str
    size: float  # its largest glyph
    y: float  # the bar: on the line's maths axis, a quarter em over the baseline
    latex: str = ""

    def write(self, line_size: float | None) -> None:
        """\\frac where Word shrinks the parts (an inline fraction: about 0.72 of the line), \\dfrac
        where they are as large as the line (a display fraction) or there is no line to compare with."""
        full = line_size is None or self.size >= SCRIPT * line_size
        self.latex = rf"\{'dfrac' if full else 'frac'}{{{self.num}}}{{{self.den}}}"


@dataclass(eq=False)
class _Item:
    """One emitted piece of a line, for spacing: kind ord / op / punct / open / close / bar / frac / text."""

    latex: str
    kind: str
    x0: float
    x1: float


@dataclass
class Piece:
    """build() result: a run of plain text (a label "b)" before the equation) or the equation."""

    kind: str  # "text" | "math"
    text: str
    size: float  # math: its line's size (an equation of fractions only: their parts' size, written \dfrac)
    bold: bool = False
    italic: bool = False
    color: str | None = None
    font: str | None = None


def _math_alnum(c: str) -> tuple[str, str]:
    """(latex, kind) of a Unicode math letter or digit: 𝑚 → m, 𝒃 → \\boldsymbol{b}, 𝛼 → \\alpha, 𝜀 → \\varepsilon.
    Script, fraktur, double-struck, sans-serif and monospace letters are not read."""
    if c == "ℎ":
        return "h", "ord"
    name = unicodedata.name(c, "")
    other = ("ITALIC", "SCRIPT", "FRAKTUR", "DOUBLE-STRUCK", "SANS-SERIF", "MONOSPACE")
    styles = [s for s in MATH_STYLES if name.startswith(f"MATHEMATICAL {s} ")
              and name[len(f"MATHEMATICAL {s} "):].split()[0] not in other]  # "BOLD SCRIPT" is not "BOLD"
    if len(styles) != 1:
        raise Unsure(f"letter {c!r}")
    style = styles[0]
    # NFKC folds the symbol forms (ϵ ϑ ϕ ϰ ϱ ϖ) into the letters: those go by name
    variant = GREEK_VARIANTS.get(name[len(f"MATHEMATICAL {style} "):]) if "SYMBOL" in name else None
    plain = variant or unicodedata.normalize("NFKC", c)
    if len(plain) != 1:
        raise Unsure(f"letter {c!r}")
    if plain.isascii() and plain.isalnum():
        letter = plain
    elif plain in GREEK_NAMES:
        letter = "\\" + GREEK_NAMES[plain]
    elif plain in GREEK_CAPITAL_LATIN:
        letter = GREEK_CAPITAL_LATIN[plain]
    elif plain in ("∂", "∇"):
        letter = r"\partial" if plain == "∂" else r"\nabla"
    else:
        raise Unsure(f"letter {c!r}")
    return MATH_STYLES[style].format(letter), "ord"


def _is_math_alnum(c: str) -> bool:
    return c == "ℎ" or 0x1D400 <= ord(c) <= 0x1D7FF


def _texty(g: Glyph) -> bool:
    """An upright letter: Word's normal text in an equation (units "kg", words "Hiệu suất")."""
    return g.c.isalpha() and not _is_math_alnum(g.c) and g.c not in SYMBOLS and not g.accents


def _atom(g: Glyph) -> tuple[str, str]:
    c = g.c
    if _is_math_alnum(c):
        return _math_alnum(c)
    if c.isdigit() and c.isascii():
        return c, "ord"
    if c in OPERATORS:
        return OPERATORS[c], "op"
    if c in PUNCT:
        return PUNCT[c], "punct"
    if c in OPENERS:
        return OPENERS[c], "open"
    if c in CLOSERS:
        return CLOSERS[c], "close"
    if c in SYMBOLS:
        return SYMBOLS[c], "bar" if c in "|‖" else "ord"
    if c.isalpha() and c.isascii():
        return rf"\mathrm{{{c}}}", "ord"
    if unicodedata.normalize("NFKC", c) in GREEK_NAMES or c == "µ":
        return rf"\mathrm{{{unicodedata.normalize('NFKC', c)}}}", "ord"  # upright: Ω of ohms, μ of micro
    if c.isalpha() and unicodedata.category(c) in ("Ll", "Lu") and ord(c) < 0x2000:
        return c, "ord"  # a Vietnamese letter among math letters ("giây")
    raise Unsure(f"symbol {c!r}")


def _accent(latex: str, accents: list[str]) -> str:
    for a in accents:
        latex = rf"\{a}{{{latex}}}"
    return latex


def _escape_text(s: str) -> str:
    out = []
    for ch in s:
        if ch in "%&#$_{}":
            out.append("\\" + ch)
        elif ch in "\\^~":
            raise Unsure(f"text {ch!r}")
        else:
            out.append(ch)
    return "".join(out)


def _layout(glyphs: list[Glyph], bars: list[Bar], depth: int = 0, outer_size: float | None = None) -> str:
    """LaTeX of glyphs set on one line, with fractions (bars) and scripts. outer_size: size of the
    line this one sits in (a numerator's line), for a line of fractions only."""
    if depth > 8:
        raise Unsure("nesting")
    # Fractions: a bar with no longer bar over or under it holds everything centred in its span.
    def inside(b: Bar, o: Bar) -> bool:
        return o is not b and o.x1 - o.x0 > b.x1 - b.x0 + 0.5 and o.x0 - 1 <= b.x0 and b.x1 <= o.x1 + 1 and abs(o.y - b.y) > 1

    outer = sorted((b for b in bars if not any(inside(b, o) for o in bars)), key=lambda b: b.x0)
    for a, b in pairwise(outer):
        if b.x0 < a.x1 - 1:
            raise Unsure("stacked bars")
    tokens: list = []
    used: set[int] = set()
    for b in outer:
        members = [g for g in glyphs if b.x0 - 1 <= g.cx <= b.x1 + 1]
        inner = [o for o in bars if o is not b and b.x0 - 1 <= (o.x0 + o.x1) / 2 <= b.x1 + 1 and inside(o, b)]
        num = [g for g in members if g.cy < b.y]
        den = [g for g in members if g.cy > b.y]
        if not num or not den or len(num) + len(den) != len(members):
            raise Unsure("bar without numerator or denominator")
        part_size = max(g.size for g in members)
        num_latex = _layout(num, [o for o in inner if o.y < b.y], depth + 1, part_size)
        den_latex = _layout(den, [o for o in inner if o.y > b.y], depth + 1, part_size)
        tokens.append(_Frac(min(b.x0, *(g.x0 for g in members)), max(b.x1, *(g.x1 for g in members)),
                            num_latex, den_latex, part_size, b.y))
        used.update(id(g) for g in members)
    if len([o for o in bars if not any(o is b or inside(o, b) for b in outer)]):
        raise Unsure("bar left over")
    rest = [g for g in glyphs if id(g) not in used]
    if not rest:
        for t in tokens:
            t.write(outer_size)
        return _emit(sorted(tokens, key=lambda t: t.x0), {}, {}, max(t.size for t in tokens))
    size = max(g.size for g in rest)
    base_y = median(g.oy for g in rest if g.size >= 0.9 * size)
    for t in tokens:
        t.write(size)
        # a fraction in an exponent ("2^{-t/T}") sits high: scripts holding fractions are not read
        if abs(base_y - t.y - AXIS * size) > ON_BASE * size:
            raise Unsure("fraction off the maths axis")
    scripts: list[tuple[Glyph, str]] = []
    for g in rest:
        dy = g.oy - base_y
        if g.size >= SCRIPT * size:
            if abs(dy) > ON_BASE * size:
                raise Unsure("glyph off the baseline")  # a second line, a raised symbol
            tokens.append(g)
        elif dy < -0.1 * size:
            scripts.append((g, "sup"))
        elif dy > 0.45 * size:
            raise Unsure("glyph far under the baseline")  # a limit under "min" (subscripts drop 0.20-0.25 em)
        elif dy > 0.06 * size:
            scripts.append((g, "sub"))
        else:
            tokens.append(g)  # smaller text on the baseline
    for g in rest:
        # a small glyph back on the baseline right after a script is a script of that script ("x_{a^2}")
        if g.size < SCRIPT * size and abs(g.oy - base_y) <= 0.1 * size and any(
            s.size > g.size + 0.3 and -0.1 * size <= g.x0 - s.x1 <= 0.1 * size for s, _ in scripts
        ):
            raise Unsure("script of a script")
    tokens.sort(key=lambda t: t.x0)
    attached: dict[int, dict[str, list[Glyph]]] = {}  # id(token) -> {"sup"/"sub"/"presup"/"presub": glyphs}
    for g, role in sorted(scripts, key=lambda gr: gr[0].x0):
        # a script set under or over a symbol (limits of "lim", a condition over an arrow) is not one of its scripts
        if any(t.x0 + 0.05 * size < g.cx < t.x1 - 0.1 * size for t in tokens):
            raise Unsure("script under or over a symbol")
        left = [t for t in tokens if t.x0 < g.cx - 0.05 * size]
        owner = left[-1] if left else None
        pre = owner is None
        if owner is not None and g.x0 < owner.x1 - 0.25 * size and not any(
            gs for r, gs in attached.get(id(owner), {}).items() if r == role
        ):
            raise Unsure("script starting under its symbol")  # Word cuts in at most 0.21 em (W_s)
        if owner is not None:
            right_edge = owner.x1  # past the owner's scripts so far ("v′₁": the subscript after the prime)
            for r, gs in attached.get(id(owner), {}).items():
                if not r.startswith("pre"):
                    right_edge = max([right_edge, *(s.x1 for s in gs)])
            kind = _atom(owner)[1] if isinstance(owner, Glyph) else "ord"
            if kind == "op" and g.x0 - right_edge <= EMPTY_BASE * size:
                raise Unsure("script of an operator")  # "A →ᵗ B": not the next symbol's prescript
            if kind in ("op", "open", "punct") or g.x0 - right_edge > EMPTY_BASE * size:
                pre = True
        if pre:
            nxt = [t for t in tokens if t.x0 >= g.x1 - 0.5]
            if not nxt:
                raise Unsure("script without a base")
            attached.setdefault(id(nxt[0]), {}).setdefault("pre" + role, []).append(g)
        else:
            attached.setdefault(id(owner), {}).setdefault(role, []).append(g)
    for t in tokens:
        r = attached.get(id(t), {})
        if isinstance(t, Glyph) and t.c.isdigit() and "sub" in r and "sup" in r:
            # a number's own scripts start together; a stack ending together belongs to the symbol after it
            # ("2 ¹₀n": Word sets the nuclide's prescripts flush against the coefficient)
            starts = abs(min(g.x0 for g in r["sub"]) - min(g.x0 for g in r["sup"]))
            ends = abs(max(g.x1 for g in r["sub"]) - max(g.x1 for g in r["sup"]))
            if starts > 0.05 * size or ends < 0.05 * size:
                raise Unsure("scripts between a number and a symbol")
    script_latex: dict[int, dict[str, str]] = {}
    for tid, roles in attached.items():
        script_latex[tid] = {role: _layout(gs, [], depth + 1, size) for role, gs in roles.items()}
    script_right = {tid: max(g.x1 for gs in roles.values() for g in gs) for tid, roles in attached.items()}
    return _emit(tokens, script_latex, script_right, size)


def _text_runs(tokens: list, scripted: set[int]) -> list[tuple[int, int]]:
    """Index ranges [i, j) of the tokens that are runs of upright words."""
    runs = []
    i = 0
    n = len(tokens)
    plain = lambda t: isinstance(t, Glyph) and id(t) not in scripted and not t.accents  # noqa: E731
    while i < n:
        t = tokens[i]
        if not (plain(t) and _texty(t)):
            i += 1
            continue
        j = i + 1
        while j < n and plain(tokens[j]) and (_texty(tokens[j]) or tokens[j].c in TEXT_PUNCT):
            j += 1
        # trailing punctuation stays in the run only as ":" "," "." or a ")" it opened
        while j > i + 1 and tokens[j - 1].c in TEXT_PUNCT and not _texty(tokens[j - 1]):
            run_text = "".join(t.c for t in tokens[i:j])
            last = tokens[j - 1].c
            if last in ":,." or (last == ")" and run_text.count("(") == run_text.count(")")):
                break
            j -= 1
        runs.append((i, j))
        i = j
    return runs


def _emit(tokens: list, script_latex: dict, script_right: dict, size: float) -> str:
    scripted = set(script_latex)
    items: list[_Item] = []
    text_at = {i: j for i, j in _text_runs(tokens, scripted)}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if i in text_at:
            j = text_at[i]
            run = tokens[i:j]
            letters = [g for g in run if _texty(g)]
            spaced = any(b.x0 - a.x1 >= SPACE * size for a, b in pairwise(run))
            if len(letters) >= 2 and (spaced or any(not g.c.isascii() for g in letters)):
                text = run[0].c
                for a, b in pairwise(run):
                    text += (" " if b.x0 - a.x1 >= SPACE * size else "") + b.c
                items.append(_Item(rf"\text{{{_escape_text(text)}}}", "text", run[0].x0, run[-1].x1))
                i = j
                continue
            if all(g.c.isascii() for g in letters):
                # units "kg", "J/kgK": upright letters, punctuation stays math
                k = i
                while k < j:
                    if _texty(tokens[k]):
                        m = k
                        while m < j and _texty(tokens[m]):
                            m += 1
                        items.append(_Item(rf"\mathrm{{{''.join(g.c for g in tokens[k:m])}}}", "ord", tokens[k].x0, tokens[m - 1].x1))
                        k = m
                    else:
                        latex, kind = _atom(tokens[k])
                        items.append(_Item(latex, kind, tokens[k].x0, tokens[k].x1))
                        k += 1
                i = j
                continue
        if isinstance(t, _Frac):
            latex, kind = t.latex, "frac"
            if id(t) in scripted and set(script_latex[id(t)]) & {"sup", "sub"}:
                latex = "{" + latex + "}"
        else:
            latex, kind = _atom(t)
            latex = _accent(latex, t.accents)
        left, right = t.x0, t.x1
        if id(t) in script_latex:
            s = script_latex[id(t)]
            pre = "".join(f"_{{{s[k]}}}" if k == "presub" else f"^{{{s[k]}}}" for k in ("presub", "presup") if k in s)
            if pre:
                latex = "{}" + pre + latex
            if kind == "op" and ("sub" in s or "sup" in s):
                kind = "ord"
            if kind == "ord" and latex.isdigit() and ("sub" in s or "sup" in s):
                # the whole number is the base: "4,1²" is (4,1)², not 4,(1²)
                digits = latex
                while items and items[-1].x1 >= left - SPACE * size and (
                    items[-1].latex.isdigit() or items[-1].latex == "," and len(items) > 1 and items[-2].latex.isdigit()
                ):
                    popped = items.pop()
                    digits, left = popped.latex + digits, popped.x0
                latex = "{" + digits + "}" if len(digits) > 1 else digits
            if "sub" in s:
                latex += f"_{{{s['sub']}}}"
            if "sup" in s:
                # primes are superscripts already: "t′" is t'
                latex += s["sup"] if set(s["sup"]) == {"'"} else f"^{{{s['sup']}}}"
            right = max(right, script_right[id(t)])
        items.append(_Item(latex, kind, left, right))
        i += 1
    out = ""
    for n, it in enumerate(items):
        if n:
            prev = items[n - 1]
            gap = it.x0 - prev.x1
            if (
                gap >= SPACE * size
                and prev.kind not in ("op", "open", "bar", "punct", "frac")  # Word pads fractions itself
                and it.kind not in ("op", "punct", "close", "bar", "frac")
                and not it.latex.startswith("{}")
            ) or (prev.kind == "punct" and gap >= PUNCT_SPACE * size and it.kind not in ("op", "close")):
                out += r"\ "
            elif it.latex[:1].isalpha() and COMMAND_END.search(out):
                out += " "  # "\Delta U": a command name must not run into the next letter
        out += it.latex
    return out


def _attach_accents(glyphs: list[Glyph]) -> list[Glyph]:
    """Combining marks go onto their letter. A mark drawn with a width (native.py reads where it is
    drawn, since the text layer moves every mark to the end of the glyph before it) must be centred
    over one letter: a stretched hat or arrow over "ABC" is not a mark of "C". A mark without width
    sits at the right edge of its letter (its ink reaches back over it), as the text layer has it."""
    marks = [g for g in glyphs if g.c in COMBINING]
    rest = [g for g in glyphs if g.c not in COMBINING]
    pieces: dict[int, tuple[Glyph, list[Glyph]]] = {}
    for m in marks:
        if m.drawn:
            on_line = [g for g in rest if g.size >= SCRIPT * m.size and abs(g.oy - m.oy) <= ON_BASE * m.size]
            cands = sorted(on_line, key=lambda g: abs(g.cx - m.cx))[:1]
        else:
            cands = [g for g in rest if abs(g.x1 - m.x0) <= 0.6 and abs(g.oy - m.oy) <= 0.12 * m.size and g.size >= SCRIPT * m.size]
        if len(cands) != 1:
            raise Unsure("accent without its letter")
        pieces.setdefault(id(cands[0]), (cands[0], []))[1].append(m)
    for g, ms in pieces.values():
        kinds = {COMBINING[m.c] for m in ms}
        # Word draws a stretched arrow in pieces: two over a capital, more over several letters
        if len(kinds) > 1 or len(ms) > 2:
            raise Unsure("accent over more than one letter")
        if any(m.drawn for m in ms):
            x0, x1 = min(m.x0 for m in ms), max(m.x1 for m in ms)
            width = max(g.x1 - g.x0, 1.0)
            if abs((x0 + x1) / 2 - g.cx) > 0.25 * width or x1 - x0 > 1.8 * width:
                raise Unsure("accent over more than one letter")
        g.accents.append(kinds.pop())
    return rest


def lead_width(pieces: list[Piece]) -> float:
    """Room an equation takes left of its first glyph: Word's empty base before a leading
    prescript ("⁰C" is {}^{0}C). Its glyph boxes leave it out, line widths need it."""
    if pieces and pieces[0].kind == "math" and pieces[0].text.startswith("{}"):
        return EMPTY_BASE_WIDTH * pieces[0].size
    return 0.0


def build(glyphs: list[Glyph], bars: list[Bar], uprights: list[tuple[float, float, float, float]], others: int = 0) -> list[Piece] | None:
    """The formula as pieces (text label, equation, trailing text), or None to keep its picture.

    glyphs: the formula's visible glyphs; bars: thin horizontal strokes (fraction bars);
    uprights: upright strokes (x0, y0, x1, y1) such as drawn absolute-value bars; others: count
    of any other strokes of the formula.
    """
    try:
        return _build(glyphs, bars, uprights, others)
    except Exception:  # noqa: BLE001 – a layout this module trips over keeps its picture, the file still converts
        return None


def why(glyphs, bars, uprights, others: int = 0) -> str:
    """The reason build() keeps the picture ("" when it does not): for tests and tuning."""
    try:
        _build(glyphs, bars, uprights, others)
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"
    return ""


def _xml_char(ch: str) -> bool:
    """A character a .docx can hold (no controls, surrogates, U+FFFE/U+FFFF or private use)."""
    return ch in "\t\n" or (ord(ch) >= 0x20 and unicodedata.category(ch) not in ("Cs", "Co", "Cn"))


def _text_pieces(gs: list[Glyph], size: float, before: str = "", after: str = "") -> list[Piece]:
    """A label or trailing text taken off the equation as plain runs, one per style ("b)": 𝒃 in bold
    italic, ")" upright)."""
    pieces: list[Piece] = []
    for n, g in enumerate(gs):
        ch, bold, italic, font = g.c, g.bold, g.italic, g.font
        if not _xml_char(ch):
            raise Unsure(f"text {ch!r}")
        if _is_math_alnum(ch):
            m = re.fullmatch(r"(?:\\(mathbf|boldsymbol)\{)?([A-Za-z0-9])\}?", _math_alnum(ch)[0])
            if not m:
                raise Unsure("label letter")
            ch, bold, italic = m.group(2), bold or m.group(1) is not None, italic or m.group(1) != "mathbf"
            font = "Cambria"  # Cambria Math has no bold or italic face: Word would fake them
        if n and g.x0 - gs[n - 1].x1 >= SPACE * size:
            ch = " " + ch
        last = pieces[-1] if pieces else None
        if last and (last.bold, last.italic, last.color, last.font) == (bold, italic, g.color, font):
            last.text += ch
        else:
            pieces.append(Piece("text", ch, g.size, bold=bold, italic=italic, color=g.color, font=font))
    pieces[0].text = before + pieces[0].text
    pieces[-1].text += after
    return pieces


def _build(glyphs: list[Glyph], bars: list[Bar], uprights, others: int) -> list[Piece]:
    if others:
        raise Unsure("strokes other than bars")
    if not any(g.math_font for g in glyphs):
        raise Unsure("no glyph of a Unicode math font")  # MathType's Times + Symbol: a picture
    fields = ("c", "x0", "y0", "x1", "y1", "oy", "size", "font", "bold", "italic", "color", "math_font", "drawn")
    glyphs = [Glyph(**{k: getattr(g, k) for k in fields}) for g in glyphs]  # accents go onto copies
    size = max(g.size for g in glyphs)
    for x0, y0, x1, y1 in uprights:
        line = [g for g in glyphs if g.size >= SCRIPT * size and y0 - 1 <= g.oy <= y1 + 1]
        if not line:
            raise Unsure("upright stroke")
        glyphs.append(Glyph("|", x0, y0, x1, y1, median(g.oy for g in line), size, font=line[0].font, color=line[0].color))
    glyphs = _attach_accents(glyphs)
    # The equation's own line: glyphs of full size outside the fractions, on one baseline.
    outside = [g for g in glyphs if g.size >= SCRIPT * size and not any(b.x0 - 1 <= g.cx <= b.x1 + 1 for b in bars)]
    base_y = median(g.oy for g in outside) if outside else None
    line = sorted((g for g in outside if abs(g.oy - base_y) <= ON_BASE * size), key=lambda g: g.x0)
    gap = lambda a, b: b.x0 - a.x1 >= SPACE * size  # noqa: E731
    # "- ΔU = …": a hyphen set apart before the equation is the line's dash, not a minus;
    # glyphs of a text font before or after it are text too
    k = 0
    while k < len(line) - 1 and (not line[k].math_font or line[k].c == "-" and gap(line[k], line[k + 1])):
        k += 1
    # "𝒃) 61 …": a label in a colour of its own
    if k < len(line) and line[k].color != line[-1].color:
        m = k
        while m < len(line) and line[m].color == line[k].color:
            m += 1
        rest = [g for g in glyphs if g not in line[:m]]
        if m < len(line) and line[m - 1].c in ").:" and all(g.color == line[m].color for g in rest):
            k = m
    m = len(line)
    while m > k + 1 and not line[m - 1].math_font:
        m -= 1
    lead, tail = line[:k], line[m:]
    body = [g for g in glyphs if g not in lead and g not in tail]
    if lead and max(g.x1 for g in lead) > min(g.x0 for g in body) + 0.5:
        raise Unsure("label overlaps the formula")
    if tail and min(g.x0 for g in tail) < max(g.x1 for g in body) - 0.5:
        raise Unsure("trailing text overlaps the formula")
    # the label's own scripts would be left in the equation ("(a+b)" in red, its ² in black)
    if lead and min(g.x0 for g in body) < line[k].x0 - 0.5 or tail and max(g.x1 for g in body) > line[m - 1].x1 + 0.5:
        raise Unsure("scripts of the label")
    depth = 0
    for g in sorted(body, key=lambda g: g.x0):
        depth += (g.c in OPENERS) - (g.c in CLOSERS)
        if depth < 0:
            break
    if depth != 0:
        raise Unsure("unbalanced brackets")  # one row of a system or a piecewise function: "\{x-y=0"
    if any(not g.math_font for g in body):
        raise Unsure("text font inside the formula")
    colors = {g.color for g in body}
    if len(colors) > 1:
        raise Unsure("several colours")
    latex = _layout(body, bars)
    color = colors.pop()
    # raises ValueError for anything the writer cannot take (docx_builder._add_math passes the same)
    latex_to_omml(latex, size_pt=size, text_font="Cambria Math", color=color)
    # the equation's size is that of its line; with only fractions on it (a lone "1/110"), the
    # line's size is unknown here (Word sets inline fractions about 0.7 of it): size 0, the caller's
    on_line = [g for g in line[k:m] if g in body]
    # fractions only (\dfrac): their parts' size; else the line's
    pieces = [Piece("math", latex, max(g.size for g in (on_line or body)), color=color, font="Cambria Math")]
    if lead:
        pieces[:0] = _text_pieces(lead, size, after=" " if gap(lead[-1], line[k]) else "")
    if tail:
        pieces += _text_pieces(tail, size, before=" " if gap(line[m - 1], tail[0]) else "")
    return pieces
