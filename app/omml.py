"""LaTeX -> Word equation (OMML).

latex_to_omml() turns the LaTeX that the formula model emits into one m:oMath element.
It covers the school maths/physics subset below and raises ValueError for anything
else, so a caller can keep the OCR text instead of writing a broken equation.

Supported: letters, digits, unicode symbols, Greek commands, operators and relations,
\\frac \\dfrac \\tfrac \\cfrac \\binom, \\sqrt[n], ^ _ and primes, \\left \\middle \\right,
paired bare brackets ( ) [ ] \\{ \\} | as growing delimiters, font commands (\\mathrm,
\\mathbf, \\mathbb...), \\text and friends, accents, \\overline \\underline, function names
(\\sin, \\log, \\lim, \\operatorname), \\sum \\int \\prod with limits, \\overset \\underset,
\\not, and the environments cases, matrix (p/b/B/v/V), array, aligned, gathered.
^{\\circ} (also {}^{\\circ}, \\circ C) becomes a degree sign, \\displaystyle is dropped,
spacing commands follow the spaces argument of latex_to_omml().
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import NamedTuple

from lxml import etree

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"
MATH_FONT = "Cambria Math"
MAX_LEN = 2000
MAX_DEPTH = 40


def _m(tag):
    return f"{{{M_NS}}}{tag}"


def _w(tag):
    return f"{{{W_NS}}}{tag}"


GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ϵ", "varepsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "ϑ", "iota": "ι", "kappa": "κ",
    "varkappa": "ϰ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "omicron": "ο", "pi": "π",
    "varpi": "ϖ", "rho": "ρ", "varrho": "ϱ", "sigma": "σ", "varsigma": "ς", "tau": "τ",
    "upsilon": "υ", "phi": "ϕ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π",
    "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}
SYMBOLS = {
    # binary operators
    "pm": "±", "mp": "∓", "times": "×", "cdot": "⋅", "cdotp": "⋅", "div": "÷", "ast": "∗",
    "star": "⋆", "circ": "∘", "bullet": "∙", "cup": "∪", "cap": "∩", "setminus": "∖",
    "wedge": "∧", "land": "∧", "vee": "∨", "lor": "∨", "oplus": "⊕", "otimes": "⊗",
    # relations
    "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "leqslant": "⩽", "geqslant": "⩾",
    "ne": "≠", "neq": "≠", "approx": "≈", "equiv": "≡", "sim": "∼", "simeq": "≃", "cong": "≅",
    "propto": "∝", "ll": "≪", "gg": "≫", "lt": "<", "gt": ">", "in": "∈", "notin": "∉",
    "ni": "∋", "subset": "⊂", "supset": "⊃", "subseteq": "⊆", "supseteq": "⊇", "perp": "⊥",
    "parallel": "∥", "mid": "∣", "nmid": "∤", "nleq": "≰", "ngeq": "≱",
    # arrows
    "to": "→", "rightarrow": "→", "leftarrow": "←", "gets": "←", "Rightarrow": "⇒",
    "Leftarrow": "⇐", "Leftrightarrow": "⇔", "leftrightarrow": "↔", "iff": "⟺", "implies": "⟹",
    "longrightarrow": "⟶", "longleftarrow": "⟵", "longleftrightarrow": "⟷",
    "Longrightarrow": "⟹", "Longleftarrow": "⟸", "Longleftrightarrow": "⟺", "mapsto": "↦",
    "uparrow": "↑", "downarrow": "↓", "rightleftharpoons": "⇌", "nearrow": "↗", "searrow": "↘",
    # other symbols
    "infty": "∞", "partial": "∂", "nabla": "∇", "forall": "∀", "exists": "∃", "emptyset": "∅",
    "varnothing": "∅", "angle": "∠", "triangle": "△", "neg": "¬", "lnot": "¬", "prime": "′",
    "degree": "°", "ldots": "…", "dots": "…", "cdots": "⋯", "vdots": "⋮", "ddots": "⋱",
    "hbar": "ℏ", "ell": "ℓ", "Re": "ℜ", "Im": "ℑ", "aleph": "ℵ", "therefore": "∴",
    "because": "∵", "square": "□", "colon": ":", "backslash": "\\",
    "%": "%", "#": "#", "&": "&", "$": "$", "_": "_",
    # delimiters (their pairing role is in DELIM_ROLE)
    "{": "{", "}": "}", "lbrace": "{", "rbrace": "}", "lbrack": "[", "rbrack": "]",
    "langle": "⟨", "rangle": "⟩", "lfloor": "⌊", "rfloor": "⌋", "lceil": "⌈", "rceil": "⌉",
    "lvert": "|", "rvert": "|", "lVert": "‖", "rVert": "‖", "vert": "|", "Vert": "‖", "|": "‖",
}
# Role of a delimiter symbol written without \left/\right: opens, closes or toggles a pair.
DELIM_ROLE = {
    "{": "open", "lbrace": "open", "lbrack": "open", "langle": "open", "lfloor": "open",
    "lceil": "open", "lvert": "open", "lVert": "open",
    "}": "close", "rbrace": "close", "rbrack": "close", "rangle": "close", "rfloor": "close",
    "rceil": "close", "rvert": "close", "rVert": "close",
    "vert": "toggle", "Vert": "toggle", "|": "toggle",
}
CHAR_ROLE = {"(": "open", "[": "open", ")": "close", "]": "close", "|": "toggle", "‖": "toggle",
             "{": "open", "}": "close", "⟨": "open", "⟩": "close", "⌊": "open", "⌋": "close",
             "⌈": "open", "⌉": "close"}
CLOSES = {")": "([", "]": "([", "}": "{", "⟩": "⟨", "⌋": "⌊", "⌉": "⌈", "|": "|", "‖": "‖"}
DELIM_CHARS = {"(": "(", ")": ")", "[": "[", "]": "]", "|": "|", "/": "/", ".": "", "<": "⟨",
               ">": "⟩"}
DELIM_CMDS = {"{": "{", "}": "}", "lbrace": "{", "rbrace": "}", "lbrack": "[", "rbrack": "]",
              "langle": "⟨", "rangle": "⟩", "lfloor": "⌊", "rfloor": "⌋", "lceil": "⌈", "rceil": "⌉",
              "vert": "|", "lvert": "|", "rvert": "|", "Vert": "‖", "lVert": "‖", "rVert": "‖",
              "|": "‖", "backslash": "\\", "uparrow": "↑", "downarrow": "↓"}
BIG = {"big", "Big", "bigg", "Bigg", "bigl", "Bigl", "biggl", "Biggl", "bigr", "Bigr", "biggr",
       "Biggr", "bigm", "Bigm", "biggm", "Biggm"}
NEGATED = {"=": "≠", "<": "≮", ">": "≯", "in": "∉", "equiv": "≢", "subset": "⊄", "supset": "⊅",
           "subseteq": "⊈", "supseteq": "⊉", "le": "≰", "leq": "≰", "ge": "≱", "geq": "≱",
           "parallel": "∦", "sim": "≁", "approx": "≉", "ni": "∌", "mid": "∤", "exists": "∄",
           "cong": "≇"}
FUNCS = {n: n for n in (
    "sin", "cos", "tan", "cot", "sec", "csc", "arcsin", "arccos", "arctan", "arccot", "sinh",
    "cosh", "tanh", "coth", "log", "ln", "lg", "exp", "det", "dim", "deg", "arg", "ker", "hom",
    "gcd", "Pr", "tg", "cotg", "arctg", "arccotg")}
LIMS = {"lim": "lim", "limsup": "lim\u2009sup", "liminf": "lim\u2009inf", "max": "max",
        "min": "min", "sup": "sup", "inf": "inf"}
NARY = {"sum": "∑", "prod": "∏", "coprod": "∐", "int": "∫", "iint": "∬", "iiint": "∭",
        "oint": "∮", "oiint": "∯", "bigcup": "⋃", "bigcap": "⋂", "bigvee": "⋁",
        "bigwedge": "⋀", "bigoplus": "⨁", "bigotimes": "⨂"}
INTEGRALS = {"int", "iint", "iiint", "oint", "oiint"}
ACCENTS = {"hat": "\u0302", "widehat": "\u0302", "check": "\u030c", "tilde": "\u0303",
           "widetilde": "\u0303", "acute": "\u0301", "grave": "\u0300", "dot": "\u0307",
           "ddot": "\u0308", "dddot": "\u20db", "breve": "\u0306", "bar": "\u0305",
           "mathring": "\u030a", "vec": "\u20d7", "overrightarrow": "\u20d7",
           "overleftarrow": "\u20d6", "overleftrightarrow": "\u20e1"}
BARS = {"overline": "top", "underline": "bot"}
# Width of each spacing command when spaces="all". With "word" any run of them is one
# no-break space (the model writes \, \: \; or ~ for the same gap before a unit).
SPACES = {",": "\u2009", ":": "\u205f", ";": "\u2005", " ": "\u00a0", "~": "\u00a0",
          "quad": "\u2003", "qquad": "\u2003\u2003", "enspace": "\u2002", "thinspace": "\u2009",
          "medspace": "\u205f", "thickspace": "\u2005", "hspace": "\u00a0"}
IGNORED = (set(SPACES) - {"hspace"}) | {
    "!", "negthinspace", "displaystyle", "textstyle", "scriptstyle", "scriptscriptstyle",
    "nonumber", "notag", "hline", "mathstrut", "strut", "allowbreak", "relax", "nobreak"}
PRIMES = {1: "′", 2: "″", 3: "‴", 4: "⁗"}
# Characters that end the argument of \sin... (FUNC_STOP) or the body of \sum... (NARY_STOP).
NARY_STOP = set("=<>≤≥≠≈≡∼≃≅∝≪≫⩽⩾∈∉∋⊂⊃⊆⊇∣→←⇒⇐⇔↔⟺⟹⟶⟵⟷⟸↦⇌,;:+−-±∓")
FUNC_STOP = NARY_STOP | set("⋅·×÷/∗*∪∩∖∧∨")
ENVS = {"cases", "dcases", "rcases", "matrix", "smallmatrix", "pmatrix", "bmatrix", "Bmatrix",
        "vmatrix", "Vmatrix", "array", "aligned", "align", "align*", "split", "gathered",
        "gather", "gather*", "eqnarray", "eqnarray*"}
MATRIX_DELIMS = {"matrix": None, "smallmatrix": None, "pmatrix": ("(", ")"), "bmatrix": ("[", "]"),
                 "Bmatrix": ("{", "}"), "vmatrix": ("|", "|"), "Vmatrix": ("‖", "‖")}


class Style(NamedTuple):
    nor: bool = False  # normal text (\text...), not math
    sty: str | None = None  # p, b, i, bi
    scr: str | None = None  # double-struck, script, fraktur, sans-serif, monospace


MATH = Style()
UPRIGHT = Style(sty="p")
MATH_FONTS = {
    "mathrm": UPRIGHT, "mathup": UPRIGHT, "mathit": Style(sty="i"), "mathbf": Style(sty="b"),
    "boldsymbol": Style(sty="bi"), "bm": Style(sty="bi"), "mathnormal": MATH,
    "mathbb": Style(sty="p", scr="double-struck"), "mathcal": Style(sty="p", scr="script"),
    "mathscr": Style(sty="p", scr="script"), "mathfrak": Style(sty="p", scr="fraktur"),
    "mathsf": Style(sty="p", scr="sans-serif"), "mathtt": Style(sty="p", scr="monospace"),
}
# Unicode math alphanumerics: (A, a, 0) code points and the letters that live elsewhere.
ALNUM = {
    "double-struck": (0x1D538, 0x1D552, 0x1D7D8, dict(zip("CHNPQRZ", "ℂℍℕℙℚℝℤ"))),
    "script": (0x1D49C, 0x1D4B6, None, dict(zip("BEFHILMRego", "ℬℰℱℋℐℒℳℛℯℊℴ"))),
    "fraktur": (0x1D504, 0x1D51E, None, dict(zip("CHIRZ", "ℭℌℑℜℨ"))),
    "sans-serif": (0x1D5A0, 0x1D5BA, 0x1D7E2, {}),
    "monospace": (0x1D670, 0x1D68A, 0x1D7F6, {}),
}
TEXT = Style(nor=True)
TEXT_FONTS = {
    "text": TEXT, "textrm": TEXT, "textup": TEXT, "textnormal": TEXT, "textmd": TEXT,
    "textsf": TEXT, "texttt": TEXT, "mbox": TEXT, "hbox": TEXT,
    "textbf": Style(nor=True, sty="b"), "textit": Style(nor=True, sty="i"),
    "emph": Style(nor=True, sty="i"),
}
TEXT_SYMBOLS = {"%": "%", "&": "&", "_": "_", "#": "#", "$": "$", "{": "{", "}": "}", " ": " ",
                ",": "\u2009", ";": "\u2004", ":": "\u205f", "quad": "\u2003",
                "qquad": "\u2003\u2003", "textbackslash": "\\", "textdegree": "°"}


# --- syntax tree -------------------------------------------------------------------------

@dataclass(eq=False)
class Char:
    ch: str
    style: Style
    cmd: str | None = None  # command it came from, e.g. "circ"
    role: str | None = None  # open / close / toggle for unpaired delimiters


@dataclass(eq=False)
class Group:
    items: list


@dataclass(eq=False)
class Frac:
    num: list
    den: list
    bar: bool = True
    display: bool = False


@dataclass(eq=False)
class Rad:
    deg: list | None
    body: list


@dataclass(eq=False)
class Scripts:
    base: object  # None for an empty base: {}^{14}C
    sub: list | None
    sup: list | None


@dataclass(eq=False)
class Pre:
    sub: list
    sup: list
    body: list


@dataclass(eq=False)
class Delim:
    beg: str
    end: str
    parts: list
    sep: str | None = None


@dataclass(eq=False)
class Acc:
    chr: str
    body: list


@dataclass(eq=False)
class Bar:
    pos: str
    body: list


@dataclass(eq=False)
class OpName:  # \sin, \lim, \operatorname{..}; becomes Func once its argument is known
    name: str
    lim: bool


@dataclass(eq=False)
class Func:
    name: list
    body: list


@dataclass(eq=False)
class Limit:
    low: bool
    body: list
    lim: list


@dataclass(eq=False)
class NaryOp:  # \sum, \int...; becomes Nary once its body is known
    chr: str
    limloc: str


@dataclass(eq=False)
class Nary:
    chr: str
    limloc: str
    sub: list | None
    sup: list | None
    body: list


@dataclass(eq=False)
class EqArr:
    rows: list


@dataclass(eq=False)
class Matrix:
    rows: list
    jcs: list


# --- parser --------------------------------------------------------------------------------

class _Parser:
    def __init__(self, src, spaces):
        self.s = src
        self.i = 0
        self.spaces = spaces
        self.style = MATH
        self.depth = 0

    def fail(self, msg):
        raise ValueError(f"{msg} (position {self.i} in {self.s!r})")

    # tokens: (kind, value); kind is cmd, char, eof or one of { } ^ _ & '
    def _ws(self):
        while self.i < len(self.s) and self.s[self.i].isspace():
            self.i += 1

    def _read(self):
        self._ws()
        s = self.s
        if self.i >= len(s):
            return ("eof", "")
        c = s[self.i]
        self.i += 1
        if c == "\\":
            if self.i >= len(s):
                self.fail("lone backslash")
            j = self.i
            if s[j].isascii() and s[j].isalpha():
                while j < len(s) and s[j].isascii() and s[j].isalpha():
                    j += 1
                name = s[self.i:j]
                if j < len(s) and s[j] == "*" and name == "operatorname":
                    name, j = name + "*", j + 1
                self.i = j
                return ("cmd", name)
            self.i += 1
            return ("cmd", s[j])
        if c in "{}^_&'":
            return (c, c)
        if c == "~":
            return ("cmd", "~")
        return ("char", c)

    def _peek(self):
        i = self.i
        tok = self._read()
        self.i = i
        return tok

    def _expect(self, kind, val=None):
        k, v = self._read()
        if k != kind or (val is not None and v != val):
            self.fail(f"expected {val or kind}")

    # lists ----------------------------------------------------------------------------------
    def parse_list(self, stops):
        """Parse items up to a stop token (left unread) and return the finished item list."""
        self.depth += 1
        if self.depth > MAX_DEPTH:
            self.fail("formula nested too deeply")
        items, opens, split, scripted = [], [], None, False
        while True:
            kind, val = self._peek()
            stop = _stop_name(kind, val, stops)
            if stop is not None:
                if stop in stops:
                    break
                self.fail(f"unexpected {stop}")
            self._read()
            if kind in ("^", "_"):
                scripted = self._script(items, opens, kind, scripted)
            elif kind == "'":
                scripted = self._primes(items, opens, scripted)
            elif kind == "cmd" and val in ("over", "choose", "atop"):
                if split:
                    self.fail("two \\over in one group")
                split, opens, scripted = (len(items), val), [], False
            elif kind == "cmd" and val in ("limits", "nolimits"):
                if items and isinstance(items[-1], NaryOp):
                    items[-1].limloc = "undOvr" if val == "limits" else "subSup"
            else:
                node = self._atom(kind, val)
                scripted = False
                if node is not None:
                    self._push(items, opens, node)
        self.depth -= 1
        if split:
            at, how = split
            if _is_empty(items[:at]) or _is_empty(items[at:]):
                self.fail(f"empty side of \\{how}")
            frac = Frac(_finish(items[:at]), _finish(items[at:]), bar=how == "over")
            return [Delim("(", ")", [[frac]]) if how == "choose" else frac]
        return _finish(items)

    def _push(self, items, opens, node):
        """Append a node, pairing delimiters written without \\left/\\right."""
        role = node.role if isinstance(node, Char) else None
        if role == "toggle":
            prev = items[-1] if items else None
            closing = (opens and opens[-1][1] == node.ch and opens[-1][0] < len(items) - 1
                       and not (isinstance(prev, Char) and (prev.ch in FUNC_STOP or prev.role == "open")))
            role = "close" if closing else "open"
        if role == "open":
            opens.append((len(items), node.ch))
        elif role == "close":
            for k in range(len(opens) - 1, -1, -1):
                at, ch = opens[k]
                if ch in CLOSES.get(node.ch, ""):
                    inner = _finish(items[at + 1:])
                    beg = items[at].ch
                    del items[at:], opens[k:]
                    items.append(Delim(beg, node.ch, [inner]))
                    return
        items.append(node)

    def _take_base(self, items, opens):
        if not items:
            return None
        node = items.pop()
        if opens and opens[-1][0] >= len(items):
            opens.pop()
        if isinstance(node, Group) and _is_empty(node.items):
            return None
        if isinstance(node, Char) and node.ch.isdigit() and not node.style.nor:
            digits = [node]
            while items and isinstance(items[-1], Char) and items[-1].ch.isdigit() \
                    and items[-1].style == node.style:
                digits.insert(0, items.pop())
            if len(digits) > 1:
                return Group(digits)
        return node

    def _script(self, items, opens, kind, scripted):
        arg = self._arg()
        if _is_empty(arg):
            return scripted
        if kind == "^" and _is_degree(arg):
            items.append(Char("°", self.style))
            return False
        slot = "sup" if kind == "^" else "sub"
        if scripted and items and isinstance(items[-1], Scripts):
            if getattr(items[-1], slot) is not None:
                self.fail("double " + ("superscript" if kind == "^" else "subscript"))
            setattr(items[-1], slot, arg)
            return True
        base = self._take_base(items, opens)
        items.append(Scripts(base, arg if slot == "sub" else None, arg if slot == "sup" else None))
        return True

    def _primes(self, items, opens, scripted):
        n = 1
        while self._peek()[0] == "'":
            self._read()
            n += 1
        sup = [Char(PRIMES.get(n, "′" * n), self.style)]
        if self._peek()[0] == "^":
            self._read()
            sup += self._arg()
        if scripted and items and isinstance(items[-1], Scripts):
            if items[-1].sup is not None:
                self.fail("double superscript")
            items[-1].sup = sup
            return True
        base = self._take_base(items, opens)
        if base is None:
            items.extend(sup)
            return False
        items.append(Scripts(base, None, sup))
        return True

    # atoms ----------------------------------------------------------------------------------
    def _arg(self):
        """One macro argument: a braced group or a single token."""
        kind, val = self._read()
        if kind == "{":
            items = self.parse_list({"}"})
            self._expect("}")
            return items
        if kind == "char":
            return [self._char(val)]
        if kind == "cmd":
            self.depth += 1  # \sqrt\sqrt... recurses without parse_list
            if self.depth > MAX_DEPTH:
                self.fail("formula nested too deeply")
            node = self._command(val)
            self.depth -= 1
            if node is None:
                self.fail("missing argument")
            return [node]
        self.fail("missing argument")

    def _atom(self, kind, val):
        if kind == "char":
            return self._char(val)
        if kind == "{":
            items = self.parse_list({"}"})
            self._expect("}")
            return Group(items)
        if kind == "cmd":
            return self._command(val)
        self.fail(f"unexpected {val!r}")

    def _char(self, c):
        if c in "#$":
            self.fail(f"unsupported character {c!r}")
        if unicodedata.category(c)[0] == "C":
            self.fail(f"control character U+{ord(c):04X}")
        if c == "-":
            c = "−"
        alt = _math_alnum(c, self.style.scr)
        if alt:
            return Char(alt, MATH)
        return Char(c, self.style, role=CHAR_ROLE.get(c) if c in "()[]|" else None)

    def _command(self, name):
        if name in IGNORED:
            if name in SPACES and self.spaces != "none":
                return Char(SPACES[name] if self.spaces == "all" else "\u00a0", self.style, cmd="space")
            return None
        if name in GREEK:
            return Char(GREEK[name], self.style, cmd=name)
        if name in SYMBOLS:
            return Char(SYMBOLS[name], self.style, cmd=name, role=DELIM_ROLE.get(name))
        if name in ("frac", "dfrac", "tfrac", "cfrac"):
            num = self._arg()
            return Frac(num, self._arg(), display=name in ("dfrac", "cfrac"))
        if name in ("binom", "dbinom", "tbinom"):
            n = self._arg()
            return Delim("(", ")", [[Frac(n, self._arg(), bar=False)]])
        if name == "sqrt":
            deg = None
            if self._peek() == ("char", "["):
                self._read()
                deg = self.parse_list({"]"})
                self._expect("char", "]")
            return Rad(deg, self._arg())
        if name == "left":
            return self._left()
        if name in BIG:
            delim = self._delim()
            if not delim:
                return None
            role = {"l": "open", "r": "close", "m": None}.get(name[-1], CHAR_ROLE.get(delim))
            return Char(delim, self.style, role=role)
        if name in MATH_FONTS:
            old, self.style = self.style, MATH_FONTS[name]
            try:
                return Group(self._arg())
            finally:
                self.style = old
        if name in TEXT_FONTS:
            return Group([Char(c, TEXT_FONTS[name]) for c in self._text_arg()])
        if name in ACCENTS:
            return Acc(ACCENTS[name], self._arg())
        if name in BARS:
            return Bar(BARS[name], self._arg())
        if name in FUNCS:
            return OpName(FUNCS[name], lim=False)
        if name in LIMS:
            return OpName(LIMS[name], lim=True)
        if name in ("operatorname", "operatorname*"):
            old, self.style = self.style, UPRIGHT
            try:
                parts = self._arg()
            finally:
                self.style = old
            chars = list(_flat(parts))
            if not chars or not all(isinstance(c, Char) and c.cmd != "space" for c in chars):
                self.fail("bad \\operatorname")
            return OpName("".join(c.ch for c in chars).replace("−", "-"), lim=name.endswith("*"))
        if name in NARY:
            return NaryOp(NARY[name], "subSup" if name in INTEGRALS else "undOvr")
        if name in ("overset", "stackrel", "underset"):
            lim = self._arg()
            return Limit(name == "underset", self._arg(), lim)
        if name in ("phantom", "vphantom", "hphantom"):
            self._arg()
            return None
        if name == "hspace":
            self._raw_group()
            return None if self.spaces == "none" else Char(SPACES[name], self.style, cmd="space")
        if name == "not":
            kind, val = self._read()
            if val in NEGATED and kind in ("char", "cmd"):
                return Char(NEGATED[val], self.style)
            self.fail("unsupported \\not")
        if name == "begin":
            return self._env()
        if name == "substack":
            self._expect("{")
            rows = self._rows("}")
            self._expect("}")
            return EqArr([_join_cells(r, None) for r in rows])
        self.fail(f"unsupported command \\{name}")

    def _delim(self):
        kind, val = self._read()
        table = DELIM_CHARS if kind == "char" else DELIM_CMDS if kind == "cmd" else {}
        if val not in table:
            self.fail("unsupported delimiter")
        return table[val]

    def _left(self):
        beg, parts, seps = self._delim(), [], []
        while True:
            parts.append(self.parse_list({"right", "middle"}))
            _, val = self._read()
            if val == "middle":
                seps.append(self._delim())
                continue
            return Delim(beg, self._delim(), parts, seps[0] if seps else None)

    def _raw_group(self):
        self._ws()
        if self.i >= len(self.s) or self.s[self.i] != "{":
            self.fail("expected {")
        end = self.s.find("}", self.i)
        if end < 0 or "{" in self.s[self.i + 1:end]:
            self.fail("bad group")
        text, self.i = self.s[self.i + 1:end].strip(), end + 1
        return text

    def _text_arg(self):
        """Argument of \\text: raw text with TeX escapes, spaces collapsed."""
        self._ws()
        s = self.s
        if self.i >= len(s) or s[self.i] != "{":
            self.fail("\\text needs a braced argument")
        self.i += 1
        out, depth = [], 1
        while True:
            if self.i >= len(s):
                self.fail("unclosed \\text")
            c = s[self.i]
            self.i += 1
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            elif c == "\\":
                j = self.i
                while j < len(s) and s[j].isascii() and s[j].isalpha():
                    j += 1
                name = s[self.i:j] or s[self.i:self.i + 1]
                self.i += len(name)
                if name not in TEXT_SYMBOLS:
                    self.fail(f"unsupported \\{name} in text")
                out.append(TEXT_SYMBOLS[name])
            elif c == "~":
                out.append("\u00a0")
            elif c in " \t\r\n":
                if not out or out[-1] != " ":
                    out.append(" ")
            elif c in "$^_&#":
                self.fail(f"unsupported {c!r} in text")
            elif unicodedata.category(c)[0] == "C":
                self.fail(f"control character U+{ord(c):04X}")
            else:
                out.append(c)
        return "".join(out)

    def _rows(self, closer):
        """Cells of an environment body up to closer (\\end or }) left unread."""
        rows = [[]]
        while True:
            rows[-1].append(self.parse_list({"&", "\\", closer}))
            kind, val = self._peek()
            if kind == "&":
                self._read()
            elif kind == "cmd" and val == "\\":
                self._read()
                if self._peek() == ("char", "["):
                    end = self.s.find("]", self.i)
                    if end < 0:
                        self.fail("bad row spacing")
                    self.i = end + 1
                rows.append([])
            else:
                break
        if len(rows) > 1 and len(rows[-1]) == 1 and not rows[-1][0]:
            rows.pop()  # \\ after the last row
        return rows

    def _env(self):
        name = self._raw_group()
        if name not in ENVS:
            self.fail(f"unsupported environment {name}")
        jcs = None
        if name == "array":
            spec = self._raw_group()
            jcs = [{"l": "left", "c": "center", "r": "right"}[c] for c in spec if c in "lcr"]
            if not jcs or any(c not in "lcr| " for c in spec):
                self.fail("unsupported array columns")
        rows = self._rows("end")
        self._expect("cmd", "end")
        if self._raw_group() != name:
            self.fail(f"\\begin{{{name}}} closed by another environment")
        ncols = max(len(r) for r in rows)
        if name in ("aligned", "align", "align*", "split", "eqnarray", "eqnarray*"):
            return EqArr([_join_cells(r, Char("&", MATH)) for r in rows])
        if name in ("gathered", "gather", "gather*"):
            if ncols > 1:
                self.fail(f"& in {name}")
            return EqArr([r[0] for r in rows])
        if name in ("cases", "dcases", "rcases"):
            inner = Matrix(rows, ["left"] * ncols)
            return Delim("{", "", [[inner]]) if name != "rcases" else Delim("", "}", [[inner]])
        if name == "array":
            jcs = (jcs + [jcs[-1]] * ncols)[:ncols]
            return Matrix(rows, jcs)
        inner = Matrix(rows, ["center"] * ncols)
        pair = MATRIX_DELIMS[name]
        return Delim(pair[0], pair[1], [[inner]]) if pair else inner


def _math_alnum(c, scr):
    """c in a \\mathbb-like alphabet as a Unicode math letter or digit, or None."""
    if scr not in ALNUM or not (c.isascii() and c.isalnum()):
        return None
    upper, lower, digit, other = ALNUM[scr]
    if c in other:
        return other[c]
    base, first = (upper, "A") if c.isupper() else (lower, "a") if c.islower() else (digit, "0")
    return chr(base + ord(c) - ord(first)) if base else None


def _stop_name(kind, val, stops):
    if kind == "eof":
        return "eof"
    if kind in ("}", "&"):
        return kind
    if kind == "cmd" and val in ("right", "middle", "\\", "end"):
        return val
    if kind == "char" and val == "]" and "]" in stops:
        return "]"
    return None


def _join_cells(cells, sep):
    out = []
    for k, cell in enumerate(cells):
        if k and sep is not None:
            out.append(sep)
        out.extend(cell)
    return out


def _flat(nodes):
    for node in nodes:
        if isinstance(node, Group):
            yield from _flat(node.items)
        else:
            yield node


def _is_empty(nodes):
    return all(isinstance(n, Group) and _is_empty(n.items) for n in nodes)


def _is_degree(nodes):
    flat = list(_flat(nodes))
    return len(flat) == 1 and isinstance(flat[0], Char) and (
        flat[0].cmd in ("circ", "degree") or flat[0].ch == "°")


def _first_char(node):
    while True:
        if isinstance(node, Char):
            return node.ch
        if isinstance(node, Scripts):
            node = node.base
        elif isinstance(node, Group) and node.items:
            node = node.items[0]
        else:
            return None


def _head(node):
    return node.base if isinstance(node, Scripts) else node


def _is_space(node):
    return isinstance(node, Char) and node.cmd == "space"


def _is_stop(node, stops):
    while isinstance(node, Group):
        if not node.items:
            return False
        node = node.items[0]
    if not isinstance(node, Char):
        return False
    return node.ch in stops or node.style.nor or stops is FUNC_STOP and node.cmd == "space"


def _tidy_spaces(flat):
    """One space at most, none at the ends or before punctuation."""
    out = []
    for k, node in enumerate(flat):
        if _is_space(node):
            nxt = next((n for n in flat[k + 1:] if not _is_space(n)), None)
            if not out or _is_space(out[-1]) or nxt is None or isinstance(nxt, Char) and nxt.ch in ".,;:":
                continue
        out.append(node)
    return out


def _operand(items, i, func):
    """Items after a function name (func) or an n-ary operator that form its argument."""
    if func and i < len(items) and isinstance(_head(items[i]), Delim):
        return items[i:i + 1], i + 1
    stops = FUNC_STOP if func else NARY_STOP
    j = i
    while j < len(items) and _is_space(items[j]):
        j += 1  # \sin\,x
    while j < len(items) and not _is_stop(items[j], stops):
        if func and isinstance(_head(items[j]), (OpName, NaryOp)):
            break
        j += 1
    if all(_is_space(n) for n in items[i:j]):
        return [], i
    return items[i:j], j


def _finish(items):
    """Attach arguments to function names and n-ary operators, resolve empty bases."""
    out, i = [], 0
    while i < len(items):
        node = items[i]
        i += 1
        head = _head(node)
        sub = node.sub if isinstance(node, Scripts) else None
        sup = node.sup if isinstance(node, Scripts) else None
        if isinstance(head, OpName):
            body, i = _operand(items, i, func=True)
            name = [Char(c, UPRIGHT) for c in head.name]
            if head.lim and sub is not None:
                if sup is not None:
                    raise ValueError(f"\\{head.name} with both limits")
                fname = [Limit(True, name, sub)]
            elif sub is not None or sup is not None:
                fname = [Scripts(Group(name), sub, sup)]
            else:
                fname = name
            out.append(Func(fname, _finish(body)) if body else Group(fname))
        elif isinstance(head, NaryOp):
            body, i = _operand(items, i, func=False)
            if body:
                out.append(Nary(head.chr, head.limloc, sub, sup, _finish(body)))
            else:
                base = Char(head.chr, MATH)
                out.append(Scripts(base, sub, sup) if sub or sup else base)
        elif isinstance(node, Scripts) and node.base is None:
            nxt = items[i] if i < len(items) else None
            if nxt is not None and not _is_stop(nxt, FUNC_STOP) \
                    and not isinstance(_head(nxt), (OpName, NaryOp)):
                out.append(Pre(sub or [], sup or [], [nxt]))
                i += 1
            else:
                out.extend((sub or []) + (sup or []))
        elif isinstance(node, Char) and node.cmd == "circ" and i < len(items) \
                and _first_char(items[i]) in ("C", "F"):
            out.append(Char("°", node.style))  # 1323 \circ C
        else:
            out.append(node)
    return out


# --- OMML writer ---------------------------------------------------------------------------

class _Writer:
    def __init__(self, size_pt, text_font, tidy):
        self.size = None if size_pt is None else str(round(float(size_pt) * 2))
        self.text_font = text_font
        self.tidy = tidy  # one space at most, none at the ends or before punctuation

    def wrpr(self, style):
        rpr = etree.Element(_w("rPr"))
        font = self.text_font if style.nor and self.text_font else MATH_FONT
        fonts = etree.SubElement(rpr, _w("rFonts"))
        for attr in ("ascii", "hAnsi") if font == MATH_FONT else ("ascii", "hAnsi", "cs"):
            fonts.set(_w(attr), font)
        if style.nor and style.sty in ("b", "bi"):
            etree.SubElement(rpr, _w("b"))
        if style.nor and style.sty in ("i", "bi"):
            etree.SubElement(rpr, _w("i"))
        if self.size:
            etree.SubElement(rpr, _w("sz")).set(_w("val"), self.size)
            etree.SubElement(rpr, _w("szCs")).set(_w("val"), self.size)
        return rpr

    def props(self, parent, tag, values=()):
        pr = etree.SubElement(parent, _m(tag))
        for name, val in values:
            etree.SubElement(pr, _m(name)).set(_m("val"), val)
        etree.SubElement(pr, _m("ctrlPr")).append(self.wrpr(MATH))
        return pr

    def arg(self, parent, tag, nodes, display=False):
        el = etree.SubElement(parent, _m(tag))
        if display:
            etree.SubElement(etree.SubElement(el, _m("argPr")), _m("scrLvl")).set(_m("val"), "0")
        self.nodes(el, nodes or [])
        return el

    def nodes(self, parent, nodes):
        flat = list(_flat(nodes))
        if self.tidy:
            flat = _tidy_spaces(flat)
        buf = []
        for node in flat:
            if isinstance(node, Char):
                if buf and buf[-1].style != node.style:
                    self.run(parent, buf)
                    buf = []
                buf.append(node)
                continue
            if buf:
                self.run(parent, buf)
                buf = []
            self.node(parent, node)
        if buf:
            self.run(parent, buf)

    def run(self, parent, chars):
        style, text = chars[0].style, "".join(c.ch for c in chars)
        r = etree.SubElement(parent, _m("r"))
        if style.nor or style.sty or style.scr:
            rpr = etree.SubElement(r, _m("rPr"))
            if style.nor:
                etree.SubElement(rpr, _m("nor"))
            else:
                if style.scr:
                    etree.SubElement(rpr, _m("scr")).set(_m("val"), style.scr)
                if style.sty:
                    etree.SubElement(rpr, _m("sty")).set(_m("val"), style.sty)
        r.append(self.wrpr(style))
        t = etree.SubElement(r, _m("t"))
        t.text = text
        if text != text.strip() or "  " in text:
            t.set(XML_SPACE, "preserve")

    def node(self, parent, n):
        sub = lambda tag: etree.SubElement(parent, _m(tag))  # noqa: E731
        if isinstance(n, Frac):
            f = sub("f")
            self.props(f, "fPr", [] if n.bar else [("type", "noBar")])
            self.arg(f, "num", n.num, n.display)
            self.arg(f, "den", n.den, n.display)
        elif isinstance(n, Rad):
            rad = sub("rad")
            self.props(rad, "radPr", [] if n.deg else [("degHide", "1")])
            self.arg(rad, "deg", n.deg)
            self.arg(rad, "e", n.body)
        elif isinstance(n, Scripts):
            tag = "sSub" if n.sup is None else "sSup" if n.sub is None else "sSubSup"
            s = sub(tag)
            self.props(s, tag + "Pr")
            self.arg(s, "e", [n.base])
            if n.sub is not None:
                self.arg(s, "sub", n.sub)
            if n.sup is not None:
                self.arg(s, "sup", n.sup)
        elif isinstance(n, Pre):
            s = sub("sPre")
            self.props(s, "sPrePr")
            self.arg(s, "sub", n.sub)
            self.arg(s, "sup", n.sup)
            self.arg(s, "e", n.body)
        elif isinstance(n, Delim):
            d = sub("d")
            vals = []
            if n.beg != "(":
                vals.append(("begChr", n.beg))
            if len(n.parts) > 1 and n.sep != "|":
                vals.append(("sepChr", n.sep or ""))
            if n.end != ")":
                vals.append(("endChr", n.end))
            self.props(d, "dPr", vals)
            for part in n.parts:
                self.arg(d, "e", part)
        elif isinstance(n, Acc):
            a = sub("acc")
            self.props(a, "accPr", [("chr", n.chr)])
            self.arg(a, "e", n.body)
        elif isinstance(n, Bar):
            b = sub("bar")
            self.props(b, "barPr", [("pos", n.pos)])
            self.arg(b, "e", n.body)
        elif isinstance(n, Func):
            f = sub("func")
            self.props(f, "funcPr")
            self.arg(f, "fName", n.name)
            self.arg(f, "e", n.body)
        elif isinstance(n, Limit):
            tag = "limLow" if n.low else "limUpp"
            lim = sub(tag)
            self.props(lim, tag + "Pr")
            self.arg(lim, "e", n.body)
            self.arg(lim, "lim", n.lim)
        elif isinstance(n, Nary):
            nary = sub("nary")
            vals = [("chr", n.chr), ("limLoc", n.limloc)]
            if n.sub is None:
                vals.append(("subHide", "1"))
            if n.sup is None:
                vals.append(("supHide", "1"))
            self.props(nary, "naryPr", vals)
            self.arg(nary, "sub", n.sub)
            self.arg(nary, "sup", n.sup)
            self.arg(nary, "e", n.body)
        elif isinstance(n, EqArr):
            eq = sub("eqArr")
            self.props(eq, "eqArrPr")
            for row in n.rows:
                self.arg(eq, "e", row)
        elif isinstance(n, Matrix):
            self.matrix(sub("m"), n)
        else:
            raise ValueError(f"cannot write {type(n).__name__}")

    def matrix(self, m, n):
        pr = etree.SubElement(m, _m("mPr"))
        etree.SubElement(pr, _m("plcHide")).set(_m("val"), "1")
        mcs = etree.SubElement(pr, _m("mcs"))
        k = 0
        while k < len(n.jcs):  # one m:mc per run of equally aligned columns
            j = k
            while j < len(n.jcs) and n.jcs[j] == n.jcs[k]:
                j += 1
            mc_pr = etree.SubElement(etree.SubElement(mcs, _m("mc")), _m("mcPr"))
            etree.SubElement(mc_pr, _m("count")).set(_m("val"), str(j - k))
            etree.SubElement(mc_pr, _m("mcJc")).set(_m("val"), n.jcs[k])
            k = j
        etree.SubElement(pr, _m("ctrlPr")).append(self.wrpr(MATH))
        for row in n.rows:
            mr = etree.SubElement(m, _m("mr"))
            for col in range(len(n.jcs)):
                self.arg(mr, "e", row[col] if col < len(row) else [])


def _strip_math_delims(s):
    for a, b in (("$$", "$$"), ("$", "$"), ("\\(", "\\)"), ("\\[", "\\]")):
        if len(s) >= len(a) + len(b) and s.startswith(a) and s.endswith(b):
            return s[len(a):len(s) - len(b)].strip()
    return s


def _trim_spaces(nodes):
    """Drop spacing at both ends of the formula, also inside groups there."""
    for end in (0, -1):
        items = nodes
        while items:
            if _is_space(items[end]):
                items.pop(end)
            elif isinstance(items[end], Group) and items[end].items:
                items = items[end].items
            elif isinstance(items[end], Group):
                items.pop(end)
            else:
                break
    return nodes


def latex_to_omml(latex: str, *, size_pt: float | None = None, text_font: str | None = None,
                  spaces: str = "word") -> etree._Element:
    """Convert one LaTeX formula to a Word equation: an m:oMath element.

    size_pt sets the font size of every run, text_font the font of \\text runs (math runs
    use Cambria Math). spaces: "word" writes each run of spacing commands (\\, \\: ~ \\quad...)
    as one no-break space, "none" drops them, "all" writes each as a Unicode space of its width.
    Raises ValueError for unsupported or malformed input.
    """
    if not isinstance(latex, str):
        raise ValueError("formula must be a string")
    if spaces not in ("none", "word", "all"):
        raise ValueError(f"spaces must be none, word or all, not {spaces!r}")
    src = _strip_math_delims(unicodedata.normalize("NFC", latex).strip())
    if len(src) > MAX_LEN:
        raise ValueError("formula too long")
    nodes = _trim_spaces(_Parser(src, spaces).parse_list({"eof"}))
    if _is_empty(nodes):
        raise ValueError("empty formula")
    root = etree.Element(_m("oMath"), nsmap={"m": M_NS, "w": W_NS})
    _Writer(size_pt, text_font, spaces == "word").nodes(root, nodes)
    return root
