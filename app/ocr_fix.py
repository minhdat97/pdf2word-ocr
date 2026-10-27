"""Fixes for Tesseract's usual misreads on Vietnamese exams, applied after OCR.

Only what the page's structure settles: an answer label run into its text
("A.0K."), digits read as letters ("I0O"), syllables run together ("phântử"),
a part heading's numeral ("PHẦN L."), a tag that differs from its many
copies ("[ĐNHỊ]" among "[ĐNH]"), a column of question numbers or answers.
It never picks between two real words ("chuyên" / "chuyển"): that needs a
better recogniser, not a rule.
"""

import re
import unicodedata
from collections import Counter

from .model import Cell

# --- Vietnamese syllables ------------------------------------------------------

ONSETS = ["ngh", "ng", "nh", "ch", "gh", "gi", "kh", "ph", "qu", "th", "tr"] + list("bcdđghklmnprstvx")
_OPEN = (
    "a e ê i o ô ơ u ư y ia ua ưa ai ao au ay âu ây eo êu iu oi ôi ơi ui ưi ưu iêu yêu uôi ươi ươu "
    "oa oe uê uy oai oay oao oeo uây uya uyu uơ"
).split()
_NUCLEI = "a ă â e ê i o ô ơ u ư y iê yê uô ươ oa oă oe uâ uê uy uyê oo".split()
_CODAS = "c ch m n ng nh p t".split()
RIMES = set(_OPEN) | {n + c for n in _NUCLEI for c in _CODAS}
_TONES = {"̀", "́", "̃", "̉", "̣"}  # grave, acute, tilde, hook, dot below


def _toneless(s: str) -> str:
    """Lower case without tone marks; â ă ê ô ơ ư đ stay."""
    d = unicodedata.normalize("NFD", s.lower())
    return unicodedata.normalize("NFC", "".join(ch for ch in d if ch not in _TONES))


def syllable(s: str) -> tuple[bool, bool]:
    """(is a Vietnamese syllable, has an onset)."""
    t = _toneless(s)
    for onset in ONSETS + [""]:
        if t.startswith(onset) and t[len(onset) :] in RIMES:
            return True, bool(onset)
    return False, False


# Loanwords written as one word whose every syllable carries a Vietnamese mark
# (those with a plain part, "canô", "nitơ", "vectơ", "mômen", are never split).
LOANWORDS = {"ôtô", "rôbốt", "điốt", "phôtôn", "côngtơ", "tôpô", "pôpơlin"}
_LOANWORDS_TONELESS = {_toneless(w) for w in LOANWORDS}  # "điốt" was compared toned and split


def split_syllables(word: str) -> str:
    """A run of letters that is no syllable but two to five run together, each
    with a Vietnamese mark ("phântử", "sựnóngchảy", "nhiệtđộcủakhí" – an answer
    option whose spaces Tesseract lost): the split with the fewest
    parts, then the fewest parts lacking an onset ("sự nóng", not "sựn óng").
    A part without marks ("ca" in "canô") means a loanword: left alone."""
    if len(word) > 24 or word.isascii() or syllable(word)[0] or _toneless(word) in _LOANWORDS_TONELESS:
        return word
    best = None

    def walk(start: int, parts: list[str], bare: int) -> None:
        nonlocal best
        if start == len(word):
            key = (len(parts), bare)
            if len(parts) > 1 and (best is None or key < best[0]):
                best = (key, list(parts))
            return
        if len(parts) == 5:
            return
        for end in range(start + 1, len(word) + 1):
            ok, onset = syllable(word[start:end])
            if ok and not word[start:end].isascii():
                walk(end, parts + [word[start:end]], bare + (not onset))

    walk(0, [], 0)
    return " ".join(best[1]) if best else word


# --- words ---------------------------------------------------------------------

_DIGIT_LIKE = str.maketrans({"I": "1", "l": "1", "|": "1", "!": "1", "O": "0", "o": "0"})
_NUMBER_LIKE = re.compile(r"^([(\[]?)([0-9IlO|o]+(?:[,.][0-9IlO|o]+)*)(.*)$")
_LABEL_RUN_ON = re.compile(r"^(\*?[A-D]\.)(?=[^\s.,])")  # "A.0K." -> "A. 0K."
_LETTERS = re.compile(r"[^\W\d_]+")


def _number(text: str) -> str:
    """Digits read as I, l, | or O inside a number ("I0O" -> "100"). The run
    needs a digit and three characters, or to start with a digit ("O2" stays)."""
    m = _NUMBER_LIKE.match(text)
    if not m or not any(ch.isdigit() for ch in m.group(2)):
        return text
    core = m.group(2)
    if not (core[0].isdigit() or len(core) >= 3) or core.isdigit():
        return text
    return m.group(1) + core.translate(_DIGIT_LIKE) + m.group(3)


UNITS = set(
    "mol cm mm km dm m g kg mg s ms h ml cal kcal kJ kW Pa kPa Hz kHz atm eV MeV rad mA mV kV lít phút giây giờ".split()
)


def _unit_space(m: re.Match) -> str:
    letters = m.group(2)
    if letters in UNITS or not letters.isascii():
        return f"{m.group(1)} {letters}"
    return m.group(0)


def fix_word(text: str) -> str:
    text = re.sub(r"(?<=\S)\.\.(?=[A-D]\.)", ". ", text)  # "273K..D.373" -> "273K. D.373"
    text = re.sub(r"(?<!\.)\.\.(?!\.)", ".", text)  # ".." is a misread "." ("..." is kept)
    parts = []
    for part in text.split(" "):
        m = _LABEL_RUN_ON.match(part)
        pieces = [m.group(1), part[m.end() :]] if m else [part]
        for piece in pieces:
            piece = _number(piece)
            # a Vietnamese word glued to a number ("và4" -> "và 4"; "cm3", "NaN3" stay)
            m = re.match(r"^([^\W\d_]+)(\d)", piece)
            if m and not m.group(1).isascii():
                piece = m.group(1) + " " + piece[m.end(1) :]
            # a number glued to its unit or to a word ("2mol", "4giờ"; "2NaN3", "3x" stay)
            piece = re.sub(r"^(\d+(?:[,.]\d+)*)([^\W\d_]+)", _unit_space, piece)
            parts.append(_LETTERS.sub(lambda w: split_syllables(w.group(0)), piece))
    return " ".join(parts)


# --- page ----------------------------------------------------------------------

_ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
_ROMAN_LIKE = str.maketrans({"l": "I", "1": "I", "|": "I", "!": "I", "L": "I", "H": "II", "U": "II"})
_TAG = re.compile(r"^\[([^\]\s]{2,6})\](.*)$")


def _part_heading(words: list, prev: int) -> int:
    """'PHẦN' + numeral at the start of a line: heading words and numeral
    repaired from the order of the parts (I, II, III; a new exam restarts at I)."""
    if len(words) < 2 or _toneless(words[0].text) not in ("phân", "phan", "phần"):
        return prev
    m = re.match(r"^([IVXLHUl1|!]{1,4})([.:]?)(.*)$", words[1].text)
    if not m:
        return prev
    raw = m.group(1).translate(_ROMAN_LIKE)
    value = _ROMAN.index(raw) + 1 if raw in _ROMAN else None
    if value is None or (value <= prev and value != 1):
        value = prev + 1
    if value > len(_ROMAN):
        return prev
    words[0].text = "PHẦN" if words[0].text.isupper() else "Phần"
    words[1].text = _ROMAN[value - 1] + m.group(2) + m.group(3)
    return value


def _edit_distance(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def fix_lines(lines: list) -> None:
    """Fix OCR words in place (lines in reading order, each with .words)."""
    for ln in lines:
        for wd in ln.words:
            wd.text = fix_word(wd.text)
    part = 0
    for ln in lines:
        part = _part_heading(ln.words, part)
    # A level tag seen many times ("[ĐNH]") and once with one letter off is that tag.
    tags = Counter(m.group(1) for ln in lines for wd in ln.words if (m := _TAG.match(wd.text)))
    common = [t for t, n in tags.items() if n >= 3]
    for ln in lines:
        for wd in ln.words:
            m = _TAG.match(wd.text)
            if not m or m.group(1) in common:
                continue
            near = [t for t in common if _edit_distance(t, m.group(1)) == 1]
            if len(near) == 1:
                wd.text = f"[{near[0]}]{m.group(2)}"


# --- tables --------------------------------------------------------------------

_DS = str.maketrans({"D": "Đ", "Ð": "Đ", "Ɖ": "Đ", "5": "S", "$": "S"})


def fix_table(cells: list[list[Cell]]) -> None:
    """Columns that are clearly question numbers (1, 2, 3...), answers A-D or
    true/false strings (ĐSĐĐ) get their odd cells read the same way."""
    if len(cells) < 3:
        return
    for c in range(len(cells[0])):
        body = [(r, row[c]) for r, row in enumerate(cells) if c < len(row)]
        texts = [cell.text.strip() for _, cell in body]
        filled = [t for t in texts if t]
        if len(filled) < 3:
            continue
        # question numbers: most cells read as row number + one offset
        offsets = Counter(
            int(t.translate(_DIGIT_LIKE)) - r for (r, _), t in zip(body, texts) if t.translate(_DIGIT_LIKE).isdigit()
        )
        if offsets:
            offset, n = offsets.most_common(1)[0]
            if n >= 0.6 * len(filled) and n >= 3:
                for (r, cell), t in zip(body, texts):
                    # only number-like cells ("1\"", "l"); a heading ("Câu") stays
                    number_like = any(ch.isdigit() for ch in t.translate(_DIGIT_LIKE))
                    if number_like and len(t) <= 4 and t != str(r + offset) and r + offset > 0:
                        cell.text = str(r + offset)
                continue
        if sum(bool(re.fullmatch(r"[A-D]", t)) for t in filled) >= 0.6 * len(filled):
            for _, cell in body:
                t = cell.text.strip()
                fixed = {"a": "A", "b": "B", "c": "C", "d": "D", "8": "B"}.get(t, t)
                if re.fullmatch(r"[A-D]", fixed):
                    cell.text = fixed
            continue
        if sum(bool(re.fullmatch(r"[ĐS]{2,6}", t)) for t in filled) >= 0.6 * len(filled):
            for _, cell in body:
                fixed = cell.text.strip().translate(_DS)
                if re.fullmatch(r"[ĐS]{2,6}", fixed):
                    cell.text = fixed
