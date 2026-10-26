"""Convert text set in legacy Vietnamese font encodings (VNI, TCVN3/ABC) to Unicode.

Documents typed with fonts such as "VNI-Times" or ".VnTime" store Vietnamese
letters as Latin-1 look-alikes, so a PDF's text layer reads "ÑEÀ KIEÅM TRA"
(VNI) or "§Ò KiÓm tra" (TCVN3) instead of "ĐỀ KIỂM TRA". Detection uses the
font name first and, for VNI only, falls back to characters that never occur
in Unicode Vietnamese.
"""

import re
import unicodedata

_ACUTE, _GRAVE, _HOOK, _TILDE, _DOT = "́", "̀", "̉", "̃", "̣"
_CIRCUMFLEX, _BREVE, _HORN = "̂", "̆", "̛"
_TONES = [("sac", _ACUTE), ("huyen", _GRAVE), ("hoi", _HOOK), ("nga", _TILDE), ("nang", _DOT)]


def _nfc(*parts: str) -> str:
    return unicodedata.normalize("NFC", "".join(parts))


def _vni_table() -> dict[str, str]:
    """VNI-Windows: a base letter followed by one modifier character."""
    tone_mod = {
        False: {"sac": "ù", "huyen": "ø", "hoi": "û", "nga": "õ", "nang": "ï"},
        True: {"sac": "Ù", "huyen": "Ø", "hoi": "Û", "nga": "Õ", "nang": "Ï"},
    }
    circ_mod = {
        False: {"": "â", "sac": "á", "huyen": "à", "hoi": "å", "nga": "ã", "nang": "ä"},
        True: {"": "Â", "sac": "Á", "huyen": "À", "hoi": "Å", "nga": "Ã", "nang": "Ä"},
    }
    breve_mod = {
        False: {"": "ê", "sac": "é", "huyen": "è", "hoi": "ú", "nga": "ü", "nang": "ë"},
        True: {"": "Ê", "sac": "É", "huyen": "È", "hoi": "Ú", "nga": "Ü", "nang": "Ë"},
    }
    table: dict[str, str] = {"ñ": "đ", "Ñ": "Đ"}
    for upper in (False, True):
        case = str.upper if upper else str.lower
        # Modifiers follow the case of the text, but mixed case (e.g. "Ñaø")
        # also occurs, so register lowercase modifiers on uppercase bases too.
        for mod_upper in {upper, False}:
            tones, circ, breve = tone_mod[mod_upper], circ_mod[mod_upper], breve_mod[mod_upper]
            for base in "aeouy":
                b = case(base)
                for name, mark in _TONES:
                    table[b + tones[name]] = _nfc(b, mark)
            for base in "aeo":
                b = case(base)
                for name, mark in [("", "")] + _TONES:
                    table[b + circ[name]] = _nfc(b, _CIRCUMFLEX, mark)
            a = case("a")
            for name, mark in [("", "")] + _TONES:
                table[a + breve[name]] = _nfc(a, _BREVE, mark)
            for horn_char, base in (("Ô" if upper else "ô", "o"), ("Ö" if upper else "ö", "u")):
                for name, mark in _TONES:
                    table[horn_char + tones[name]] = _nfc(case(base), _HORN, mark)
    # Single-character forms.
    table.update({"ô": "ơ", "Ô": "Ơ", "ö": "ư", "Ö": "Ư"})
    table.update({"æ": "ỉ", "ó": "ĩ", "ò": "ị", "Æ": "Ỉ", "Ó": "Ĩ", "Ò": "Ị", "î": "ỵ", "Î": "Ỵ"})
    return table


_TCVN3 = dict(
    zip(
        "¸µ¶·¹¨¾»¼½Æ©ÊÇÈÉË®ÐÌÎÏÑªÕÒÓÔÖÝ×ØÜÞãßáâä«èåæçé¬íêëìîóïñòô­øõö÷ùýúûüþ¡¢§£¤¥¦",
        "áàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵĂÂĐÊÔƠƯ",
        strict=True,
    )
)

_VNI = _vni_table()
_VNI_RE = re.compile("|".join(sorted(map(re.escape, _VNI), key=len, reverse=True)))
_TCVN3_RE = re.compile("|".join(map(re.escape, _TCVN3)))

# Present in VNI text, absent from Unicode Vietnamese.
_VNI_SIGNALS = set("øØûÛïÏñÑåÅäÄëËæÆîÎ")
_VNI_DIGRAPHS = re.compile(r"[aeoAEO][âÂ]|[aA][êÊ]")
_UNICODE_VIETNAMESE = re.compile(r"[ăĂđĐơƠưƯẠ-ỹ]")


def _base_font(font: str) -> str:
    return font.split("+", 1)[-1]


def legacy_encoding(font: str) -> str | None:
    name = _base_font(font)
    if re.match(r"VNI[-_ ]", name, re.I):
        return "vni"
    if re.match(r"[._]?Vn[A-Z]", name):
        return "tcvn3"
    return None


def looks_like_vni(text: str) -> bool:
    if _UNICODE_VIETNAMESE.search(text):
        return False
    score = sum(ch in _VNI_SIGNALS for ch in text) + len(_VNI_DIGRAPHS.findall(text))
    return score >= 2


def vni_to_unicode(text: str) -> str:
    return _VNI_RE.sub(lambda m: _VNI[m.group()], text)


def tcvn3_to_unicode(text: str, uppercase_font: bool = False) -> str:
    out = _TCVN3_RE.sub(lambda m: _TCVN3[m.group()], text)
    return out.upper() if uppercase_font else out


def _unicode_family(font: str) -> str:
    return "Times New Roman" if re.search(r"tim", font, re.I) else "Arial"


def decode_span(text: str, font: str) -> tuple[str, str | None]:
    """Return (unicode text, replacement font family or None if not legacy)."""
    encoding = legacy_encoding(font)
    if encoding is None and looks_like_vni(text):
        encoding = "vni"
    if encoding == "vni":
        return vni_to_unicode(text), _unicode_family(font)
    if encoding == "tcvn3":
        style_free = re.split(r"[-,]", _base_font(font))[0]
        return tcvn3_to_unicode(text, style_free.endswith("H")), _unicode_family(font)
    return text, None
