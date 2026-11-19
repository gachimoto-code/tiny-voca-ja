"""Normalize Arabic-numeral expressions in Japanese text to their readings.

ASR systems (Whisper included) write numbers as digits ("2026年11月19日"),
while reference texts may use kanji numerals ("十一月十九日"). The kana-CER
scorer drops digits entirely, which makes the same sounds score as errors
on one side only. Converting digit expressions to readings on BOTH sides
before kana conversion removes that notation artifact.
"""

import re

_DIGIT = {"0": "ぜろ", "1": "いち", "2": "に", "3": "さん", "4": "よん",
          "5": "ご", "6": "ろく", "7": "なな", "8": "はち", "9": "きゅう"}
_ONES = ["", "いち", "に", "さん", "よん", "ご", "ろく", "なな", "はち", "きゅう"]


def int_reading(n):
    """Reading of an integer 0..9999 (no counter suffix)."""
    if n == 0:
        return "ぜろ"
    out = []
    th, rem = divmod(n, 1000)
    if th:
        out.append({1: "せん", 3: "さんぜん", 8: "はっせん"}.get(th, _ONES[th] + "せん"))
    hu, rem = divmod(rem, 100)
    if hu:
        out.append({1: "ひゃく", 3: "さんびゃく", 6: "ろっぴゃく", 8: "はっぴゃく"}.get(hu, _ONES[hu] + "ひゃく"))
    te, on = divmod(rem, 10)
    if te:
        out.append("じゅう" if te == 1 else _ONES[te] + "じゅう")
    if on:
        out.append(_ONES[on])
    return "".join(out)


_MONTH = {4: "しがつ", 7: "しちがつ", 9: "くがつ"}
_DAY = {1: "ついたち", 2: "ふつか", 3: "みっか", 4: "よっか", 5: "いつか",
        6: "むいか", 7: "なのか", 8: "ようか", 9: "ここのか", 10: "とおか",
        14: "じゅうよっか", 20: "はつか", 24: "にじゅうよっか"}


def _pun(n):
    """分 suffix: ぷん after 1,3,4,6,8 and after round tens, else ふん."""
    if n % 10 == 0 and n > 0:
        base = int_reading(n)
        return base[:-1] + "っぷん" if base.endswith("う") else base + "っぷん"
    if n % 10 in (1, 3, 4, 6, 8):
        base = int_reading(n)
        return base[:-1] + "っぷん" if n % 10 in (1, 6, 8) else base + "ぷん"
    return int_reading(n) + "ふん"


def normalize_numerals(text):
    # Unit notation the ASR prefers (%, kg) vs spelled-out reference words:
    # map the symbols to the words so both sides carry them into kana space.
    text = re.sub(r"\s*[%％]", "パーセント", text)
    text = re.sub(r"(?<=\d)\s*kg", "キログラム", text, flags=re.IGNORECASE)

    def rep(pattern, func):
        nonlocal text
        text = re.sub(pattern, lambda m: func(int(m.group(1))), text)

    rep(r"(\d+)年", lambda n: int_reading(n) + "ねん")
    rep(r"(\d+)月", lambda n: _MONTH.get(n, int_reading(n) + "がつ"))
    rep(r"(\d+)日", lambda n: _DAY.get(n, int_reading(n) + "にち"))
    rep(r"(\d+)時", lambda n: int_reading(n) + "じ")
    rep(r"(\d+)分", _pun)
    rep(r"(\d+)人", lambda n: {1: "ひとり", 2: "ふたり", 4: "よにん"}.get(n, int_reading(n) + "にん"))
    rep(r"(\d+)歳", lambda n: {1: "いっさい", 8: "はっさい", 10: "じゅっさい", 20: "はたち"}.get(n, int_reading(n) + "さい"))
    return text


if __name__ == "__main__":
    import sys
    for t in sys.argv[1:]:
        print(t, "->", normalize_numerals(t))
