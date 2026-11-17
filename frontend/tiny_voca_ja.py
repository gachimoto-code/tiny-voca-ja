#!/usr/bin/env python3
"""tiny-voca 日本語フロントエンド試作 v0（再学習なし）

英語版 tiny-voca の音素在庫（76記号）の範囲で日本語テキストを音素列に変換し、
既存の重み（F2/M2）のまま日本語を読ませる実験用スクリプト。

変換の流れ:
  日本語テキスト -> 数字の読み展開 -> pykakasi で漢字をひらがな化
  -> モーラ単位で IPA 記号（モデル在庫内）に近似マッピング -> モデル推論

割り切り（v0）:
  - ピッチアクセントは扱わない（stress は全て 0）
  - 母音: あ=æ い=i う=ʊ え=ɛ お=ɔ、長音は あ=ɑː い=iː う=uː え=eɪ お=oː
  - ふ=f、ら行=ɾ、つ=t+s のように最寄りの在庫音素で近似
  - ん=n（後続音による同化はなし）、っ=次の子音を重ねる
"""
import argparse
import os
import re
import sys

os.environ.setdefault("DIZIONE_DATA", os.path.expanduser("~/tinyvoca/weights/phonemizer"))
os.environ.setdefault("TINY_VOCA_FRONTEND", "own")
sys.path.insert(0, os.path.expanduser("~/tinyvoca/repo/src"))

# ---------- 数字の読み ----------
DIGITS = ["ぜろ", "いち", "に", "さん", "よん", "ご", "ろく", "なな", "はち", "きゅう"]
HUNDREDS = {1: "ひゃく", 2: "にひゃく", 3: "さんびゃく", 4: "よんひゃく", 5: "ごひゃく",
            6: "ろっぴゃく", 7: "ななひゃく", 8: "はっぴゃく", 9: "きゅうひゃく"}
THOUSANDS = {1: "せん", 2: "にせん", 3: "さんぜん", 4: "よんせん", 5: "ごせん",
             6: "ろくせん", 7: "ななせん", 8: "はっせん", 9: "きゅうせん"}


def read_4digits(n):
    out = ""
    th, rem = divmod(n, 1000)
    if th:
        out += THOUSANDS[th]
    hu, rem = divmod(rem, 100)
    if hu:
        out += HUNDREDS[hu]
    te, on = divmod(rem, 10)
    if te:
        out += ("" if te == 1 else DIGITS[te]) + "じゅう"
    if on:
        out += DIGITS[on]
    return out


def read_number(n):
    if n == 0:
        return DIGITS[0]
    out = ""
    oku, rem = divmod(n, 100_000_000)
    if oku:
        out += read_number(oku) + "おく"
    man, rem = divmod(rem, 10_000)
    if man:
        out += read_4digits(man) + "まん"
    if rem:
        out += read_4digits(rem)
    return out


def expand_numbers(text):
    def repl(m):
        return read_number(int(m.group(0).replace(",", "")))
    return re.sub(r"\d[\d,]*", repl, text)


# ---------- かな -> 音素 ----------
# (子音, 母音) 母音キーは a/i/u/e/o
MORA = {}
_ROWS = {
    "": "あいうえお",
    "k": "かきくけこ", "g": "がぎぐげご",
    "s": "さしすせそこ",  # placeholder fixed below
}
_ROWS = {
    "": "あいうえお",
    "k": "かきくけこ", "g": "がぎぐげご",
    "s": "さしすせそこ".replace("そこ", "せそこ") if False else "さしすせそ",
    "z": "ざじずぜぞ",
    "t": "たちつてと", "d": "だぢづでど",
    "n": "なにぬねの",
    "h": "はひふへほ", "b": "ばびぶべぼ", "p": "ぱぴぷぺぽ",
    "m": "まみむめも",
    "y": "やゆよ",
    "r": "らりるれろ",
    "w": "わを",
}
_VOWELS = "aiueo"
for _cons, _kanas in _ROWS.items():
    for _v, _kana in zip(_VOWELS, _kanas):
        MORA[_kana] = (_cons, _v)
MORA["ん"] = ("N", None)
MORA["や"] = ("y", "a"); MORA["ゆ"] = ("y", "u"); MORA["よ"] = ("y", "o")
MORA["わ"] = ("w", "a"); MORA["を"] = ("w", "o")
# 特殊な読み
MORA["し"] = ("sh", "i"); MORA["じ"] = ("j", "i"); MORA["ぢ"] = ("j", "i")
MORA["ち"] = ("ch", "i"); MORA["つ"] = ("ts", "u"); MORA["づ"] = ("z", "u")
MORA["ふ"] = ("f", "u")
# 小さい母音単体（外来音の後半など）
for _k, _v in zip("ぁぃぅぇぉ", "aiueo"):
    MORA[_k] = ("", _v)

CONS_IPA = {
    "": [], "k": ["k"], "g": ["ɡ"], "s": ["s"], "z": ["z"], "sh": ["ʃ"], "j": ["dʒ"],
    "t": ["t"], "d": ["d"], "ch": ["tʃ"], "ts": ["t", "s"], "n": ["n"], "h": ["h"],
    "f": ["f"], "b": ["b"], "p": ["p"], "m": ["m"], "y": ["j"], "r": ["ɾ"], "w": ["w"],
    "v": ["v"],
}
VOWEL_IPA = {"a": "æ", "i": "i", "u": "ʊ", "e": "ɛ", "o": "ɔ"}
VOWEL_LONG_IPA = {"a": "ɑː", "i": "iː", "u": "uː", "e": "eɪ", "o": "oː"}
_SMALL_Y = {"ゃ": "a", "ゅ": "u", "ょ": "o"}
# 拗音の子音上書き（し+ゃ = sha など）
_YOUON_CONS = {"sh": "sh", "j": "dʒ", "ch": "tʃ"}
# 外来音コンボ: 直前のモーラと小母音の組み合わせ
_FOREIGN = {
    ("f", "a"): (["f"], "a"), ("f", "i"): (["f"], "i"), ("f", "e"): (["f"], "e"), ("f", "o"): (["f"], "o"),
    ("t", "i"): (["t"], "i"), ("d", "i"): (["d"], "i"), ("t", "u"): (["t"], "u"), ("d", "u"): (["d"], "u"),
    ("sh", "e"): (["ʃ"], "e"), ("j", "e"): (["dʒ"], "e"), ("ch", "e"): (["tʃ"], "e"),
    ("w", "i"): (["w"], "i"), ("w", "e"): (["w"], "e"), ("w", "o"): (["w"], "o"),
    ("v", None): (["v"], None),
}
_PUNCT = {"。": ".", "、": ",", "？": "?", "!": "!", "！": "!", "・": ",", "「": '"', "」": '"',
          ".": ".", ",": ",", "?": "?", ":": ":", ";": ";"}


def kata_to_hira(text):
    out = []
    for ch in text:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        else:
            out.append(ch)
    return "".join(out)


def kana_to_symbols(kana):
    """ひらがな文字列 -> モデルの記号列（単語境界なし、句読点含む）"""
    symbols = []
    i = 0
    geminate = False
    prev_vowel = None  # 直前モーラの母音キー（長音処理用）
    while i < len(kana):
        ch = kana[i]
        if ch in _PUNCT:
            symbols.append(_PUNCT[ch])
            prev_vowel = None
            i += 1
            continue
        if ch == " " or ch == "　":
            symbols.append("<sp>")
            prev_vowel = None
            i += 1
            continue
        if ch == "っ":
            geminate = True
            i += 1
            continue
        if ch == "ー":
            if prev_vowel is not None and symbols:
                # 直前の母音記号を長母音記号に差し替える
                for j in range(len(symbols) - 1, -1, -1):
                    if symbols[j] in VOWEL_IPA.values():
                        symbols[j] = VOWEL_LONG_IPA[prev_vowel]
                        break
            i += 1
            continue
        if ch not in MORA:
            i += 1  # 未対応文字は読み飛ばす
            continue
        cons, vowel = MORA[ch]
        nxt = kana[i + 1] if i + 1 < len(kana) else ""
        if nxt in _SMALL_Y and cons not in ("", "y", "w"):
            # 拗音: 子音 + j + 母音
            base = _YOUON_CONS.get(cons, cons)
            cons_tokens = CONS_IPA[base] if isinstance(CONS_IPA.get(base), list) else CONS_IPA[cons]
            vowel = _SMALL_Y[nxt]
            tokens = list(cons_tokens) + ["j", VOWEL_IPA[vowel]]
            i += 2
        elif nxt != "" and nxt in "ぁぃぅぇぉ":
            tokens = CONS_IPA[cons] + [VOWEL_IPA[MORA[nxt][1]]]
            vowel = MORA[nxt][1]
            i += 2
        elif False:
            # 外来音（ふぁ、てぃ等）: 子音はそのまま、母音は小母音
            vowel = MORA[nxt][1]
            tokens = CONS_IPA[cons] + [VOWEL_IPA[vowel]]
            i += 2
        else:
            if cons == "N":
                tokens = ["n"]
                vowel = None
            else:
                tokens = list(CONS_IPA[cons])
                if vowel is not None:
                    tokens.append(VOWEL_IPA[vowel])
            i += 1
        if geminate and tokens:
            # 促音: 子音を重ねる（子音がない場合は ʔ）
            first = tokens[0]
            if first in ("æ", "i", "ʊ", "ɛ", "ɔ"):
                tokens = ["ʔ"] + tokens
            else:
                tokens = [first] + tokens
            geminate = False
        # 長音化: お段+う、え段+い（おう、えい）は長母音に
        if prev_vowel == "o" and ch == "う" and cons == "":
            for j in range(len(symbols) - 1, -1, -1):
                if symbols[j] == VOWEL_IPA["o"]:
                    symbols[j] = VOWEL_LONG_IPA["o"]
                    break
            prev_vowel = "o"
            continue
        if prev_vowel == "e" and ch == "い" and cons == "":
            for j in range(len(symbols) - 1, -1, -1):
                if symbols[j] == VOWEL_IPA["e"]:
                    symbols[j] = VOWEL_LONG_IPA["e"]
                    break
            prev_vowel = "e"
            continue
        symbols.extend(tokens)
        prev_vowel = vowel
    return symbols


def fix_particle_reading(hira):
    """助詞の「は」「へ」の読み替え（セグメント単位のヒューリスティクス）。

    pykakasi は「これは」「わたしは」「では」「こんにちは」のように
    助詞を前の語と1セグメントにまとめることがあるため、末尾の
    は→わ、へ→え を置き換える。笑い声（はは等）のような反復は除外。
    """
    if not hira:
        return hira
    if len(hira) == 1:
        return {"は": "わ", "へ": "え"}.get(hira, hira)  # 単独セグメントは助詞
    if set(hira) == {"は"} or set(hira) == {"へ"}:
        return hira  # 反復のみのセグメント（笑い声など）はそのまま
    if hira.endswith("は"):
        hira = hira[:-1] + "わ"
    elif hira.endswith("へ"):
        hira = hira[:-1] + "え"
    return hira


def text_to_phonemized(text):
    import pykakasi
    from tiny_voca.text import Phonemized, SYMBOL_TO_ID

    text = expand_numbers(text)
    kakasi = pykakasi.kakasi()
    symbols = ["<bos>"]
    for seg in kakasi.convert(text):
        hira = fix_particle_reading(kata_to_hira(seg["hira"]))
        seg_symbols = kana_to_symbols(hira)
        if not seg_symbols:
            continue
        if symbols[-1] not in ("<bos>",) and seg_symbols[0] not in _PUNCT.values():
            symbols.append("<sp>")
        symbols.extend(seg_symbols)
    symbols.append("<eos>")
    symbols = [s for s in symbols if s in SYMBOL_TO_ID]
    ids = tuple(SYMBOL_TO_ID[s] for s in symbols)
    stress = tuple(0 for _ in symbols)
    return Phonemized(tuple(symbols), ids, stress, ())


def main():
    parser = argparse.ArgumentParser(description="tiny-voca 日本語試作 v0")
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice", choices=("F2", "M2"), default="F2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=os.path.expanduser("~/tinyvoca/weights/tiny-voca.pt"))
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    import soundfile as sf
    from tiny_voca.infer import _voice, limit_peak, load_model

    phones = text_to_phonemized(args.text)
    print("symbols:", " ".join(phones.symbols))
    model = load_model(args.checkpoint)
    wave, _details = _voice(model, phones, args.voice, args.speed, 0.0, 0, 1.0)
    wave = limit_peak(wave)
    sf.write(args.output, wave.numpy(), 24000)
    print(f"saved: {args.output} ({len(wave)/24000:.2f}s)")


if __name__ == "__main__":
    main()
