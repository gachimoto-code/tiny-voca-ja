"""ASR round-trip check for tiny-voca-ja Phase 5.
Transcribe a WAV with local faster-whisper small (language=ja),
kana-normalize reference and hypothesis with pykakasi, report char-level CER.
Usage: python asr_cer.py <wav> "<reference text>"
"""
import sys, unicodedata
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from faster_whisper import WhisperModel
from pykakasi import kakasi

def kana(text):
    k = kakasi()
    k.setMode("J", "H")  # kanji -> hiragana
    k.setMode("K", "H")  # katakana -> hiragana
    conv = k.getConverter()
    out = conv.do(text)
    # keep only hiragana (drop punctuation, spaces, latin, digits)
    out = "".join(ch for ch in out if "ぁ" <= ch <= "ゖ" or ch in "ーゝゞ")
    return unicodedata.normalize("NFKC", out)

def cer(ref, hyp):
    if not ref:
        return 1.0 if hyp else 0.0
    # Levenshtein
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[len(hyp)] / len(ref)

def main():
    wav, ref_text = sys.argv[1], sys.argv[2]
    # Load + resample to 16kHz mono ourselves (bypasses faster-whisper's av
    # decoder, which is broken with the installed PyAV 19).
    audio, sr = sf.read(wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = resample_poly(audio, 16000, sr).astype(np.float32)
    model = WhisperModel("models/whisper-small", device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio, language="ja", beam_size=5)
    hyp_text = "".join(s.text for s in segments).strip()
    ref_k, hyp_k = kana(ref_text), kana(hyp_text)
    print(f"reference : {ref_text}")
    print(f"hypothesis: {hyp_text}")
    print(f"ref kana  : {ref_k} ({len(ref_k)} chars)")
    print(f"hyp kana  : {hyp_k} ({len(hyp_k)} chars)")
    print(f"CER (kana): {cer(ref_k, hyp_k):.4f}  ({cer(ref_k, hyp_k)*100:.1f}%)")

if __name__ == "__main__":
    main()
