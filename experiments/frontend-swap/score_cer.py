# Score the pyopenjtalk-plus front-end clips with faster-whisper large-v3.
# Same references and kana-CER method as the baseline (stock front end)
# scores, so the two runs are directly comparable. Each clip also gets a
# numeral-normalized CER (numeral_norm.py): digit expressions are converted
# to readings on both sides first, which removes the notation artifact
# described in this folder's README.
#
# Usage: python score_cer.py [audio_dir]
# Expects <id>_oj.wav files in audio_dir (default: ./audio next to this file).
# The ASR model is $FW_MODEL or "large-v3" (a local faster-whisper model
# directory also works). WAVs are decoded directly and resampled to 16 kHz
# with scipy, so no ffmpeg/PyAV dependency is needed.
import json, os, sys, time, unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from numeral_norm import normalize_numerals

HERE = Path(__file__).resolve().parent
AUDIO = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "audio"

CLIPS = [
    ("p7", "p7_oj.wav", "ブラウザを開いて、メールを読んでください。"),
    ("p8", "p8_oj.wav", "テストが通ったら、新しいブランチを作って変更をプッシュし、プルリクエストを開きます。"),
    ("p9", "p9_oj.wav", "今日は2026年11月19日木曜日です。会議は午後3時30分からで、参加者は12人です。"),
    ("s01", "s01_oj.wav", "こんにちは、これは日本語の音声合成のテストです。"),
    ("s02", "s02_oj.wav", "はじめまして。私は日本語で話す、小さな音声合成モデルです。"),
    ("s03", "s03_oj.wav", "今日は十一月十九日、木曜日です。東京の天気は、晴れのち曇りでしょう。"),
    ("s04", "s04_oj.wav", "人工知能が、日本語の文章を読み上げます。声の大きさも、速さも、自由に変えられます。"),
]

def kana(text):
    from pykakasi import kakasi
    k = kakasi()
    k.setMode("J", "H")
    k.setMode("K", "H")
    out = k.getConverter().do(text)
    out = "".join(ch for ch in out if "ぁ" <= ch <= "ゖ" or ch in "ーゝゞ")
    return unicodedata.normalize("NFKC", out)

def cer(ref, hyp):
    if not ref:
        return 1.0 if hyp else 0.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[len(hyp)] / len(ref)

def load16k(path):
    import wave
    import numpy as np
    from scipy.signal import resample_poly
    from math import gcd
    with wave.open(path, "rb") as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        g = gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    return x

def main():
    from faster_whisper import WhisperModel
    t0 = time.time()
    model = WhisperModel(os.environ.get("FW_MODEL", "large-v3"),
                         device="cpu", compute_type="int8")
    print(f"model loaded in {time.time()-t0:.1f}s", flush=True)
    results = []
    for cid, fname, ref in CLIPS:
        path = AUDIO / fname
        t0 = time.time()
        segments, info = model.transcribe(load16k(str(path)), language="ja")
        hyp = "".join(s.text for s in segments).strip()
        dt = time.time() - t0
        rk, hk = kana(ref), kana(hyp)
        c = cer(rk, hk)
        c_norm = cer(kana(normalize_numerals(ref)), kana(normalize_numerals(hyp)))
        results.append({"id": cid, "file": fname, "reference": ref, "hypothesis": hyp,
                        "ref_kana": rk, "hyp_kana": hk, "cer": round(c, 4),
                        "cer_numeral_normalized": round(c_norm, 4),
                        "transcribe_sec": round(dt, 1)})
        print(f"{cid}: CER {c*100:.1f}% (normalized {c_norm*100:.1f}%) hyp={hyp}", flush=True)
    out = {"asr": "faster-whisper large-v3 int8 CPU", "frontend": "pyopenjtalk-plus",
           "clips": results,
           "mean_cer_bench_p7_p9": round(sum(r["cer"] for r in results[:3]) / 3, 4),
           "mean_cer_phase5_s01_s04": round(sum(r["cer"] for r in results[3:]) / 4, 4),
           "mean_cer_numeral_normalized": round(sum(r["cer_numeral_normalized"] for r in results) / len(results), 4)}
    out_path = HERE / "scores_oj.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"WROTE {out_path}")

if __name__ == "__main__":
    main()
