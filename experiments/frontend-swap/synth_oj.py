"""Synthesize the 7 evaluation sentences with the pyopenjtalk-plus front end.

Usage:
    python synth_oj.py <checkpoint.pt> [output_dir]

The checkpoint defaults to $TINY_VOCA_CHECKPOINT or runs/ja-v1/inference.pt.
Loading the model runs the stock front end's fingerprint check, which needs
eSpeak NG to be importable by the tiny_voca package; if your eSpeak NG lives
in a non-standard prefix, set ESPEAK_HOME to it (bin/, lib/, share/espeak-ng-data).
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

if os.environ.get("ESPEAK_HOME"):
    ESPEAK = Path(os.environ["ESPEAK_HOME"])
    os.environ["PATH"] = str(ESPEAK / "bin") + os.pathsep + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = str(ESPEAK / "lib")
    os.environ["ESPEAK_DATA_PATH"] = str(ESPEAK / "share" / "espeak-ng-data")

import soundfile as sf

from tiny_voca.infer import load_model, _voice, limit_peak
from oj_frontend import phonemize_oj

CHECKPOINT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    os.environ.get("TINY_VOCA_CHECKPOINT", "runs/ja-v1/inference.pt"))
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "audio"
OUT.mkdir(parents=True, exist_ok=True)

ITEMS = [
    ("p7", "F2", "ブラウザを開いて、メールを読んでください。"),
    ("p8", "F2", "テストが通ったら、新しいブランチを作って変更をプッシュし、プルリクエストを開きます。"),
    ("p9", "F2", "今日は2026年11月19日木曜日です。会議は午後3時30分からで、参加者は12人です。"),
    ("s01", "F2", "こんにちは、これは日本語の音声合成のテストです。"),
    ("s02", "F2", "はじめまして。私は日本語で話す、小さな音声合成モデルです。"),
    ("s03", "M2", "今日は十一月十九日、木曜日です。東京の天気は、晴れのち曇りでしょう。"),
    ("s04", "F2", "人工知能が、日本語の文章を読み上げます。声の大きさも、速さも、自由に変えられます。"),
]


def main():
    model = load_model(CHECKPOINT, "cpu")
    sr = model.config.sample_rate
    for name, voice, text in ITEMS:
        phones = phonemize_oj(text)
        wav, _details = _voice(model, phones, voice, 1.0, 0.0, 0, 1.0)
        wav = limit_peak(wav)
        out = OUT / f"{name}_oj.wav"
        sf.write(out, wav.numpy(), sr)
        print(f"{name} ({voice}): {len(wav)/sr:.2f}s unknown={phones.unknown} -> {out}")


if __name__ == "__main__":
    main()
