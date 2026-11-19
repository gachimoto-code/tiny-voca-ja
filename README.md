# tiny-voca-ja

A Japanese text-to-speech model built with the tiny-voca recipe: trained from scratch on synthetic speech only — it has never heard a real human voice.

## 日本語まとめ

tiny-voca（n4ze3m）の日本語版です。人間の声を一切使わず、Qwen3-TTS が生成した合成音声だけを教師データにして学習しました。パラメータ数は 4,538,827、重みは 18MB、CPU で実時間より速く合成できます。モデルとデータセットは Hugging Face で公開しています（下記リンク）。

## Links

- **Model:** https://huggingface.co/gachimoto/tiny-voca-ja
- **Dataset:** https://huggingface.co/datasets/gachimoto/tiny-voca-ja
- **Original (English) tiny-voca:** https://github.com/n4ze3m/tiny-voca

## Features

- 4,538,827 parameters; the inference checkpoint is 18 MB
- Synthesizes faster than real time on CPU — no GPU needed
- 24 kHz mono output
- Two synthetic voices: F2 and M2

## How it was made

1. **Corpus** — 8,500 Japanese sentences from Tatoeba, Common Voice, and hand-written templates.
2. **Teacher audio** — Qwen3-TTS generated 16,321 clips in two designed voices (F2/M2).
3. **Screening** — each clip was transcribed with Whisper and kept only if the reading matched the text (kana-normalised CER).
4. **Annotation** — forced alignment with the prj-beatrice Japanese phoneme aligner produced the training labels; 11,561 clips passed (about 11.5 hours of audio).
5. **Training** — 104,000 steps on a RunPod RTX 4090.

## What changed for Japanese

- The phoneme inventory grew from 76 to 84 symbols for Japanese (pyopenjtalk-plus based G2P; see `src/tiny_voca/text.py`).
- `src/tiny_voca/align.py` implements the prj-beatrice phoneme aligner used to annotate the Japanese data.
- `scripts/generate_audio_ja.py` generates the Japanese teacher audio.

## Quickstart (inference)

Download the checkpoint `tiny-voca-ja.pt` from the [model page](https://huggingface.co/gachimoto/tiny-voca-ja), install this package, and run:

```bash
pip install .
python -m tiny_voca.infer --checkpoint tiny-voca-ja.pt --text "こんにちは、世界。" --voice F2 --output out.wav
```

`--voice` is `F2` or `M2`. `--text-file <path>` reads the text from a UTF-8 file instead of `--text`. The console script `tiny-voca-speak` accepts the same arguments.

## Cost

About **$20 total (≈ ¥3,190)**, computed from instance prices × run time: teacher generation ≈ $9.18, training ≈ $4.69, everything else ≈ $5.82.

## Limits

In the same honest spirit as the original:

- Numbers and dates are read unreliably — one verification sentence (a date reading) scored CER 29.2%.
- A front-end swap experiment ([experiments/frontend-swap](experiments/frontend-swap/README.md)) traced most of that to the reading layer: with pyopenjtalk-plus readings, the Arabic-numeral date sentence scores CER 0.0% on the same checkpoint, while the kanji-numeral date sentence above stays at 25.0%.
- Pitch accent is not modelled, so the accent sounds foreign.
- A formal 16-sentence evaluation has not been run yet; in a 4-sentence spot check, 3 sentences scored CER 0.0% and the date sentence failed as above.

## Support

It cost about $20 to make. If you like it, you can buy me a coffee: https://ko-fi.com/gachimoto3

## Credits & License

Based on [tiny-voca](https://github.com/n4ze3m/tiny-voca) by n4ze3m (Apache-2.0). Teacher model: Qwen3-TTS (Apache-2.0). Texts: Tatoeba (CC BY 2.0 FR) and Common Voice (CC0). The code in this repository is released under the Apache-2.0 license — see `LICENSE`.
