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

## v2 fine-tune (2026-11-20)

The model on Hugging Face is now **v2**: the v1 checkpoint fine-tuned for +10,000 steps (114,000 total) on the same corpus, re-annotated end to end with pyopenjtalk-plus labels and beatrice forced alignment (15,558 of 16,321 clips accepted). In the formal 16-sentence evaluation (Whisper large-v3, matched front end), the numeral-normalized CER improved from 1.6% to 1.0% and UTMOS from 3.67 to 3.73. The checkpoint's front-end identity changed to pyopenjtalk-plus (`oj-1`); see [VERSIONS.md](VERSIONS.md) for the version registry, checkpoint hashes and the full numbers.

## Vocoder pilot (2026-11-21)

The stock waveform head is a fixed source-filter, so its graininess cannot be trained away. As a pilot, a mel + HiFi-GAN vocoder was trained on ground-truth mels from the teacher audio to measure the ceiling. UTMOS (utmos22_strong, 24 test clips): teacher 4.06 vs vocoded 3.23 when trained from scratch, and 3.52 when initialised from the official pretrained HiFi-GAN V2 generator (90,000 steps each). The grain is reduced but not removed. Code and Mac instructions: [vocoder/README.md](vocoder/README.md). Weights are on Hugging Face under `vocoder/gen_final.pt`.

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

- Numbers and dates can be misread by the stock front end — in an Arabic-numeral date sentence it read 「今日は」 as the greeting 「こんにちは」 and got the weekday wrong (CER 2.6%). A front-end swap experiment ([experiments/frontend-swap](experiments/frontend-swap/README.md)) fixes that class of error on the same checkpoint (CER 0.0%).
- Correction to an earlier claim: the kanji-numeral date sentence once reported here as "CER 29.2%" was mostly a scoring artifact — the ASR writes dates as digits while the reference is read as kanji numerals, so matching sounds were counted as errors. With numeral notation normalized on both sides, that sentence scores CER 0.0% with the stock front end and 2.1% with the swapped one.
- Pitch accent is not modelled, so the accent sounds foreign.
- The formal 16-sentence evaluation has since been run (2026-11-19): with the stock front end, mean CER was 6.6% raw and 1.6% after numeral normalization; the v2 fine-tune above scores 6.0% raw and 1.0% normalized with its matched front end.

## Support

It cost about $20 to make. If you like it, you can buy me a coffee: https://ko-fi.com/gachimoto3

## Credits & License

Based on [tiny-voca](https://github.com/n4ze3m/tiny-voca) by n4ze3m (Apache-2.0). Teacher model: Qwen3-TTS (Apache-2.0). Texts: Tatoeba (CC BY 2.0 FR) and Common Voice (CC0). The code in this repository is released under the Apache-2.0 license — see `LICENSE`.
