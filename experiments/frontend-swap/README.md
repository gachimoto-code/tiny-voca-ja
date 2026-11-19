# Front-end swap experiment: pyopenjtalk-plus readings

The stock Japanese front end (`tiny_voca.text.phonemize`) builds readings with
pykakasi and phonemes with eSpeak NG. That reading layer makes real mistakes —
most visibly it reads 「今日は」 as the greeting 「こんにちわ」 — and the model
faithfully learns and reproduces those readings. This experiment asks how much
of the model's date/number weakness is the *reading layer* rather than the
acoustic model, by swapping only the inference-time front end.

## Method

- `oj_frontend.py` — phonemizes text with pyopenjtalk-plus (`g2p_prosody`,
  OpenJTalk dictionary + accent phrases) and maps its phonemes onto the
  checkpoint's eSpeak-style token inventory, so the trained model consumes
  them unchanged. Accent-phrase boundaries become `<sp>`; stress is set on the
  first vowel of each accent phrase (an approximation of the eSpeak word
  stress). Unknown phonemes: zero on all 7 evaluation sentences.
- `synth_oj.py` — re-synthesizes the 7 evaluation sentences (the 3 tts-bench
  Japanese prompts + the 4 Phase-5 spot-check sentences) with the same
  checkpoint (`runs/ja-v1/inference.pt`), same voices and seed.
- `score_cer.py` — transcribes both sets with faster-whisper large-v3
  (int8, CPU) and computes kana-normalized CER, the same method as the
  baseline scores. It also reports a numeral-normalized CER
  (`numeral_norm.py`): ASR systems write dates as digits ("11月19日")
  while the reference may use kanji numerals ("十一月十九日"), and the kana
  scorer drops digits — so the same sounds score as errors on one side
  only. Normalizing digit expressions to readings on both sides removes
  that artifact, which turned out to dominate the kanji-numeral sentence.

Note: pyopenjtalk-plus was *not* the source of the training tokens — during
training-data preparation it was used on the alignment side (the beatrice
aligner's label space), while the token sequences themselves came from the
pykakasi + eSpeak front end. This swap is therefore deliberately a
train/inference mismatch test.

## Results (kana CER, stock front end → pyopenjtalk-plus)

Raw CER first, then the numeral-normalized CER in parentheses (see
`score_cer.py` above for why both are reported):

| Sentence | Stock | OpenJTalk |
|---|---|---|
| p7 ブラウザを開いて、メールを読んでください。 | 0.0% (0.0%) | 0.0% (0.0%) |
| p8 テストが通ったら…プルリクエストを開きます。 | 0.0% (0.0%) | 0.0% (0.0%) |
| p9 今日は2026年11月19日木曜日です。会議は午後3時30分からで、参加者は12人です。 | 2.6% (2.8%) | **0.0% (0.0%)** |
| s01 こんにちは、これは日本語の音声合成のテストです。 | 0.0% (0.0%) | 0.0% (0.0%) |
| s02 はじめまして。私は日本語で話す、小さな音声合成モデルです。 | 0.0% (0.0%) | 0.0% (0.0%) |
| s03 今日は十一月十九日、木曜日です。東京の天気は、晴れのち曇りでしょう。 | 29.2% (0.0%) | **25.0% (2.1%)** |
| s04 人工知能が、日本語の文章を読み上げます… | 0.0% (0.0%) | 0.0% (0.0%) |

Bench mean (p7–p9): 0.87% → 0.0%. Spot-check mean (s01–s04): 7.3% → 6.25%
raw; numeral-normalized means over all 7 sentences: stock 0.4%, swapped 0.3%.

## Reading of the results

- p9 reaching an exact match settles the main question: for ordinary
  sentences, the date/time/number errors were caused by the reading layer,
  not by the acoustic model. The same checkpoint renders the correct phoneme
  sequence correctly.
- s03 needs a correction to how it was first reported (as a "date reading
  failure" at CER 29.2%): almost all of that number is the digit-notation
  artifact. Normalized, the stock front end scores 0.0% on s03 — the sounds
  were right — and the swapped front end scores 2.1% (「東京」 misheard as
  「きょう」, plus a stray particle), the one place where the train/inference
  front-end mismatch still costs something. A third system trained
  consistently on an OpenJTalk front end (ayousanz's tiny-voca-ja
  tsukuyomi-chan, P1) scores raw 29.2% → normalized 0.0% on the same
  sentence, confirming the artifact is in the metric, not the models.
- Caveats: only 7 sentences; the stress marks are an accent-phrase
  approximation, not the training front end's word stress; the kana
  conversion reads the reference's 「今日は」 as 「こんにちは」 too, so the
  greeting error is invisible to this metric and was verified from the raw
  transcription text; the large-v3 scoring for the swapped set was run on a
  local CPU VM with the same model and quantization as the baseline run
  (the baseline was scored on a separate Windows PC).

## Run it

```bash
pip install pyopenjtalk-plus soundfile faster-whisper pykakasi scipy
python experiments/frontend-swap/synth_oj.py runs/ja-v1/inference.pt
python experiments/frontend-swap/score_cer.py
```
