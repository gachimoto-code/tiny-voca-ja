# ボコーダ実験（メル＋HiFi-GAN）

tiny-voca-ja の音のザラザラは、波形生成部がソース・フィルタ式（数式で固定され、学習されない）であることに由来します。このフォルダは、その波形生成部を「メルスペクトログラム＋ニューラルボコーダ（HiFi-GAN）」に置き換えられるかを検証したパイロットのコードと結果です。

ここで測っているのは「正解メル（本物の音声から計算したメル）→波形」の上限です。tiny-voca 本体にメル予測を組み込む作業は未実施で、この結果は「ボコーダ単体でどこまで行けるか」の目安です。

## 結果（UTMOS、utmos22_strong、テスト24文）

| ラウンド | 条件 | 教師音声 | ボコード後 | 差 |
|---|---|---:|---:|---:|
| 第1ラウンド（2026-11-20） | スクラッチ、90,000ステップ | 4.06 | 3.23 | −0.82 |
| 第2ラウンド（2026-11-21） | 公式HiFi-GAN VCTK_V2生成器で初期化、90,000ステップ | 4.06 | 3.52 | −0.54 |

結論：事前学習済み初期化で改善はしましたが、ザラザラは完全には消えませんでした。

## v3評価サンプルでの実測（2026-11-21）

第2ラウンドの生成器を、tiny-voca-ja v3（oj-en-1、134,000ステップ）の評価サンプル（モデルの素出力）に通した測定です。上の表が「正解メル→波形」の上限測定なのに対し、こちらはモデル自身の出力に対する実測です。UTMOS は utmos22_strong。

| サンプル | 原音 | ボコーダ版 | 差 |
|---|---:|---:|---:|
| 英語1 M2 | 3.25 | 4.28 | +1.04 |
| 英語2 M2 | 3.53 | 3.97 | +0.44 |
| 英語3 M2 | 3.54 | 4.20 | +0.66 |
| 日本語1 M2 | 3.57 | 4.20 | +0.63 |
| 日本語2 M2 | 3.60 | 3.66 | +0.06 |
| 混在1 M2 | 3.64 | 4.25 | +0.61 |
| 混在2 M2 | 4.07 | 3.99 | −0.07 |
| 混在3 M2 | 3.63 | 4.19 | +0.55 |
| 英語1 F2 | 3.85 | 3.16 | −0.69 |

結論：男性 M2 は8本中7本が改善し、平均 +0.50 でした。低い声ほどソース・フィルタのザラつきが目立ち、ボコーダの効きも大きい傾向です。一方、女性 F2 は −0.69 と悪化しました（Web の実出力での先行測定でも F2 は −0.16 で、傾向は一貫しています）。実運用の推奨は、ボコーダは男性の声にだけ適用することです。

## 重み

Hugging Face の `gachimoto/tiny-voca-ja` に置いています（第2ラウンドの生成器、3.8MB）。

- https://huggingface.co/gachimoto/tiny-voca-ja/blob/main/vocoder/gen_final.pt
- 評価の内訳：同じリポジトリの `vocoder/utmos_comparison_r1.json` / `vocoder/utmos_comparison_r2.json`

## Mac で試す

手持ちの音声をボコーダに通して、元の音声と聴き比べられます（＝ボコーダの上限を耳で確認できます）。

前提：24kHz・モノラルの wav。違う形式の音声は ffmpeg で変換してください。

```bash
ffmpeg -i input.wav -ar 24000 -ac 1 test24k.wav
```

環境（Python 3.12 の venv を推奨します。Mac の標準 python3 が新しい版だと一部パッケージの wheel が無いためです）:

```bash
brew install python@3.12
python3.12 -m venv .venv-vocoder
.venv-vocoder/bin/pip install torch soundfile
```

実行：

```bash
# 重み gen_final.pt を Hugging Face からダウンロードして、カレントに置く
echo "/path/to/test24k.wav" > list.txt
.venv-vocoder/bin/python vocoder/vocode.py --ckpt gen_final.pt --list list.txt --out-dir vocoded_out
```

`vocoded_out/` に再合成された wav が出ます。

## 仕様

- メル：24kHz、n_fft 1024、hop 256、80バンド、fmax 12,000Hz（`mel.py`）
- 生成器：HiFi-GAN V2 相当の自前実装（`models.py`）。学習は RunPod RTX 4090 で実施（`vocode.py` は推論用）
- 学習データ：tiny-voca-ja の教師音声（Qwen3-TTS 合成、24kHz）
