# tiny-voca-ja

[tiny-voca](https://github.com/n4ze3m/tiny-voca)（n4ze3m、Apache-2.0）の日本語版を作るプロジェクトです。

tiny-voca は、人間の発話データを一切使わず Qwen3-TTS の合成音声だけで学習した 4.5M パラメータ・18MB の英語 TTS モデルです。本プロジェクトでは、原作者の公開レシピを日本語に置き換え、日本語データでゼロから学習したモデルを目指します。

## 状態（2026-11-17）

- 試作 v0.1（再学習なし）: 稼働中。日本語フロントエンドが日本語テキストを英語版の音素在庫（76記号）へ近似変換し、英語版の重み（F2/M2）のまま日本語を読ませる実験実装。単語の形は聞き取れるが外国語訛りが強い、という水準
- 本格版: 計画確定、準備中。計画書は [PLAN.md](PLAN.md)
  - Phase 0: 学習用コーパス構築と RunPod ランブック作成（ローカル作業、進行中）
  - Phase 1 以降: RunPod の RTX 4090 で Qwen3-TTS による教師音声生成 → 学習

## 試作 v0.1 の使い方

前提: 上流 tiny-voca のリポジトリと重み、Python 環境（torch、soundfile、pykakasi）。

```bash
# 上流リポジトリを clone し、重みを取得しておく
git clone https://github.com/n4ze3m/tiny-voca.git
# 重みは Hugging Face の n4ze3m/tiny-voca から取得し weights/ に置く

DIZIONE_DATA=<weights>/phonemizer python frontend/tiny_voca_ja.py \
  --text "こんにちは、これは日本語対応のテストです。" \
  --voice F2 --output sample.wav
```

変換の流れ: 数字の読み展開 → pykakasi で漢字をひらがな化（助詞「は→ワ」「へ→エ」の読み替え含む）→ モーラごとに最寄りの IPA 記号へ写像 → 既存モデルで推論。ピッチアクセントは扱いません。

## 構成

- `frontend/tiny_voca_ja.py` — 試作 v0.1 の日本語フロントエンド
- `PLAN.md` — 本格版の計画書（工程、費用見積もり、設計判断、リスク）
- `corpus/` — 学習用テキストコーパス（Phase 0 で追加予定）
- `runpod/` — RunPod 実行ランブック（Phase 0 で追加予定）

## ライセンス

- 本リポジトリのコードと文書: Apache-2.0（`LICENSE`）
- 上流 tiny-voca: Apache-2.0（コード・重み）、データセットは CC BY 4.0
- 学習コーパスの出典とライセンスは `corpus/` 内に記録します（Common Voice 文: CC0、Tatoeba 文: CC BY 2.0 / CC0 など）
