# コーパス v1 統計（STATS）

構築日: 2026-11-17 / 構築スクリプト: phase0/build_corpus.py（seed=20261117）

## ソース別件数

| ソース | 取得件数（生） | フィルタ後 | 採用 |
|---|---|---|---|
| Tatoeba 日本語文 | 248,989 | 214,428 | 5,900 |
| Common Voice 日本語文 | 9,469 | 2,956 | 2,000 |
| プロジェクト作成テンプレート | 620（生成数） | - | 600 |
| **合計** | | | **8,500** |

## split 別件数

| split | 合計 | tatoeba | common_voice | template |
|---|---|---|---|---|
| train | 8,000 | 5,554 | 1,882 | 564 |
| validation | 250 | 173 | 59 | 18 |
| test | 250 | 173 | 59 | 18 |

分割は文単位。同一文（空白除去で正規化）が複数 split に入らないことをスクリプト内で assert 検証済み。

## 文字数分布（全文）

- 最小 5 / 最大 60 / 平均 19.9 / 中央値 17.0
- 0-9文字: 466文
- 10-19文字: 4,673文
- 20-29文字: 2,167文
- 30-39文字: 735文
- 40-49文字: 331文
- 50-59文字: 120文
- 60-69文字: 8文

## ライセンス内訳

- CC-BY-2.0-FR: 5,900文
- CC0-1.0: 2,000文
- project-own: 600文

- train: CC-BY-2.0-FR 5,554, CC0-1.0 1,882, project-own 564
- validation: CC-BY-2.0-FR 173, CC0-1.0 59, project-own 18
- test: CC-BY-2.0-FR 173, CC0-1.0 59, project-own 18

## カテゴリ内訳

- general: 7,900文
- template_numbers: 118文
- template_money: 108文
- template_dates: 104文
- template_names: 95文
- template_questions: 88文
- template_time: 87文

## 総モーラ数と音声量の概算

- 総モーラ数（pykakasi のひらがな化長から概算）: 約 190,131 モーラ
- 推定音声量: 1声あたり約 7.0 時間（7.5モーラ/秒と仮定）、2声で約 14.1 時間
- 参考: 英語版は 17,371 クリップで 19.1 時間。本コーパスは文数（8,500文×2声=17,000クリップ）は同規模だが、平均文字数 19.9 と短めのため総時間は約14時間と少なめになる見込み。学習量が不足する場合は、v2 で長めの文の比率を上げるか文数を増やす。

## 出典とライセンスの判定方法

- Tatoeba: https://downloads.tatoeba.org/exports/per_language/jpn/jpn_sentences.tsv.bz2（2026-11-17 取得、248,989文）。文単位ライセンスは同梱の jpn_sentences_CC0.tsv.bz2 に含まれる id を CC0-1.0 とし、それ以外は Tatoeba の既定ライセンス CC BY 2.0 FR とした。なお日本語の CC0 指定文は2文のみで、いずれも英字を含むためフィルタで除外され、採用文に CC0 の Tatoeba 文は含まれていない。
- Common Voice: 公式 GitHub リポジトリ common-voice/common-voice の server/data/ja/sentence-collector.txt（2026-11-17 取得、9,468文）。録音プロンプトとして検証済みの文の一覧で、Common Voice の文は CC0。データセット本体（音声）は 2025-10 から Mozilla Data Collective 経由（同意手続きあり）に移行したが、文ファイル自体は追加手続きなしで取得できたため併用した。音声は一切ダウンロードしていない。
- テンプレート: プロジェクト作成（generate_templates.py でプログラム生成）。ライセンスは project-own。

## フィルタ条件（Tatoeba / Common Voice 共通）

- NFKC 正規化後、5〜60文字
- ひらがな/カタカナ/漢字を少なくとも1文字含む
- URL・@ を含む文、英字を含む文、数字を含む文を除外（数字の読みはテンプレート文でカバー）
- 日本語文字と許容句読点（、。！？「」『』・ー…等）以外の記号を含む文を除外、記号比率30%超を除外
- 文末が 。！？（閉じ括弧可）で終わる文のみ
- 正規化テキストで重複除去（出典横断、Common Voice を優先）

## フィールド

各行: id, text, origin, source_license, category, source_id, source_url, character_count, word_count, split。
word_count は generate_audio.py の speaking_rate 検査向けの概算（モーラ数/2.5）。character_count は文字数。
