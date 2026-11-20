# tiny-voca-ja モデルバージョン管理

tiny-voca-ja の学習済みモデルの版管理台帳。チェックポイント本体は `~/workspace/tiny-voca-ja-pod-data/tiny-voca-ja/tiny-voca/runs/` 配下に保管。

## v1（現行・公開中）— runs/ja-v1

- 学習完了: 2026-11-19、総ステップ数 104,000（旧フロントエンド注釈データ 11,561クリップ）
- `inference.pt`（18.2MB）sha256 先頭16桁: `34e6c68cd6d62720`
- 公開先: Hugging Face `gachimoto/tiny-voca-ja`、tiny-voca-ja-web の ONNX モデルとして稼働中
- 評価（16文、faster-whisper large-v3 で採点）:
  - stock 条件: 生CER 6.62% / 数値正規化後 1.61%
  - OJ差し替え条件: 生CER 6.51% / 正規化後 1.56%、UTMOS 3.67

## v2（評価済み・未公開）— runs/ja-v2-oj

- ファインチューン実施: 2026-11-20、v1 から +10,000ステップ（総 114,000）。OJ再注釈データ 15,558クリップ（train 14,652 / validation 453 / test 453）で学習
- 注意点: 学習率の実効値は承認値 1e-4 の約1/40（約2e-6）だった（train.py の減衰式が絶対ステップ基準だったため）。それでも下記の通り改善した
- チェックポイント:
  - 110,000ステップ: `inference-110000.pt`（18.2MB）sha256 先頭16桁: `8da7aea30bcb8e6f`
  - 114,000ステップ: `inference.pt`（18.2MB）sha256 先頭16桁: `bace068fa0c0a529`
- 評価（16文、OJ条件、現行と同条件で再計測）:
  - 110k / 114k ともに 生CER 5.99% / 正規化後 1.04%（現行 1.56% から改善）
  - UTMOS: 110k 3.68 / 114k 3.73（現行 3.67）
  - 改善が耳で分かる例: E02「この電車は、博多行きの特急ですか」— 現行は「特急」を誤読（ASR書き起こし「都急」）、v2 は正しく発音
- 採用候補: 114,000ステップ
- 状態: 展開完了（2026-11-20）。ONNX書き出し済み（検証: 相関1.000000、最大誤差0.0005）、Hugging Face更新済み（commit 0e577969）、tiny-voca-ja-web のモデル差し替え済み（Mac側コミット 8e1f45eb）。LR修正版の再学習は「v2が現行を上回らなければ実行」の条件が未達のため実施しない
- 評価スコア: `~/workspace/tts-bench-ja/results/scores_eval16_v2.json` / `utmos_eval16_v2.json`
- 評価音声: `~/workspace/tiny-voca-ja/frontend-swap/eval16-v2/`（current-oj / v2-110k-oj / v2-114k-oj、各16文）

## 予定

- 第2弾ファインチューン（未定）: 追加教師文172文＋英語ルートB（英語音素）をまとめて実施する計画。データ保持のため RunPod ボリューム x7z4fk7ar0 は残置中
