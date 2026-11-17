# RunPod 実行ランブック（tiny-voca 日本語版 / GPUセッション当日手順）

作成: 2026-11-17 / 対象計画: ~/tinyvoca/japanese/PLAN.md
前提: Phase 0 完了（コーパス v1 = phase0/corpus/{train,validation,test}.jsonl、計8,500文）。
予算上限: **$30**。これを超えそうになったら作業を止めて藤本さんに相談する。

凡例: コマンドは原則そのまま実行できるもの。原作者リポジトリに存在しない手順や
未検証の API は **** と明記した。当日、実行前にその場で公式ドキュメントと突き合わせること。

---

## 0. 事前準備（Pod 起動前）

- [] RunPod アカウントに $30 以上チャージされていること（藤本さん側の作業）
- [] コーパス3ファイル（合計 約2.9MB）を Pod へ運ぶ準備
- 配置先: Pod 上の `/workspace/tiny-voca-ja/corpus/`
- 運び方はいずれか: RunPod の Web UI ファイルアップロード / `scp` / Pod 上で `git clone` した作業リポジトリへの転送
- [] Hugging Face の公開モデルだけを使うので HF トークンは不要
（Qwen3-TTS 2モデル、アライナー、Whisper small はいずれも公開・ライセンス同意不要のはず。ダウンロード時にゲート表示が出たら止まって確認）

## 1. Pod の起動（選択項目）

| 項目 | 選択 | 備考 |
|---|---|---|
| Cloud | **Community Cloud を優先** | RTX 4090 $0.34/時 目安（2026-11-17 調査）。在庫がなければ Secure Cloud（$0.69/時 目安）に切替 |
| GPU | RTX 4090 24GB ×1 | 下記 VRAM 内訳のとおり 24GB で足りる |
| コンテナイメージ | RunPod 公式 PyTorch テンプレート | 正確なタグは起動時に選ぶ。Python 3.10+ / CUDA 12.x 系であること |
| Volume Disk | **80GB**、マウント先 `/workspace` | 根拠は下記。作業データは必ず Volume 側に置く（コンテナディスクは Pod 終了で消える） |
| Container Disk | 30GB 目安 | イメージ展開用 |
| その他 | SSH / Jupyter のどちらかで入れること | 長時間処理は `nohup` + ログファイルで回す |

### ディスク 80GB の根拠

| 内訳 | 概算 |
|---|---|
| Qwen3-TTS-12Hz-1.7B-Base（教師） | 約4GB（1.7B パラメータ bf16） |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign（声設計） | 約4GB |
| wav2vec2-lv-60-espeak-cv-ft（アライナー） | 約1.3GB（large 317M パラメータ fp32） |
| Whisper small（選別・評価用） | 約0.5GB |
| 生成 FLAC（約14時間分、24kHz/16bit） | 約2GB |
| prepare の特徴量（wave float32 + mel + pitch、約14時間分） | 約6〜8GB |
| 学習チェックポイント（step-*.pt を 10,000 歩ごとに保持 + last.pt） | 約4GB |
| pip キャッシュ・中間ファイル・余裕 | 約10GB |
| **小計** | **約32〜34GB** → 失敗時の再生成・別 run の共存を見て 80GB |

### VRAM 24GB の根拠

- 音声生成: Qwen3-TTS 1.7B bf16 推論で約5〜6GB（バッチ8）
- アノテーション: wav2vec2-large を half で載せて約2〜3GB
- 学習: tiny-voca 本体 4.5M + GAN 判別器 2.7M と小さい。バッチはフレーム数指定（--batch-frames 9000）で数GB規模
- 同時に載せない前提なら 24GB で十分

## 2. 環境構築（Pod 上、目安 30〜45分）

```bash
# システム依存（text.py が espeak-ng バイナリを必須とする）
apt-get update && apt-get install -y espeak-ng ffmpeg git
espeak-ng --version # バージョンを記録する（フロントエンドの identity に入る）

# 作業場所（Volume 上）
mkdir -p /workspace/tiny-voca-ja && cd /workspace/tiny-voca-ja

# 原作者リポジトリ（README の手順）
git clone https://github.com/n4ze3m/tiny-voca.git
cd tiny-voca
pip install -e ".[prepare,annotate,evaluate]"
# prepare = librosa / pyarrow / huggingface-hub
# annotate = transformers / torchaudio / scipy
# evaluate = faster-whisper / jiwer / librosa / scipy

# 音声生成スクリプト専用の依存（pyproject の extras には含まれない）
pip install qwen-tts
python3 -c "from qwen_tts import Qwen3TTSModel; print('qwen-tts OK')" # import 名と PyPI パッケージ名

# モデルの事前取得
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base
hf download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
hf download facebook/wav2vec2-lv-60-espeak-cv-ft
# ↑ アライナーは align.py の PhoneRecognizer が local_files_only=True で読むため、
# annotate の前に必ず HF キャッシュへ入れておくこと（実行時ダウンロードはされない）
python3 -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')" # Whisper small（多言語版）の事前取得
```

コーパスを `/workspace/tiny-voca-ja/corpus/` に配置し、生成スクリプト用に
全 split を1ファイルへ結合する（各行に split フィールドが入っているのでそのまま使える）:

```bash
cd /workspace/tiny-voca-ja
cat corpus/train.jsonl corpus/validation.jsonl corpus/test.jsonl > scripts_ja.jsonl
wc -l scripts_ja.jsonl # 8500 になること
```

## 3. Phase 0.5 スパイク（目安 30分）— フロントエンド決定

目的: eSpeak NG 日本語が出す音素と、(a) tiny-voca の76記号、(b) アライナーのラベル集合の一致確認。
ここで案A（eSpeak 日本語）か案B（かな変換器）かを決める。

```bash
cd /workspace/tiny-voca-ja
# train から 500 文を抽出
python3 - <<'EOF'
import json
rows = [json.loads(l) for l in open('corpus/train.jsonl', encoding='utf-8')][:500]
open('spike500.txt', 'w', encoding='utf-8').write('\n'.join(r['text'] for r in rows) + '\n')
EOF

# 音素化（src/tiny_voca/text.py の espeak 呼び出しと同一形式。声だけ ja に差し替え）
espeak-ng -q --ipa --sep=_ -v ja --stdin < spike500.txt > spike500.ipa

# 出現音素の一覧と頻度
tr ' ' '\n' < spike500.ipa | tr '_' '\n' | sed 's/[ˈˌ]//g' | grep -v '^$' | sort | uniq -c | sort -rn > spike500.phones
cat spike500.phones

# 76記号・アライナーラベルとの突き合わせ
python3 - <<'EOF'
import json, sys
sys.path.insert(0, 'tiny-voca/src')
from tiny_voca.text import SYMBOLS, FALLBACK
emitted = set()
for line in open('spike500.ipa', encoding='utf-8'):
for word in line.split():
emitted.update(u.replace('ˈ', '').replace('ˌ', '') for u in word.split('_') if u)
emitted.discard('')
vocab = json.load(open('aligner_vocab.json', encoding='utf-8')) # Phase 0 で取得済みの控え、または hf download した snapshot の vocab.json
print('espeak-ja 出現音素数:', len(emitted))
print('76記号にも FALLBACK にもない音素:', sorted(emitted - set(SYMBOLS) - set(FALLBACK)))
print('アライナーラベルにない音素:', sorted(emitted - set(vocab)))
EOF
```

判定基準:
- 「76記号+FALLBACK でカバーでき、アライナーラベルにもある」音素が大勢なら **案A** で進む。
その場合のコード改修: `src/tiny_voca/text.py` の `ESPEAK_VOICE = "en-us"` を `"ja"` に変更し、
日本語固有音素（ɯ、ɸ、tɕ、dʑ、ɕ、ʑ、ɴ 等）は SYMBOLS 末尾への append + FALLBACK 追加で対応する。
- カバーできない音素が多い場合は **案B**（試作v0のかな変換器をアノテーション側フロントエンドにする）。
ただし案Bではアライナーの `label()` が PHONE_SET の各記号にラベルを要求する点に注意
（align.py の PhoneRecognizer は、記号が vocab にも ALIGNER_LABEL_FALLBACK にもないと例外を投げる）。

**Phase 0 での事前確認（2026-11-17、ローカルで実施済み）**:
アライナーの vocab.json（392ラベル）を取得して照合したところ、日本語 eSpeak 音素の
ɯ / ɸ / tɕ / dʑ / ɕ / ʑ / ɴ と長母音 aː / iː / eː / oː はすべてラベルに存在した（ɯː のみ無し → ɯ への写像で対処）。
つまり案Aの見通しは明るい。残る確認は「espeak-ng -v ja が実際にどの音素を出すか」だけで、上記のスパイクで確定する。

## 4. Phase 1 声の設計（目安 30分）

リポジトリに VoiceDesign 用のスクリプトは無いため、Qwen3-TTS 公式の使い方に沿って小さなスクリプトを書く。
API 名は公式 README で当日確認すること。

```python
# make_voices.py（想定スニペット: Qwen/Qwen3-TTS 公式リポジトリの VoiceDesign 例に合わせる）
from qwen_tts import Qwen3TTSModel
model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
device_map="cuda:0", dtype="bfloat16", attn_implementation="sdpa")
# 女性: 「落ち着いた大人の声」と明示する（英語版 F2 が想定より若く聞こえた反省）
# 男性: 落ち着いた低めの声
# 各声で参照文（30〜60文字程度の自然な文）を読み上げ、voices/F2.wav, voices/M2.wav に保存
```

成果物の形式（scripts/generate_audio.py の読み方から確定している事項）:
- `--references` ディレクトリに `F2.wav` と `M2.wav` を置く
- `--voices` の voices.json は `{"F2": {"text": "<F2.wav の読み上げ文>"}, "M2": {"text": "<M2.wav の読み上げ文>"}}`
という形（generate_audio.py が `references[v]["text"]` を参照音声のテキストとして使う）

## 5. Phase 2 教師音声の生成と選別（目安 10〜16時間・無人運転）

### 5-1. generate_audio.py の日本語化（前提改修）

`scripts/generate_audio.py` は英語前提でハードコードされている。コピーして以下の4点を直す:

| 箇所（実在コード） | 英語版 | 日本語版 |
|---|---|---|
| `generate_voice_clone(..., language=["English"] * len(batch))` | English | `"Japanese"` |
| `WhisperModel("small.en",...)` | 英語専用 small.en | 多言語版 `"small"` |
| `EnglishTextNormalizer` + `jiwer.wer`（WER 15% で棄却） | 単語誤り率 | かな正規化した文同士の **CER（文字誤り率）15%** に変更（PLAN Phase 2 の決定） |
| speaking_rate 検査 `0.5 <= word_count/duration <= 7` | 英語の語数前提 | コーパス側で word_count をモーラ数/2.5 の概算にしてあるので、そのまま使える想定スモークで棄却率を見て判断 |

再開可能性: 生成結果は `<root>/state.sqlite` に記録され、同じコマンドの再実行で続きから処理される。
停止は `<root>/STOP` ファイルを作るか、`--deadline-file`（`{"deadline": <UNIX時刻>}`）で自動停止できる（どちらも実在の仕様）。

### 5-2. 実行

```bash
cd /workspace/tiny-voca-ja/tiny-voca
mkdir -p../voices../gen

# スモーク（20文×2声だけ）
python3 scripts/generate_audio.py \
--scripts../scripts_ja.jsonl --root../gen \
--references../voices --voices../voices/voices.json \
--limit 20
cat../gen/progress.json # accepted が出ていること、review の理由が想定内であること

# 予算デッドライン（例: ここから14時間後）
python3 -c "import json,time; json.dump({'deadline': time.time()+14*3600}, open('../deadline.json','w'))"

# 本番（無人運転）
nohup python3 scripts/generate_audio.py \
--scripts../scripts_ja.jsonl --root../gen \
--references../voices --voices../voices/voices.json \
--deadline-file../deadline.json \
>../gen.log 2>&1 &
tail -f../gen.log # progress.json が定期的に更新される
```

完了条件: `../gen/progress.json` の stage が `complete`（または deadline で `paused` → 再実行で継続）、
`../gen/manifest.jsonl`（accepted のみ）と `../gen/review.jsonl` が出力される。
review 率が 30% を超えて続く場合は止めて相談する（読み・選別の前提が崩れている可能性）。

## 6. Phase 3 前処理とアノテーション（目安 1〜2時間）

```bash
cd /workspace/tiny-voca-ja/tiny-voca
export TINY_VOCA_FRONTEND=espeak # 案A の場合（text.py の既定は espeak。案B の場合は実装したフロントエンド名）

# prepare は行の split フィールドと --split が一致しないとエラーになるため、split ごとに分ける
python3 - <<'EOF'
import json
rows = [json.loads(l) for l in open('../gen/manifest.jsonl', encoding='utf-8')]
for split in ('train', 'validation', 'test'):
with open(f'../gen/manifest-{split}.jsonl', 'w', encoding='utf-8') as f:
for r in rows:
if r['split'] == split:
f.write(json.dumps(r, ensure_ascii=False) + '\n')
EOF

for split in train validation test; do
python -m tiny_voca.prepare --inputs../gen/manifest-$split.jsonl --split $split --output features
done

python -m tiny_voca.annotate --features features --output annotations --dataset-revision ja-v1 --limit 50
# ↑ まず --limit 50 でスモーク（棄却理由と QC を確認）。問題なければ外して本番（--resume で継続可）
python -m tiny_voca.annotate --features features --output annotations --dataset-revision ja-v1 --resume
```

確認点: annotations の棄却率。英語版の train 棄却は 40/8,226 件程度。
日本語で桁違いに高い場合は Phase 0.5 のフロントエンド判断に戻る。

## 7. Phase 4 学習（目安 8〜10時間・無人運転）

README の公開コマンドが基本形。ただし train.py には1回の起動あたりの時間上限
`--max-seconds`（既定 3600 秒）があるため、ループで `--resume` を繰り返すか、上限を大きくして回す。

```bash
cd /workspace/tiny-voca-ja/tiny-voca
python3 -c "import json,time; json.dump({'deadline': time.time()+11*3600}, open('../deadline-train.json','w'))"

nohup python -m tiny_voca.train \
--train annotations/train.jsonl --validation annotations/validation.jsonl \
--output runs/ja-v1 --steps 104000 --gan-start 8000 --gan-ratio 0.7 --teacher-final 0 \
--max-seconds 39600 --deadline-file../deadline-train.json \
>../train.log 2>&1 &
```

- チェックポイントは `runs/ja-v1/last.pt`（再開用）と `runs/ja-v1/inference.pt`（推論用、保存のたびに自動で書き出される）、
10,000 歩ごとの `step-*.pt` が残る
- 途中で Pod を止めた場合は、同じ出力先に `--resume runs/ja-v1/last.pt` を付けて再実行する（レシピ・データが同じであることが条件）
- loss が発散する・NaN が続く場合は止めて相談する

## 8. Phase 5 評価と仕上げ（目安 30分 + ローカル確認）

```bash
cd /workspace/tiny-voca-ja/tiny-voca
# 評価（--tag 必須。ASR は日本語なので多言語版 small を指定）
python -m tiny_voca.evaluate --checkpoint runs/ja-v1/inference.pt \
--annotations annotations --output eval/ja-v1 --tag ja-v1 --asr-model small
# 注意: evaluate.py の NOVEL_SENTENCES（未知文評価）は英語文がハードコードされている。
# 日本語の未知文評価はこの16文を日本語に差し替えてから行う。差し替え前は未知文の数値を使わないこと。

# ONNX 書き出し（README の手順）
python -m tiny_voca.export --checkpoint runs/ja-v1/inference.pt --output../tiny-voca-ja.onnx
```

成果物の回収（Pod を消す前に必ずローカルへ）:
- `runs/ja-v1/inference.pt`（推論用チェックポイント）
- `tiny-voca-ja.onnx`
- `eval/ja-v1/`（評価結果一式）
- `gen/manifest.jsonl` / `gen/review.jsonl` / `gen/progress.json`（データセットの記録）
- `annotations/` のサマリ（annotations.json）と各 split の jsonl
- `voices/`（F2.wav / M2.wav / voices.json）— 再生成の元になるので必須
- ログ（gen.log / train.log）

回収後、ローカル（このマシン）で CPU 推論を確認して完了とする
（試作v0と同じ文での聞き比べは記事用に別途）。

## 9. コスト監視と停止条件

- **予算上限 $30**。RunPod の Billing / 使用量画面で累積を確認するタイミング:
環境構築後、Phase 2 のスモーク後、Phase 2 完了時、Phase 4 の折り返し（約52,000 步）、Phase 5 前
- フェーズ別コスト目安（Community $0.34/時換算）: 構築+0.5+声設計 約$1 / 生成 約$5 / アノテーション 約$1 / 学習 約$3 / 評価 約$0.5。
Secure（$0.69/時）だと約2倍。どちらでも見積もり総額 $17〜32 の範囲
- 停止条件（いずれかで即停止して相談）:
- 累積コストが **$25** に達した（残り $5 を予備として残す）
- Phase 2 の review 率が 30% 超で継続する
- 学習の loss 発散・NaN 継続
- 想定外の有料オプションやライセンス同意画面が出た
- **課金の止め方**: RunPod は Pod が「停止中」でも Volume のストレージ課金が続く。
作業をしない時間は Pod を Stop、すべての成果物を回収し終えたら **Terminate（削除）** して初めて課金が止まる。
「止めたつもりで放置」が一番の無駄。セッション終了時は必ず Terminate まで行うこと

## 10. 当日チェックリスト（時系列）

1. [] Pod 起動（Community / RTX 4090 / Volume 80GB / PyTorch テンプレート）
2. [] 環境構築（§2）— espeak-ng のバージョンを記録
3. [] コーパス配置と scripts_ja.jsonl 結合（8,500行）
4. [] Phase 0.5 スパイク → 案A/案B を決定し、必要なら text.py を改修
5. [] Phase 1 声設計 → voices/ 一式（女性は「落ち着いた大人の声」）
6. [] Phase 2 スモーク（--limit 20）→ review 率と音を確認 → 本番生成（deadline 設定）
7. [] Phase 3 prepare / annotate（--limit 50 スモーク → 本番）→ 棄却率を確認
8. [] Phase 4 学習（deadline 設定、--resume の回し方を確認）
9. [] Phase 5 評価（NOVEL_SENTENCES 差し替え後）→ ONNX 書き出し
10. [] 成果物をローカルへ回収 → **Pod を Terminate**
11. [] 累積コストを記録して報告
