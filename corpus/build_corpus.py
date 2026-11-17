#!/usr/bin/env python3
"""tiny-voca 日本語版 学習用テキストコーパス v1 構築スクリプト

ソース:
  - Tatoeba 日本語文 (work/jpn_sentences.tsv.bz2)
    ライセンスは文単位: work/jpn_sentences_CC0.tsv.bz2 に含まれる id は CC0 1.0、
    それ以外は Tatoeba の既定ライセンス CC BY 2.0 FR。
  - Common Voice 日本語文 (work/cv_ja_sentences.txt)
    公式 GitHub リポジトリ common-voice/common-voice の server/data/ja/
    sentence-collector.txt（録音プロンプト用に検証済みの文、CC0 1.0）。
    2025-10 からデータセット本体は Mozilla Data Collective 経由に移行したが、
    この文ファイルは追加手続きなしで取得できたため併用する。
  - プロジェクト作成テンプレート文 (generate_templates.py)

出力: corpus/{train,validation,test}.jsonl と corpus/STATS.md
実行: ~/musicgen/venv/bin/python build_corpus.py  (pykakasi をモーラ数概算に使う)
"""
import bz2
import json
import math
import random
import re
import statistics
import sys
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
OUT = HERE / "corpus"
SEED = 20261117

TARGET_TOTAL = 8500
SPLIT_TARGETS = {"train": 8000, "validation": 250, "test": 250}
# ソース別サンプル数（Tatoeba を主ソースにする）
SOURCE_TARGETS = {"tatoeba": 5900, "common_voice": 2000, "template": 600}

sys.path.insert(0, str(HERE))
from generate_templates import generate_templates  # noqa: E402

# ---------- フィルタ ----------
JP_CHAR = re.compile(r"[ぁ-んァ-ヶ一-龯々]")
LATIN = re.compile(r"[A-Za-zＡ-Ｚａ-ｚ]")
DIGIT = re.compile(r"[0-9０-９]")
URL_LIKE = re.compile(r"(https?://|www\.|@)")
# 許容する記号（これ以外の文字を含む文は除外）
ALLOWED_EXTRA = set("、。「」『』・ー…〜:：;；()（）!?！？　 ")
END_PUNCT = re.compile(r"[。！？][」』）)]?$")


def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def acceptable(text):
    """読み上げコーパスとして許容できる文か"""
    if not (5 <= len(text) <= 60):
        return False
    if not JP_CHAR.search(text):
        return False
    if URL_LIKE.search(text):
        return False
    if LATIN.search(text):  # 英字は読みが不安定なので v1 では除外
        return False
    if DIGIT.search(text):  # 数字はテンプレート側でカバーする
        return False
    punct = 0
    for ch in text:
        if JP_CHAR.match(ch):
            continue
        if ch in ALLOWED_EXTRA:
            punct += 1
            continue
        return False  # 想定外の記号を含む
    if punct / len(text) > 0.3:  # 記号過多
        return False
    if not END_PUNCT.search(text):
        return False
    return True


# ---------- モーラ数概算 ----------
_kakasi = None


def mora_estimate(text):
    """pykakasi のひらがな化長からモーラ数を概算（小書きかなは差し引く）"""
    global _kakasi
    if _kakasi is None:
        import pykakasi
        _kakasi = pykakasi.kakasi()
    hira = "".join(seg["hira"] for seg in _kakasi.convert(text))
    small = set("ゃゅょぁぃぅぇぉゎャュョァィゥェォヮ")
    n = 0
    for ch in hira:
        if re.match(r"[ぁ-んァ-ヶ]", ch):
            n += 0 if ch in small else 1
    return max(n, 1)


def word_count_estimate(mora):
    """generate_audio.py の speaking_rate 検査 (0.5〜7 words/s) に合わせた概算。
    日本語の平均発話速度を約7.5モーラ/秒と仮定し word_count = mora/2.5 とすると
    比率は約3.0 words/s 相当になり、検査範囲の中央に収まる。"""
    return max(1, round(mora / 2.5))


# ---------- ソース読み込み ----------
def load_tatoeba():
    cc0_ids = set()
    with bz2.open(WORK / "jpn_sentences_CC0.tsv.bz2", "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts and parts[0].isdigit():
                cc0_ids.add(parts[0])
    rows, raw = [], 0
    with bz2.open(WORK / "jpn_sentences.tsv.bz2", "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            raw += 1
            sid, text = parts[0], normalize(parts[2])
            if not acceptable(text):
                continue
            rows.append({
                "id": f"tat-{sid}",
                "text": text,
                "origin": "tatoeba",
                "source_license": "CC0-1.0" if sid in cc0_ids else "CC-BY-2.0-FR",
                "category": "general",
                "source_id": sid,
                "source_url": f"https://tatoeba.org/en/sentences/show/{sid}",
            })
    return rows, raw


def load_common_voice():
    rows, raw = [], 0
    with open(WORK / "cv_ja_sentences.txt", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            raw += 1
            text = normalize(line)
            if not acceptable(text):
                continue
            rows.append({
                "id": f"cv-{i:05d}",
                "text": text,
                "origin": "common_voice",
                "source_license": "CC0-1.0",
                "category": "general",
                "source_id": f"sentence-collector.txt line {i}",
                "source_url": "https://github.com/common-voice/common-voice/blob/main/server/data/ja/sentence-collector.txt",
            })
    return rows, raw


def load_templates():
    rows = []
    for t in generate_templates():
        text = normalize(t["text"])
        rows.append({
            "id": t["id"],
            "text": text,
            "origin": "template",
            "source_license": "project-own",
            "category": t["category"],
            "source_id": t["id"],
            "source_url": "",
        })
    return rows


def main():
    rng = random.Random(SEED)
    tat, tat_raw = load_tatoeba()
    cv, cv_raw = load_common_voice()
    tpl = load_templates()
    pools = {"tatoeba": tat, "common_voice": cv, "template": tpl}
    pool_sizes = {k: len(v) for k, v in pools.items()}
    print("raw:", {"tatoeba": tat_raw, "common_voice": cv_raw, "template": len(tpl)})
    print("after filter:", pool_sizes)

    # 重複除去（出典横断、CC0 の Common Voice → テンプレート → Tatoeba の優先順）
    seen = {}
    for origin in ("common_voice", "template", "tatoeba"):
        kept = []
        for row in pools[origin]:
            key = row["text"].replace(" ", "")
            if key in seen:
                continue
            seen[key] = row["id"]
            kept.append(row)
        pools[origin] = kept
    print("after dedupe:", {k: len(v) for k, v in pools.items()})

    # サンプリングと split（ソース別に比例配分し、総数を目標に合わせる）
    splits = {s: [] for s in SPLIT_TARGETS}
    sampled_counts = {}
    leftovers = {}
    for origin, target in SOURCE_TARGETS.items():
        pool = pools[origin]
        rng.shuffle(pool)
        chosen = pool[:target]
        leftovers[origin] = pool[target:]
        sampled_counts[origin] = len(chosen)
        # split 配分: validation/test を先に切り、残りを train
        n_val = round(len(chosen) * SPLIT_TARGETS["validation"] / TARGET_TOTAL)
        n_test = round(len(chosen) * SPLIT_TARGETS["test"] / TARGET_TOTAL)
        splits["validation"].extend(chosen[:n_val])
        splits["test"].extend(chosen[n_val:n_val + n_test])
        splits["train"].extend(chosen[n_val + n_test:])
    # 総数の端数を調整: validation/test の過不足は train との間で tatoeba 文を移動し、
    # 最後に train を未使用プールとの増減で合わせる
    for split in ("validation", "test"):
        target = SPLIT_TARGETS[split]
        diff = target - len(splits[split])
        if diff < 0:  # 過剰分を train へ
            move = [r for r in splits[split] if r["origin"] == "tatoeba"][: -diff]
            ids = {r["id"] for r in move}
            splits[split] = [r for r in splits[split] if r["id"] not in ids]
            splits["train"].extend(move)
        elif diff > 0:  # 不足分を train から
            move = [r for r in splits["train"] if r["origin"] == "tatoeba"][:diff]
            ids = {r["id"] for r in move}
            splits["train"] = [r for r in splits["train"] if r["id"] not in ids]
            splits[split].extend(move)
    diff = SPLIT_TARGETS["train"] - len(splits["train"])
    if diff > 0:
        splits["train"].extend(leftovers["tatoeba"][:diff])
        sampled_counts["tatoeba"] += diff
    elif diff < 0:
        tat_train = [r for r in splits["train"] if r["origin"] == "tatoeba"]
        drop = {r["id"] for r in tat_train[: -diff]}
        splits["train"] = [r for r in splits["train"] if r["id"] not in drop]
        sampled_counts["tatoeba"] += diff
    for split, target in SPLIT_TARGETS.items():
        assert len(splits[split]) == target, (split, len(splits[split]), target)

    # split 間の重複がないことを検証
    texts = {}
    for split, rows in splits.items():
        for r in rows:
            key = r["text"].replace(" ", "")
            assert key not in texts, f"split leak: {key}"
            texts[key] = split

    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for split, rows in splits.items():
        rng.shuffle(rows)
        with open(OUT / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                mora = mora_estimate(r["text"])
                rec = dict(r)
                rec["character_count"] = len(r["text"])
                rec["word_count"] = word_count_estimate(mora)
                rec["split"] = split
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                rec["_mora"] = mora  # 統計用（ファイルには書き出さない）
                all_rows.append(rec)
        print(split, len(rows))

    write_stats(pools_raw={"tatoeba": tat_raw, "common_voice": cv_raw},
                pool_sizes=pool_sizes, sampled_counts=sampled_counts,
                splits=splits, all_rows=all_rows, template_generated=len(tpl))
    print("done")


def write_stats(pools_raw, pool_sizes, sampled_counts, splits, all_rows, template_generated):
    def dist(values):
        return {
            "min": min(values), "max": max(values),
            "mean": round(statistics.mean(values), 1),
            "median": statistics.median(values),
        }

    chars = [r["character_count"] for r in all_rows]
    buckets = Counter()
    for c in chars:
        b = (c // 10) * 10
        buckets[f"{b}-{b + 9}"] += 1
    total_mora = sum(r["_mora"] for r in all_rows)
    est_hours_1voice = total_mora / 7.5 / 3600
    lic = Counter(r["source_license"] for r in all_rows)
    origin_counts = Counter(r["origin"] for r in all_rows)
    cat_counts = Counter(r["category"] for r in all_rows)
    split_origin = {s: Counter(r["origin"] for r in rows) for s, rows in splits.items()}
    split_lic = {s: Counter(r["source_license"] for r in rows) for s, rows in splits.items()}

    lines = []
    lines.append("# コーパス v1 統計（STATS）")
    lines.append("")
    lines.append("構築日: 2026-11-17 / 構築スクリプト: phase0/build_corpus.py（seed=20261117）")
    lines.append("")
    lines.append("## ソース別件数")
    lines.append("")
    lines.append("| ソース | 取得件数（生） | フィルタ後 | 採用 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Tatoeba 日本語文 | {pools_raw['tatoeba']:,} | {pool_sizes['tatoeba']:,} | {sampled_counts['tatoeba']:,} |")
    lines.append(f"| Common Voice 日本語文 | {pools_raw['common_voice']:,} | {pool_sizes['common_voice']:,} | {sampled_counts['common_voice']:,} |")
    lines.append(f"| プロジェクト作成テンプレート | {template_generated:,}（生成数） | - | {sampled_counts['template']:,} |")
    lines.append(f"| **合計** | | | **{len(all_rows):,}** |")
    lines.append("")
    lines.append("## split 別件数")
    lines.append("")
    lines.append("| split | 合計 | tatoeba | common_voice | template |")
    lines.append("|---|---|---|---|---|")
    for s in ("train", "validation", "test"):
        c = split_origin[s]
        lines.append(f"| {s} | {sum(c.values()):,} | {c['tatoeba']:,} | {c['common_voice']:,} | {c['template']:,} |")
    lines.append("")
    lines.append("分割は文単位。同一文（空白除去で正規化）が複数 split に入らないことをスクリプト内で assert 検証済み。")
    lines.append("")
    lines.append("## 文字数分布（全文）")
    lines.append("")
    d = dist(chars)
    lines.append(f"- 最小 {d['min']} / 最大 {d['max']} / 平均 {d['mean']} / 中央値 {d['median']}")
    for b in sorted(buckets, key=lambda x: int(x.split('-')[0])):
        lines.append(f"- {b}文字: {buckets[b]:,}文")
    lines.append("")
    lines.append("## ライセンス内訳")
    lines.append("")
    for k, v in lic.most_common():
        lines.append(f"- {k}: {v:,}文")
    lines.append("")
    for s in ("train", "validation", "test"):
        lines.append(f"- {s}: " + ", ".join(f"{k} {v:,}" for k, v in split_lic[s].most_common()))
    lines.append("")
    lines.append("## カテゴリ内訳")
    lines.append("")
    for k, v in cat_counts.most_common():
        lines.append(f"- {k}: {v:,}文")
    lines.append("")
    lines.append("## 総モーラ数と音声量の概算")
    lines.append("")
    lines.append(f"- 総モーラ数（pykakasi のひらがな化長から概算）: 約 {total_mora:,} モーラ")
    lines.append(f"- 推定音声量: 1声あたり約 {est_hours_1voice:.1f} 時間（7.5モーラ/秒と仮定）、2声で約 {est_hours_1voice * 2:.1f} 時間")
    lines.append("- 参考: 英語版は 17,371 クリップで 19.1 時間。本コーパスは文数（8,500文×2声=17,000クリップ）は同規模だが、平均文字数 19.9 と短めのため総時間は約14時間と少なめになる見込み。学習量が不足する場合は、v2 で長めの文の比率を上げるか文数を増やす。")
    lines.append("")
    lines.append("## 出典とライセンスの判定方法")
    lines.append("")
    lines.append("- Tatoeba: https://downloads.tatoeba.org/exports/per_language/jpn/jpn_sentences.tsv.bz2（2026-11-17 取得、248,989文）。文単位ライセンスは同梱の jpn_sentences_CC0.tsv.bz2 に含まれる id を CC0-1.0 とし、それ以外は Tatoeba の既定ライセンス CC BY 2.0 FR とした。なお日本語の CC0 指定文は2文のみで、いずれも英字を含むためフィルタで除外され、採用文に CC0 の Tatoeba 文は含まれていない。")
    lines.append("- Common Voice: 公式 GitHub リポジトリ common-voice/common-voice の server/data/ja/sentence-collector.txt（2026-11-17 取得、9,468文）。録音プロンプトとして検証済みの文の一覧で、Common Voice の文は CC0。データセット本体（音声）は 2025-10 から Mozilla Data Collective 経由（同意手続きあり）に移行したが、文ファイル自体は追加手続きなしで取得できたため併用した。音声は一切ダウンロードしていない。")
    lines.append("- テンプレート: プロジェクト作成（generate_templates.py でプログラム生成）。ライセンスは project-own。")
    lines.append("")
    lines.append("## フィルタ条件（Tatoeba / Common Voice 共通）")
    lines.append("")
    lines.append("- NFKC 正規化後、5〜60文字")
    lines.append("- ひらがな/カタカナ/漢字を少なくとも1文字含む")
    lines.append("- URL・@ を含む文、英字を含む文、数字を含む文を除外（数字の読みはテンプレート文でカバー）")
    lines.append("- 日本語文字と許容句読点（、。！？「」『』・ー…等）以外の記号を含む文を除外、記号比率30%超を除外")
    lines.append("- 文末が 。！？（閉じ括弧可）で終わる文のみ")
    lines.append("- 正規化テキストで重複除去（出典横断、Common Voice を優先）")
    lines.append("")
    lines.append("## フィールド")
    lines.append("")
    lines.append("各行: id, text, origin, source_license, category, source_id, source_url, character_count, word_count, split。")
    lines.append("word_count は generate_audio.py の speaking_rate 検査向けの概算（モーラ数/2.5）。character_count は文字数。")
    (OUT / "STATS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
