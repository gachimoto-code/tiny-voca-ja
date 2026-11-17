#!/usr/bin/env python3
"""プロジェクト作成テンプレート文の生成（英語版 batch 2 のカテゴリ設計に倣う）

カテゴリ: numbers / dates / money / time / names / questions
数字表記の制約: 試作フロントエンドの expand_numbers が読める範囲
= 整数のみ（カンマ区切り可）。小数・分数・パーセント記号・英字混じりは使わない。
"""
import datetime
import random

WEEKDAYS = ["月曜日", "火曜日", "水曜日", "木曜日", "金曜日", "土曜日", "日曜日"]


def generate_templates():
    rng = random.Random(20261117)
    out = []
    counters = {}

    def add(category, text):
        counters[category] = counters.get(category, 0) + 1
        out.append({"id": f"tpl-{category}-{counters[category]:03d}",
                    "text": text, "category": f"template_{category}"})

    def fill(category, target, maker):
        seen = set()
        attempts = 0
        while len(seen) < target and attempts < target * 60:
            attempts += 1
            text = maker()
            if text in seen or not (5 <= len(text) <= 60):
                continue
            seen.add(text)
            add(category, text)

    # ---------- numbers ----------
    item_counter = [
        ("りんご", "個"), ("みかん", "個"), ("卵", "個"), ("トマト", "個"),
        ("本", "冊"), ("ノート", "冊"), ("雑誌", "冊"),
        ("鉛筆", "本"), ("傘", "本"), ("花", "本"), ("ビール", "本"),
        ("切手", "枚"), ("はがき", "枚"), ("チケット", "枚"), ("写真", "枚"),
        ("牛乳", "本"), ("パン", "個"), ("ケーキ", "個"), ("椅子", "脚"),
        ("車", "台"), ("パソコン", "台"), ("自転車", "台"),
        ("猫", "匹"), ("犬", "匹"), ("鳥", "羽"), ("コーヒー", "杯"),
    ]
    small = list(range(1, 31))
    mid = [32, 40, 45, 50, 55, 60, 70, 75, 80, 90, 99]
    big = [100, 120, 150, 200, 250, 300, 400, 500, 640, 800, 1000, 1200, 2000, 3000, 5000]

    def make_number():
        kind = rng.randrange(6)
        if kind == 0:
            item, counter = rng.choice(item_counter)
            n = rng.choice(small + mid)
            return rng.choice([
                f"{item}を{n}{counter}買いました。",
                f"{item}が{n}{counter}あります。",
                f"{item}を{n}{counter}ください。",
            ])
        if kind == 1:
            n = rng.choice(small + mid + big)
            return rng.choice([
                f"参加者は{n}人でした。",
                f"全部で{n}人集まりました。",
                f"生徒は{n}人います。",
            ])
        if kind == 2:
            n = rng.choice([100, 120, 150, 180, 200, 240, 256, 300, 320, 400, 480, 500, 512, 600, 800, 1000])
            return rng.choice([
                f"この本は{n}ページあります。",
                f"レポートは{n}ページになりました。",
            ])
        if kind == 3:
            n = rng.choice(list(range(1, 41)))
            return rng.choice([
                f"部屋は{n}階にあります。",
                f"{n}階まで階段で上がりました。",
                f"エレベーターで{n}階まで行きます。",
            ])
        if kind == 4:
            n = rng.choice(list(range(2, 90)))
            return rng.choice([
                f"彼は{n}歳になりました。",
                f"彼女は{n}歳です。",
                f"{n}歳の男の子です。",
            ])
        n = rng.choice(list(range(1, 31)) + [42, 50, 100])
        return rng.choice([
            f"今日は{n}キロ走りました。",
            f"毎日{n}回練習しています。",
            f"あと{n}個で終わります。",
        ])

    fill("numbers", 120, make_number)

    # ---------- dates ----------
    def rand_date():
        y = rng.choice([2024, 2025, 2026, 2027, 2028])
        m = rng.randrange(1, 13)
        d = rng.randrange(1, 29)
        return y, m, d

    def make_date():
        y, m, d = rand_date()
        kind = rng.randrange(6)
        if kind == 0:
            return f"{y}年{m}月{d}日に生まれました。"
        if kind == 1:
            return f"今日は{y}年{m}月{d}日です。"
        if kind == 2:
            wd = WEEKDAYS[datetime.date(y, m, d).weekday()]
            return f"{y}年{m}月{d}日は{wd}です。"
        if kind == 3:
            return rng.choice([
                f"締め切りは{m}月{d}日です。",
                f"会議は{m}月{d}日に開かれます。",
                f"出発は{m}月{d}日の予定です。",
            ])
        if kind == 4:
            return rng.choice([
                f"{y}年{m}月{d}日から始まります。",
                f"{y}年{m}月{d}日まで有効です。",
                f"{m}月{d}日までに提出してください。",
            ])
        return rng.choice([
            f"結婚記念日は{m}月{d}日です。",
            f"次の休みは{m}月{d}日です。",
            f"{m}月{d}日に引っ越します。",
        ])

    fill("dates", 110, make_date)

    # ---------- money ----------
    amounts = ([100, 120, 150, 180, 200, 250, 300, 350, 400, 450, 500, 550, 600,
                700, 750, 800, 900, 980]
               + [1000, 1200, 1500, 1800, 2000, 2500, 3000, 3500, 4000, 4500,
                  5000, 6000, 7000, 8000, 9000, 9800]
               + [10000, 12000, 12800, 15000, 18000, 20000, 25000, 30000, 35000,
                  42000, 50000, 58000, 64000, 75000, 80000, 98000, 100000, 120000,
                  150000, 200000, 250000, 300000, 500000, 1000000])

    def fmt_amount(n):
        if n >= 10000 and rng.random() < 0.5:
            return f"{n:,}"
        return str(n)

    def make_money():
        a = fmt_amount(rng.choice(amounts))
        return rng.choice([
            f"これは{a}円です。",
            f"合計は{a}円になりました。",
            f"{a}円の本を買いました。",
            f"家賃は月{a}円です。",
            f"{a}円を払いました。",
            f"お釣りは{a}円です。",
            f"昼食は{a}円でした。",
            f"この時計は{a}円です。",
            f"{a}円貸してください。",
            f"予算は{a}円までです。",
        ])

    fill("money", 110, make_money)

    # ---------- time ----------
    def make_time():
        h = rng.randrange(0, 24)
        minute = rng.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
        kind = rng.randrange(6)
        if kind == 0:
            return f"今は{h}時{minute}分です。" if minute else f"今は{h}時です。"
        if kind == 1:
            return f"{h}時{minute}分に起きました。" if minute else f"{h}時に起きました。"
        if kind == 2:
            return rng.choice([
                f"会議は{h}時からです。",
                f"店は{h}時に開きます。",
                f"授業は{h}時に終わります。",
            ])
        if kind == 3:
            return f"{h}時{minute}分の電車に乗ります。" if minute else f"{h}時の電車に乗ります。"
        if kind == 4:
            hh = rng.randrange(1, 13)
            ap = rng.choice(["午前", "午後"])
            return f"{ap}{hh}時{minute}分に着きます。" if minute else f"{ap}{hh}時に着きます。"
        n = rng.choice(list(range(1, 13)))
        m2 = rng.choice([10, 15, 20, 30, 40, 45, 50])
        return rng.choice([
            f"{n}時間かかりました。",
            f"ここから{m2}分歩きます。",
            f"あと{m2}分で始まります。",
        ])

    fill("time", 90, make_time)

    # ---------- names ----------
    persons = [
        "田中健太", "佐藤美咲", "鈴木翔太", "高橋葵", "渡辺大輝", "山本陽菜",
        "中村結衣", "小林蓮", "加藤さくら", "吉田悠真", "山田花", "松本颯太",
        "井上凛", "木村朝陽", "林美月", "清水湊", "森樹", "池田澪",
        "橋本航", "阿部心春", "藤本賢志",
    ]
    places = [
        "熊本市", "東京都", "大阪市", "阿蘇山", "熊本城", "通潤橋", "天草市",
        "菊池渓谷", "福岡市", "京都市", "札幌市", "名古屋市", "水前寺公園",
        "横浜市", "神戸市", "広島市", "仙台市", "鹿児島市", "別府市", "長崎市",
    ]
    companies = [
        "株式会社あおぞら", "株式会社ひまわり商事", "株式会社さくら工務店",
        "株式会社みらい電子", "株式会社くまもと食品", "株式会社はやぶさ運送",
    ]

    def make_name():
        kind = rng.randrange(5)
        if kind == 0:
            p = rng.choice(persons)
            return rng.choice([
                f"{p}さんは先生です。",
                f"{p}さんに会いました。",
                f"{p}さんから電話がありました。",
                f"{p}さんは熊本の出身です。",
            ])
        if kind == 1:
            pl = rng.choice(places)
            return rng.choice([
                f"{pl}へ行きました。",
                f"{pl}は有名な場所です。",
                f"{pl}に住んでいます。",
                f"明日{pl}へ出張します。",
            ])
        if kind == 2:
            p, pl = rng.choice(persons), rng.choice(places)
            return f"{p}さんは{pl}に住んでいます。"
        if kind == 3:
            c = rng.choice(companies)
            return rng.choice([
                f"{c}で働いています。",
                f"{c}の製品です。",
            ])
        p = rng.choice(persons)
        return rng.choice([
            f"{p}さん、おはようございます。",
            f"{p}さんの意見を聞きました。",
        ])

    fill("names", 100, make_name)

    # ---------- questions ----------
    def make_question():
        kind = rng.randrange(6)
        if kind == 0:
            pl = rng.choice(places)
            return rng.choice([
                f"{pl}はどこですか。",
                f"{pl}までどのくらいかかりますか。",
                f"{pl}へはどう行きますか。",
            ])
        if kind == 1:
            p = rng.choice(persons)
            return rng.choice([
                f"{p}さんは来ますか。",
                f"{p}さんを知っていますか。",
                f"{p}さんは元気ですか。",
            ])
        if kind == 2:
            h = rng.randrange(1, 24)
            return rng.choice([
                "何時から始まりますか。",
                f"{h}時に始まりますか。",
                "何時に終わりますか。",
            ])
        if kind == 3:
            return rng.choice([
                "これはいくらですか。",
                "全部でいくらになりますか。",
                "カードで払えますか。",
            ])
        if kind == 4:
            return rng.choice([
                "今日は雨ですか。",
                "明日は晴れますか。",
                "窓を開けてもいいですか。",
                "手伝ってもらえますか。",
            ])
        item, _c = rng.choice(item_counter)
        n = rng.choice(small)
        return rng.choice([
            f"{item}はありますか。",
            f"{item}を{n}つください。",
            "これは何ですか。",
        ])

    fill("questions", 90, make_question)
    return out


if __name__ == "__main__":
    rows = generate_templates()
    from collections import Counter
    print(len(rows), Counter(r["category"] for r in rows))
    for r in rows[:5]:
        print(r)
