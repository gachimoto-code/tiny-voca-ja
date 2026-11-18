"""Measure where the beatrice recogniser's CTC emissions sit inside a phone.

Same voicing-flip method as ``calibrate_aligner_timing.py`` (which see for the
rationale), run on the beatrice label sequences: for phone transitions that
must flip the pitch tracker's voicing (vowel -> voiceless fricative, voiceless
fricative -> vowel), compare the time of the flip with the START of the next
phone's first emission frame. The suggested constant is
``BEATRICE_ONSET_OFFSET_SECONDS`` in ``tiny_voca/align.py``. Transitions are
taken between consecutive g2p phones with no pau between them, so the pair is
also adjacent in the aligned sequence.
"""

import argparse
import collections
import json
from pathlib import Path
import numpy as np
import torch

from tiny_voca.align import ALIGNER_RATE, ALIGNER_STRIDE, BeatriceRecognizer, token_spans
from tiny_voca.config import Config

VOWELS = {"a", "i", "u", "e", "o", "I", "U"}
FRICATIVES, STOPS = {"f", "s", "sh"}, {"p", "t", "k"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import pyopenjtalk
    c = Config()
    recognizer = BeatriceRecognizer(args.device)
    rows = [json.loads(line) for line in (args.features / (args.split + ".jsonl")).read_text(encoding="utf-8").split("\n") if line.strip()][:args.count]
    offsets, runs = collections.defaultdict(list), collections.Counter()
    for row in rows:
        stored = torch.load(args.features / row["path"], map_location="cpu", weights_only=True)
        voiced = stored["prosody"][:, 1].numpy() > 0.5
        with_pau = pyopenjtalk.g2p(row["text"]).split()
        names = [p for p in with_pau if p not in ("pau", "sil")]
        log_probs = recognizer.log_probs(stored["wave"].numpy(), c.sample_rate)
        spans = token_spans(log_probs, [recognizer.label(s) for s in names], recognizer.blank)
        runs.update(last - first + 1 for first, last in spans)
        flips = [(f, bool(voiced[f - 1])) for f in range(1, len(voiced)) if voiced[f - 1] != voiced[f]]
        # Adjacent phone pairs in the g2p output (a pau between them breaks adjacency).
        index = -1
        previous = None  # (name, align index) of the preceding phone when adjacent
        for phone in with_pau:
            if phone in ("pau", "sil"):
                previous = None
                continue
            index += 1
            if previous is not None:
                a, b = previous[0], phone
                if a in VOWELS and b in FRICATIVES:
                    kind, before = "vowel>fricative", True
                elif a in VOWELS and b in STOPS:
                    kind, before = "vowel>stop", True
                elif a in FRICATIVES and b in VOWELS:
                    kind, before = "fricative>vowel", False
                elif a in STOPS and b in VOWELS:
                    kind, before = "stop>vowel", False
                else:
                    kind = None
                if kind is not None:
                    emission = spans[index][0] * ALIGNER_STRIDE / ALIGNER_RATE
                    times = [(f - 0.5) * c.hop / c.sample_rate - emission for f, was_voiced in flips if was_voiced == before]
                    near = [t for t in times if abs(t) < 0.15]
                    offsets[(row["speaker"], kind)].append(min(near, key=abs) if near else None)
            previous = (phone, index)
    report = {"utterances": len(rows), "emission_run_lengths": dict(sorted(runs.items())), "transitions": {}}
    for (voice, kind), values in sorted(offsets.items()):
        found = np.array([v for v in values if v is not None]) * 1000
        report["transitions"][f"{voice} {kind}"] = {
            "count": len(values), "flip_found": round(len(found) / len(values), 3), "median_ms": round(float(np.median(found)), 1),
            "mean_ms": round(float(found.mean()), 1), "std_ms": round(float(found.std()), 1),
            "within_11ms_of_median": round(float((np.abs(found - np.median(found)) <= 11).mean()), 3),
            "within_22ms_of_median": round(float((np.abs(found - np.median(found)) <= 22).mean()), 3)}
    clean = [report["transitions"][f"{v} {k}"]["median_ms"] for v in c.speakers for k in ("vowel>fricative", "fricative>vowel")
             if f"{v} {k}" in report["transitions"]]
    report["suggested_onset_offset_seconds"] = round(float(np.mean(clean)) / 1000, 4) if clean else None
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
