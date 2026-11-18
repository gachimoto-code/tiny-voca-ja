"""Offline phoneme-level forced alignment. Annotation time only.

A pretrained multilingual phoneme recogniser (wav2vec 2.0 fine-tuned with CTC on
eSpeak phoneme labels, ``facebook/wav2vec2-lv-60-espeak-cv-ft``) labels the
existing recordings. It is used ONLY to annotate training data: none of its
weights, features or outputs enter the TTS model, which never sees it.

Why this replaces the learned Gaussian aligner: that objective normalised scores
over the token positions of one utterance, so a frequent token whose key matched
the average spectrum (first BOS, later the word spaces) won the softmax on most
frames and every other token was left with single frames. Here each frame is
scored against phoneme classes shared by the whole language, and the transcript
only constrains their order.

Per utterance:
1. CTC Viterbi alignment of the eSpeak phoneme sequence (20 ms frames).
2. This recogniser emits a label once, at the START of the phone (measured on
   our recordings against pitch-tracker voicing flips: a phone begins about
   19 ms before the start of its first emission frame, for both voices; see
   ``scripts/calibrate_aligner_timing.py``). A phone therefore lasts from its
   own onset to the next phone's onset.
3. Silence is found from signal energy at acoustic resolution and given to the
   explicit pause-capable token at that position (BOS, EOS, punctuation, word
   boundary). Tokens without a pause get zero frames.
4. Durations are counted on the acoustic frame grid, so they sum to the frame
   count exactly and every phoneme has at least one frame.

``refine.py`` can then sharpen the boundaries with a classifier trained from
scratch on these labels.

Alignment version 3 adds a second recogniser for Japanese:
``prj-beatrice/japanese-hubert-base-phoneme-ctc`` (HuBERT-base fine-tuned with
CTC on ReazonSpeech with pyopenjtalk-plus phoneme labels). The eSpeak
recogniser above never learned Japanese acoustics -- on this corpus its
per-phone posterior means sat near 0.27 and every utterance failed QC -- while
the beatrice recogniser measures 0.92 on the same recordings. Its label
inventory differs from the front end's, so the flow is: g2p the text with
pyopenjtalk-plus, force-align that label sequence, then transfer the phone
spans to the front end's token sequence with a deterministic mora-level
alignment (``transfer_groups``) and feed ``assemble`` pseudo-spans. Utterances
whose two readings genuinely diverge (different kanji readings, rendaku) do
not transfer and are rejected, which is the honest failure mode. QC
probabilities are measured in the beatrice label space; every other check is
unchanged. The eSpeak path is kept intact for rollback and comparison.
"""

from dataclasses import dataclass
import math
import numpy as np

from .text import PHONE_SET, PUNCTUATION, SP, UNK, VOICELESS, VOWEL_SET

ALIGNER_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
BEATRICE_MODEL = "prj-beatrice/japanese-hubert-base-phoneme-ctc-v2"
BEATRICE_REVISION = "f38383fcd1871577f8bb26d7007452916bd2db9d"
ALIGNER_RATE = 16000
ALIGNER_STRIDE = 320          # samples at 16 kHz: 20 ms
ALIGNER_CENTER = 200          # a frame's centre: 12.5 ms after its start
ALIGNMENT_VERSION = 3         # 3: beatrice recogniser + span transfer added
PAUSING = (",", ".", "?", "!", ";", ":", "—", "…")
# Every inventory phone is a label of the recogniser (checked when it loads);
# only the unknown-phone token needs a stand-in.
ALIGNER_LABEL_FALLBACK = {UNK: "<unk>"}


@dataclass(frozen=True)
class AlignmentSettings:
    onset_offset_seconds: float = -0.019   # phone onset relative to the start of its first emission frame
    min_pause_frames: int = 7              # ~75 ms of silence: less stays with the neighbouring phones (stop closures)
    silence_below_peak_db: float = 45.0
    silence_above_floor_db: float = 8.0
    silence_ceiling_below_peak_db: float = 30.0
    energy_window: int = 512
    aligner_model: str = ALIGNER_MODEL     # which recogniser these settings belong to

    def to_dict(self):
        values = {key: value for key, value in self.__dict__.items() if key != "aligner_model"}
        return {"alignment_version": ALIGNMENT_VERSION, "aligner": self.aligner_model, **values}


def ctc_viterbi(log_probs, labels, blank=0):
    """Best CTC path. Returns per-frame extended-state indices (token k is state 2k+1).

    Plain numpy reference, also used when torchaudio's implementation is absent.
    """
    frames, count = log_probs.shape[0], len(labels)
    if count < 1:
        raise ValueError("no labels to align")
    states = 2 * count + 1
    extended = np.full(states, blank, dtype=np.int64)
    extended[1::2] = labels
    skip = np.zeros(states, dtype=bool)
    skip[3::2] = extended[3::2] != extended[1:-2:2]
    score = np.full(states, -np.inf)
    score[0] = log_probs[0, blank]
    score[1] = log_probs[0, extended[1]]
    back = np.zeros((frames, states), dtype=np.int8)
    for t in range(1, frames):
        one = np.concatenate(([-np.inf], score[:-1]))
        two = np.concatenate(([-np.inf, -np.inf], score[:-2]))
        two[~skip] = -np.inf
        options = np.stack((score, one, two))
        choice = options.argmax(0)
        score = options[choice, np.arange(states)] + log_probs[t, extended]
        back[t] = choice
    state = states - 1 if score[states - 1] >= score[states - 2] else states - 2
    if not np.isfinite(score[state]):
        raise ValueError("audio is too short for this phoneme sequence")
    path = np.empty(frames, dtype=np.int64)
    for t in range(frames - 1, -1, -1):
        path[t] = state
        state -= back[t, state]
    return path


def spans_from_states(states, count):
    spans = []
    for index in range(count):
        where = np.flatnonzero(states == 2 * index + 1)
        if not where.size:
            raise ValueError("forced alignment did not visit every phoneme")
        spans.append((int(where[0]), int(where[-1])))
    return spans


def token_spans(log_probs, labels, blank=0):
    """Inclusive ``(first, last)`` emission frames of every label on the best CTC path."""
    count = len(labels)
    repeats = sum(1 for a, b in zip(labels[:-1], labels[1:]) if a == b)
    if log_probs.shape[0] < count + repeats:
        raise ValueError("audio is too short for this phoneme sequence")
    try:
        import torch
        from torchaudio.functional import forced_align
    except ImportError:
        return spans_from_states(ctc_viterbi(log_probs, np.asarray(labels), blank), count)
    path, _ = forced_align(torch.from_numpy(np.ascontiguousarray(log_probs, dtype=np.float32))[None],
                           torch.tensor([list(labels)], dtype=torch.long), blank=blank)
    spans, previous = [], blank
    for frame, label in enumerate(path[0].tolist()):
        if label != blank and (previous == blank or label != previous):
            spans.append([frame, frame])
        elif label != blank:
            spans[-1][1] = frame
        previous = label
    if len(spans) != count:
        raise ValueError("forced alignment did not visit every phoneme")
    return [(a, b) for a, b in spans]


def frame_energy_db(wave, hop, window=512):
    """RMS level per acoustic frame, window centred on the frame centre i*hop."""
    frames = len(wave) // hop
    padded = np.pad(np.asarray(wave, dtype=np.float64), (window // 2, window // 2))
    squares = np.concatenate(([0.0], np.cumsum(padded ** 2)))
    starts = np.arange(frames) * hop
    mean = (squares[starts + window] - squares[starts]) / window
    return 10 * np.log10(np.maximum(mean, 1e-12))


def silence_mask(level_db, settings):
    loud, floor = np.percentile(level_db, 95), np.percentile(level_db, 2)
    threshold = min(loud - settings.silence_ceiling_below_peak_db,
                    max(loud - settings.silence_below_peak_db, floor + settings.silence_above_floor_db))
    return level_db < threshold, float(threshold)


def pause_carrier(symbols, group):
    """Index of the token that receives a pause inside a run of non-phone tokens."""
    for wanted in (PAUSING, PUNCTUATION, (SP,)):
        for index in group:
            if symbols[index] in wanted:
                return index
    return group[0]


def assemble(symbols, spans, frames, silent, hop, sample_rate, settings):
    """Durations in acoustic frames for every token (sums to ``frames``).

    Acoustic frame i is centred on sample ``i * hop`` (the convention of the
    stored mel/pitch features and of the synthesis frames).
    """
    phones = [i for i, s in enumerate(symbols) if s in PHONE_SET or s == UNK]
    count = len(phones)
    if count != len(spans) or count < 1:
        raise ValueError("phoneme count differs from alignment")
    stride = ALIGNER_STRIDE / ALIGNER_RATE
    emission = np.array([first for first, _ in spans], dtype=np.float64) * stride
    seconds = np.arange(frames) * hop / sample_rate
    # One protected frame per phone at its emission centre; phones cannot lose it.
    keep = np.clip(np.round((emission + ALIGNER_CENTER / ALIGNER_RATE) * sample_rate / hop).astype(np.int64), 0, frames - 1)
    for k in range(1, count):
        keep[k] = max(keep[k], keep[k - 1] + 1)
    if keep[-1] > frames - 1:
        raise ValueError("audio is too short for this phoneme sequence")
    starts = np.minimum(np.searchsorted(seconds, emission + settings.onset_offset_seconds, side="left"), keep)
    starts[0] = 0
    for k in range(1, count):
        starts[k] = max(starts[k], keep[k - 1] + 1)
    token = np.empty(frames, dtype=np.int64)
    bounds = np.append(starts, frames)
    for k in range(count):
        token[bounds[k]:bounds[k + 1]] = phones[k]
    # A pause runs from the first to the last silent frame between two protected
    # frames. Whatever lies between them (a breath, a stop release) is not part
    # of either phone's steady sound and stays with the pause.
    quiet = np.flatnonzero(silent[:keep[0]])
    if quiet.size:                                      # leading silence -> BOS
        token[:quiet[-1] + 1] = 0
    quiet = np.flatnonzero(silent[keep[-1] + 1:]) + keep[-1] + 1
    if quiet.size:                                      # trailing silence -> EOS
        token[quiet[0]:] = len(symbols) - 1
    for k in range(count - 1):                          # interior pauses -> pause-capable token
        left, right = phones[k], phones[k + 1]
        if right == left + 1:
            continue
        quiet = np.flatnonzero(silent[keep[k] + 1:keep[k + 1]]) + keep[k] + 1
        if quiet.size < settings.min_pause_frames:
            continue
        token[keep[k] + 1:quiet[0]] = left
        token[quiet[0]:quiet[-1] + 1] = pause_carrier(symbols, range(left + 1, right))
        token[quiet[-1] + 1:keep[k + 1]] = right
    if np.any(np.diff(token) < 0):
        raise ValueError("alignment is not monotonic")
    durations = np.bincount(token, minlength=len(symbols)).astype(np.int64)
    if durations.sum() != frames or np.any(durations[phones] < 1):
        raise ValueError("durations do not cover the acoustic frames")
    return durations


def quality(symbols, durations, spans, log_probs, labels, voiced, level_db, hop, sample_rate, token_probability=None):
    """Label-free checks of one alignment against the recording itself.

    ``token_probability`` may be supplied when the aligned label space is not
    the token space (the beatrice path measures posteriors over its own
    labels); the two probability fields then describe that label space while
    every other field still describes the tokens and their durations.
    """
    durations = np.asarray(durations)
    ends = np.cumsum(durations)
    starts = ends - durations
    if token_probability is None:
        token_probability = np.array([float(np.exp(log_probs[a:b + 1, label]).mean()) for (a, b), label in zip(spans, labels)])
    else:
        token_probability = np.asarray(token_probability, dtype=np.float64)
    phone_frames = np.array([d for s, d in zip(symbols, durations) if s in PHONE_SET or s == UNK])

    def voiced_share(wanted):
        total = hits = 0
        for symbol, start, end in zip(symbols, starts, ends):
            if wanted(symbol) and end - start >= 3:      # interior frames only
                total += end - start - 2
                hits += float(voiced[start + 1:end - 1].sum())
        return round(hits / total, 4) if total else None

    pause = np.zeros(len(voiced), dtype=bool)
    for symbol, start, end in zip(symbols, starts, ends):
        if symbol not in PHONE_SET and symbol != UNK:
            pause[start:end] = True
    speech_seconds = float((~pause).sum() * hop / sample_rate)
    return {
        "token_probability_mean": round(float(token_probability.mean()), 4),
        "weak_phone_fraction": round(float((token_probability < 0.1).mean()), 4),
        "max_phone_frames": int(phone_frames.max()),
        "one_frame_phone_fraction": round(float((phone_frames == 1).mean()), 4),
        "phones_per_second": round(len(phone_frames) / max(speech_seconds, 1e-6), 2),
        "pause_frames": int(pause.sum()),
        "leading_frames": int(durations[0]),
        "trailing_frames": int(durations[-1]),
        "vowel_voiced": voiced_share(lambda s: s in VOWEL_SET),
        "voiceless_voiced": voiced_share(lambda s: s in VOICELESS),
        "pause_voiced": round(float(voiced[pause].mean()), 4) if pause.any() else None,
        "pause_level_db": round(float(level_db[pause].mean()), 1) if pause.any() else None,
        "speech_level_db": round(float(np.percentile(level_db[~pause], 95)), 1) if (~pause).any() else None,
    }


class PhoneRecognizer:
    """The pretrained annotation model. Loaded from the local HF cache only."""

    def __init__(self, device="cuda", name=ALIGNER_MODEL, local_files_only=True):
        import json
        from pathlib import Path
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2ForCTC
        self.torch, self.device = torch, torch.device(device)
        self.model = Wav2Vec2ForCTC.from_pretrained(name, local_files_only=local_files_only).to(self.device).eval()
        if self.device.type == "cuda":
            self.model = self.model.half()
        vocabulary = Path(hf_hub_download(name, "vocab.json", local_files_only=local_files_only))
        self.vocabulary = json.loads(vocabulary.read_text(encoding="utf-8"))
        self.blank = int(self.model.config.pad_token_id)
        self.name = name
        self.revision = vocabulary.parent.name   # snapshot directory = commit hash
        missing = [p for p in sorted(PHONE_SET) if p not in self.vocabulary and p not in ALIGNER_LABEL_FALLBACK]
        if missing:
            raise ValueError(f"recogniser has no label for inventory phones {missing}")

    def label(self, symbol):
        symbol = symbol if symbol in self.vocabulary else ALIGNER_LABEL_FALLBACK[symbol]
        return int(self.vocabulary[symbol])

    def log_probs(self, wave, sample_rate):
        from scipy.signal import resample_poly
        torch = self.torch
        factor = math.gcd(sample_rate, ALIGNER_RATE)
        audio = resample_poly(np.asarray(wave, dtype=np.float64), ALIGNER_RATE // factor, sample_rate // factor)
        audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)
        with torch.inference_mode():
            values = torch.from_numpy(audio.astype(np.float32))[None].to(self.device, self.model.dtype)
            return self.model(values).logits[0].float().log_softmax(-1).cpu().numpy()


def align_utterance(recognizer, phonemized, wave, voiced, config, settings, text=None):
    """Return ``(durations, qc)`` for one recording or raise ``ValueError``."""
    if isinstance(recognizer, BeatriceRecognizer):
        if text is None:
            raise ValueError("the beatrice aligner needs the utterance text")
        return _align_utterance_beatrice(recognizer, phonemized, text, wave, voiced, config, settings)
    symbols = phonemized.symbols
    frames = len(wave) // config.hop
    if len(wave) != frames * config.hop or len(voiced) != frames:
        raise ValueError("waveform length must be frames * hop")
    labels = [recognizer.label(s) for s in symbols if s in PHONE_SET or s == UNK]
    log_probs = recognizer.log_probs(wave, config.sample_rate)
    spans = token_spans(log_probs, labels, recognizer.blank)
    level = frame_energy_db(wave, config.hop, settings.energy_window)
    silent, threshold = silence_mask(level, settings)
    durations = assemble(symbols, spans, frames, silent, config.hop, config.sample_rate, settings)
    qc = quality(symbols, durations, spans, log_probs, labels, np.asarray(voiced) > 0.5, level, config.hop, config.sample_rate)
    qc["silence_threshold_db"] = round(threshold, 1)
    return durations.tolist(), qc


# ---------------------------------------------------------------------------
# Beatrice path: Japanese phoneme CTC (HuBERT) + span transfer to the front
# end's tokens. Added in alignment version 3; the eSpeak path above is the
# original and stays available for rollback.
# ---------------------------------------------------------------------------

# Onset offset for the beatrice recogniser, measured with the same voicing-flip
# method as the eSpeak value (scripts/calibrate_beatrice_timing.py, both
# voices, 100 validation utterances, 2026-11-18): vowel->fricative medians
# -33.3/-28.0 ms, fricative->vowel +10.7/+22.7 ms, whose mean is -7.0 ms; the
# stop transitions bracket -12 ms, corroborating a much smaller offset than
# the eSpeak recogniser's -19 ms.
BEATRICE_ONSET_OFFSET_SECONDS = -0.007

# beatrice label -> front-end phones it can realise as a single token.
BEATRICE_MAP1 = {
    "a": ("æ",), "i": ("i",), "u": ("ɯᵝ",), "e": ("eɪ",), "o": ("oʊ",),
    "I": ("i",), "U": ("ɯᵝ",),                       # devoiced vowels fold onto the plain vowel
    "k": ("k",), "g": ("ɡ",), "s": ("s",), "z": ("z",), "t": ("t",), "d": ("d",),
    "n": ("n",), "h": ("h",), "b": ("b",), "p": ("p",), "m": ("m",),
    "y": ("j",), "r": ("ɽ",), "w": ("w", "h"),       # w: particle は read "ha" by the front end
    "f": ("ɸ", "f"), "j": ("dʑ",), "v": ("v",),
    "sh": ("ɕ",), "ch": ("tɕ",), "ky": ("kʲ", "k"),
    # A palatalised label can also surface as its plain consonant when the
    # front end does not palatalise (e.g. hy -> h in ひょう).
    "gy": ("ɡ",), "hy": ("h",), "by": ("b",), "py": ("p",), "my": ("m",),
    "ny": ("n",), "ry": ("ɽ",), "fy": ("ɸ", "f"), "dy": ("d",),
    "kw": ("k",), "gw": ("ɡ",),
    "N": ("ɴ", "ŋ", "n", "uː"),                     # moraic nasal, context form chosen by the front end
    "cl": ("ʔ",),                                    # stray geminate realised as a glottal stop
}
# beatrice label -> front-end phone pairs it can realise.
BEATRICE_MAP2 = {
    "ts": (("t", "s"),),
    "gy": (("ɡ", "j"),), "hy": (("h", "j"),), "by": (("b", "j"),), "py": (("p", "j"),),
    "my": (("m", "j"),), "ny": (("n", "j"),), "ry": (("ɽ", "j"),),
    "fy": (("ɸ", "j"), ("f", "j")), "dy": (("d", "j"),),
    "kw": (("k", "w"),), "gw": (("ɡ", "w"),),
    "sh": (("ɕ", "j"),), "ch": (("tɕ", "j"),), "j": (("dʑ", "j"),),
    # Particle へ: pyopenjtalk reads it as a bare "e"; the front end sometimes
    # keeps the h when its segment-level particle fix missed it.
    "e": (("h", "eɪ"),),
}
# beatrice pair -> single front-end token (long-vowel merges).
BEATRICE_MERGE21 = {("i", "i"): ("iː",), ("u", "u"): ("uː",), ("o", "o"): ("oː",), ("a", "a"): ("ɑː",)}
# beatrice pair -> front-end pair, beyond two 1:1 steps (おう / えい spellings).
BEATRICE_BLOCK22 = {("o", "o"): (("oʊ", "ɯᵝ"),), ("e", "e"): (("eɪ", "i"),)}
# Base consonant of a front-end phone, for the geminate copy of palatal units.
BEATRICE_BASE = {"kʲ": "k", "tɕ": "t"}


def _beatrice_images(label):
    images = [(token,) for token in BEATRICE_MAP1.get(label, ())]
    images.extend(BEATRICE_MAP2.get(label, ()))
    return images


def transfer_groups(beatrice_labels, phones):
    """Align a beatrice label sequence to the front end's phone sequence.

    Returns a list of ``(beatrice_indices, phone_indices)`` groups partitioning
    both sequences, or raises ``ValueError`` when no deterministic alignment
    exists (the two readings of the text genuinely diverge). Allowed steps are
    the measured correspondences in the tables above plus the geminate rule:
    ``(cl, X)`` realises as ``(head, *image(X))`` where head is ``ʔ`` or the
    base consonant of X's first image phone.
    """
    bea, tiny = list(beatrice_labels), list(phones)
    n, m = len(bea), len(tiny)
    back = [[None] * (m + 1) for _ in range(n + 1)]
    back[0][0] = ()
    for i in range(n + 1):
        for j in range(m + 1):
            if back[i][j] is None:
                continue
            if i < n:
                for image in _beatrice_images(bea[i]):
                    nj = j + len(image)
                    if nj <= m and tuple(tiny[j:nj]) == image:
                        back[i + 1][nj] = (i, j, ((i,), tuple(range(j, nj))))
            if i + 2 <= n:
                pair = (bea[i], bea[i + 1])
                if pair in BEATRICE_MERGE21 and j < m and tiny[j] in BEATRICE_MERGE21[pair]:
                    back[i + 2][j + 1] = (i, j, ((i, i + 1), (j,)))
                if pair in BEATRICE_BLOCK22:
                    for image in BEATRICE_BLOCK22[pair]:
                        if tuple(tiny[j:j + 2]) == image:
                            back[i + 2][j + 2] = (i, j, ((i, i + 1), (j, j + 1)))
                if bea[i] == "cl":
                    for image in _beatrice_images(bea[i + 1]):
                        for head in {"ʔ", BEATRICE_BASE.get(image[0], image[0])}:
                            target = (head,) + image
                            nj = j + len(target)
                            if nj <= m and tuple(tiny[j:nj]) == target:
                                back[i + 2][nj] = (i, j, ((i, i + 1), tuple(range(j, nj))))
    if back[n][m] is None:
        raise ValueError("beatrice labels do not transfer to the front-end phones (readings diverge)")
    groups, i, j = [], n, m
    while (i, j) != (0, 0):
        i, j, group = back[i][j][0], back[i][j][1], back[i][j][2]
        groups.append(group)
    return groups[::-1]


def _transferred_spans(groups, spans, total_frames):
    """Pseudo-spans in aligner frames for every front-end phone.

    A group's aligner-frame range (its first label's onset to the next group's
    onset) is divided among its phones as evenly as possible; ``assemble``
    only reads the first element and enforces one protected acoustic frame per
    phone, so equal onsets inside a short group are safe.
    """
    onsets = [spans[group[0][0]][0] for group in groups] + [total_frames]
    pseudo = []
    for index, (_, phones) in enumerate(groups):
        start, end = onsets[index], onsets[index + 1]
        count = len(phones)
        for k in range(count):
            first = start + (end - start) * k // count
            last = start + (end - start) * (k + 1) // count - 1
            pseudo.append((first, max(first, last)))
    return pseudo


def beatrice_label_sequence(text):
    """pyopenjtalk-plus g2p of the raw text, pau/sil excluded (the measured
    configuration; pauses are placed acoustically by ``assemble``)."""
    import pyopenjtalk
    return [phone for phone in pyopenjtalk.g2p(text).split() if phone not in ("pau", "sil")]


def beatrice_labeler_version():
    from importlib.metadata import version
    return version("pyopenjtalk-plus")


class BeatriceRecognizer:
    """Japanese phoneme CTC recogniser (HuBERT-base). Local HF cache only,
    model and vocabulary pinned as a pair by revision."""

    def __init__(self, device="cuda", name=BEATRICE_MODEL, revision=BEATRICE_REVISION, local_files_only=True):
        import json
        from pathlib import Path
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForCTC
        self.torch, self.device = torch, torch.device(device)
        self.model = AutoModelForCTC.from_pretrained(name, revision=revision, local_files_only=local_files_only).to(self.device).eval()
        if self.device.type == "cuda":
            self.model = self.model.half()
        vocabulary = Path(hf_hub_download(name, "vocab.json", revision=revision, local_files_only=local_files_only))
        self.vocabulary = json.loads(vocabulary.read_text(encoding="utf-8"))
        self.blank = int(self.vocabulary["PAD"])
        if self.blank != int(self.model.config.pad_token_id):
            raise ValueError("beatrice vocabulary PAD does not match the model config")
        self.name = name
        self.revision = vocabulary.parent.name   # snapshot directory = commit hash

    def label(self, symbol):
        if symbol not in self.vocabulary:
            raise ValueError(f"beatrice recogniser has no label for {symbol!r}")
        return int(self.vocabulary[symbol])

    def log_probs(self, wave, sample_rate):
        from scipy.signal import resample_poly
        torch = self.torch
        factor = math.gcd(sample_rate, ALIGNER_RATE)
        audio = resample_poly(np.asarray(wave, dtype=np.float64), ALIGNER_RATE // factor, sample_rate // factor)
        audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)
        with torch.inference_mode():
            values = torch.from_numpy(audio.astype(np.float32))[None].to(self.device, self.model.dtype)
            return self.model(values).logits[0].float().log_softmax(-1).cpu().numpy()


def _align_utterance_beatrice(recognizer, phonemized, text, wave, voiced, config, settings):
    """Beatrice flow: g2p labels -> forced alignment -> span transfer ->
    the shared ``assemble``/``quality`` machinery on pseudo-spans."""
    symbols = phonemized.symbols
    frames = len(wave) // config.hop
    if len(wave) != frames * config.hop or len(voiced) != frames:
        raise ValueError("waveform length must be frames * hop")
    phones = [s for s in symbols if s in PHONE_SET or s == UNK]
    if UNK in phones:
        raise ValueError("front end produced an unknown phoneme; no beatrice label can carry it")
    beatrice_labels = beatrice_label_sequence(text)
    labels = [recognizer.label(p) for p in beatrice_labels]
    log_probs = recognizer.log_probs(wave, config.sample_rate)
    spans = token_spans(log_probs, labels, recognizer.blank)
    groups = transfer_groups(beatrice_labels, phones)
    pseudo = _transferred_spans(groups, spans, log_probs.shape[0])
    token_probability = np.array([float(np.exp(log_probs[a:b + 1, label]).mean())
                                  for (a, b), label in zip(spans, labels)])
    level = frame_energy_db(wave, config.hop, settings.energy_window)
    silent, threshold = silence_mask(level, settings)
    durations = assemble(symbols, pseudo, frames, silent, config.hop, config.sample_rate, settings)
    qc = quality(symbols, durations, pseudo, log_probs, labels, np.asarray(voiced) > 0.5, level,
                 config.hop, config.sample_rate, token_probability=token_probability)
    qc["silence_threshold_db"] = round(threshold, 1)
    return durations.tolist(), qc
