"""Vocode WAV files through a trained generator checkpoint (ground-truth mel
-> waveform). Used to measure the vocoder ceiling for the pilot.

Usage:
  python vocode.py --ckpt runs/pilot1/ckpt-120000.pt --list filelist_test.txt \
      --out-dir vocoded_test --limit 20
"""
import argparse
import os

import soundfile as sf
import torch

import mel as melmod
import models


def load_generator(ckpt_path, device):
    gen = models.Generator().to(device)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "generator" in state:
        state = state["generator"]
    gen.load_state_dict(state)
    gen.remove_weight_norm()
    gen.eval()
    return gen


@torch.no_grad()
def vocode_file(gen, wav_path, device):
    wav, _ = sf.read(wav_path, dtype="float32")
    t = torch.tensor(wav, device=device)[None, :]
    m = melmod.mel_spectrogram(t)
    y = gen(m)[0, 0].float().cpu().numpy()
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = load_generator(args.ckpt, device)
    with open(args.list) as f:
        paths = [l.strip() for l in f if l.strip()][:args.limit]
    os.makedirs(args.out_dir, exist_ok=True)
    for p in paths:
        y = vocode_file(gen, p, device)
        name = os.path.splitext(os.path.basename(p))[0]
        sf.write(os.path.join(args.out_dir, name + ".wav"), y, melmod.SR)
        print("vocoded", name, f"{len(y)} samples")
    print("DONE", len(paths))


if __name__ == "__main__":
    main()
