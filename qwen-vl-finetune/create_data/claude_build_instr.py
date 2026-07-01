#!/usr/bin/env python3
"""
Cách B — build per-episode interleaved training data where the INSTRUCTION is
repeated in EVERY turn (not only the first).

Why:
  In the current data (claude_build_long.py) the instruction appears once, at the
  first turn. With a 2048-token sliding window (~12 frames), the instruction
  scrolls out of attention and the model must rely on LaCT/anchor memory to keep
  the task in mind. Repeating a lightweight instruction reminder on every turn
  guarantees the task specification is always inside the recent attention window,
  so even a purely windowed model never "forgets" what to do on long episodes.

Output schema is identical to claude_build_long.py:
    {"id": ..., "conversations": [human/gpt ...], "images": [...]}

IMPORTANT — keep train/eval in sync:
  If you retrain on the data produced here, you MUST change the evaluator's
  NEXT_TURN in src/evaluation_spatial_ttt_nav.py to match:

      NEXT_TURN = "Instruction: {instruction}\nObservation: <image>"

  and format it with the instruction each step:

      turn_text = (FIRST_TURN.format(instruction=instruction) if is_first_step
                   else NEXT_TURN.format(instruction=instruction))

  (The current evaluator uses a bare "Observation: <image>" for NEXT_TURN, which
  matches the OLD data only.)

Usage:
    python claude_build_instr.py \
        --data_root /mnt/data/vmo-ai-task/anhdh35/JanusVLN \
        --src_json  /mnt/data/vmo-ai-task/anhdh35/JanusVLN/train_r2r_rxr.json \
        --out_json  /mnt/data/vmo-ai-task/anhdh35/JanusVLN/train_r2r_rxr_episode_instr.json
"""
import os
import json
import argparse
from functools import partial
from multiprocessing import Pool
from collections import Counter

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **k):
        return it

# Reuse all the parsing / instruction-map / chunking helpers from the original
# builder so the two scripts cannot silently diverge on episode enumeration.
import claude_build_long as base

# ----------------------------------------------------------------------
# Prompt templates  (KEEP IN SYNC WITH INFERENCE!)
# ----------------------------------------------------------------------
FIRST_TURN = base.FIRST_TURN                                   # full preamble + instruction
NEXT_TURN = "Instruction: {instruction}\nObservation: <image>"  # NEW: instruction every step


def process_episode(task, max_frames, stride, data_root):
    ep_dir, instruction = task
    frames = []
    for fn in os.listdir(ep_dir):
        m = base.FRAME_RE.search(fn)
        if not m:
            continue
        idx, act = int(m.group(1)), m.group(2)
        if act in base.VALID_ACTIONS:
            frames.append((idx, act, fn))
    if not frames:
        return [], "no_valid_frames"
    frames.sort(key=lambda x: x[0])

    rel_dir = os.path.relpath(ep_dir, data_root).replace("\\", "/")
    key = base.episode_key_from_path(ep_dir)
    chunks = base.make_chunks(len(frames), max_frames, stride)

    samples = []
    for (s, e) in chunks:
        seg = frames[s:e]
        convs, imgs = [], []
        for j, (idx, act, fn) in enumerate(seg):
            imgs.append(f"{rel_dir}/{fn}")
            # difference vs claude_build_long: NEXT_TURN also carries the instruction
            human = (FIRST_TURN.format(instruction=instruction) if j == 0
                     else NEXT_TURN.format(instruction=instruction))
            convs.append({"from": "human", "value": human})
            convs.append({"from": "gpt", "value": act})
        sid = key if len(chunks) == 1 else f"{key}::{s}-{e}"
        samples.append({"id": sid, "conversations": convs, "images": imgs})
    return samples, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=base.DATA_ROOT)
    ap.add_argument("--traj_subdir", default=base.TRAJ_SUBDIR)
    ap.add_argument("--src_json", default=base.SRC_JSON)
    ap.add_argument("--out_json",
                    default=os.path.join(base.DATA_ROOT, "train_r2r_rxr_episode_instr.json"))
    ap.add_argument("--datasets", nargs="+", default=base.DATASETS)
    ap.add_argument("--split", default=base.SPLIT)
    ap.add_argument("--max_frames", type=int, default=base.MAX_FRAMES)
    ap.add_argument("--stride", type=int, default=base.STRIDE)
    ap.add_argument("--num_workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0, help="debug: only N episodes")
    args = ap.parse_args()

    imap = base.build_instruction_map(args.src_json)

    print("[2/3] Enumerating episode folders ...")
    ep_dirs = []
    for ds in args.datasets:
        base_dir = os.path.join(args.data_root, args.traj_subdir, ds, args.split)
        if not os.path.isdir(base_dir):
            print(f"      WARN: missing {base_dir}")
            continue
        for name in os.listdir(base_dir):
            d = os.path.join(base_dir, name)
            if os.path.isdir(d):
                ep_dirs.append(d)
    print(f"      found {len(ep_dirs)} episode folders")

    tasks, missing = [], 0
    for d in ep_dirs:
        instr = imap.get(base.episode_key_from_path(d))
        if instr is None:
            missing += 1
            continue
        tasks.append((d, instr))
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"      usable episodes: {len(tasks)}  (no instruction: {missing})")

    print(f"[3/3] Building samples with {args.num_workers} workers ...")
    worker = partial(process_episode, max_frames=args.max_frames,
                     stride=args.stride, data_root=args.data_root)
    all_samples, status = [], Counter()
    with Pool(args.num_workers) as pool:
        for samples, st in tqdm(pool.imap_unordered(worker, tasks, chunksize=16),
                                total=len(tasks)):
            status[st] += 1
            all_samples.extend(samples)

    lens = [len(s["images"]) for s in all_samples]
    buckets = Counter()
    for L in lens:
        buckets[("1-8" if L <= 8 else "9-32" if L <= 32 else
                 "33-64" if L <= 64 else "65-128" if L <= 128 else ">128")] += 1
    print("\n================ SUMMARY ================")
    print(f"episodes ok / no_frames : {status['ok']} / {status['no_valid_frames']}")
    print(f"output samples          : {len(all_samples)}")
    if lens:
        print(f"frames/sample min/mean/max: {min(lens)} / {sum(lens)/len(lens):.1f} / {max(lens)}")
        print(f"length buckets          : {dict(buckets)}")
    print(f"writing -> {args.out_json}")
    with open(args.out_json, "w") as f:
        json.dump(all_samples, f, ensure_ascii=False)
    print("done.")


if __name__ == "__main__":
    main()