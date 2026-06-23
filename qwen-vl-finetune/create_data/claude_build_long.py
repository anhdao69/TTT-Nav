#!/usr/bin/env python3
"""
Build per-episode interleaved (Path A) training data for Spatial-TTT VLN.

Input  : disk trajectory frames + the existing train_r2r_rxr.json (for instructions)
Output : train_r2r_rxr_long.json  (one sample per episode/chunk, frames+actions interleaved)
"""
import os
import re
import json
import argparse
from functools import partial
from multiprocessing import Pool
from collections import Counter

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **k): return it

# ----------------------------------------------------------------------
# Defaults (edit or pass via CLI)
# ----------------------------------------------------------------------
DATA_ROOT    = "/mnt/data/vmo-ai-task/anhdh35/JanusVLN"
TRAJ_SUBDIR  = "data/trajectory_data"
SRC_JSON     = os.path.join(DATA_ROOT, "train_r2r_rxr.json")
OUT_JSON     = os.path.join(DATA_ROOT, "train_r2r_rxr_episode.json")
DATASETS     = ["R2R", "RxR"]
SPLIT        = "train"
MAX_FRAMES   = 128          # cap per sample; lower to ~96 if you OOM on 40GB
STRIDE       = 96           # overlap stride for episodes longer than MAX_FRAMES

VALID_ACTIONS = {"MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"}
FRAME_RE = re.compile(r"step_(\d+)_(.+)\.png$")
INSTR_RE = re.compile(r"Your task is to (.*?)\n\s*You should take", re.DOTALL)

# ----------------------------------------------------------------------
# Prompt templates  (KEEP IN SYNC WITH INFERENCE!)
# ----------------------------------------------------------------------
FIRST_TURN = (
    "You are a visual-language navigation agent moving through an indoor "
    "environment. Follow the instruction below. At every step you receive one "
    "new observation image and must output exactly one action.\n"
    "Instruction: {instruction}\n"
    "Action space: MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, STOP.\n"
    "Observation: <image>"
)
NEXT_TURN = "Observation: <image>"


def episode_key_from_path(path: str) -> str | None:
    """Last 3 dir components -> 'R2R/train/1' (works for disk dir or json image path)."""
    d = os.path.dirname(path.replace("\\", "/")) if path.endswith(".png") else path.replace("\\", "/")
    parts = [p for p in d.rstrip("/").split("/") if p]
    return "/".join(parts[-3:]) if len(parts) >= 3 else None


def build_instruction_map(src_json: str) -> dict:
    print(f"[1/3] Reading instructions from {src_json} ...")
    with open(src_json, "r") as f:
        data = json.load(f)
    imap = {}
    no_match = 0
    for item in data:
        imgs = item.get("images") or item.get("image")
        if not imgs:
            continue
        if isinstance(imgs, str):
            imgs = [imgs]
        key = episode_key_from_path(imgs[0])
        if key is None or key in imap:
            continue
        human = next((t.get("value", "") for t in item.get("conversations", [])
                      if t.get("from") == "human"), "")
        m = INSTR_RE.search(human)
        if m:
            imap[key] = m.group(1).strip()
        else:
            no_match += 1
    print(f"      episodes with instruction: {len(imap)}  (regex misses: {no_match})")
    return imap


def make_chunks(n: int, max_frames: int, stride: int):
    """Return list of (start, end). One chunk if short; overlapping windows if long."""
    if n <= max_frames:
        return [(0, n)]
    chunks, start = [], 0
    while start < n:
        end = min(start + max_frames, n)
        chunks.append((start, end))
        if end >= n:
            break
        start += stride
    return chunks


def process_episode(task, max_frames, stride, data_root):
    ep_dir, instruction = task
    # collect + parse frames
    frames = []
    for fn in os.listdir(ep_dir):
        m = FRAME_RE.search(fn)
        if not m:
            continue
        idx, act = int(m.group(1)), m.group(2)
        if act in VALID_ACTIONS:
            frames.append((idx, act, fn))
    if not frames:
        return [], "no_valid_frames"
    frames.sort(key=lambda x: x[0])

    rel_dir = os.path.relpath(ep_dir, data_root).replace("\\", "/")   # data/trajectory_data/R2R/train/1
    key = episode_key_from_path(ep_dir)
    chunks = make_chunks(len(frames), max_frames, stride)

    samples = []
    for (s, e) in chunks:
        seg = frames[s:e]
        convs, imgs = [], []
        for j, (idx, act, fn) in enumerate(seg):
            imgs.append(f"{rel_dir}/{fn}")
            human = FIRST_TURN.format(instruction=instruction) if j == 0 else NEXT_TURN
            convs.append({"from": "human", "value": human})
            convs.append({"from": "gpt", "value": act})
        sid = key if len(chunks) == 1 else f"{key}::{s}-{e}"
        samples.append({"id": sid, "conversations": convs, "images": imgs})
    return samples, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=DATA_ROOT)
    ap.add_argument("--traj_subdir", default=TRAJ_SUBDIR)
    ap.add_argument("--src_json", default=SRC_JSON)
    ap.add_argument("--out_json", default=OUT_JSON)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--num_workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0, help="debug: only N episodes")
    args = ap.parse_args()

    imap = build_instruction_map(args.src_json)

    # enumerate episode dirs from disk
    print("[2/3] Enumerating episode folders ...")
    ep_dirs = []
    for ds in args.datasets:
        base = os.path.join(args.data_root, args.traj_subdir, ds, args.split)
        if not os.path.isdir(base):
            print(f"      WARN: missing {base}")
            continue
        for name in os.listdir(base):
            d = os.path.join(base, name)
            if os.path.isdir(d):
                ep_dirs.append(d)
    print(f"      found {len(ep_dirs)} episode folders")

    # attach instruction; drop episodes with none
    tasks, missing = [], 0
    for d in ep_dirs:
        instr = imap.get(episode_key_from_path(d))
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

    # stats
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